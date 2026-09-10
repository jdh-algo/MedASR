"""
Qwen3-ASR worker(每卡一个进程,CUDA_VISIBLE_DEVICES 钉卡,仅绑 127.0.0.1,无鉴权)。
一个 vLLM 进程同端口同时服务:
    POST /offline {path, lang} -> {lang, detected_lang, text, segments, has_speaker, duration, engine}
    WS   /ws/asr               -> 流式(建会话→喂 Int16 PCM→增量文本;协议见 README.md)
    GET  /health
并发:micro_batch=false(默认)时 vLLM 同步 generate 非线程安全,靠 loop 级 _gen_lock + engine._MODEL_LOCK
双锁全进程串行(每卡并发=1);micro_batch=true 时改由 engine 内单 batcher 线程独占 generate、拆双锁,
本层放开并发喂(多会话 feed 并行入 batcher 才有得批),并把线程池上限提到 gen_threadpool。
启动:CUDA_VISIBLE_DEVICES=0 MEDASR_WORKER_PORT=9300 python -m asrsvc.worker.qwen_app
"""
import asyncio
import contextlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from asrsvc import config
from asrsvc.engines import qwen as qwen_engine
from asrsvc.worker import audio_guard

SPOOL = Path(config.MEDASR_SPOOL_DIR).resolve()
JOB_TIMEOUT = config.MEDASR_JOB_TIMEOUT
MAX_BYTES = int(config.MEDASR_DOWNLOAD_CONFIG.get("max_bytes", 0) or 0)
MAX_SECONDS = int(config.MEDASR_LIMITS.get("max_audio_seconds", 0) or 0)
SESSION_MAX_BYTES = int(config.MEDASR_LIMITS.get("ws_session_max_bytes", 0) or 0)
IDLE_TIMEOUT = int(config.MEDASR_LIMITS.get("ws_idle_timeout", 0) or 0)

_state = {"model": None, "ready": False, "sessions": 0}
_gen_lock = asyncio.Lock()


def _gen_ctx():
    """generate 串行上下文。micro_batch 关 → _gen_lock(loop 级串行);开 → 空上下文(batcher 已串行)。"""
    return contextlib.nullcontext() if config.MEDASR_MICRO_BATCH else _gen_lock


def _resolve_in_spool(path: str) -> Path:
    try:
        rp = Path(path).resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(400, "路径无效")
    try:
        rp.relative_to(SPOOL)
    except ValueError:
        raise HTTPException(403, "路径越权")
    if not rp.is_file():
        raise HTTPException(404, "文件不存在")
    return rp


@asynccontextmanager
async def lifespan(app):
    print(f"[medasr-worker] 加载 Qwen3-ASR {config.MEDASR_ASR_MODEL}(vLLM, gpu_util={config.MEDASR_GPU_UTIL})…", flush=True)
    if config.MEDASR_MICRO_BATCH:
        anyio.to_thread.current_default_thread_limiter().total_tokens = config.MEDASR_GEN_THREADPOOL
        print(f"[medasr-worker] micro_batch=on(batch_invariant={config.MEDASR_BATCH_INVARIANT}, "
              f"backend={config.MEDASR_ATTENTION_BACKEND}, max_batch={config.MEDASR_MAX_MICRO_BATCH}, "
              f"threadpool={config.MEDASR_GEN_THREADPOOL})", flush=True)
    try:
        _state["model"] = await run_in_threadpool(qwen_engine.build_qwen_model)
        _state["ready"] = True
        print("[medasr-worker] 就绪 ✓(双语单模型,离线+流式)", flush=True)
    except Exception as e:
        print(f"[medasr-worker] 加载失败,保持 not-ready:{type(e).__name__}: {e}", flush=True)
    if config.MEDASR_VAD_ENABLED:
        try:
            from asrsvc.engines import qwen_vad
            t0 = time.time()
            m = await run_in_threadpool(qwen_vad.build_vad_model)
            print(f"[medasr-worker] VAD 预热{'完成' if m else '跳过'}({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[medasr-worker] VAD 预热失败(退回旧判据):{type(e).__name__}: {e}", flush=True)
    yield


app = FastAPI(title="MedASR Worker", lifespan=lifespan)


class OfflineReq(BaseModel):
    path: str
    lang: str = config.MEDASR_DEFAULT_LANG


@app.get("/health")
async def health():
    out = {"ready": _state["ready"], "engine": "medasr", "langs": ["zh", "yue", "auto"],
           "sessions": _state["sessions"]}
    b = getattr(_state["model"], "_batcher", None)
    if b is not None:
        try:
            out.update(b.gpu_stats())
        except Exception:
            pass
    return out


def _do_offline(path: Path, lang: str) -> dict:
    audio_guard.probe(str(path), MAX_BYTES, MAX_SECONDS)
    return qwen_engine.transcribe_file(_state["model"], path, lang=lang)


@app.post("/offline")
async def offline(req: OfflineReq):
    if not _state["ready"]:
        raise HTTPException(503, "模型未就绪")
    path = _resolve_in_spool(req.path)
    try:
        async with _gen_ctx():
            return await asyncio.wait_for(
                run_in_threadpool(_do_offline, path, req.lang), timeout=JOB_TIMEOUT)
    except audio_guard.AudioRejected as e:
        raise HTTPException(422, f"音频不合规:{e}")
    except asyncio.TimeoutError:
        raise HTTPException(504, "识别超时")


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    if not _state["ready"]:
        await ws.send_text(json.dumps({"type": "error", "msg": "模型未就绪"}))
        await ws.close()
        return
    sess = None
    total_bytes = 0

    def ensure(lang):
        nonlocal sess
        if sess is None:
            sess = qwen_engine.QwenStreamSession(_state["model"], lang or config.MEDASR_DEFAULT_LANG)
            _state["sessions"] += 1
        return sess

    await ws.send_text(json.dumps({"type": "ready"}))
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=IDLE_TIMEOUT or None)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "error", "msg": "空闲超时"}))
                break
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is not None:
                total_bytes += len(data)
                if SESSION_MAX_BYTES and total_bytes > SESSION_MAX_BYTES:
                    await ws.send_text(json.dumps({"type": "error", "msg": "会话数据超限"}))
                    break
                ensure(config.MEDASR_DEFAULT_LANG)
                async with _gen_ctx():
                    deltas = await run_in_threadpool(sess.feed_pcm, data)
                for delta in deltas:
                    await ws.send_text(json.dumps({"type": "text", "delta": delta, "final": False}))
                continue
            text = msg.get("text")
            if text is None:
                continue
            try:
                evt = json.loads(text)
            except (ValueError, TypeError):
                evt = {}
            e = evt.get("event") if isinstance(evt, dict) else None
            if e == "start":
                ensure(evt.get("lang") or config.MEDASR_DEFAULT_LANG)
            elif e == "eof":
                if sess is not None:
                    async with _gen_ctx():
                        tail = await run_in_threadpool(sess.finalize)
                    if tail:
                        await ws.send_text(json.dumps({"type": "text", "delta": tail, "final": True}))
                await ws.send_text(json.dumps({"type": "done"}))
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[medasr-worker] 会话异常:{type(e).__name__}: {e}", flush=True)
        try:
            await ws.send_text(json.dumps({"type": "error", "msg": "识别服务内部错误"}))
        except Exception:
            pass
    finally:
        if sess is not None:
            sk = getattr(sess, "overload_skips", 0)
            dp = getattr(sess, "dropped_sec", 0.0)
            if sk or dp:
                print(f"[medasr-worker] 会话降级统计:过载跳过 {sk} 步,丢弃音频 {dp:.1f}s", flush=True)
            hg = getattr(sess, "head_gated_sec", 0.0)
            if hg:
                print(f"[medasr-worker] 开头门控:挡掉开头 {hg:.1f}s 非语音(静音/回铃),未喂模型", flush=True)
            rb = getattr(sess, "relatch_breaks", 0)
            print(f"[medasr-worker] 会话解闩统计:回缩按住 {getattr(sess, 'shrink_holds', 0)} 步,"
                  f"解闩 {rb} 次(压掉不重发 {getattr(sess, 'relatch_swallowed', 0)} 字),"
                  f"大改写留接缝 {getattr(sess, 'resync_seams', 0)} 次,"
                  f"最长停滞 {getattr(sess, 'max_stall', 0)} 步"
                  f"(停滞换段 {getattr(sess, 'stall_rolls', 0)} 次)", flush=True)
            _state["sessions"] = max(0, _state["sessions"] - 1)
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    host = config.MEDASR_WORKER_HOST
    port = int(os.environ.get("MEDASR_WORKER_PORT", str(config.MEDASR_WORKER_BASE_PORT)))
    ws_max = int(config.MEDASR_LIMITS.get("ws_max_msg_bytes", 1 << 20) or (1 << 20))
    print(f"[medasr-worker] serve http://{host}:{port}  (POST /offline, WS /ws/asr)", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning", ws_max_size=ws_max,
                ws_ping_interval=None)
