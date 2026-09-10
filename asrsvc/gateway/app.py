"""
ASR 网关(唯一对外端口)。单一共享密钥鉴权 + SSRF 安全下载 + 调度,前面建议再挂 nginx/TLS。

鉴权:除 /healthz /health 外均需 `Authorization: Bearer <MEDASR_API_KEY>`(WS 见 auth.check_ws_key);
      未配置 API key 时业务请求保持关闭。

路由(识别引擎:Qwen3-ASR,vLLM 双语单模型,离线+流式共用):
  GET  /healthz                 → {ok:true}(无鉴权)
  GET  /health                  → 各 worker 就绪/负载快照(无鉴权)
  POST /v1/asr/offline          → {url, lang} 离线识别
  WS   /ws/asr                  → 流式识别
启动:python -m asrsvc.gateway.app
"""
import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import websockets
from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from asrsvc import config
from asrsvc.common import logging as _logging
from asrsvc.gateway import auth, download
from asrsvc.gateway.dispatch import (OfflineBusy, UpstreamError, WorkerReject,
                                     dispatcher)

MAX_WS = int(config.MEDASR_LIMITS.get("max_ws_connections", 256) or 256)
WS_MAX_MSG = int(config.MEDASR_LIMITS.get("ws_max_msg_bytes", 1 << 20) or (1 << 20))
MAX_REQ_BYTES = int(config.MEDASR_LIMITS.get("max_request_bytes", 1 << 20) or (1 << 20))
MAX_URL_LEN = 2048
DL_MAX_CONCURRENT = int(config.MEDASR_DOWNLOAD_CONFIG.get("max_concurrent", 8) or 8)
VALID_LANGS = {"zh", "yue", "auto"}

_ws_active = 0
_dl_sem = asyncio.Semaphore(DL_MAX_CONCURRENT)


def _new_request_id() -> str:
    """本次请求唯一标识:req_<本地时间戳>-<随机>,可读、可按时间 grep 日志、同秒并发不撞。"""
    return "req_" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


@asynccontextmanager
async def lifespan(app):
    _logging.install()
    await dispatcher.start()
    print(f"[gateway] 就绪 ✓ {config.summary()}", flush=True)
    try:
        yield
    finally:
        await dispatcher.stop()


app = FastAPI(title="MedASR Gateway", lifespan=lifespan)

if config.MEDASR_CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=config.MEDASR_CORS_ORIGINS,
                       allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])


@app.middleware("http")
async def limit_body(request, call_next):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_REQ_BYTES:
        return JSONResponse({"detail": "请求体过大"}, status_code=413)
    return await call_next(request)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/health")
async def health():
    snap = dispatcher.snapshot()
    return {"ok": True, "engine": "medasr", **snap}


class OfflineReq(BaseModel):
    url: str = Field(..., max_length=MAX_URL_LEN)
    lang: str = config.MEDASR_DEFAULT_LANG


async def _handle_offline(req: "OfflineReq", disp) -> dict:
    """离线识别通用流程:校验→SSRF 安全下载→分发到指定池→finally 删文件。"""
    url = (req.url or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "仅支持 http/https 音频链接")
    lang = (req.lang or config.MEDASR_DEFAULT_LANG).lower()
    if lang not in VALID_LANGS:
        raise HTTPException(400, "lang 仅支持 zh(普通话)/ yue(粤语)/ auto(自动)")

    try:
        async with _dl_sem:
            dl = await run_in_threadpool(download.download, url)
    except download.SSRFBlocked as e:
        raise HTTPException(400, f"链接被安全策略拒绝:{e}")
    except download.DownloadError as e:
        raise HTTPException(502, f"下载失败:{e}")

    path = dl["path"]
    try:
        result = await disp.dispatch_offline(path, lang)
    except OfflineBusy:
        raise HTTPException(429, "离线队列已满,请稍后重试")
    except WorkerReject as e:
        raise HTTPException(e.status, e.detail)
    except UpstreamError as e:
        raise HTTPException(502, f"识别服务不可用:{e}")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return {"ok": True, "download_size": dl["size"], **result}


@app.post("/v1/asr/offline")
async def offline(req: OfflineReq, _: bool = Depends(auth.require_key)):
    """离线识别(Qwen3-ASR,双语单模型,自带标点,无说话人/时间戳)。"""
    return await _handle_offline(req, dispatcher)


@app.websocket("/ws/asr")
async def ws_proxy(client: WebSocket):
    """流式识别(Qwen3-ASR)。"""
    await _handle_ws(client, dispatcher)


async def _handle_ws(client: WebSocket, disp):
    global _ws_active
    if _ws_active >= MAX_WS:
        await client.close(code=4429)
        return
    _ws_active += 1
    try:
        if not await auth.check_ws_key(client):
            return
        rid = _new_request_id()
        try:
            await client.send_text(json.dumps({"type": "session", "request_id": rid}))
        except Exception:
            return
        await _proxy_stream(client, disp)
    finally:
        _ws_active = max(0, _ws_active - 1)


async def _proxy_stream(client: WebSocket, disp):
    up = None
    idx = None
    tried = set()
    while True:
        picked = disp.pick_stream_target(exclude=tried)
        if picked is None:
            break
        idx, url = picked
        try:
            up = await websockets.connect(url, max_size=WS_MAX_MSG, open_timeout=8,
                                          ping_interval=None, proxy=None)

            break
        except Exception:
            disp.release_stream(idx)
            tried.add(idx)
            idx = None
    if up is None:
        if not tried and disp.has_ready_stream():
            code, msg = 4429, "流式容量已满,请稍后重试"
        else:
            code, msg = 1013, "无法连接流式 ASR 实例池"
        try:
            await client.send_text(json.dumps({"type": "error", "msg": msg}))
        finally:
            await client.close(code=code)
        return

    why = {}

    async def c2u():
        try:
            while True:
                msg = await client.receive()
                if msg.get("type") == "websocket.disconnect":
                    why.setdefault("c2u", "客户端断开")
                    break
                b = msg.get("bytes")
                if b is not None:
                    await up.send(b)
                    continue
                t = msg.get("text")
                if t is not None:
                    await up.send(t)
        except Exception as e:
            why.setdefault("c2u", f"{type(e).__name__}: {str(e)[:200]}")

    async def u2c():
        try:
            async for m in up:
                if isinstance(m, (bytes, bytearray)):
                    await client.send_bytes(m)
                else:
                    await client.send_text(m)
            why.setdefault("u2c", "上游正常结束(worker 关闭了连接)")
        except Exception as e:
            why.setdefault("u2c", f"{type(e).__name__}: {str(e)[:200]}")

    t1 = asyncio.create_task(c2u())
    t2 = asyncio.create_task(u2c())
    try:
        _, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if why != {"c2u": "客户端断开"}:
            print(f"[gateway] 流式会话结束 reason={why}", flush=True)
        for closer in (up.close, client.close):
            try:
                await closer()
            except Exception:
                pass
    finally:
        disp.release_stream(idx)


if __name__ == "__main__":
    import uvicorn
    host, port = config.MEDASR_HOST, config.MEDASR_PORT
    log_level = os.environ.get("MEDASR_LOG_LEVEL", "info")
    print(f"[gateway] serve http://{host}:{port}  (log_level={log_level}, access_log=on)", flush=True)
    uvicorn.run(app, host=host, port=port, log_level=log_level, access_log=True, ws_max_size=WS_MAX_MSG,
                ws_ping_interval=None)
