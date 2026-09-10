"""MedASR configuration loader (environment variables override YAML)."""
import copy
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_YAML = ROOT / "config" / "asr_server.yaml"

_DEFAULT_API_KEY = ""

_DEFAULTS = {
    "gateway": {
        "host": "0.0.0.0",
        "port": 18080,
        "cors_origins": [],
    },
    "worker": {
        "host": "127.0.0.1",
        "upstream_host": "127.0.0.1",
    },
    "auth": {
        "api_key": _DEFAULT_API_KEY,
    },
    "download": {
        "spool_dir": str(ROOT / "data" / "spool"),
        "max_bytes": 500 * 1024 * 1024,
        "proxy": "",
        "timeout": 300,
        "connect_timeout": 10,
        "max_concurrent": 8,
        "max_redirects": 3,
        "allow_hosts": [],
        "block_private": True,
        "block_cidrs": [],
    },
    "limits": {
        "max_audio_seconds": 7200,
        "max_request_bytes": 1 * 1024 * 1024,
        "offline_queue_max": 64,
        "max_ws_connections": 256,
        "max_stream_sessions_per_card": 2,
        "ws_max_msg_bytes": 1 * 1024 * 1024,
        "ws_session_max_bytes": 1024 * 1024 * 1024,
        "ws_auth_timeout": 10,
        "ws_idle_timeout": 120,
    },
    "medasr": {
        "cards": "auto",
        "worker_base_port": 9300,
        "asr_model": "Qwen/Qwen3-ASR-1.7B",
        "gpu_util": 0.8,
        "chunk_sec": 4.0,
        "stream_window_sec": 0.0,
        "rollover_max_sec": 360.0,
        "rollover_min_sec": 300.0,
        "seam_rms": 0.0,
        "rollover_ctx_chars": 192,
        "rollover_on_shrink": False,
        "stall_steps": 0,
        "emit_unlatch": True,
        "head_gate": True,
        "silence_rms": 0.002,
        "silence_rel": 0.05,
        "silence_roll_sec": 30.0,
        "vad_enabled": True,
        "vad_model": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "vad_chunk_ms": 200,
        "vad_silence_ms": 800,
        "preroll_sec": 2.0,
        "step_timeout": 20.0,
        "max_chunk_mult": 4,
        "warmup_sec": 4.0,
        "warmup_chunk_sec": 1.0,
        "max_lag_sec": 120.0,
        "stream_max_tokens": 64,
        "offline_max_tokens": 512,
        "micro_batch": True,
        "batch_invariant": True,
        "attention_backend": "TRITON_ATTN",
        "max_micro_batch": 16,
        "gen_threadpool": 128,
        "incr_encode": True,
        "incr_block_sec": 8.0,
        "max_len": 40960,
        "offline_concurrency": 4,
        "offline_chunk_sec": 30,
        "job_timeout": 1800,
    },
    "default_lang": "zh",
}


def _deep_merge(base: dict, over: dict):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        elif v is not None:
            base[k] = v


def _load_yaml(p: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise RuntimeError(f"failed to load configuration {p}: {e}") from e


def _b(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


_ENV_MAP = [
    (("gateway", "host"), "MEDASR_HOST", str),
    (("gateway", "port"), "MEDASR_PORT", int),
    (("worker", "host"), "MEDASR_WORKER_HOST", str),
    (("worker", "upstream_host"), "MEDASR_UPSTREAM_HOST", str),
    (("auth", "api_key"), "MEDASR_API_KEY", str),
    (("download", "spool_dir"), "MEDASR_SPOOL_DIR", str),
    (("download", "proxy"), "MEDASR_DOWNLOAD_PROXY", str),
    (("download", "max_bytes"), "MEDASR_DOWNLOAD_MAX_BYTES", int),
    (("download", "timeout"), "MEDASR_DOWNLOAD_TIMEOUT", int),
    (("download", "connect_timeout"), "MEDASR_DOWNLOAD_CONNECT_TIMEOUT", int),
    (("download", "max_concurrent"), "MEDASR_DOWNLOAD_MAX_CONCURRENT", int),
    (("download", "max_redirects"), "MEDASR_DOWNLOAD_MAX_REDIRECTS", int),
    (("limits", "max_audio_seconds"), "MEDASR_MAX_AUDIO_SECONDS", int),
    (("limits", "max_request_bytes"), "MEDASR_MAX_REQUEST_BYTES", int),
    (("limits", "offline_queue_max"), "MEDASR_OFFLINE_QUEUE_MAX", int),
    (("limits", "max_ws_connections"), "MEDASR_MAX_WS_CONNECTIONS", int),
    (("limits", "ws_max_msg_bytes"), "MEDASR_WS_MAX_MSG_BYTES", int),
    (("limits", "ws_session_max_bytes"), "MEDASR_WS_SESSION_MAX_BYTES", int),
    (("limits", "ws_auth_timeout"), "MEDASR_WS_AUTH_TIMEOUT", int),
    (("limits", "ws_idle_timeout"), "MEDASR_WS_IDLE_TIMEOUT", int),
    (("medasr", "cards"), "MEDASR_CARDS", str),
    (("medasr", "worker_base_port"), "MEDASR_WORKER_BASE_PORT", int),
    (("medasr", "asr_model"), "MEDASR_ASR_MODEL", str),
    (("medasr", "gpu_util"), "MEDASR_GPU_UTIL", float),
    (("medasr", "chunk_sec"), "MEDASR_CHUNK_SEC", float),
    (("medasr", "stream_window_sec"), "MEDASR_STREAM_WINDOW_SEC", float),
    (("medasr", "rollover_max_sec"), "MEDASR_ROLLOVER_MAX_SEC", float),
    (("medasr", "rollover_min_sec"), "MEDASR_ROLLOVER_MIN_SEC", float),
    (("medasr", "seam_rms"), "MEDASR_SEAM_RMS", float),
    (("medasr", "silence_rms"), "MEDASR_SILENCE_RMS", float),
    (("medasr", "silence_rel"), "MEDASR_SILENCE_REL", float),
    (("medasr", "silence_roll_sec"), "MEDASR_SILENCE_ROLL_SEC", float),
    (("medasr", "vad_model"), "MEDASR_VAD_MODEL", str),
    (("medasr", "vad_chunk_ms"), "MEDASR_VAD_CHUNK_MS", int),
    (("medasr", "vad_silence_ms"), "MEDASR_VAD_SILENCE_MS", int),
    (("medasr", "preroll_sec"), "MEDASR_PREROLL_SEC", float),
    (("medasr", "step_timeout"), "MEDASR_STEP_TIMEOUT", float),
    (("medasr", "max_chunk_mult"), "MEDASR_MAX_CHUNK_MULT", int),
    (("medasr", "warmup_sec"), "MEDASR_WARMUP_SEC", float),
    (("medasr", "warmup_chunk_sec"), "MEDASR_WARMUP_CHUNK_SEC", float),
    (("medasr", "max_lag_sec"), "MEDASR_MAX_LAG_SEC", float),
    (("medasr", "rollover_ctx_chars"), "MEDASR_ROLLOVER_CTX_CHARS", int),
    (("medasr", "stall_steps"), "MEDASR_STALL_STEPS", int),
    (("medasr", "stream_max_tokens"), "MEDASR_STREAM_MAX_TOKENS", int),
    (("medasr", "offline_max_tokens"), "MEDASR_OFFLINE_MAX_TOKENS", int),
    (("medasr", "attention_backend"), "MEDASR_ATTENTION_BACKEND", str),
    (("medasr", "max_micro_batch"), "MEDASR_MAX_MICRO_BATCH", int),
    (("medasr", "gen_threadpool"), "MEDASR_GEN_THREADPOOL", int),
    (("medasr", "incr_block_sec"), "MEDASR_INCR_BLOCK_SEC", float),
    (("medasr", "max_len"), "MEDASR_MAX_LEN", int),
    (("medasr", "offline_concurrency"), "MEDASR_OFFLINE_CONCURRENCY", int),
    (("medasr", "offline_chunk_sec"), "MEDASR_OFFLINE_CHUNK_SEC", float),
    (("medasr", "job_timeout"), "MEDASR_JOB_TIMEOUT", int),
    (("default_lang",), "MEDASR_DEFAULT_LANG", str),
]


def _apply_env(c: dict):
    e = os.environ.get
    for path, env, typ in _ENV_MAP:
        val = e(env)
        if val is None:
            continue
        if val == "" and path != ("auth", "api_key"):
            continue
        try:
            val = typ(val)
        except (ValueError, TypeError):
            print(f"[config] 环境变量 {env}={val!r} 类型无效,忽略", flush=True)
            continue
        d = c
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = val
    if e("MEDASR_BLOCK_PRIVATE") is not None:
        c["download"]["block_private"] = _b(e("MEDASR_BLOCK_PRIVATE"))
    if e("MEDASR_ALLOW_HOSTS"):
        c["download"]["allow_hosts"] = [h.strip() for h in e("MEDASR_ALLOW_HOSTS").split(",") if h.strip()]
    if e("MEDASR_BLOCK_CIDRS"):
        c["download"]["block_cidrs"] = [n.strip() for n in e("MEDASR_BLOCK_CIDRS").split(",") if n.strip()]
    if e("MEDASR_CORS_ORIGINS"):
        c["gateway"]["cors_origins"] = [o.strip() for o in e("MEDASR_CORS_ORIGINS").split(",") if o.strip()]
    if e("MEDASR_MICRO_BATCH") is not None:
        c["medasr"]["micro_batch"] = _b(e("MEDASR_MICRO_BATCH"))
    if e("MEDASR_BATCH_INVARIANT") is not None:
        c["medasr"]["batch_invariant"] = _b(e("MEDASR_BATCH_INVARIANT"))
    if e("MEDASR_INCR_ENCODE") is not None:
        c["medasr"]["incr_encode"] = _b(e("MEDASR_INCR_ENCODE"))
    if e("MEDASR_VAD_ENABLED") is not None:
        c["medasr"]["vad_enabled"] = _b(e("MEDASR_VAD_ENABLED"))
    if e("MEDASR_EMIT_UNLATCH") is not None:
        c["medasr"]["emit_unlatch"] = _b(e("MEDASR_EMIT_UNLATCH"))
    if e("MEDASR_HEAD_GATE") is not None:
        c["medasr"]["head_gate"] = _b(e("MEDASR_HEAD_GATE"))
    if e("MEDASR_ROLLOVER_ON_SHRINK") is not None:
        c["medasr"]["rollover_on_shrink"] = _b(e("MEDASR_ROLLOVER_ON_SHRINK"))
    if e("MEDASR_MAX_STREAM_PER_CARD") is not None and e("MEDASR_MAX_STREAM_PER_CARD") != "":
        try:
            c["limits"]["max_stream_sessions_per_card"] = int(e("MEDASR_MAX_STREAM_PER_CARD"))
        except (ValueError, TypeError):
            print(f"[config] 环境变量 MEDASR_MAX_STREAM_PER_CARD={e('MEDASR_MAX_STREAM_PER_CARD')!r} 非整数,忽略", flush=True)


def _build() -> dict:
    c = copy.deepcopy(_DEFAULTS)
    if _YAML.exists():
        _deep_merge(c, _load_yaml(_YAML))
    _apply_env(c)
    return c


def _detect_gpu_cards() -> list:
    """Detect visible GPU identifiers without importing CUDA libraries."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return [c.strip() for c in cvd.split(",") if c.strip() not in ("", "-1")]
    try:
        import re
        n = len([p for p in os.listdir("/dev") if re.fullmatch(r"nvidia\d+", p)])
        if n > 0:
            return [str(i) for i in range(n)]
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            n = len([ln for ln in r.stdout.splitlines() if ln.strip() != ""])
            if n > 0:
                return [str(i) for i in range(n)]
    except Exception:
        pass
    return []


_C = _build()

MEDASR_HOST = _C["gateway"]["host"]
MEDASR_PORT = int(_C["gateway"]["port"])
MEDASR_CORS_ORIGINS = list(_C["gateway"].get("cors_origins") or [])

MEDASR_WORKER_HOST = _C["worker"]["host"]
MEDASR_UPSTREAM_HOST = _C["worker"]["upstream_host"]

MEDASR_API_KEY = str(_C["auth"].get("api_key", "") or "").strip()
MEDASR_AUTH_REQUIRED = True

MEDASR_DOWNLOAD_CONFIG = dict(_C["download"])
MEDASR_SPOOL_DIR = MEDASR_DOWNLOAD_CONFIG["spool_dir"]

MEDASR_LIMITS = dict(_C["limits"])
MEDASR_DEFAULT_LANG = _C.get("default_lang", "zh")

MEDASR_CONFIG = dict(_C["medasr"])
MEDASR_ASR_MODEL = MEDASR_CONFIG.get("asr_model", "")
MEDASR_GPU_UTIL = float(MEDASR_CONFIG.get("gpu_util", 0.3) or 0.3)
MEDASR_CHUNK_SEC = float(MEDASR_CONFIG.get("chunk_sec", 1.0) or 1.0)
MEDASR_STREAM_WINDOW_SEC = max(0.0, float(MEDASR_CONFIG.get("stream_window_sec", 0.0) or 0.0))
MEDASR_ROLLOVER_MAX_SEC = max(0.0, float(MEDASR_CONFIG.get("rollover_max_sec", 0.0) or 0.0))
MEDASR_ROLLOVER_MIN_SEC = max(0.0, float(MEDASR_CONFIG.get("rollover_min_sec", 60.0) or 0.0))
MEDASR_SEAM_RMS = max(0.0, float(MEDASR_CONFIG.get("seam_rms", 0.0) or 0.0))
MEDASR_ROLLOVER_CTX_CHARS = max(0, int(MEDASR_CONFIG.get("rollover_ctx_chars", 0) or 0))
MEDASR_ROLLOVER_ON_SHRINK = bool(MEDASR_CONFIG.get("rollover_on_shrink", False))
MEDASR_STALL_STEPS = max(0, int(MEDASR_CONFIG.get("stall_steps", 0) or 0))
MEDASR_EMIT_UNLATCH = bool(MEDASR_CONFIG.get("emit_unlatch", True))
MEDASR_HEAD_GATE = bool(MEDASR_CONFIG.get("head_gate", True))
MEDASR_SILENCE_RMS = max(0.0, float(MEDASR_CONFIG.get("silence_rms", 0.0) or 0.0))
MEDASR_SILENCE_REL = min(0.5, max(0.0, float(MEDASR_CONFIG.get("silence_rel", 0.05) or 0.0)))
MEDASR_SILENCE_ROLL_SEC = max(0.0, float(MEDASR_CONFIG.get("silence_roll_sec", 30.0) or 0.0))
MEDASR_VAD_ENABLED = bool(MEDASR_CONFIG.get("vad_enabled", True))
MEDASR_VAD_CHUNK_MS = max(10, int(MEDASR_CONFIG.get("vad_chunk_ms", 200) or 200))
MEDASR_VAD_SILENCE_MS = max(50, int(MEDASR_CONFIG.get("vad_silence_ms", 800) or 800))
MEDASR_PREROLL_SEC = max(0.0, float(MEDASR_CONFIG.get("preroll_sec", 2.0) or 0.0))
MEDASR_STEP_TIMEOUT = max(0.0, float(MEDASR_CONFIG.get("step_timeout", 20.0) or 0.0))
MEDASR_MAX_CHUNK_MULT = max(1, int(MEDASR_CONFIG.get("max_chunk_mult", 4) or 1))
MEDASR_WARMUP_SEC = max(0.0, float(MEDASR_CONFIG.get("warmup_sec", 0) or 0))
MEDASR_WARMUP_CHUNK_SEC = min(float(MEDASR_CONFIG.get("warmup_chunk_sec", 0) or 0), MEDASR_CHUNK_SEC)
MEDASR_MAX_LAG_SEC = max(0.0, float(MEDASR_CONFIG.get("max_lag_sec", 120.0) or 0.0))



def _find_vad_model():
    """VAD 模型查找顺序:配置显式指定 > 项目 models/ > modelscope 本地缓存。
    都找不到就返回 ""(上层据此关闭 VAD 并退回旧判据,不影响服务启动)。"""
    p = (MEDASR_CONFIG.get("vad_model") or "").strip()
    if p:
        if os.path.isabs(p):
            return p
        local = ROOT / p
        return str(local) if local.exists() else p
    name = "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    for cand in (str(ROOT / "models" / name),
                 os.path.expanduser(f"~/.cache/modelscope/hub/models/iic/{name}")):
        if os.path.exists(cand):
            return cand
    return ""


MEDASR_VAD_MODEL = _find_vad_model()

MEDASR_STREAM_MAX_TOKENS = max(1, int(MEDASR_CONFIG.get("stream_max_tokens", 48) or 48))
MEDASR_OFFLINE_MAX_TOKENS = max(1, int(MEDASR_CONFIG.get("offline_max_tokens", 512) or 512))
MEDASR_MICRO_BATCH = bool(MEDASR_CONFIG.get("micro_batch", False))
MEDASR_BATCH_INVARIANT = bool(MEDASR_CONFIG.get("batch_invariant", True))
MEDASR_ATTENTION_BACKEND = str(MEDASR_CONFIG.get("attention_backend", "TRITON_ATTN") or "TRITON_ATTN").strip()
MEDASR_MAX_MICRO_BATCH = max(1, int(MEDASR_CONFIG.get("max_micro_batch", 16) or 16))
MEDASR_GEN_THREADPOOL = max(1, int(MEDASR_CONFIG.get("gen_threadpool", 128) or 128))
MEDASR_INCR_ENCODE = bool(MEDASR_CONFIG.get("incr_encode", False))
MEDASR_INCR_BLOCK_SEC = max(8.0, round(float(MEDASR_CONFIG.get("incr_block_sec", 8.0) or 8.0) / 8.0) * 8.0)
MEDASR_MAX_LEN = int(MEDASR_CONFIG.get("max_len", 32768) or 32768)
MEDASR_OFFLINE_CONCURRENCY = max(1, int(MEDASR_CONFIG.get("offline_concurrency", 4) or 4))
MEDASR_OFFLINE_CHUNK_SEC = float(MEDASR_CONFIG.get("offline_chunk_sec", 30) or 30)
MEDASR_JOB_TIMEOUT = int(MEDASR_CONFIG.get("job_timeout", 1800) or 1800)
MEDASR_WORKER_BASE_PORT = int(MEDASR_CONFIG.get("worker_base_port", 9300))
_medasr_cards_raw = str(MEDASR_CONFIG.get("cards", "")).strip()
if _medasr_cards_raw.lower() in ("", "auto", "all"):
    MEDASR_CARDS = _detect_gpu_cards()
    MEDASR_CARDS_AUTO = True
else:
    MEDASR_CARDS = [d.strip() for d in _medasr_cards_raw.split(",") if d.strip() != ""]
    MEDASR_CARDS_AUTO = False
if len(set(MEDASR_CARDS)) != len(MEDASR_CARDS):
    _dup = [c for c in dict.fromkeys(MEDASR_CARDS) if MEDASR_CARDS.count(c) > 1]
    print(f"[config] medasr.cards 有重复卡号 {_dup},同卡多实例必 OOM(gpu_util 是总显存比例);已去重", flush=True)
    MEDASR_CARDS = list(dict.fromkeys(MEDASR_CARDS))
MEDASR_INSTANCE_PLAN = [(MEDASR_CARDS[i], MEDASR_WORKER_BASE_PORT + i) for i in range(len(MEDASR_CARDS))]
MEDASR_OFFLINE_UPSTREAMS = [f"http://{MEDASR_UPSTREAM_HOST}:{p}" for _, p in MEDASR_INSTANCE_PLAN]
MEDASR_STREAM_UPSTREAMS = [f"ws://{MEDASR_UPSTREAM_HOST}:{p}/ws/asr" for _, p in MEDASR_INSTANCE_PLAN]


def summary() -> dict:
    """打印/自检用的非敏感配置快照(不含明文密钥)。"""
    return {
        "gateway": f"{MEDASR_HOST}:{MEDASR_PORT}",
        "auth_required": MEDASR_AUTH_REQUIRED,
        "medasr_cards": MEDASR_CARDS,
        "medasr_cards_auto": MEDASR_CARDS_AUTO,
        "medasr_ports": [p for _, p in MEDASR_INSTANCE_PLAN],
        "medasr_offline_concurrency": MEDASR_OFFLINE_CONCURRENCY,
        "medasr_micro_batch": MEDASR_MICRO_BATCH,
        "medasr_batch_invariant": MEDASR_BATCH_INVARIANT if MEDASR_MICRO_BATCH else None,
        "default_lang": MEDASR_DEFAULT_LANG,
        "block_private": MEDASR_DOWNLOAD_CONFIG.get("block_private", True),
    }
