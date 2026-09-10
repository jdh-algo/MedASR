"""CPU-based FSMN-VAD endpoint detection for streaming rollover."""
import os
import threading

import numpy as np

from asrsvc import config

SAMPLE_RATE = 16000

_MODEL_LOCK = threading.Lock()
_MODEL = None
_MODEL_TRIED = False


def build_vad_model():
    """Load the shared FSMN-VAD model, returning None when unavailable."""
    global _MODEL, _MODEL_TRIED
    if _MODEL is not None or _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True
    if not config.MEDASR_VAD_ENABLED:
        return None
    path = config.MEDASR_VAD_MODEL
    if not path:
        print("[vad] 未配置模型,VAD 关闭", flush=True)
        return None
    try:
        from funasr import AutoModel
        _MODEL = AutoModel(model=path, device="cpu", disable_update=True)
        print(f"[vad] FSMN-VAD 就绪(CPU):{path}", flush=True)
    except Exception as e:
        print(f"[vad] 加载失败,VAD 关闭:{type(e).__name__}: {e}", flush=True)
        _MODEL = None
    return _MODEL


class EndpointDetector:
    """Per-session streaming endpoint detector."""

    __slots__ = ("model", "cache", "_buf", "_chunk", "_fed_samples")

    def __init__(self, model):
        self.model = model
        self.cache = {}
        self._buf = np.zeros(0, dtype=np.float32)
        self._chunk = max(1, int(config.MEDASR_VAD_CHUNK_MS * SAMPLE_RATE / 1000))
        self._fed_samples = 0

    def feed(self, samples: np.ndarray):
        """Return the latest speech endpoint in milliseconds, or None."""
        if self.model is None:
            return None
        samples = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if samples.size:
            self._buf = np.concatenate((self._buf, samples))
        endpoint_ms = None
        while self._buf.size >= self._chunk:
            chunk = np.ascontiguousarray(self._buf[:self._chunk])
            self._buf = self._buf[self._chunk:]
            try:
                with _MODEL_LOCK:
                    res = self.model.generate(
                        input=chunk, cache=self.cache, is_final=False,
                        chunk_size=config.MEDASR_VAD_CHUNK_MS,
                        max_end_silence_time=config.MEDASR_VAD_SILENCE_MS,
                        disable_pbar=True, disable_log=True)
            except Exception:
                return None
            self._fed_samples += len(chunk)
            for _start_ms, end_ms in (res[0].get("value", []) if res else []):
                if end_ms is not None and end_ms >= 0:
                    endpoint_ms = int(end_ms)
        return endpoint_ms


def make_detector():
    """Create a detector, or return None when VAD is unavailable."""
    m = build_vad_model()
    return EndpointDetector(m) if m is not None else None
