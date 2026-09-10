# Copyright 2026 The Alibaba Qwen team.
# Copyright 2026 MedASR contributors.
# SPDX-License-Identifier: Apache-2.0
# Portions of this file are adapted from Qwen3-ASR and modified for MedASR.
"""Qwen3-ASR adapter for offline and long-running streaming inference."""
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import copy
import threading
import time
from collections import deque

import numpy as np

from asrsvc import config

_INVARIANT = config.MEDASR_BATCH_INVARIANT and (config.MEDASR_MICRO_BATCH or config.MEDASR_INCR_ENCODE)
if _INVARIANT:
    os.environ["VLLM_BATCH_INVARIANT"] = "1"

def _incr():
    """Import the optional incremental encoder lazily."""
    from asrsvc.engines import qwen_incr
    return qwen_incr


def _vad():
    """Import the optional VAD adapter lazily."""
    from asrsvc.engines import qwen_vad
    return qwen_vad



SAMPLE_RATE = 16000
CHUNK_SEC = config.MEDASR_CHUNK_SEC
CHUNK_SAMPLES = int(round(CHUNK_SEC * SAMPLE_RATE))
OFFLINE_CHUNK_SEC = config.MEDASR_OFFLINE_CHUNK_SEC

_MODEL_LOCK = threading.Lock()

_SEAM_PUNCT = ("。", "!", "?", "!", "?", ";", ";", "…")

_PREROLL_GUARD_CHARS = 32
_PREROLL_MIN_MATCH = 4

_TONE_BAND_HZ = 30.0
_TONE_RATIO = 0.5
_TONE_FLATNESS = 0.02

_MAX_TAIL_REWRITE = 16

_CHUNK_MULT_DECAY = 10
_EOF_RETRIES = 3


def _is_nonspeech(chunk: np.ndarray, rms_floor: float) -> bool:
    """Detect silence or a narrow-band tone during session startup."""
    if chunk.size < 512:
        return False
    rms = float(np.sqrt(np.mean(np.square(chunk))))
    if rms < max(rms_floor, 1e-4):
        return True
    win = chunk * np.hanning(chunk.size)
    S = np.abs(np.fft.rfft(win))
    peak = float(S.max())
    if peak <= 0.0:
        return True
    P = S * S
    total = float(P.sum())
    if total <= 0.0:
        return True
    f = np.fft.rfftfreq(chunk.size, 1.0 / SAMPLE_RATE)
    i = int(np.argmax(S))
    ratio = float(P[np.abs(f - f[i]) <= _TONE_BAND_HZ].sum() / total)
    Sp = S + 1e-12
    flat = float(np.exp(np.log(Sp).mean()) / Sp.mean())
    return ratio > _TONE_RATIO and flat < _TONE_FLATNESS


def _overlap_len(prev_tail: str, cur: str) -> int:
    """Return the longest overlap between the prior suffix and current prefix."""
    m = min(len(prev_tail), len(cur))
    for k in range(m, _PREROLL_MIN_MATCH - 1, -1):
        if prev_tail[-k:] == cur[:k]:
            return k
    return 0



class QwenBusyError(RuntimeError):
    """Raised when a queued or running generation step exceeds its deadline."""

    def __init__(self, msg, started=False):
        super().__init__(msg)
        self.started = started


class _Item:
    """One pending generation item."""
    __slots__ = ("inp", "sp", "event", "result", "error", "started", "abandoned")

    def __init__(self, inp, sp):
        self.inp = inp
        self.sp = sp
        self.event = threading.Event()
        self.result = None
        self.error = None
        self.started = False
        self.abandoned = False


class _Batcher:
    """Serialize vLLM generation while dynamically batching compatible requests."""

    def __init__(self, real_generate, max_batch):
        self._real = real_generate
        self._max = max(1, int(max_batch))
        self._q = deque()
        self._cv = threading.Condition()
        self._alive = True
        self._gpu = deque(maxlen=512)
        self._gpu_lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="medasr-batcher", daemon=True)
        self._thread.start()

    def gpu_stats(self):
        """Return recent GPU-time and batch-size statistics."""
        with self._gpu_lock:
            rows = list(self._gpu)
        if not rows:
            return {"gpu_sec_per_audio_sec": None, "batch_avg": None, "samples": 0}
        gpu = sum(t for t, _ in rows)
        audio = sum(n for _, n in rows) * CHUNK_SEC
        return {"gpu_sec_per_audio_sec": round(gpu / audio, 4) if audio else None,
                "batch_avg": round(sum(n for _, n in rows) / len(rows), 2),
                "samples": len(rows)}

    def dispatch(self, prompts, sampling_params=None, use_tqdm=False, **kw):
        """Queue compatible requests and return outputs in input order."""
        plist = list(prompts) if isinstance(prompts, (list, tuple)) else [prompts]
        if isinstance(sampling_params, (list, tuple)):
            sps = list(sampling_params)
        else:
            sps = [sampling_params] * len(plist)
        items = [_Item(p, s) for p, s in zip(plist, sps)]
        with self._cv:
            self._q.extend(items)
            self._cv.notify()
        budget = config.MEDASR_STEP_TIMEOUT
        deadline = (time.monotonic() + budget) if budget > 0 else None
        outs = []
        for it in items:
            if deadline is None:
                it.event.wait()
            elif not it.event.wait(max(0.0, deadline - time.monotonic())):
                with self._cv:
                    it.abandoned = True
                    started = it.started
                    if not started:
                        try:
                            self._q.remove(it)
                        except ValueError:
                            started = it.started
                raise QwenBusyError("Qwen3-ASR 单步排队/推理超时", started=started)
            if it.error is not None:
                raise it.error
            outs.append(it.result)
        return outs

    def _take_batch(self):
        """Take up to the configured number of compatible queued items."""
        first = self._q.popleft()
        batch = [first]
        key = getattr(first.sp, "max_tokens", None)
        while self._q and len(batch) < self._max and getattr(self._q[0].sp, "max_tokens", None) == key:
            batch.append(self._q.popleft())
        return batch

    def _loop(self):
        while True:
            with self._cv:
                while self._alive and not self._q:
                    self._cv.wait()
                if not self._alive and not self._q:
                    return
                batch = self._take_batch()
                for it in batch:
                    it.started = True
            try:
                _t0 = time.time()
                outs = self._real([it.inp for it in batch],
                                  sampling_params=[it.sp for it in batch], use_tqdm=False)
                with self._gpu_lock:
                    self._gpu.append((time.time() - _t0, len(batch)))
                for it, o in zip(batch, outs):
                    it.result = o
                    it.event.set()
            except Exception as e:
                for it in batch:
                    it.error = e
                    it.event.set()

    def close(self):
        with self._cv:
            self._alive = False
            self._cv.notify_all()


def _map_lang(lang):
    """Map public language codes to Qwen3-ASR language names."""
    s = str(lang).lower()
    if s in ("yue", "canto", "cantonese", "粤语", "粤"):
        return "Cantonese"
    if s in ("auto", "", "und", "multi", "detect"):
        return None
    return "Chinese"


def _unmap_lang(name) -> str:
    """Qwen 语种名 → 对外短码(Chinese→zh / Cantonese→yue / 其它→原名小写)。"""
    n = str(name or "").strip().lower()
    if n == "chinese":
        return "zh"
    if n == "cantonese":
        return "yue"
    return n or "unknown"


def build_qwen_model(gpu_util=None):
    """Load Qwen3-ASR with separate offline and streaming sampling settings."""
    from qwen_asr import Qwen3ASRModel
    from vllm import SamplingParams
    kwargs = dict(
        model=config.MEDASR_ASR_MODEL,
        gpu_memory_utilization=gpu_util if gpu_util is not None else config.MEDASR_GPU_UTIL,
        max_model_len=config.MEDASR_MAX_LEN,
        max_new_tokens=config.MEDASR_OFFLINE_MAX_TOKENS,
    )
    if _INVARIANT:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        kwargs["attention_backend"] = getattr(AttentionBackendEnum, config.MEDASR_ATTENTION_BACKEND)
    if config.MEDASR_INCR_ENCODE:
        _incr().patch_vllm()
        kwargs.update(_incr().llm_kwargs())
    model = Qwen3ASRModel.LLM(**kwargs)
    model._sp_offline = model.sampling_params
    model._sp_stream = SamplingParams(temperature=0.0, max_tokens=config.MEDASR_STREAM_MAX_TOKENS)
    if config.MEDASR_MICRO_BATCH:
        model.sampling_params = model._sp_offline
        model._batcher = _Batcher(model.model.generate, config.MEDASR_MAX_MICRO_BATCH)
        model.model.generate = model._batcher.dispatch
        model._stream_view = copy.copy(model)
        model._stream_view.sampling_params = model._sp_stream
    return model


def transcribe_file(model, audio_path, lang="zh") -> dict:
    """Transcribe an audio file in chunks and return the normalized result."""
    import librosa
    from qwen_asr.inference.utils import split_audio_into_chunks

    language = _map_lang(lang)
    wav, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    duration = round(len(wav) / SAMPLE_RATE, 2)
    parts = split_audio_into_chunks(wav=wav, sr=SAMPLE_RATE, max_chunk_sec=float(OFFLINE_CHUNK_SEC))
    if not parts:
        return {"lang": lang, "detected_lang": lang, "text": "", "segments": [],
                "has_speaker": False, "duration": duration, "engine": "medasr"}
    if config.MEDASR_MICRO_BATCH:
        res = model.transcribe(audio=[(c, SAMPLE_RATE) for c, _ in parts],
                               language=[language] * len(parts))
    else:
        with _MODEL_LOCK:
            model.sampling_params = model._sp_offline
            res = model.transcribe(audio=[(c, SAMPLE_RATE) for c, _ in parts],
                                   language=[language] * len(parts))
    segs = []
    t = 0.0
    detected_counts = {}
    for i, (c, _) in enumerate(parts):
        seg_dur = len(c) / SAMPLE_RATE
        r = res[i] if i < len(res) else None
        txt = ((getattr(r, "text", "") if r is not None else "") or "").strip()
        dl = getattr(r, "language", "") if r is not None else ""
        if dl:
            detected_counts[dl] = detected_counts.get(dl, 0) + 1
        if txt:
            segs.append({"id": len(segs), "start": round(t, 3), "end": round(t + seg_dur, 3),
                         "spk": 0, "text": txt})
        t += seg_dur
    detected = max(detected_counts, key=detected_counts.get) if detected_counts else ""
    return {"lang": lang, "detected_lang": _unmap_lang(detected) if detected else lang,
            "text": "".join(s["text"] for s in segs),
            "segments": segs, "has_speaker": False, "duration": duration, "engine": "medasr"}


class QwenStreamSession:
    """Maintain decoding state and stable incremental output for one stream."""

    def __init__(self, model, lang="zh"):
        self.model = model._stream_view if config.MEDASR_MICRO_BATCH else model
        self._lang = _map_lang(lang)
        _init_chunk = (config.MEDASR_WARMUP_CHUNK_SEC
                       if (config.MEDASR_WARMUP_CHUNK_SEC > 0 and config.MEDASR_WARMUP_SEC > 0)
                       else CHUNK_SEC)
        self.state = self.model.init_streaming_state(language=self._lang, chunk_size_sec=_init_chunk)
        self._buf = np.zeros(0, dtype=np.float32)
        self._emitted = 0
        self._prev_full = ""
        self._done = False
        self._unlatch = config.MEDASR_EMIT_UNLATCH
        self._sent = ""
        self.shrink_holds = 0
        self.relatch_breaks = 0
        self.relatch_swallowed = 0
        self.resync_seams = 0
        self._window_samples = int(config.MEDASR_STREAM_WINDOW_SEC * SAMPLE_RATE) if config.MEDASR_STREAM_WINDOW_SEC > 0 else 0
        self._roll_max = config.MEDASR_ROLLOVER_MAX_SEC
        self._roll_min = min(config.MEDASR_ROLLOVER_MIN_SEC, self._roll_max) if self._roll_max > 0 else 0.0
        self._seam_rms = config.MEDASR_SEAM_RMS
        self._ctx_chars = config.MEDASR_ROLLOVER_CTX_CHARS
        self._roll_shrink = config.MEDASR_ROLLOVER_ON_SHRINK
        self._last_rms = 0.0
        self.segments = 0
        if self._roll_max > 0 and self._window_samples:
            self._window_samples = 0
        self._sil_rms = config.MEDASR_SILENCE_RMS
        self._sil_rel = config.MEDASR_SILENCE_REL
        self._sil_roll = config.MEDASR_SILENCE_ROLL_SEC
        self._peak_rms = 0.0
        self._silent_sec = 0.0
        self._sil_pending_roll = False
        self._head_gate = config.MEDASR_HEAD_GATE
        self.head_gated_sec = 0.0
        self._speech_text = ""
        self._mel = None
        if config.MEDASR_INCR_ENCODE:
            self._window_samples = 0
            self._mel = _incr().MelStream(self.model.processor.feature_extractor, id(self))
        self._vad = None
        if config.MEDASR_VAD_ENABLED and self._roll_max > 0:
            self._vad = _vad().make_detector()
        self._vad_endpoint = False
        _pre = config.MEDASR_PREROLL_SEC
        if self._roll_max > 0:
            _pre = min(_pre, self._roll_max / 4.0)
        self._preroll_samples = int(_pre * SAMPLE_RATE)

        self._recent = np.zeros(0, dtype=np.float32)
        self._preroll_guard = ""
        self._new_since_roll = 0
        self._chunk_mult = 1
        self._max_chunk_mult = config.MEDASR_MAX_CHUNK_MULT
        self._warm_samples = (int(config.MEDASR_WARMUP_SEC * SAMPLE_RATE)
                              if config.MEDASR_WARMUP_CHUNK_SEC > 0 else 0)
        self._warm_step = int(round(config.MEDASR_WARMUP_CHUNK_SEC * SAMPLE_RATE))
        self._fed_samples = 0
        self._ok_streak = 0
        self._max_lag_samples = int(config.MEDASR_MAX_LAG_SEC * SAMPLE_RATE)
        self.overload_skips = 0
        self.dropped_sec = 0.0
        self._stall_steps = config.MEDASR_STALL_STEPS
        self._stall = 0
        self.stall_rolls = 0
        self.max_stall = 0





    @staticmethod
    def _pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
        if not pcm_bytes:
            return np.zeros(0, dtype=np.float32)
        n = len(pcm_bytes) - (len(pcm_bytes) % 2)
        return np.frombuffer(pcm_bytes[:n], dtype=np.int16).astype(np.float32) / 32768.0

    @staticmethod
    def _common_prefix_len(a: str, b: str) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _emit_delta(self, final=False):
        cur = self.state.text or ""
        if self._preroll_guard and cur:
            k = _overlap_len(self._preroll_guard, cur)
            if k >= _PREROLL_MIN_MATCH:
                if k > self._emitted:
                    self._sent = cur[:k]
                    self._emitted = k
                self._preroll_guard = ""
            elif len(cur) > _PREROLL_GUARD_CHARS * 2:
                self._preroll_guard = ""
        if final:
            commit = len(cur)
        elif not self._unlatch:
            commit = max(self._emitted, self._common_prefix_len(self._prev_full, cur))
        else:
            if len(cur) < len(self._prev_full):
                self.shrink_holds += 1
                commit = self._emitted
            else:
                commit = max(self._emitted, self._common_prefix_len(self._prev_full, cur))
            if len(cur) < self._emitted:
                p = self._common_prefix_len(self._sent, cur)
                self.relatch_breaks += 1
                self._emitted = p
                commit = max(p, self._common_prefix_len(self._prev_full, cur))
        if commit > len(cur):
            commit = len(cur)
        self._prev_full = cur
        if commit <= self._emitted:
            return ""
        self._emitted = commit
        return self._send(cur, commit)

    def _send(self, cur: str, commit: int) -> str:
        """Return only text not already emitted by the append-only protocol."""
        sent = self._sent
        start = self._common_prefix_len(sent, cur)
        if start < len(sent):
            if start > 0 and len(sent) - start <= _MAX_TAIL_REWRITE:
                self.relatch_swallowed += len(sent) - start
                start = len(sent)
            else:
                self.resync_seams += 1
        if commit <= start:
            return ""
        d = cur[start:commit]
        self._sent = cur[:commit]
        return d


    def _apply_window(self):
        """Limit accumulated audio when the legacy sliding window is enabled."""
        if self._window_samples and self.state.audio_accum.shape[0] > self._window_samples:
            self.state.audio_accum = self.state.audio_accum[-self._window_samples:]

    def _accum_sec(self) -> float:
        if self._mel is not None:
            return self._mel.n_samples / SAMPLE_RATE
        return self.state.audio_accum.shape[0] / SAMPLE_RATE

    def _segment_sec(self) -> float:
        """Return new audio duration in the current segment, excluding pre-roll."""
        return self._new_since_roll / SAMPLE_RATE

    def _good_seam(self, cur: str) -> bool:
        """Return whether the current position is a suitable segment boundary."""
        if self._vad is not None:
            return self._vad_endpoint
        if self._seam_rms > 0.0 and self._last_rms <= self._seam_rms:
            return True
        return cur.endswith(_SEAM_PUNCT)


    def _should_rollover(self) -> bool:
        """Return whether the current stream should start a new segment."""
        if self._roll_shrink and len(self.state.text or "") < self._emitted:
            return True
        if self._stall_steps > 0 and self._stall >= self._stall_steps:
            self.stall_rolls += 1
            return True
        if self._roll_max <= 0:
            return False
        if self._accum_sec() >= self._roll_max:
            return True
        return (self._segment_sec() >= self._roll_min
                and self._good_seam(self.state.text or ""))

    def _rollover(self, tail_text=None, preroll=True) -> str:
        """Flush the current segment and initialize a fresh streaming state."""
        prev_text = self.state.text or ""
        tail = self._emit_delta(final=True) if tail_text is None else tail_text[self._emitted:]
        buf = self.state.buffer
        ctx = prev_text[-self._ctx_chars:] if self._ctx_chars else ""
        self.state = self.model.init_streaming_state(
            context=ctx, language=self._lang, chunk_size_sec=CHUNK_SEC)
        self.state.buffer = buf
        self._emitted, self._prev_full, self._sent = 0, "", ""
        if self._mel is not None:
            self._mel = _incr().MelStream(self.model.processor.feature_extractor, id(self))
        self._preroll_guard = prev_text[-_PREROLL_GUARD_CHARS:]
        if preroll and self._preroll_samples > 0 and self._recent.size > 0:
            pre = np.ascontiguousarray(self._recent[-self._preroll_samples:])
            if self._mel is not None:
                self._mel.add(pre)
            else:
                self.state.audio_accum = np.concatenate([self.state.audio_accum, pre])
        self._vad_endpoint = False
        self._new_since_roll = 0
        self._stall = 0
        self.segments += 1
        return tail

    def _roll_at_silence(self) -> str:
        """Roll over at a silence boundary without carrying silent pre-roll."""
        snap, self._speech_text = self._speech_text, ""
        self._silent_sec = 0.0
        return self._rollover(tail_text=snap, preroll=False)

    def _feed_chunk(self, pcm):
        """Decode one audio chunk through the configured streaming path."""
        if self._mel is not None:
            self._stream_call(_incr().streaming_transcribe, self.model, pcm, self.state, self._mel)
        else:
            self._stream_call(self.model.streaming_transcribe, pcm, self.state)

    def _stream_call(self, fn, *args):
        """Run one streaming generation step with the required serialization."""
        if config.MEDASR_MICRO_BATCH:
            fn(*args)
        else:
            with _MODEL_LOCK:
                self.model.sampling_params = self.model._sp_stream
                fn(*args)

    def feed_pcm(self, pcm_bytes: bytes) -> list:
        return self.feed_float(self._pcm16_to_float32(pcm_bytes))

    def feed_float(self, samples: np.ndarray) -> list:
        if self._done or samples.size == 0:
            return []
        if self._vad is not None:
            if self._vad.feed(samples) is not None:
                self._vad_endpoint = True
        if self._preroll_samples > 0:
            self._recent = np.concatenate([self._recent, samples])[-self._preroll_samples:]
        self._buf = np.concatenate([self._buf, samples])

        deltas = []
        if self._max_lag_samples > 0 and self._buf.size > self._max_lag_samples:
            drop = self._buf.size - self._max_lag_samples
            self._buf = self._buf[drop:]
            self.dropped_sec += drop / SAMPLE_RATE
        while True:
            step = CHUNK_SAMPLES * self._chunk_mult
            if self._warm_samples and self._fed_samples < self._warm_samples:
                step = min(step, self._warm_step)
            elif self._warm_samples and self.state.chunk_size_samples != CHUNK_SAMPLES:
                self.state.chunk_size_samples = CHUNK_SAMPLES
            if self._buf.size < step:
                break
            chunk = self._buf[:step]
            rms = (float(np.sqrt(np.mean(np.square(chunk))))
                   if (self._sil_rms > 0.0 or self._seam_rms > 0.0) else 0.0)
            if self._seam_rms > 0.0:
                self._last_rms = rms
            if self._sil_rms > 0.0:
                self._peak_rms = max(rms, self._peak_rms * 0.995)
                silent = rms < max(self._sil_rms, self._sil_rel * self._peak_rms)
            else:
                silent = False
            if not silent and self._sil_pending_roll:
                self._sil_pending_roll = False
                t = self._roll_at_silence()
                if t:
                    deltas.append(t)
            if self._head_gate:
                if _is_nonspeech(chunk, self._sil_rms):
                    self.head_gated_sec += step / SAMPLE_RATE
                    self._buf = self._buf[step:]
                    continue
                self._head_gate = False
            try:
                self._feed_chunk(chunk)
            except QwenBusyError as e:
                self.overload_skips += 1
                if self._chunk_mult < self._max_chunk_mult:
                    self._chunk_mult += 1
                self._ok_streak = 0
                if not e.started:
                    break
                self._buf = self._buf[step:]
                self._new_since_roll += step
                self._fed_samples += step
                continue
            self._buf = self._buf[step:]
            self._new_since_roll += step
            self._fed_samples += step
            self._ok_streak += 1
            if self._chunk_mult > 1 and self._ok_streak >= _CHUNK_MULT_DECAY:
                self._chunk_mult -= 1
                self._ok_streak = 0
            self._apply_window()
            if silent:
                self._silent_sec += step / SAMPLE_RATE
                if self._sil_roll > 0.0 and self._silent_sec >= self._sil_roll:
                    t = self._roll_at_silence()
                    self._sil_pending_roll = True
                    if t:
                        deltas.append(t)
            else:
                self._silent_sec = 0.0
                self._speech_text = self.state.text or ""
                d = self._emit_delta()
                if d:
                    deltas.append(d)
                self._stall = 0 if d else self._stall + 1
                if self._stall > self.max_stall:
                    self.max_stall = self._stall
                if self._should_rollover():
                    t = self._rollover()
                    if t:
                        deltas.append(t)
        return deltas

    def finalize(self) -> str:
        if self._done:
            return ""
        self._done = True
        if self._buf.size > 0:
            for _ in range(_EOF_RETRIES):
                try:
                    self._feed_chunk(self._buf)
                    break
                except QwenBusyError as e:
                    self.overload_skips += 1
                    if e.started:
                        break
            self._buf = np.zeros(0, dtype=np.float32)
            self._apply_window()
        try:
            if self._mel is not None:
                self._stream_call(_incr().finish_streaming_transcribe,
                                  self.model, self.state, self._mel)
            else:
                self._stream_call(self.model.finish_streaming_transcribe, self.state)
        except QwenBusyError:
            self.overload_skips += 1
        return self._emit_delta(final=True)
