# Copyright 2026 The Alibaba Qwen team.
# Copyright 2026 MedASR contributors.
# SPDX-License-Identifier: Apache-2.0
# Portions of this file are adapted from Qwen3-ASR and modified for MedASR.
"""Incremental Qwen3-ASR feature extraction and reusable prefix caching."""
import itertools

import numpy as np
import torch

from asrsvc import config

HOP = 160
N_FFT = 400
FRAMES_PER_SEC = 100
TOK_PER_SEC = 13
_PAD = "<|audio_pad|>"
_SID = itertools.count()


def patch_vllm():
    """Enable precomputed audio features in the supported Qwen3-ASR/vLLM stack."""
    from itertools import chain, repeat

    from qwen_asr.core.transformers_backend import processing_qwen3_asr as HP
    from qwen_asr.core.vllm_backend import qwen3_asr as P

    if getattr(P, "_asrsvc_dict_items", False):
        return

    P.Qwen3ASRProcessingInfo.get_data_parser = lambda self: P.Qwen3ASRMultiModalDataParser(
        target_sr=self.get_feature_extractor().sampling_rate)

    _orig = HP.Qwen3ASRProcessor.replace_multimodal_special_tokens
    HP.Qwen3ASRProcessor.replace_multimodal_special_tokens = (
        lambda self, text, audio_lengths: _orig(self, text, chain(audio_lengths, repeat(1))))
    P._asrsvc_dict_items = True


def llm_kwargs():
    """Return vLLM options required by incremental encoding."""
    max_items = int(config.MEDASR_MAX_LEN / (TOK_PER_SEC * config.MEDASR_INCR_BLOCK_SEC)) + 2
    return {"limit_mm_per_prompt": {"audio": max_items}, "enable_mm_embeds": True}


class MelStream:
    """Per-session incremental log-mel cache."""

    _GROW = 30_000

    def __init__(self, feature_extractor, sid=""):
        self.sid = f"{sid}#{next(_SID)}"
        self._mf = torch.from_numpy(feature_extractor.mel_filters).to(torch.float32)
        self._win = torch.hann_window(N_FFT)
        self.raw = torch.empty(128, 0)
        self.norm = torch.empty(128, 0)
        self.gmax = -float("inf")
        self.epoch = 0
        self.n_samples = 0
        self.nf = 0
        self._done = 0
        self._tail = np.empty(0, dtype=np.float32)
        self._tail_at = 0
        self.n_prompt = self.n_cached = 0

    def _raw_of(self, x):
        w = torch.from_numpy(x).to(torch.float32).unsqueeze(0)
        s = torch.stft(w, N_FFT, HOP, window=self._win, return_complex=True)
        return torch.clamp(self._mf.T @ (s[..., :-1].abs() ** 2), min=1e-10).log10()[0]

    def _reserve(self, nf):
        if self.raw.shape[1] >= nf:
            return
        cap = ((nf // self._GROW) + 1) * self._GROW
        for name in ("raw", "norm"):
            old = getattr(self, name)
            new = torch.empty(128, cap)
            if old.shape[1]:
                new[:, :old.shape[1]] = old
            setattr(self, name, new)

    def add(self, samples: np.ndarray):
        """Append float32 audio samples and update the feature cache."""
        self._tail = np.concatenate([self._tail, np.asarray(samples, dtype=np.float32)])
        self.n_samples += len(samples)
        nf = self.n_samples // HOP
        if nf <= self.nf:
            return
        self._reserve(nf)

        prev_done = self._done
        a = max(0, prev_done - 2)
        s = a * HOP - self._tail_at
        assert s >= 0, "音频尾巴保留不足,无法重算未定稿帧"
        block = self._raw_of(self._tail[s:])
        keep = 0 if a == 0 else 2
        self.raw[:, a + keep: nf] = block[:, keep: nf - a]

        self.nf = nf
        self._done = min(nf, (self.n_samples - N_FFT // 2) // HOP + 1)
        keep_from = max(0, (self._done - 2) * HOP)
        self._tail = self._tail[keep_from - self._tail_at:]
        self._tail_at = keep_from

        g = float(self.raw[:, :nf].max())
        if g > self.gmax:
            self.gmax, self.epoch, lo = g, self.epoch + 1, 0
        else:
            lo = 0 if a == 0 else prev_done
        v = self.raw[:, lo:nf]
        self.norm[:, lo:nf] = (torch.maximum(v, torch.tensor(self.gmax - 8.0)) + 4.0) / 4.0

    def mm(self, block_frames):
        """Split cached features into reusable multimodal items."""
        n_full = min(self.nf // block_frames, self._done // block_frames)
        lens = [block_frames] * n_full
        uuids = [f"{self.sid}:{self.epoch}:{i}" for i in range(n_full)]
        r = self.nf - n_full * block_frames
        if r >= FRAMES_PER_SEC:
            lens.append(r)
            uuids.append(f"{self.sid}:{self.epoch}:t{self.nf}")
        elif r and lens:
            lens[-1] += r
            uuids[-1] = f"{self.sid}:{self.epoch}:t{self.nf}"
        elif r:
            lens, uuids = [r], [f"{self.sid}:{self.epoch}:t{self.nf}"]
        return self.norm[:, :self.nf].contiguous(), lens, uuids


def _mm_data(feats, lens):
    m = max(lens)
    mask = torch.zeros(len(lens), m, dtype=torch.long)
    for i, L in enumerate(lens):
        mask[i, :L] = 1
    return {"audio": {"input_audio_features": feats,
                      "audio_feature_lengths": torch.tensor(lens, dtype=torch.long),
                      "feature_attention_mask": mask}}


def _rollback_prefix(tok, state, guard_replacement):
    """Build a stable text prefix while avoiding partial replacement characters."""
    if state.chunk_id < state.unfixed_chunk_num:
        return ""
    cur_ids = tok.encode(state._raw_decoded)
    k = int(state.unfixed_token_num)
    if not guard_replacement:
        return tok.decode(cur_ids[: max(1, len(cur_ids) - k)])
    while True:
        end_idx = max(0, len(cur_ids) - k)
        prefix = tok.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
        if "�" not in prefix:
            return prefix
        k += 1


def _step(model, state, mel, gen, guard=True):
    """Build the prompt, generate one step, and update streaming state."""
    from qwen_asr.inference.utils import parse_asr_output

    prefix = _rollback_prefix(model.processor.tokenizer, state, guard)
    feats, lens, uuids = mel.mm(int(config.MEDASR_INCR_BLOCK_SEC * FRAMES_PER_SEC))
    inp = {"prompt": state.prompt_raw.replace(_PAD, _PAD * len(lens)) + prefix,
           "multi_modal_data": _mm_data(feats, lens),
           "multi_modal_uuids": {"audio": uuids}}
    out = gen([inp], sampling_params=model.sampling_params, use_tqdm=False)
    mel.n_prompt = len(out[0].prompt_token_ids or [])
    mel.n_cached = int(getattr(out[0], "num_cached_tokens", -1) or -1)
    state._raw_decoded = prefix + out[0].outputs[0].text
    state.language, state.text = parse_asr_output(
        state._raw_decoded, user_language=state.force_language)
    state.chunk_id += 1


def streaming_transcribe(model, pcm, state, mel: MelStream, generate=None):
    """Process new PCM data using cached incremental features."""
    gen = generate if generate is not None else model.model.generate
    x = np.asarray(pcm)
    if x.ndim != 1:
        x = x.reshape(-1)
    x = (x.astype(np.float32) / 32768.0) if x.dtype == np.int16 else x.astype(np.float32, copy=False)
    if x.shape[0]:
        state.buffer = np.concatenate([state.buffer, x], axis=0)

    while state.buffer.shape[0] >= state.chunk_size_samples:
        mel.add(state.buffer[: state.chunk_size_samples])
        state.buffer = state.buffer[state.chunk_size_samples:]
        _step(model, state, mel, gen)
    return state


def finish_streaming_transcribe(model, state, mel: MelStream, generate=None):
    """Flush the final partial chunk and update streaming state."""
    if state.buffer is None or state.buffer.shape[0] == 0:
        return state
    tail = state.buffer
    state.buffer = np.zeros((0,), dtype=np.float32)
    if len(tail) % HOP:
        tail = np.pad(tail, (0, HOP - len(tail) % HOP))
    mel.add(tail)
    _step(model, state, mel, generate if generate is not None else model.model.generate,
          guard=False)
    return state
