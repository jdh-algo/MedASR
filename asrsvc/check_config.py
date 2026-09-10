"""Validate combinations of performance-sensitive configuration values."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    quiet = "--quiet" in sys.argv
    from asrsvc import config as c

    warns = []

    def g(name, default=None):
        return getattr(c, name, default)

    if not c.MEDASR_API_KEY:
        warns.append("MEDASR_API_KEY 未设置，业务请求将被拒绝")

    if not g("MEDASR_INCR_ENCODE"):
        warns.append("MEDASR_INCR_ENCODE=False → 「越说越慢」与「显存持续上涨」会回来"
                     "(库内每块重喂全部累积音频,O(n²))。应设 MEDASR_INCR_ENCODE=1")
    if not g("MEDASR_MICRO_BATCH"):
        warns.append("MEDASR_MICRO_BATCH=False → 每卡并发退回 1(所有 generate 过 _MODEL_LOCK 串行)。"
                     "应设 MEDASR_MICRO_BATCH=1")
    if g("MEDASR_MICRO_BATCH") and not g("MEDASR_BATCH_INVARIANT"):
        warns.append("micro_batch 开但 batch_invariant 关可能导致批处理输出不一致。"
                     "应设 MEDASR_BATCH_INVARIANT=1")
    rmin = float(g("MEDASR_ROLLOVER_MIN_SEC", 0) or 0)
    if 0 < rmin < 300:
        warns.append(f"MEDASR_ROLLOVER_MIN_SEC={rmin:.0f}s 过短，频繁换段可能降低上下文一致性；"
                     f"建议至少 300 秒，并同时设置 rollover_max")
    chunk = float(g("MEDASR_CHUNK_SEC", 0) or 0)
    smt = int(g("MEDASR_STREAM_MAX_TOKENS", 0) or 0)
    need = int(-(-(2 * chunk * 6.0 + 5 + 8) // 16) * 16) if chunk else 0
    if need and smt < need:
        warns.append(f"MEDASR_STREAM_MAX_TOKENS={smt} 在 chunk_sec={chunk} 下会**静默截断输出**"
                     f"(前 2 步要整段吐出 {2*chunk:.0f}s 音频)。应 ≥{need},"
                     f"设 MEDASR_STREAM_MAX_TOKENS={need}")
    warm_sec = float(g("MEDASR_WARMUP_SEC", 0) or 0)
    warm_chunk = float(g("MEDASR_WARMUP_CHUNK_SEC", 0) or 0)
    warm_on = warm_sec > 0 and warm_chunk > 0
    first_char = 2 * (warm_chunk if warm_on else chunk)
    mx = int(g("MEDASR_MAX_LEN", 0) or 0)
    if mx and mx < 32768:
        warns.append(f"MEDASR_MAX_LEN={mx} 偏小 → 单段上下文不足约 {mx/18/60:.0f} 分钟就要换段")
    if not g("MEDASR_EMIT_UNLATCH"):
        warns.append("MEDASR_EMIT_UNLATCH=False 可能导致文本回缩后不再输出；"
                     "建议设 MEDASR_EMIT_UNLATCH=1")
    if not g("MEDASR_HEAD_GATE"):
        warns.append("MEDASR_HEAD_GATE=False 时，开头静音或纯音可能影响后续流式输出；"
                     "建议设 MEDASR_HEAD_GATE=1")
    if not g("MEDASR_SILENCE_RMS"):
        warns.append("MEDASR_SILENCE_RMS=0 → 静默恢复关闭，长静默后可能不再输出。建议 0.002")

    if not quiet or warns:
        print("[config] 生效配置:")
        rows = [
            ("增量编码 incr_encode", g("MEDASR_INCR_ENCODE")),
            ("跨会话微批 micro_batch", g("MEDASR_MICRO_BATCH")),
            ("批不变内核 batch_invariant", g("MEDASR_BATCH_INVARIANT")),
            ("流式块大小 chunk_sec", f"{chunk}s (首字≈{first_char:.1f}s)"),
            ("开头暖机 warmup", f"前 {warm_sec:.0f}s 用 {warm_chunk}s 块" if warm_on else "关"),
            ("解码上限 stream_max_tokens", f"{smt} (chunk={chunk} 需 ≥{need})"),
            ("换段软顶 rollover_min", f"{rmin:.0f}s"),
            ("换段硬顶 rollover_max", f"{float(g('MEDASR_ROLLOVER_MAX_SEC', 0) or 0):.0f}s"),
            ("换段带上下文 ctx_chars", g("MEDASR_ROLLOVER_CTX_CHARS")),
            ("上下文上限 max_model_len", f"{mx} (约 {mx/18/60:.0f} 分钟)" if mx else mx),
            ("提交指针解闩 emit_unlatch", g("MEDASR_EMIT_UNLATCH")),
            ("指针停滞自愈 stall_steps",
             f"{_ss} 步(≈{_ss * chunk:.0f}s)" if (_ss := int(g("MEDASR_STALL_STEPS", 0) or 0)) else "关"),
            ("开头门控 head_gate", g("MEDASR_HEAD_GATE")),
            ("VAD 端点", g("MEDASR_VAD_ENABLED")),
            ("pre-roll 重叠", f"{g('MEDASR_PREROLL_SEC')}s"),
            ("静默守护 silence_rms", g("MEDASR_SILENCE_RMS")),
            ("显存占比 gpu_util", g("MEDASR_GPU_UTIL")),
            ("每卡准入上限", c.MEDASR_LIMITS.get("max_stream_sessions_per_card")),
            ("网关端口", c.MEDASR_PORT),
        ]
        for k, v in rows:
            print(f"    {k:28s}= {v}")

    if warns:
        print()
        print(f"[config] !! {len(warns)} 项需要注意:")
        for w in warns:
            print(f"    - {w}")
        return 1
    if not quiet:
        print("[config] 检查通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
