"""
音频守卫:转写前探测文件体积与时长,拒绝超大/超长(防解码炸弹与资源耗尽)。
优先用 ffprobe(支持 mp3/m4a 等 soundfile 读不到时长的格式),回退 soundfile。
"""
import json
import shutil
import subprocess
from pathlib import Path


class AudioRejected(Exception):
    """音频不合规(体积/时长超限、无法解析)。"""


def _ffprobe_duration(path: str):
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        if out.returncode != 0:
            return None
        d = json.loads(out.stdout.decode("utf-8", "ignore") or "{}")
        dur = (d.get("format") or {}).get("duration")
        return float(dur) if dur is not None else None
    except Exception:
        return None


def _soundfile_duration(path: str):
    try:
        import soundfile as sf
        info = sf.info(str(path))
        if info.samplerate:
            return info.frames / float(info.samplerate)
    except Exception:
        return None
    return None


def probe(path: str, max_bytes: int, max_seconds: int) -> dict:
    """校验并返回 {size_bytes, duration}。超限抛 AudioRejected。"""
    p = Path(path)
    if not p.is_file():
        raise AudioRejected(f"文件不存在:{path}")
    size = p.stat().st_size
    if size <= 0:
        raise AudioRejected("空文件")
    if max_bytes and size > max_bytes:
        raise AudioRejected(f"文件过大:{size} > {max_bytes} 字节")

    dur = _ffprobe_duration(str(p))
    if dur is None:
        dur = _soundfile_duration(str(p))
    if dur is None:
        raise AudioRejected("无法解析音频时长(格式不支持或文件损坏)")
    if max_seconds and dur > max_seconds:
        raise AudioRejected(f"音频过长:{dur:.0f}s > {max_seconds}s")
    return {"size_bytes": size, "duration": round(dur, 2)}
