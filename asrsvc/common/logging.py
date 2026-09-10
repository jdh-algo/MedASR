"""
日志:密钥脱敏 + 从访问日志中清洗 URL 上的 key/token 查询参数。
"""
import logging
import re

_TOKEN_RE = re.compile(r"(medasr_[A-Za-z0-9_\-]{6})[A-Za-z0-9_\-]+", re.IGNORECASE)
_QS_TOKEN_RE = re.compile(r"([?&](?:key|token)=)[^&\s\"']+", re.IGNORECASE)


def redact(text: str) -> str:
    """Redact MedASR tokens and key/token query parameters."""
    if not text:
        return text
    text = _TOKEN_RE.sub(r"\1***", text)
    text = _QS_TOKEN_RE.sub(r"\1***", text)
    return text


class RedactFilter(logging.Filter):
    """logging 过滤器:对 msg 做脱敏(挂到 uvicorn.access / root)。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = redact(record.msg)
            if record.args:
                record.args = tuple(
                    redact(a) if isinstance(a, str) else a for a in record.args
                )
        except Exception:
            pass
        return True


def install():
    """给 uvicorn 访问日志与 root 装上脱敏过滤器。"""
    f = RedactFilter()
    for name in ("", "uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).addFilter(f)
