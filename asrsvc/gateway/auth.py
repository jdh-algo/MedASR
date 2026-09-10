"""
网关鉴权:单一共享密钥(所有调用方共用一个 key)。

- HTTP:请求头 `Authorization: Bearer <key>`。
- WS 密钥传输优先级:子协议 `Sec-WebSocket-Protocol: bearer,<key>` > 查询参数 `?key=`(兼容 `?token=`)
  > 首帧 `{"event":"auth","token":"<key>"}`。
- `config.MEDASR_API_KEY` 为空时拒绝所有业务请求；健康检查保持公开。
- 校验用 `secrets.compare_digest`(常量时间比较,防计时侧信道)。

失败:HTTP 401;WS 以应用码 4401 关闭。所有校验在业务处理(离线下载、WS 拨号上游)之前完成。
"""
import asyncio
import json
import secrets

from fastapi import HTTPException, Request, WebSocket

from asrsvc import config

_WS_AUTH_TIMEOUT = int(config.MEDASR_LIMITS.get("ws_auth_timeout", 10) or 10)


def _bearer(authorization: str) -> str:
    if not authorization:
        return ""
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return ""


def _key_ok(presented) -> bool:
    """常量时间比较；未配置服务端密钥时拒绝请求。
    对任意输入都安全:非字符串直接拒绝;统一按 UTF-8 bytes 比较,
    避免 secrets.compare_digest 遇非 ASCII 字符抛 TypeError(否则未鉴权者可用畸形
    Authorization 头 / 首帧 token 触发 HTTP 500 或 WS handler 崩溃、绕过 4401)。"""
    if not config.MEDASR_API_KEY:
        return False
    if not isinstance(presented, str) or not presented:
        return False
    return secrets.compare_digest(presented.encode("utf-8"), config.MEDASR_API_KEY.encode("utf-8"))


async def require_key(request: Request) -> bool:
    """FastAPI 依赖:校验 Authorization: Bearer <key> 是否等于共享密钥。"""
    if _key_ok(_bearer(request.headers.get("Authorization", ""))):
        return True
    raise HTTPException(401, "密钥无效或缺失")


async def check_ws_key(ws: WebSocket) -> bool:
    """握手阶段校验 WS 密钥。成功返回 True 并已 accept;失败已 close(4401),返回 False。"""
    subs = list(ws.scope.get("subprotocols") or [])
    key = None
    accept_sub = None
    if "bearer" in subs:
        i = subs.index("bearer")
        if i + 1 < len(subs):
            key = subs[i + 1]
            accept_sub = "bearer"
    if not key:
        key = ws.query_params.get("key") or ws.query_params.get("token")

    if key is not None:
        await ws.accept(subprotocol=accept_sub)
    else:
        await ws.accept()
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=_WS_AUTH_TIMEOUT)
        except (asyncio.TimeoutError, Exception):
            await ws.close(code=4401)
            return False
        txt = msg.get("text") if isinstance(msg, dict) else None
        if txt:
            try:
                evt = json.loads(txt)
                if isinstance(evt, dict) and evt.get("event") == "auth":
                    key = evt.get("token") or evt.get("key")
            except (ValueError, TypeError):
                key = None

    if _key_ok(key or ""):
        return True
    await ws.close(code=4401)
    return False
