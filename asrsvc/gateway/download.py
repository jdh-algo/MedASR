"""
音频 URL 下载器(纯 Python,不用 wget/curl 子进程)。用于把用户给的音频 URL 拉到本地 spool。

下载策略(不做 DNS 分类,避免误判):
  **先走代理 → 代理失败再直连 → 都失败才报错。**
  - 可选 HTTP/HTTPS 正向代理
    (证书仍校验目标主机名)。代理连不上/CONNECT 被拒/返回 ≥400 视为“代理失败”,切直连。
  - 直连:解析并**钉到已校验 IP**(HTTPS 仍按主机名做 SNI/证书)。
  - 未配置 proxy → 直接走直连。

安全(最低限度,不影响正常公网下载):
  - 仅 http/https;可选 host 白名单。
  - URL 主机若是 **IP 字面量** 且属内网/元数据(169.254.169.254 等)→ 直接拒绝(无需 DNS)。
  - **直连兜底**这步仍做 SSRF 校验(解析全部 IP,内网/保留一律拒绝,除非 block_private=false)+ 钉 IP,
    避免“代理失败回退直连”被用作 SSRF 通道。
  - 每跳重定向重新校验;限跳数。流式落盘带体积/超时上限;mkstemp 0600。
  - SSRF 命中是硬拦截,绝不因此回退到另一种方式。
下载只应在网关进程调用;worker 永不碰 URL。
"""
import http.client
import ipaddress
import os
import socket
import ssl
import tempfile
import time
import urllib.parse
from pathlib import Path

from asrsvc import config

_DL = config.MEDASR_DOWNLOAD_CONFIG
_SPOOL = Path(config.MEDASR_SPOOL_DIR)
_PROXY = str(_DL.get("proxy", "") or "").strip()
_EXTRA_CIDRS = [ipaddress.ip_network(c) for c in _DL.get("block_cidrs", [])]
_ALWAYS_BLOCK = [ipaddress.ip_network("100.64.0.0/10")]
_REDIRECT_CODES = (301, 302, 303, 307, 308)


class DownloadError(Exception):
    pass


class SSRFBlocked(DownloadError):
    pass


def _is_internal_ip(ip_str: str) -> bool:
    """IP 是否属内网/保留/不可作公网目标(与 block_private 无关)。无法解析视为内网(保守)。"""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
            or ip.is_multicast or ip.is_unspecified):
        return True
    for net in _ALWAYS_BLOCK + _EXTRA_CIDRS:
        if ip.version == net.version and ip in net:
            return True
    return False


def _ip_allowed(ip_str: str) -> bool:
    """SSRF 策略下该 IP 是否放行:block_private 关闭时全放行;否则内网 IP 一律拒绝。"""
    if not _DL.get("block_private", True):
        return True
    return not _is_internal_ip(ip_str)


def _literal_guard(host: str):
    """host 若是 IP 字面量,直接按 SSRF 策略校验(无需 DNS);域名放过(交由代理/直连兜底)。"""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return
    if not _ip_allowed(host):
        raise SSRFBlocked(f"目标地址被拦截(内网/元数据):{host}")


def _resolve_and_check(host: str, port: int) -> list:
    """解析 host 所有地址并按 SSRF 策略校验;全部通过才返回 [(family, ip)],否则抛。仅直连兜底用。"""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise DownloadError(f"域名解析失败:{host} ({e})")
    ips = []
    for family, _, _, _, sockaddr in infos:
        ip = sockaddr[0]
        if not _ip_allowed(ip):
            raise SSRFBlocked(f"目标地址被拦截(疑似内网/元数据):{host} → {ip}")
        ips.append((family, ip))
    if not ips:
        raise DownloadError(f"域名无可用地址:{host}")
    return ips


def _check_host_allowlist(host: str):
    allow = [h.lower() for h in (_DL.get("allow_hosts") or [])]
    if allow and host.lower() not in allow:
        raise SSRFBlocked(f"host 不在白名单:{host}")


def _open_pinned(host: str, port: int, is_https: bool, timeout: int, ips: list):
    """直连:连接到已校验 IP;HTTPS 用主机名做 SNI + 证书校验。返回 conn。"""
    last = None
    for family, ip in ips:
        try:
            raw = socket.create_connection((ip, port), timeout=timeout)
        except OSError as e:
            last = e
            continue
        if is_https:
            ctx = ssl.create_default_context()
            sock = ctx.wrap_socket(raw, server_hostname=host)
        else:
            sock = raw
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.sock = sock
        return conn
    raise DownloadError(f"连接失败:{host} ({last})")


def _open_via_proxy(scheme: str, host: str, port: int, timeout: int):
    """走正向代理:HTTP 用绝对 URI,HTTPS 用 CONNECT 隧道(证书仍校验目标主机名)。返回 conn。"""
    pu = urllib.parse.urlsplit(_PROXY)
    phost, pport = pu.hostname, (pu.port or 3128)
    if not phost:
        raise DownloadError(f"代理地址无效:{_PROXY}")
    if scheme == "https":
        ctx = ssl.create_default_context()
        conn = http.client.HTTPSConnection(phost, pport, timeout=timeout, context=ctx)
        conn.set_tunnel(host, port)
    else:
        conn = http.client.HTTPConnection(phost, pport, timeout=timeout)
    return conn


def _parse(url: str):
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https"):
        raise SSRFBlocked(f"仅允许 http/https:{u.scheme or '空'}")
    host = u.hostname
    if not host:
        raise DownloadError("URL 缺少主机名")
    try:
        port = u.port or (443 if u.scheme == "https" else 80)
    except ValueError:
        raise SSRFBlocked(f"非法端口:{url}")
    path = urllib.parse.urlunsplit(("", "", u.path or "/", u.query, ""))
    return u.scheme, host, port, path


def _fetch_once(method, scheme, host, port, path, cur, timeout):
    """按 method(proxy/direct)发一次请求。成功(2xx/3xx)返回 (conn, resp);失败抛 DownloadError(以便切换)。
    direct 会先做 SSRF 解析校验(可能抛 SSRFBlocked,硬拦截不回退)。"""
    if method == "proxy":
        conn = _open_via_proxy(scheme, host, port, timeout)
        req_target = cur if scheme == "http" else (path or "/")
    else:
        ips = _resolve_and_check(host, port)
        conn = _open_pinned(host, port, scheme == "https", timeout, ips)
        req_target = path or "/"
    try:
        conn.request("GET", req_target, headers={
            "Host": host, "User-Agent": "medasr/0.1.0", "Accept": "*/*",
        })
        resp = conn.getresponse()
    except (OSError, http.client.HTTPException) as e:
        conn.close()
        raise DownloadError(f"{method} 连接失败:{type(e).__name__}: {e}")
    if resp.status >= 400:
        st = resp.status
        conn.close()
        raise DownloadError(f"{method} 返回 HTTP {st}")
    return conn, resp


def download(url: str) -> dict:
    """下载 url 到 spool,返回 {path, size, url}。失败抛 DownloadError/SSRFBlocked。调用方负责删文件。"""
    max_bytes = int(_DL.get("max_bytes", 0) or 0)
    total_timeout = int(_DL.get("timeout", 300) or 300)
    connect_timeout = int(_DL.get("connect_timeout", 10) or 10)
    max_redirects = int(_DL.get("max_redirects", 3) or 0)
    methods = (["proxy"] if _PROXY else []) + ["direct"]

    deadline = time.monotonic() + total_timeout
    cur = url
    for _hop in range(max_redirects + 1):
        if time.monotonic() > deadline:
            raise DownloadError("下载超时")
        scheme, host, port, path = _parse(cur)
        _check_host_allowlist(host)
        _literal_guard(host)
        op_timeout = max(1, min(connect_timeout, int(deadline - time.monotonic())))

        conn = resp = None
        last = None
        for m in methods:
            try:
                conn, resp = _fetch_once(m, scheme, host, port, path, cur, op_timeout)
                break
            except SSRFBlocked:
                raise
            except DownloadError as e:
                last = e
                continue
        if resp is None:
            raise last or DownloadError("下载失败(代理与直连均不可用)")

        try:
            if resp.status in _REDIRECT_CODES:
                loc = resp.getheader("Location")
                if not loc:
                    raise DownloadError(f"重定向缺少 Location(HTTP {resp.status})")
                cur = urllib.parse.urljoin(cur, loc)
                continue
            clen = resp.getheader("Content-Length")
            if clen and clen.isdigit() and max_bytes and int(clen) > max_bytes:
                raise DownloadError(f"文件过大(Content-Length {clen} > {max_bytes})")
            return _stream_to_spool(resp, cur, max_bytes, deadline)
        finally:
            conn.close()
    raise DownloadError(f"重定向次数超限(>{max_redirects})")


def _stream_to_spool(resp, url: str, max_bytes: int, deadline: float) -> dict:
    base = Path(urllib.parse.urlsplit(url).path).name or "audio"
    suffix = Path(base).suffix or ".wav"
    _SPOOL.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="dl_", suffix=suffix, dir=str(_SPOOL))
    os.chmod(tmp, 0o600)
    written = 0
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                if time.monotonic() > deadline:
                    raise DownloadError("下载超时")
                chunk = resp.read(65536)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes and written > max_bytes:
                    raise DownloadError(f"文件超过大小上限 {max_bytes} 字节")
                f.write(chunk)
        if written == 0:
            raise DownloadError("下载到空文件")
        return {"path": tmp, "size": written, "url": url}
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
