"""
调度器:把网关收到的请求分发到各个 MedASR worker。
  - 离线:槽位队列(每卡 concurrency 个槽)实现每卡并发上限 + 天然排队;超过 (总槽位 + 队列上限)
    直接 429 背压;worker 可重试错误(502/503)则故障转移到其它卡(504=识别超时不转移)。
  - 流式:轮询挑卡(跳过未就绪),由网关 WS 代理逐个尝试连接、失败顺延。
  - 后台定期轮询各 worker /health,缓存就绪/负载。
"""
import asyncio
import itertools

import httpx

from asrsvc import config

_RETRIABLE_STATUS = {502, 503}


class OfflineBusy(Exception):
    """离线队列已满,应回 429。"""


class WorkerReject(Exception):
    """worker 明确拒绝(4xx),原样上抛(如音频不合规/超时)。"""

    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


class UpstreamError(Exception):
    """所有候选 worker 都失败,应回 502/504。"""


class Dispatcher:
    def __init__(self, name, offline_upstreams, stream_upstreams,
                 offline_health, stream_health, conc, queue_max, job_timeout,
                 max_stream_per_card=0):
        self.name = name
        self._offline = list(offline_upstreams)
        self._stream = list(stream_upstreams)
        self._offline_health = list(offline_health)
        self._stream_health = list(stream_health)
        self._conc = conc
        self._queue_max = queue_max
        self._job_timeout = job_timeout
        self._max_failover = min(len(self._offline), 3) if self._offline else 1
        self._slots = None
        self._pending = 0
        self._rr = itertools.count()
        self._health = {}
        self._client = None
        self._poll_task = None
        self._max_inflight = max(1, len(self._offline)) * conc
        self._max_stream_per_card = max(0, int(max_stream_per_card))
        self._stream_active = [0] * len(self._stream)

    async def start(self):
        self._slots = asyncio.Queue()
        for idx in range(len(self._offline)):
            for _ in range(self._conc):
                self._slots.put_nowait(idx)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._job_timeout + 30, connect=10),
                                         trust_env=False)
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        if self._poll_task:
            self._poll_task.cancel()
        if self._client:
            await self._client.aclose()

    async def _poll_loop(self):
        urls = list(dict.fromkeys(self._offline_health + self._stream_health))
        while True:
            for url in urls:
                try:
                    r = await self._client.get(url, timeout=3)
                    self._health[url] = r.json() if r.status_code == 200 else {"ready": False}
                except Exception:
                    self._health[url] = {"ready": False}
            await asyncio.sleep(5)

    async def dispatch_offline(self, path: str, lang: str) -> dict:
        if not self._offline:
            raise UpstreamError(f"[{self.name}] 无离线 worker")
        if self._pending >= self._max_inflight + self._queue_max:
            raise OfflineBusy()
        self._pending += 1
        try:
            attempts = 0
            last = None
            while attempts < self._max_failover:
                idx = await self._slots.get()
                try:
                    r = await self._client.post(self._offline[idx] + "/offline",
                                                json={"path": path, "lang": lang})
                    if r.status_code == 200:
                        return r.json()
                    if r.status_code in _RETRIABLE_STATUS:
                        last = f"HTTP {r.status_code}"
                        attempts += 1
                        continue
                    raise WorkerReject(r.status_code, _detail(r))
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    last = f"{type(e).__name__}: {e}"
                    attempts += 1
                finally:
                    self._slots.put_nowait(idx)
            raise UpstreamError(f"[{self.name}] 离线 worker 均不可用:{last}")
        finally:
            self._pending -= 1

    def _ready_idx(self, i) -> bool:
        return bool(self._health.get(self._stream_health[i], {}).get("ready"))

    def pick_stream_target(self, exclude=None):
        """挑一张卡承接新流式会话:ready 且未达每卡上限中,取当前会话数最少者(并列按轮询错开)。
        选中即占位(会话计数 +1,防突发并发在校验与占位之间超额);连接失败由网关 release_stream 归还。
        返回 (idx, ws_url);无可用卡返回 None(容量已满或都不 ready)。同步方法、事件循环内调用,无竞态。"""
        exclude = exclude or set()
        n = len(self._stream)
        if n == 0:
            return None
        cands = [i for i in range(n) if i not in exclude and self._ready_idx(i)
                 and (self._max_stream_per_card == 0 or self._stream_active[i] < self._max_stream_per_card)]
        if not cands:
            return None
        spin = next(self._rr)
        idx = min(cands, key=lambda i: (self._stream_active[i], (i + spin) % n))
        self._stream_active[idx] += 1
        return idx, self._stream[idx]

    def has_ready_stream(self) -> bool:
        """是否存在 ready 的流式卡(与容量无关)。用于区分「全满(4429)」和「都不可用(1013)」。"""
        return any(self._ready_idx(i) for i in range(len(self._stream)))

    def release_stream(self, idx):
        """归还一个流式会话占位(连接失败或会话结束)。"""
        if idx is not None and 0 <= idx < len(self._stream_active) and self._stream_active[idx] > 0:
            self._stream_active[idx] -= 1

    def snapshot(self) -> dict:
        off = [self._health.get(u, {}) for u in self._offline_health]
        stm = [self._health.get(u, {}) for u in self._stream_health]
        return {
            "offline_ready": sum(1 for h in off if h.get("ready")),
            "offline_total": len(self._offline),
            "stream_ready": sum(1 for h in stm if h.get("ready")),
            "stream_total": len(self._stream),
            "offline_pending": self._pending,
            "stream_active_per_card": list(self._stream_active),
            "stream_active_total": sum(self._stream_active),
            "max_stream_per_card": self._max_stream_per_card,
        }


def _detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("detail") or r.text)[:300]
    except Exception:
        return (r.text or "")[:300]


_medasr_health = [f"http://{config.MEDASR_UPSTREAM_HOST}:{p}/health" for _, p in config.MEDASR_INSTANCE_PLAN]
dispatcher = Dispatcher(
    "medasr",
    config.MEDASR_OFFLINE_UPSTREAMS, config.MEDASR_STREAM_UPSTREAMS,
    _medasr_health, _medasr_health,
    config.MEDASR_OFFLINE_CONCURRENCY, int(config.MEDASR_LIMITS.get("offline_queue_max", 64) or 64),
    config.MEDASR_JOB_TIMEOUT,
    max_stream_per_card=int(config.MEDASR_LIMITS.get("max_stream_sessions_per_card", 0) or 0),
)
