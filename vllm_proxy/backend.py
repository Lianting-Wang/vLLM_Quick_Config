from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import aiohttp

from .config import BackendConfig
from .gpu_monitor import GpuMonitor

LOGGER = logging.getLogger(__name__)


class BackendState(StrEnum):
    OFFLINE = "offline"
    AWAKE = "awake"
    SLEEP_PENDING = "sleep_pending"
    SLEEPING = "sleeping"
    WAKING = "waking"
    ERROR = "error"


class BackendController:
    def __init__(self, config: BackendConfig, session: aiohttp.ClientSession, gpu_monitor: GpuMonitor) -> None:
        self.config = config
        self.session = session
        self.gpu_monitor = gpu_monitor
        self.state = BackendState.OFFLINE
        self.active_requests = 0
        self.last_activity_monotonic = time.monotonic()
        self.last_activity_at = datetime.now(timezone.utc)
        self.last_error = ""
        self.last_sleep_duration_ms: int | None = None
        self.last_wake_duration_ms: int | None = None
        self._state_lock = asyncio.Lock()
        self._wake_task: asyncio.Task[None] | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._closed = False

    def start(self) -> None:
        self._idle_task = asyncio.create_task(self._idle_loop(), name=f"idle-{self.config.id}")

    async def close(self) -> None:
        self._closed = True
        if self._idle_task:
            self._idle_task.cancel()
            await asyncio.gather(self._idle_task, return_exceptions=True)

    def update_config(self, config: BackendConfig) -> None:
        self.config = config

    async def _request(self, method: str, path: str, **kwargs: Any) -> aiohttp.ClientResponse:
        url = f"{self.config.upstream_url}{path}"
        return await self.session.request(method, url, **kwargs)

    async def probe(self) -> dict[str, Any]:
        sleeping: bool | None = None
        healthy = False
        error = ""
        try:
            async with await self._request("GET", "/is_sleeping", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    value = await response.json(content_type=None)
                    sleeping = bool(value if isinstance(value, bool) else value.get("is_sleeping", value.get("sleeping", False)))
            if sleeping is True:
                healthy = True
                self.state = BackendState.SLEEPING
            else:
                async with await self._request("GET", "/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    healthy = response.status < 500
                self.state = BackendState.AWAKE if healthy else BackendState.ERROR
        except Exception as exc:
            error = str(exc)
            self.state = BackendState.OFFLINE
        self.last_error = error
        return {"healthy": healthy, "is_sleeping": sleeping, "error": error}

    async def request_started(self, counts_as_activity: bool) -> None:
        async with self._state_lock:
            self.active_requests += 1
            if counts_as_activity:
                self.last_activity_monotonic = time.monotonic()
                self.last_activity_at = datetime.now(timezone.utc)

    async def request_finished(self, counts_as_activity: bool) -> None:
        async with self._state_lock:
            self.active_requests = max(0, self.active_requests - 1)
            if counts_as_activity:
                self.last_activity_monotonic = time.monotonic()
                self.last_activity_at = datetime.now(timezone.utc)

    async def ensure_awake(self) -> None:
        # Normal traffic should not incur a health round trip on every request.
        # State changes to sleeping are owned by this controller, so AWAKE is a
        # safe fast path; failures during forwarding are still returned as 502.
        if self.state == BackendState.AWAKE:
            return
        async with self._state_lock:
            if self.state == BackendState.AWAKE:
                return
            if self._wake_task and not self._wake_task.done():
                task = self._wake_task
            else:
                task = asyncio.create_task(self._wake_once(), name=f"wake-{self.config.id}")
                self._wake_task = task
        await task

    async def _wake_once(self) -> None:
        start = time.monotonic()
        try:
            probe = await self.probe()
            if probe["healthy"] and probe["is_sleeping"] is not True:
                self.state = BackendState.AWAKE
                return
            if probe["is_sleeping"] is not True:
                raise RuntimeError(probe["error"] or "vLLM backend is unavailable")
            self.state = BackendState.WAKING
            LOGGER.info("backend=%s action=wake_requested", self.config.id)
            async with await self._request(
                "POST", "/wake_up", timeout=aiohttp.ClientTimeout(total=self.config.wake_timeout_seconds)
            ) as response:
                if response.status >= 400:
                    raise RuntimeError(f"wake_up returned HTTP {response.status}: {await response.text()}")
            deadline = time.monotonic() + self.config.wake_timeout_seconds
            while time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                probe = await self.probe()
                if probe["healthy"] and probe["is_sleeping"] is not True:
                    self.state = BackendState.AWAKE
                    self.last_error = ""
                    self.last_wake_duration_ms = round((time.monotonic() - start) * 1000)
                    LOGGER.info("backend=%s action=wake_complete duration_ms=%s", self.config.id, self.last_wake_duration_ms)
                    return
            raise TimeoutError(f"backend did not wake within {self.config.wake_timeout_seconds} seconds")
        except Exception as exc:
            self.state = BackendState.ERROR
            self.last_error = str(exc)
            LOGGER.exception("backend=%s action=wake_failed", self.config.id)
            raise

    async def sleep(self, *, force: bool = False) -> dict[str, Any]:
        async with self._state_lock:
            if self.active_requests > 0 and not force:
                raise RuntimeError(f"{self.active_requests} active request(s) are still running")
            self.state = BackendState.SLEEP_PENDING
        if force:
            while self.active_requests > 0:
                await asyncio.sleep(0.2)
        probe = await self.probe()
        if probe["is_sleeping"] is True:
            return {"status": "already_sleeping"}
        if not probe["healthy"]:
            raise RuntimeError(probe["error"] or "vLLM backend is unavailable")
        start = time.monotonic()
        async with self._state_lock:
            if self.active_requests > 0:
                self.state = BackendState.AWAKE
                raise RuntimeError("a request arrived before sleep began")
            self.state = BackendState.SLEEP_PENDING
            async with await self._request(
                "POST",
                f"/sleep?level={self.config.sleep_level}",
                timeout=aiohttp.ClientTimeout(total=self.config.wake_timeout_seconds),
            ) as response:
                if response.status >= 400:
                    raise RuntimeError(f"sleep returned HTTP {response.status}: {await response.text()}")
        deadline = time.monotonic() + self.config.wake_timeout_seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            probe = await self.probe()
            if probe["is_sleeping"] is True:
                self.state = BackendState.SLEEPING
                self.last_sleep_duration_ms = round((time.monotonic() - start) * 1000)
                self.last_error = ""
                LOGGER.info("backend=%s action=sleep_complete duration_ms=%s", self.config.id, self.last_sleep_duration_ms)
                return {"status": "sleeping"}
        self.state = BackendState.ERROR
        raise TimeoutError("backend did not enter sleep mode")

    async def _idle_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(1)
            if self.config.idle_timeout_seconds <= 0:
                continue
            idle_seconds = time.monotonic() - self.last_activity_monotonic
            if self.active_requests == 0 and idle_seconds >= self.config.idle_timeout_seconds:
                try:
                    probe = await self.probe()
                    if probe["healthy"] and probe["is_sleeping"] is not True:
                        LOGGER.info("backend=%s action=auto_sleep idle_seconds=%d", self.config.id, int(idle_seconds))
                        await self.sleep()
                except Exception as exc:
                    self.last_error = str(exc)
                    await asyncio.sleep(5)

    def status(self) -> dict[str, Any]:
        idle_seconds = max(0, int(time.monotonic() - self.last_activity_monotonic))
        process_running = bool(self.gpu_monitor.process_tree(self.config.pidfile)) if self.config.pidfile else None
        remaining = max(0, self.config.idle_timeout_seconds - idle_seconds)
        return {
            "id": self.config.id,
            "name": self.config.name,
            "state": self.state.value,
            "upstream_url": self.config.upstream_url,
            "profile": self.config.profile,
            "process_running": process_running,
            "active_requests": self.active_requests,
            "last_activity_at": self.last_activity_at.isoformat(),
            "idle_seconds": idle_seconds,
            "sleep_in_seconds": remaining if self.state == BackendState.AWAKE else None,
            "last_error": self.last_error,
            "last_sleep_duration_ms": self.last_sleep_duration_ms,
            "last_wake_duration_ms": self.last_wake_duration_ms,
            "listeners": [item.to_dict() for item in self.config.listeners],
            "gpu": self.gpu_monitor.snapshot(self.config.pidfile),
        }
