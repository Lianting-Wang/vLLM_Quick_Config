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
    """Coordinate proxy traffic with vLLM sleep and wake transitions.

    There are two distinct synchronization concerns:

    * ``_state_lock`` protects local counters, state, and task references.
    * ``_transition_lock`` serializes actual upstream sleep/wake operations.

    Sleep and wake callers share one task per operation.  Serializing the
    upstream transitions is essential: a request arriving while ``/sleep`` is
    running must wait for that sleep transition to settle, then wake the model.
    Otherwise the older sleep coroutine can keep polling after the model has
    already been woken and incorrectly overwrite the state with ``ERROR``.
    """

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
        self._no_active_requests = asyncio.Condition(self._state_lock)
        self._transition_lock = asyncio.Lock()
        self._sleep_task: asyncio.Task[dict[str, Any]] | None = None
        self._wake_task: asyncio.Task[None] | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._closed = False

    def start(self) -> None:
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_loop(), name=f"idle-{self.config.id}")

    async def close(self) -> None:
        self._closed = True
        tasks = [task for task in (self._idle_task, self._sleep_task, self._wake_task) if task and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def update_config(self, config: BackendConfig) -> None:
        self.config = config

    async def _request(self, method: str, path: str, **kwargs: Any) -> aiohttp.ClientResponse:
        url = f"{self.config.upstream_url}{path}"
        return await self.session.request(method, url, **kwargs)

    def _transition_active_unlocked(self) -> bool:
        return bool(
            (self._sleep_task is not None and not self._sleep_task.done())
            or (self._wake_task is not None and not self._wake_task.done())
            or self._transition_lock.locked()
        )

    async def _observe_backend(self) -> dict[str, Any]:
        """Read upstream state without mutating the controller state."""

        sleeping: bool | None = None
        healthy = False
        error = ""
        try:
            async with await self._request(
                "GET",
                "/is_sleeping",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    value = await response.json(content_type=None)
                    if isinstance(value, bool):
                        sleeping = value
                    elif isinstance(value, dict):
                        sleeping = bool(value.get("is_sleeping", value.get("sleeping", False)))

            if sleeping is True:
                # A sleeping vLLM process is still a healthy, controllable backend.
                healthy = True
            else:
                async with await self._request(
                    "GET",
                    "/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    healthy = response.status < 500
        except Exception as exc:
            error = str(exc)

        return {"healthy": healthy, "is_sleeping": sleeping, "error": error}

    async def probe(self) -> dict[str, Any]:
        """Probe the backend and update stable state only when no transition owns it.

        The admin UI may call this method while a sleep or wake operation is in
        progress.  Such a probe must not replace ``SLEEP_PENDING`` or ``WAKING``
        with a transient observation.
        """

        result = await self._observe_backend()
        async with self._state_lock:
            if not self._transition_active_unlocked():
                if result["is_sleeping"] is True:
                    self.state = BackendState.SLEEPING
                elif result["healthy"]:
                    self.state = BackendState.AWAKE
                elif result["error"]:
                    self.state = BackendState.OFFLINE
                else:
                    self.state = BackendState.ERROR
                self.last_error = result["error"]
        return result

    async def request_started(self, counts_as_activity: bool) -> None:
        async with self._no_active_requests:
            self.active_requests += 1
            if counts_as_activity:
                self.last_activity_monotonic = time.monotonic()
                self.last_activity_at = datetime.now(timezone.utc)

    async def request_finished(self, counts_as_activity: bool) -> None:
        async with self._no_active_requests:
            self.active_requests = max(0, self.active_requests - 1)
            if counts_as_activity:
                self.last_activity_monotonic = time.monotonic()
                self.last_activity_at = datetime.now(timezone.utc)
            if self.active_requests == 0:
                self._no_active_requests.notify_all()

    async def ensure_awake(self) -> None:
        """Ensure the backend is awake, sharing one wake operation across callers."""

        async with self._state_lock:
            sleep_in_progress = self._sleep_task is not None and not self._sleep_task.done()
            if self.state == BackendState.AWAKE and not sleep_in_progress:
                return

            if self._wake_task is not None and not self._wake_task.done():
                task = self._wake_task
            else:
                task = asyncio.create_task(self._wake_once(), name=f"wake-{self.config.id}")
                self._wake_task = task

        # Shield the shared operation so cancellation of one client request does
        # not cancel the wake needed by every other waiting request.
        try:
            await asyncio.shield(task)
        finally:
            async with self._state_lock:
                if self._wake_task is task and task.done():
                    self._wake_task = None

    async def _wake_once(self) -> None:
        start = time.monotonic()
        try:
            # A wake waits behind an in-flight sleep.  This is the key ordering
            # guarantee for request-during-sleep races.
            async with self._transition_lock:
                result = await self._observe_backend()
                if result["healthy"] and result["is_sleeping"] is not True:
                    async with self._state_lock:
                        self.state = BackendState.AWAKE
                        self.last_error = ""
                    return
                if result["is_sleeping"] is not True:
                    raise RuntimeError(result["error"] or "vLLM backend is unavailable")

                async with self._state_lock:
                    self.state = BackendState.WAKING
                    self.last_error = ""

                LOGGER.info("backend=%s action=wake_requested", self.config.id)
                async with await self._request(
                    "POST",
                    "/wake_up",
                    timeout=aiohttp.ClientTimeout(total=self.config.wake_timeout_seconds),
                ) as response:
                    if response.status >= 400:
                        raise RuntimeError(f"wake_up returned HTTP {response.status}: {await response.text()}")

                deadline = time.monotonic() + self.config.wake_timeout_seconds
                while time.monotonic() < deadline:
                    result = await self._observe_backend()
                    if result["healthy"] and result["is_sleeping"] is not True:
                        duration = round((time.monotonic() - start) * 1000)
                        async with self._state_lock:
                            self.state = BackendState.AWAKE
                            self.last_error = ""
                            self.last_wake_duration_ms = duration
                        LOGGER.info("backend=%s action=wake_complete duration_ms=%s", self.config.id, duration)
                        return
                    await asyncio.sleep(self._poll_interval())

                raise TimeoutError(f"backend did not wake within {self.config.wake_timeout_seconds} seconds")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with self._state_lock:
                self.state = BackendState.ERROR
                self.last_error = str(exc)
            LOGGER.exception("backend=%s action=wake_failed", self.config.id)
            raise

    async def sleep(self, *, force: bool = False) -> dict[str, Any]:
        """Put the backend to sleep, deduplicating concurrent sleep callers.

        ``force=True`` means "wait for active requests to drain, then sleep".
        It does not terminate requests.  New traffic can still postpone the
        transition; once ``/sleep`` has started, new traffic waits and wakes the
        model immediately after the sleep transition completes.
        """

        async with self._state_lock:
            if self._sleep_task is not None and not self._sleep_task.done():
                task = self._sleep_task
            else:
                if self.active_requests > 0 and not force:
                    raise RuntimeError(f"{self.active_requests} active request(s) are still running")
                self.state = BackendState.SLEEP_PENDING
                self.last_error = ""
                task = asyncio.create_task(self._sleep_once(force=force), name=f"sleep-{self.config.id}")
                self._sleep_task = task

        try:
            return await asyncio.shield(task)
        finally:
            async with self._state_lock:
                if self._sleep_task is task and task.done():
                    self._sleep_task = None

    async def _sleep_once(self, *, force: bool) -> dict[str, Any]:
        start = time.monotonic()
        try:
            while True:
                if force:
                    async with self._no_active_requests:
                        await self._no_active_requests.wait_for(lambda: self.active_requests == 0 or self._closed)
                    if self._closed:
                        raise asyncio.CancelledError

                async with self._transition_lock:
                    # Recheck after obtaining the transition lock.  A normal
                    # sleep request is cancelled cleanly if traffic won the race
                    # before the upstream sleep command was sent.
                    async with self._state_lock:
                        if self.active_requests > 0:
                            if force:
                                retry_after_unlock = True
                            else:
                                self.state = BackendState.AWAKE
                                self.last_error = ""
                                return {"status": "cancelled", "reason": "request_arrived"}
                        else:
                            retry_after_unlock = False
                            self.state = BackendState.SLEEP_PENDING
                            self.last_error = ""

                    if retry_after_unlock:
                        continue

                    result = await self._observe_backend()
                    if result["is_sleeping"] is True:
                        async with self._state_lock:
                            self.state = BackendState.SLEEPING
                            self.last_error = ""
                        return {"status": "already_sleeping"}
                    if not result["healthy"]:
                        raise RuntimeError(result["error"] or "vLLM backend is unavailable")

                    # One final local check narrows the window before POST.  If a
                    # request arrives after this point, it waits behind this
                    # transition and then wakes the model.
                    async with self._state_lock:
                        if self.active_requests > 0:
                            if force:
                                retry_after_unlock = True
                            else:
                                self.state = BackendState.AWAKE
                                self.last_error = ""
                                return {"status": "cancelled", "reason": "request_arrived"}
                        else:
                            retry_after_unlock = False

                    if retry_after_unlock:
                        continue

                    LOGGER.info("backend=%s action=sleep_requested", self.config.id)
                    async with await self._request(
                        "POST",
                        f"/sleep?level={self.config.sleep_level}",
                        timeout=aiohttp.ClientTimeout(total=self.config.wake_timeout_seconds),
                    ) as response:
                        if response.status >= 400:
                            raise RuntimeError(f"sleep returned HTTP {response.status}: {await response.text()}")

                    deadline = time.monotonic() + self.config.wake_timeout_seconds
                    while time.monotonic() < deadline:
                        result = await self._observe_backend()
                        if result["is_sleeping"] is True:
                            duration = round((time.monotonic() - start) * 1000)
                            async with self._state_lock:
                                self.state = BackendState.SLEEPING
                                self.last_error = ""
                                self.last_sleep_duration_ms = duration
                            LOGGER.info("backend=%s action=sleep_complete duration_ms=%s", self.config.id, duration)
                            return {"status": "sleeping"}
                        await asyncio.sleep(self._poll_interval())

                    raise TimeoutError("backend did not enter sleep mode")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with self._state_lock:
                self.state = BackendState.ERROR
                self.last_error = str(exc)
            LOGGER.exception("backend=%s action=sleep_failed", self.config.id)
            raise

    def _poll_interval(self) -> float:
        return min(0.25, max(0.02, self.config.wake_timeout_seconds / 20))

    async def _idle_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(1)
            if self.config.idle_timeout_seconds <= 0:
                continue

            async with self._state_lock:
                idle_seconds = time.monotonic() - self.last_activity_monotonic
                should_sleep = (
                    self.active_requests == 0
                    and idle_seconds >= self.config.idle_timeout_seconds
                    and self.state not in {BackendState.SLEEPING, BackendState.SLEEP_PENDING, BackendState.WAKING}
                    and not self._transition_active_unlocked()
                )

            if not should_sleep:
                continue

            try:
                LOGGER.info("backend=%s action=auto_sleep idle_seconds=%d", self.config.id, int(idle_seconds))
                await self.sleep()
            except RuntimeError:
                # Traffic can arrive between the idle check and sleep().  That
                # is a normal race, not a backend error.
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                async with self._state_lock:
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
