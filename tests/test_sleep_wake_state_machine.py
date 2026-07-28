from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import AsyncIterator

import aiohttp
import pytest
from aiohttp import web

from vllm_proxy.backend import BackendController, BackendState
from vllm_proxy.config import BackendConfig, ListenerConfig
from vllm_proxy.proxy import create_listener_app


class DummyGpuMonitor:
    def process_tree(self, _pidfile: str) -> set[int]:
        return set()

    def snapshot(self, _pidfile: str) -> dict[str, object]:
        return {"available": False, "gpus": []}


@dataclass
class FakeVLLM:
    """Deterministic fake of the vLLM sleep/wake management endpoints."""

    sleeping: bool = False
    sleep_delay: float = 0.05
    wake_delay: float = 0.05
    fail_next_sleep: bool = False
    fail_next_wake: bool = False
    sleep_calls: int = 0
    wake_calls: int = 0
    inference_calls: int = 0
    sleep_started: asyncio.Event = field(default_factory=asyncio.Event)
    wake_started: asyncio.Event = field(default_factory=asyncio.Event)
    stream_started: asyncio.Event = field(default_factory=asyncio.Event)
    stream_release: asyncio.Event = field(default_factory=asyncio.Event)

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/is_sleeping", self.is_sleeping)
        app.router.add_get("/health", self.health)
        app.router.add_post("/sleep", self.sleep)
        app.router.add_post("/wake_up", self.wake_up)
        app.router.add_post("/v1/chat/completions", self.chat)
        return app

    async def is_sleeping(self, _request: web.Request) -> web.Response:
        return web.json_response(self.sleeping)

    async def health(self, _request: web.Request) -> web.Response:
        return web.Response(status=503 if self.sleeping else 200, text="ok")

    async def sleep(self, _request: web.Request) -> web.Response:
        self.sleep_calls += 1
        self.sleep_started.set()
        if self.fail_next_sleep:
            self.fail_next_sleep = False
            return web.Response(status=500, text="injected sleep failure")
        await asyncio.sleep(self.sleep_delay)
        self.sleeping = True
        return web.json_response({"status": "sleeping"})

    async def wake_up(self, _request: web.Request) -> web.Response:
        self.wake_calls += 1
        self.wake_started.set()
        if self.fail_next_wake:
            self.fail_next_wake = False
            return web.Response(status=500, text="injected wake failure")
        await asyncio.sleep(self.wake_delay)
        self.sleeping = False
        return web.json_response({"status": "awake"})

    async def chat(self, request: web.Request) -> web.StreamResponse:
        self.inference_calls += 1
        if self.sleeping:
            return web.json_response({"error": "model is sleeping"}, status=503)
        payload = await request.json()
        if payload.get("stream_test"):
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            await response.write(b'data: {"part":1}\n\n')
            self.stream_started.set()
            await self.stream_release.wait()
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        return web.json_response({"ok": True})


@contextlib.asynccontextmanager
async def serve(app: web.Application) -> AsyncIterator[str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets  # type: ignore[union-attr]
    port = sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def make_config(
    backend_id: str,
    upstream_url: str,
    *,
    idle_timeout_seconds: int = 0,
    wake_timeout_seconds: int = 2,
) -> BackendConfig:
    return BackendConfig(
        id=backend_id,
        name=backend_id,
        upstream_url=upstream_url,
        idle_timeout_seconds=idle_timeout_seconds,
        wake_timeout_seconds=wake_timeout_seconds,
        sleep_level=1,
        listeners=[ListenerConfig(host="127.0.0.1", port=1)],
    )


@contextlib.asynccontextmanager
async def controller_for(
    fake: FakeVLLM,
    *,
    backend_id: str = "test",
    idle_timeout_seconds: int = 0,
) -> AsyncIterator[BackendController]:
    async with serve(fake.app()) as upstream_url:
        async with aiohttp.ClientSession() as session:
            controller = BackendController(
                make_config(
                    backend_id,
                    upstream_url,
                    idle_timeout_seconds=idle_timeout_seconds,
                ),
                session,
                DummyGpuMonitor(),  # type: ignore[arg-type]
            )
            await controller.probe()
            try:
                yield controller
            finally:
                await controller.close()


@pytest.mark.asyncio
async def test_burst_requests_share_one_wake_command() -> None:
    fake = FakeVLLM(sleeping=True, wake_delay=0.1)
    async with controller_for(fake) as controller:
        assert controller.state == BackendState.SLEEPING
        await asyncio.gather(*(controller.ensure_awake() for _ in range(25)))
        assert fake.wake_calls == 1
        assert controller.state == BackendState.AWAKE
        assert controller.status()["transition_counters"]["wake_commands"] == 1
        assert controller.status()["transition_counters"]["wake_completions"] == 1


@pytest.mark.asyncio
async def test_concurrent_sleep_requests_share_one_sleep_command() -> None:
    """Two admin/idle sleep triggers must not POST /sleep twice."""

    fake = FakeVLLM(sleeping=False, sleep_delay=0.15)
    async with controller_for(fake) as controller:
        results = await asyncio.gather(controller.sleep(), controller.sleep())
        assert fake.sleep_calls == 1, (
            "Concurrent sleep operations sent more than one /sleep command. "
            "The controller needs a shared _sleep_task, analogous to _wake_task."
        )
        assert all(result["status"] in {"sleeping", "already_sleeping"} for result in results)
        assert controller.state == BackendState.SLEEPING
        assert controller.status()["transition_counters"]["sleep_commands"] == 1
        assert controller.status()["transition_counters"]["sleep_completions"] == 1


@pytest.mark.asyncio
async def test_request_arriving_during_sleep_transition_finishes_cleanly() -> None:
    """A request racing with sleep must wake and must not leave ERROR behind."""

    fake = FakeVLLM(sleeping=False, sleep_delay=0.2, wake_delay=0.05)
    async with controller_for(fake) as controller:
        async with serve(create_listener_app(controller, "passthrough")) as proxy_url:
            sleep_task = asyncio.create_task(controller.sleep())
            await asyncio.wait_for(fake.sleep_started.wait(), timeout=1)

            async with aiohttp.ClientSession() as client:
                async with client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={"messages": []},
                ) as response:
                    assert response.status == 200
                    assert await response.json() == {"ok": True}

            sleep_result = await asyncio.gather(sleep_task, return_exceptions=True)
            assert not isinstance(sleep_result[0], BaseException), (
                "The sleep operation timed out or failed after a request woke the model. "
                f"Observed: {sleep_result[0]!r}"
            )
            assert controller.state == BackendState.AWAKE
            assert controller.last_error == ""
            assert fake.inference_calls == 1
            assert fake.wake_calls == 1


@pytest.mark.asyncio
async def test_active_stream_blocks_auto_sleep_until_stream_finishes() -> None:
    fake = FakeVLLM(sleeping=False)
    async with controller_for(fake, idle_timeout_seconds=1) as controller:
        controller.start()
        async with serve(create_listener_app(controller, "passthrough")) as proxy_url:
            async with aiohttp.ClientSession() as client:
                response = await client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={"messages": [], "stream_test": True},
                )
                await asyncio.wait_for(fake.stream_started.wait(), timeout=1)
                first_chunk = await asyncio.wait_for(response.content.readany(), timeout=1)
                assert first_chunk

                await asyncio.sleep(1.3)
                assert controller.active_requests == 1
                assert fake.sleep_calls == 0

                fake.stream_release.set()
                await response.read()
                response.release()

                deadline = asyncio.get_running_loop().time() + 4.0
                while fake.sleep_calls == 0 and asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(0.05)
                assert controller.active_requests == 0
                assert fake.sleep_calls == 1
                assert controller.state in {BackendState.SLEEP_PENDING, BackendState.SLEEPING}

                deadline = asyncio.get_running_loop().time() + 2.0
                while controller.state != BackendState.SLEEPING and asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(0.05)
                assert controller.state == BackendState.SLEEPING


@pytest.mark.asyncio
async def test_backends_keep_independent_idle_timers() -> None:
    fake_a = FakeVLLM(sleeping=False)
    fake_b = FakeVLLM(sleeping=False)

    async with serve(fake_a.app()) as upstream_a, serve(fake_b.app()) as upstream_b:
        async with aiohttp.ClientSession() as session:
            controller_a = BackendController(
                make_config("a", upstream_a, idle_timeout_seconds=1),
                session,
                DummyGpuMonitor(),  # type: ignore[arg-type]
            )
            controller_b = BackendController(
                make_config("b", upstream_b, idle_timeout_seconds=1),
                session,
                DummyGpuMonitor(),  # type: ignore[arg-type]
            )
            await controller_a.probe()
            await controller_b.probe()
            controller_a.start()
            controller_b.start()
            try:
                await asyncio.sleep(0.7)
                await controller_a.request_started(True)
                await controller_a.request_finished(True)

                await asyncio.sleep(1.0)
                assert fake_b.sleep_calls == 1
                assert fake_a.sleep_calls == 0
                assert controller_b.state == BackendState.SLEEPING
                assert controller_a.state == BackendState.AWAKE

                await asyncio.sleep(1.2)
                assert fake_a.sleep_calls == 1
                assert controller_a.state == BackendState.SLEEPING
            finally:
                await controller_a.close()
                await controller_b.close()


@pytest.mark.asyncio
async def test_failed_wake_can_be_retried() -> None:
    fake = FakeVLLM(sleeping=True, fail_next_wake=True)
    async with controller_for(fake) as controller:
        with pytest.raises(RuntimeError, match="wake_up returned HTTP 500"):
            await controller.ensure_awake()
        assert controller.state == BackendState.ERROR

        await controller.ensure_awake()
        assert fake.wake_calls == 2
        assert controller.state == BackendState.AWAKE
        assert controller.last_error == ""


@pytest.mark.asyncio
async def test_probe_during_sleep_does_not_clobber_transition_state() -> None:
    """An admin probe must not turn SLEEP_PENDING back into AWAKE."""

    fake = FakeVLLM(sleeping=False, sleep_delay=0.2)
    async with controller_for(fake) as controller:
        sleep_task = asyncio.create_task(controller.sleep())
        await asyncio.wait_for(fake.sleep_started.wait(), timeout=1)
        assert controller.state == BackendState.SLEEP_PENDING

        result = await controller.probe()
        assert result == {"healthy": True, "is_sleeping": False, "error": ""}
        assert controller.state == BackendState.SLEEP_PENDING

        assert (await sleep_task)["status"] == "sleeping"
        assert controller.state == BackendState.SLEEPING


@pytest.mark.asyncio
async def test_cancelled_wake_waiter_does_not_cancel_shared_wake() -> None:
    """Disconnecting one client must not cancel the wake used by other clients."""

    fake = FakeVLLM(sleeping=True, wake_delay=0.2)
    async with controller_for(fake) as controller:
        first = asyncio.create_task(controller.ensure_awake())
        await asyncio.wait_for(fake.wake_started.wait(), timeout=1)
        second = asyncio.create_task(controller.ensure_awake())

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        await asyncio.wait_for(second, timeout=2)
        assert fake.wake_calls == 1
        assert controller.state == BackendState.AWAKE
        assert controller.last_error == ""


@pytest.mark.asyncio
async def test_cancelled_sleep_waiter_does_not_cancel_shared_sleep() -> None:
    """Cancelling one admin request must not cancel the shared sleep operation."""

    fake = FakeVLLM(sleeping=False, sleep_delay=0.2)
    async with controller_for(fake) as controller:
        first = asyncio.create_task(controller.sleep())
        await asyncio.wait_for(fake.sleep_started.wait(), timeout=1)
        second = asyncio.create_task(controller.sleep())

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        result = await asyncio.wait_for(second, timeout=2)
        assert result["status"] == "sleeping"
        assert fake.sleep_calls == 1
        assert controller.state == BackendState.SLEEPING
        assert controller.last_error == ""


@pytest.mark.asyncio
async def test_force_sleep_waits_for_active_requests_then_sleeps() -> None:
    fake = FakeVLLM(sleeping=False)
    async with controller_for(fake) as controller:
        await controller.request_started(True)
        task = asyncio.create_task(controller.sleep(force=True))

        await asyncio.sleep(0.1)
        assert fake.sleep_calls == 0
        assert controller.state == BackendState.SLEEP_PENDING

        await controller.request_finished(True)
        result = await asyncio.wait_for(task, timeout=2)
        assert result["status"] == "sleeping"
        assert fake.sleep_calls == 1
        assert controller.state == BackendState.SLEEPING


@pytest.mark.asyncio
async def test_request_winning_before_sleep_command_cancels_sleep_cleanly() -> None:
    """Traffic arriving before POST /sleep should prevent the sleep command."""

    fake = FakeVLLM(sleeping=False)
    async with controller_for(fake) as controller:
        # Hold the transition lock so the request can deterministically arrive
        # after sleep() is scheduled but before it can inspect active_requests.
        await controller._transition_lock.acquire()
        try:
            sleep_task = asyncio.create_task(controller.sleep())
            await asyncio.sleep(0)
            await controller.request_started(True)
            wake_task = asyncio.create_task(controller.ensure_awake())
        finally:
            controller._transition_lock.release()

        sleep_result = await asyncio.wait_for(sleep_task, timeout=2)
        await asyncio.wait_for(wake_task, timeout=2)
        await controller.request_finished(True)

        assert sleep_result == {"status": "cancelled", "reason": "request_arrived"}
        assert fake.sleep_calls == 0
        assert fake.wake_calls == 0
        assert controller.state == BackendState.AWAKE
        assert controller.last_error == ""


@pytest.mark.asyncio
async def test_repeated_sleep_wake_cycles_leave_no_stale_transition() -> None:
    fake = FakeVLLM(sleeping=False, sleep_delay=0.005, wake_delay=0.005)
    async with controller_for(fake) as controller:
        for _ in range(20):
            sleep_result = await controller.sleep()
            assert sleep_result["status"] == "sleeping"
            assert controller.state == BackendState.SLEEPING
            await controller.ensure_awake()
            assert controller.state == BackendState.AWAKE
            assert controller.last_error == ""

        assert fake.sleep_calls == 20
        assert fake.wake_calls == 20


@pytest.mark.asyncio
async def test_failed_sleep_can_be_retried() -> None:
    fake = FakeVLLM(sleeping=False, fail_next_sleep=True)
    async with controller_for(fake) as controller:
        with pytest.raises(RuntimeError, match="sleep returned HTTP 500"):
            await controller.sleep()
        assert controller.state == BackendState.ERROR

        result = await controller.sleep()
        assert result["status"] == "sleeping"
        assert fake.sleep_calls == 2
        assert controller.state == BackendState.SLEEPING
        assert controller.last_error == ""
