from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from .backend import BackendController
from .config import DEFAULT_CONFIG_PATH, ProxyConfig, load_config, save_config, validate_config
from .gpu_monitor import GpuMonitor
from .proxy import create_listener_app

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")


class ProxyServer:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config: ProxyConfig | None = None
        self.session: aiohttp.ClientSession | None = None
        self.gpu_monitor = GpuMonitor()
        self.backends: dict[str, BackendController] = {}
        self.listener_runners: list[web.AppRunner] = []
        self.admin_runner: web.AppRunner | None = None
        self.events: set[asyncio.Queue[str]] = set()
        self.event_task: asyncio.Task[None] | None = None
        self.shutdown_event = asyncio.Event()

    async def start(self) -> None:
        self.config = await load_config(self.config_path)
        connector = aiohttp.TCPConnector(limit=0, enable_cleanup_closed=True)
        self.session = aiohttp.ClientSession(connector=connector)
        await self._build_backends()
        await self._start_listeners()
        await self._start_admin()
        self.event_task = asyncio.create_task(self._event_loop(), name="status-events")

    async def _build_backends(self) -> None:
        assert self.config and self.session
        for config in self.config.backends:
            controller = BackendController(config, self.session, self.gpu_monitor)
            controller.start()
            self.backends[config.id] = controller
            await controller.probe()

    async def _start_listeners(self) -> None:
        for backend in self.backends.values():
            for listener in backend.config.listeners:
                if not listener.enabled:
                    continue
                app = create_listener_app(backend, listener.mode)
                runner = web.AppRunner(app, access_log=None)
                await runner.setup()
                site = web.TCPSite(runner, listener.host, listener.port)
                await site.start()
                self.listener_runners.append(runner)
                LOGGER.info(
                    "listener backend=%s mode=%s address=%s:%s",
                    backend.config.id,
                    listener.mode,
                    listener.host,
                    listener.port,
                )

    def _authorized(self, request: web.Request) -> bool:
        assert self.config
        token = self.config.admin.token
        if not token:
            return True
        provided = request.headers.get("Authorization", "")
        if provided.startswith("Bearer "):
            provided = provided[7:]
        if not provided:
            provided = request.query.get("token", "")
        return provided == token

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler: Any) -> web.StreamResponse:
        if request.path.startswith("/api/") and not self._authorized(request):
            raise web.HTTPUnauthorized(text="Invalid admin token")
        return await handler(request)

    async def _start_admin(self) -> None:
        assert self.config
        app = web.Application(middlewares=[self._auth_middleware])
        app.router.add_get("/", self.index)
        app.router.add_get("/app.js", self.static_file)
        app.router.add_get("/style.css", self.static_file)
        app.router.add_get("/api/status", self.api_status)
        app.router.add_get("/api/config", self.api_get_config)
        app.router.add_put("/api/config", self.api_put_config)
        app.router.add_get("/api/events", self.api_events)
        app.router.add_post("/api/backends/{backend_id}/sleep", self.api_sleep)
        app.router.add_post("/api/backends/{backend_id}/wake", self.api_wake)
        app.router.add_post("/api/backends/{backend_id}/probe", self.api_probe)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, self.config.admin.host, self.config.admin.port).start()
        self.admin_runner = runner
        LOGGER.info("admin address=http://%s:%s", self.config.admin.host, self.config.admin.port)

    async def close(self) -> None:
        if self.event_task:
            self.event_task.cancel()
            await asyncio.gather(self.event_task, return_exceptions=True)
        for runner in self.listener_runners:
            await runner.cleanup()
        if self.admin_runner:
            await self.admin_runner.cleanup()
        for controller in self.backends.values():
            await controller.close()
        if self.session:
            await self.session.close()
        self.gpu_monitor.close()

    async def restart_runtime(self, new_config: ProxyConfig) -> None:
        for runner in self.listener_runners:
            await runner.cleanup()
        self.listener_runners.clear()

        old_ids = set(self.backends)
        new_ids = {backend.id for backend in new_config.backends}
        for removed in old_ids - new_ids:
            await self.backends.pop(removed).close()

        assert self.session
        for backend_config in new_config.backends:
            controller = self.backends.get(backend_config.id)
            if controller:
                controller.update_config(backend_config)
            else:
                controller = BackendController(backend_config, self.session, self.gpu_monitor)
                controller.start()
                self.backends[backend_config.id] = controller
                await controller.probe()
        self.config = new_config
        await self._start_listeners()

    def _backend(self, request: web.Request) -> BackendController:
        backend_id = request.match_info["backend_id"]
        backend = self.backends.get(backend_id)
        if not backend:
            raise web.HTTPNotFound(text=f"Unknown backend: {backend_id}")
        return backend

    async def index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "index.html")

    async def static_file(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / request.path.lstrip("/"))

    async def api_status(self, request: web.Request) -> web.Response:
        return web.json_response({"backends": [backend.status() for backend in self.backends.values()]})

    async def api_get_config(self, request: web.Request) -> web.Response:
        assert self.config
        data = self.config.to_dict()
        if data["admin"].get("token"):
            data["admin"]["token"] = ""
            data["admin"]["token_configured"] = True
        return web.json_response(data)

    async def api_put_config(self, request: web.Request) -> web.Response:
        assert self.config
        try:
            data = await request.json()
            # An empty token means “keep the current secret” when one is already configured.
            submitted_token = str(data.get("admin", {}).get("token", ""))
            if not submitted_token and self.config.admin.token:
                data.setdefault("admin", {})["token"] = self.config.admin.token
            new_config = ProxyConfig.from_dict(data)
            validate_config(new_config)
            if (new_config.admin.host, new_config.admin.port) != (self.config.admin.host, self.config.admin.port):
                raise ValueError("Changing the admin host or port requires restarting the proxy")
            await save_config(new_config, self.config_path)
            await self.restart_runtime(new_config)
            return web.json_response({"status": "saved"})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def api_sleep(self, request: web.Request) -> web.Response:
        backend = self._backend(request)
        force = request.query.get("force", "false").lower() in {"1", "true", "yes"}
        try:
            return web.json_response(await backend.sleep(force=force))
        except RuntimeError as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)

    async def api_wake(self, request: web.Request) -> web.Response:
        backend = self._backend(request)
        try:
            await backend.ensure_awake()
            return web.json_response({"status": "awake"})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)

    async def api_probe(self, request: web.Request) -> web.Response:
        return web.json_response(await self._backend(request).probe())

    async def api_events(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=2)
        self.events.add(queue)
        try:
            await queue.put(json.dumps({"backends": [item.status() for item in self.backends.values()]}))
            while True:
                payload = await queue.get()
                await response.write(f"data: {payload}\n\n".encode())
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.events.discard(queue)
        return response

    async def _event_loop(self) -> None:
        while True:
            payload = json.dumps({"backends": [item.status() for item in self.backends.values()]})
            for queue in tuple(self.events):
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    pass
            await asyncio.sleep(2)


async def async_main(config_path: Path) -> None:
    server = ProxyServer(config_path)
    loop = asyncio.get_running_loop()
    for signame in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signame, server.shutdown_event.set)
    await server.start()
    try:
        await server.shutdown_event.wait()
    finally:
        await server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM smart sleep proxy")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(async_main(args.config))


if __name__ == "__main__":
    main()
