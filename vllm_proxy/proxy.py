from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import aiohttp
from aiohttp import web

from .backend import BackendController

LOGGER = logging.getLogger(__name__)

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
BLOCKED_PATHS = {
    "/sleep",
    "/wake_up",
    "/is_sleeping",
    "/collective_rpc",
    "/server_info",
    "/reset_prefix_cache",
    "/reset_mm_cache",
    "/reset_encoder_cache",
}


def filtered_headers(headers: aiohttp.typedefs.LooseHeaders, *, response: bool = False) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower in HOP_BY_HOP_HEADERS or lower in {"host", "content-length"}:
            continue
        if lower == "content-encoding":
            # aiohttp transparently decodes request and response bodies; do not
            # forward a stale encoding header for the decoded stream.
            continue
        result[key] = value
    return result


def inject_no_thinking(body: bytes, content_type: str, path: str) -> bytes:
    if path != "/v1/chat/completions" or "application/json" not in content_type.lower():
        return body
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    kwargs = payload.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    else:
        kwargs = dict(kwargs)
    kwargs["enable_thinking"] = False
    payload["chat_template_kwargs"] = kwargs
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


async def proxy_request(request: web.Request) -> web.StreamResponse:
    backend: BackendController = request.app["backend"]
    mode: str = request.app["listener_mode"]
    if request.path in BLOCKED_PATHS:
        raise web.HTTPNotFound()

    counts_as_activity = request.path not in backend.config.idle_ignore_paths
    await backend.request_started(counts_as_activity)
    try:
        try:
            await backend.ensure_awake()
        except Exception as exc:
            return web.json_response(
                {
                    "error": {
                        "type": "backend_wake_failed",
                        "message": "The model server could not be awakened.",
                        "backend": backend.config.id,
                        "detail": str(exc),
                    }
                },
                status=503,
            )

        body = await request.read()
        if mode == "no_thinking":
            body = inject_no_thinking(body, request.headers.get("Content-Type", ""), request.path)
        upstream_url = f"{backend.config.upstream_url}{request.rel_url}"
        headers = filtered_headers(request.headers)
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=None)
        upstream = await backend.session.request(
            request.method,
            upstream_url,
            headers=headers,
            data=body if body else None,
            allow_redirects=False,
            timeout=timeout,
        )
        response = web.StreamResponse(status=upstream.status, reason=upstream.reason)
        for key, value in filtered_headers(upstream.headers, response=True).items():
            response.headers[key] = value
        await response.prepare(request)
        try:
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
        finally:
            upstream.release()
        await response.write_eof()
        return response
    except asyncio.CancelledError:
        raise
    except (aiohttp.ClientError, OSError) as exc:
        LOGGER.exception("proxy failure backend=%s path=%s", backend.config.id, request.path)
        return web.json_response(
            {"error": {"type": "upstream_error", "message": str(exc), "backend": backend.config.id}},
            status=502,
        )
    finally:
        await backend.request_finished(counts_as_activity)


def create_listener_app(backend: BackendController, mode: str) -> web.Application:
    app = web.Application(client_max_size=1024**3)
    app["backend"] = backend
    app["listener_mode"] = mode
    app.router.add_route("*", "/{tail:.*}", proxy_request)
    return app
