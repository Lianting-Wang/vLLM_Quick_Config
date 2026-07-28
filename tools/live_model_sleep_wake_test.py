from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp

# Allow running the file directly from the repository without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vllm_proxy.auth import load_admin_credentials  # noqa: E402
from vllm_proxy.config import BackendConfig, ListenerConfig, ProxyConfig, load_config  # noqa: E402


@dataclass(slots=True)
class StepResult:
    name: str
    backend: str | None
    status: str
    duration_seconds: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunReport:
    started_at: str
    finished_at: str = ""
    config_file: str = ""
    admin_url: str = ""
    selected_backends: list[str] = field(default_factory=list)
    results: list[StepResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    restored_config: bool = False
    restored_backend_states: bool = False
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config_file": self.config_file,
            "admin_url": self.admin_url,
            "selected_backends": self.selected_backends,
            "results": [asdict(item) for item in self.results],
            "warnings": self.warnings,
            "restored_config": self.restored_config,
            "restored_backend_states": self.restored_backend_states,
            "success": self.success,
        }


class LiveTestFailure(RuntimeError):
    pass


def local_url(host: str, port: int) -> str:
    normalized = host.strip()
    if normalized in {"0.0.0.0", "localhost"}:
        normalized = "127.0.0.1"
    elif normalized == "::":
        normalized = "::1"
    if ":" in normalized and not normalized.startswith("["):
        normalized = f"[{normalized}]"
    return f"http://{normalized}:{port}"


def listener_url(listener: ListenerConfig) -> str:
    return local_url(listener.host, listener.port)


def choose_listener(backend: BackendConfig) -> ListenerConfig:
    enabled = [item for item in backend.listeners if item.enabled]
    if not enabled:
        raise LiveTestFailure(f"backend {backend.id!r} has no enabled listener")
    return next((item for item in enabled if item.mode == "passthrough"), enabled[0])


def latest_proxy_log(project_root: Path) -> Path | None:
    log_dir = Path(os.environ.get("VLLM_PROXY_LOG_DIR", project_root / "logs"))
    candidates = sorted(log_dir.glob("proxy_*.log"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def count_log_event(path: Path | None, backend_id: str, action: str) -> int | None:
    if path is None or not path.is_file():
        return None
    needle = f"backend={backend_id} action={action}"
    try:
        return path.read_text(encoding="utf-8", errors="replace").count(needle)
    except OSError:
        return None


def extract_model_memory(status: dict[str, Any]) -> float | None:
    gpu = status.get("gpu")
    if not isinstance(gpu, dict) or not gpu.get("available"):
        return None
    value = gpu.get("model_process_memory_mib")
    return float(value) if isinstance(value, (int, float)) else None


class LiveModelTester:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config_path = Path(args.config).expanduser().resolve()
        self.config: ProxyConfig | None = None
        self.original_config_json: dict[str, Any] | None = None
        self.initial_states: dict[str, str] = {}
        self.report = RunReport(
            started_at=datetime.now(timezone.utc).isoformat(),
            config_file=str(self.config_path),
        )
        self.admin_url = ""
        self.admin_session: aiohttp.ClientSession | None = None
        self.proxy_session: aiohttp.ClientSession | None = None
        self.api_headers: dict[str, str] = {"Content-Type": "application/json"}
        if args.api_key:
            self.api_headers["Authorization"] = f"Bearer {args.api_key}"
        self.proxy_log = latest_proxy_log(PROJECT_ROOT)
        self.model_ids: dict[str, str] = {}

    async def run(self) -> RunReport:
        self.config = await load_config(self.config_path)
        selected = self._selected_backend_configs()
        self.report.selected_backends = [item.id for item in selected]
        self.admin_url = self.args.admin_url or local_url(self.config.admin.host, self.config.admin.port)
        self.report.admin_url = self.admin_url

        timeout = aiohttp.ClientTimeout(total=self.args.request_timeout)
        cookie_jar = aiohttp.CookieJar(unsafe=True)
        self.admin_session = aiohttp.ClientSession(timeout=timeout, cookie_jar=cookie_jar)
        self.proxy_session = aiohttp.ClientSession(timeout=timeout)

        try:
            await self._login()
            self.original_config_json = await self._get_config_json()
            statuses = await self._status_map()
            for backend in selected:
                status = statuses.get(backend.id)
                if status is None:
                    raise LiveTestFailure(f"backend {backend.id!r} is not present in the running proxy")
                self.initial_states[backend.id] = str(status.get("state", "unknown"))

            await self._step("preflight", None, lambda: self._preflight(selected))

            for backend in selected:
                await self._test_backend(backend)

            if not self.args.skip_auto_idle:
                for backend in selected:
                    await self._step(
                        "automatic idle sleep and request-triggered wake",
                        backend.id,
                        lambda backend=backend: self._test_auto_idle(backend),
                    )

            if len(selected) >= 2 and not self.args.skip_independent_timers:
                await self._step(
                    "independent backend idle timers",
                    None,
                    lambda: self._test_independent_timers(selected[0], selected[1]),
                )

            self.report.success = all(item.status == "passed" for item in self.report.results)
        finally:
            await self._restore_all()
            if self.admin_session:
                await self.admin_session.close()
            if self.proxy_session:
                await self.proxy_session.close()
            self.report.success = (
                self.report.success
                and self.report.restored_config
                and self.report.restored_backend_states
            )
            self.report.finished_at = datetime.now(timezone.utc).isoformat()
            self._write_report()

        return self.report

    def _selected_backend_configs(self) -> list[BackendConfig]:
        assert self.config
        by_id = {item.id: item for item in self.config.backends}
        requested = self.args.backend or list(by_id)
        missing = [item for item in requested if item not in by_id]
        if missing:
            raise LiveTestFailure(f"unknown backend id(s): {', '.join(missing)}")
        return [by_id[item] for item in requested]

    async def _step(
        self,
        name: str,
        backend: str | None,
        operation: Callable[[], Awaitable[dict[str, Any] | None]],
    ) -> dict[str, Any] | None:
        label = f"[{backend}] {name}" if backend else name
        print(f"\n==> {label}", flush=True)
        started = time.monotonic()
        try:
            detail = await operation() or {}
        except Exception as exc:
            duration = time.monotonic() - started
            self.report.results.append(
                StepResult(name=name, backend=backend, status="failed", duration_seconds=duration, detail={"error": str(exc)})
            )
            print(f"FAIL ({duration:.2f}s): {exc}", flush=True)
            raise
        duration = time.monotonic() - started
        self.report.results.append(
            StepResult(name=name, backend=backend, status="passed", duration_seconds=duration, detail=detail)
        )
        print(f"PASS ({duration:.2f}s)", flush=True)
        return detail

    async def _login(self) -> None:
        assert self.admin_session
        credentials = load_admin_credentials(self.config_path)
        async with self.admin_session.post(
            f"{self.admin_url}/api/login",
            json={"password": credentials.password},
        ) as response:
            body = await response.text()
            if response.status != 200:
                raise LiveTestFailure(f"admin login failed: HTTP {response.status}: {body}")
        async with self.admin_session.get(f"{self.admin_url}/api/session") as response:
            payload = await response.json(content_type=None)
            if response.status != 200 or not payload.get("authenticated"):
                raise LiveTestFailure(f"admin session was not established: {payload}")

    async def _admin_json(
        self,
        method: str,
        path: str,
        *,
        expected: set[int] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        assert self.admin_session
        accepted = {200} if expected is None else expected
        async with self.admin_session.request(method, f"{self.admin_url}{path}", json=json_body) as response:
            text = await response.text()
            try:
                payload = json.loads(text) if text else {}
            except json.JSONDecodeError:
                payload = {"raw": text}
            if response.status not in accepted:
                raise LiveTestFailure(f"{method} {path} returned HTTP {response.status}: {payload}")
            return response.status, payload

    async def _get_config_json(self) -> dict[str, Any]:
        _, payload = await self._admin_json("GET", "/api/config")
        if not isinstance(payload, dict):
            raise LiveTestFailure("admin config response is not an object")
        payload = copy.deepcopy(payload)
        if isinstance(payload.get("admin"), dict):
            payload["admin"].pop("password_configured", None)
        return payload

    async def _put_config_json(self, payload: dict[str, Any]) -> None:
        await self._admin_json("PUT", "/api/config", json_body=payload)
        await asyncio.sleep(0.5)

    async def _status_map(self) -> dict[str, dict[str, Any]]:
        _, payload = await self._admin_json("GET", "/api/status")
        backends = payload.get("backends", []) if isinstance(payload, dict) else []
        return {str(item.get("id")): item for item in backends if isinstance(item, dict)}

    async def _backend_status(self, backend_id: str) -> dict[str, Any]:
        statuses = await self._status_map()
        if backend_id not in statuses:
            raise LiveTestFailure(f"backend {backend_id!r} disappeared from proxy status")
        return statuses[backend_id]

    async def _wait_state(self, backend_id: str, expected: set[str], timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = await self._backend_status(backend_id)
            if str(last.get("state")) in expected:
                return last
            await asyncio.sleep(0.25)
        raise LiveTestFailure(
            f"backend {backend_id} did not reach {sorted(expected)} within {timeout}s; last status={last}"
        )

    async def _probe(self, backend_id: str) -> dict[str, Any]:
        _, payload = await self._admin_json("POST", f"/api/backends/{backend_id}/probe")
        return payload

    async def _wake(self, backend_id: str) -> dict[str, Any]:
        _, payload = await self._admin_json("POST", f"/api/backends/{backend_id}/wake")
        await self._wait_state(backend_id, {"awake"}, self.args.request_timeout)
        return payload

    async def _sleep(self, backend_id: str, *, force: bool = False) -> dict[str, Any]:
        query = "?force=true" if force else ""
        _, payload = await self._admin_json("POST", f"/api/backends/{backend_id}/sleep{query}")
        await self._wait_state(backend_id, {"sleeping"}, self.args.request_timeout)
        return payload

    async def _direct_sleep_state(self, backend: BackendConfig) -> bool | None:
        assert self.proxy_session
        try:
            async with self.proxy_session.get(
                f"{backend.upstream_url}/is_sleeping",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    return None
                payload = await response.json(content_type=None)
                if isinstance(payload, bool):
                    return payload
                if isinstance(payload, dict):
                    return bool(payload.get("is_sleeping", payload.get("sleeping", False)))
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None
        return None

    async def _preflight(self, selected: list[BackendConfig]) -> dict[str, Any]:
        detail: dict[str, Any] = {"proxy_log": str(self.proxy_log) if self.proxy_log else None, "backends": {}}
        for backend in selected:
            status = await self._backend_status(backend.id)
            probe = await self._probe(backend.id)
            if not probe.get("healthy") and probe.get("is_sleeping") is not True:
                raise LiveTestFailure(f"backend {backend.id} is unavailable: {probe}")
            if status.get("process_running") is False:
                self.report.warnings.append(
                    f"{backend.id}: PID-file process detection is false; GPU memory attribution may be unavailable"
                )
            detail["backends"][backend.id] = {
                "initial_state": status.get("state"),
                "probe": probe,
                "listener": listener_url(choose_listener(backend)),
            }
        return detail

    async def _test_backend(self, backend: BackendConfig) -> None:
        await self._step("manual wake and baseline inference", backend.id, lambda: self._baseline(backend))
        await self._step("manual sleep and GPU-memory release", backend.id, lambda: self._manual_sleep(backend))
        await self._step("sleeping request automatically wakes model", backend.id, lambda: self._request_wake(backend))
        await self._step("concurrent burst shares one wake transition", backend.id, lambda: self._burst_wake(backend))
        if not self.args.skip_stream:
            await self._step("active streaming request blocks manual sleep", backend.id, lambda: self._stream_guard(backend))

    async def _discover_model(self, backend: BackendConfig) -> str:
        assert self.proxy_session
        base = listener_url(choose_listener(backend))
        async with self.proxy_session.get(f"{base}/v1/models", headers=self.api_headers) as response:
            text = await response.text()
            if response.status != 200:
                raise LiveTestFailure(f"{backend.id} /v1/models returned HTTP {response.status}: {text}")
            payload = json.loads(text)
        models = payload.get("data", []) if isinstance(payload, dict) else []
        if not models or not isinstance(models[0], dict) or not models[0].get("id"):
            raise LiveTestFailure(f"{backend.id} returned no model id from /v1/models: {payload}")
        return str(models[0]["id"])

    async def _chat(
        self,
        backend: BackendConfig,
        *,
        model: str | None = None,
        prompt: str = "Reply with exactly LIVE_TEST_OK.",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        assert self.proxy_session
        model_id = model or await self._discover_model(backend)
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens or self.args.max_tokens,
            "stream": False,
        }
        base = listener_url(choose_listener(backend))
        started = time.monotonic()
        async with self.proxy_session.post(
            f"{base}/v1/chat/completions",
            headers=self.api_headers,
            json=payload,
        ) as response:
            text = await response.text()
            latency = time.monotonic() - started
            if response.status != 200:
                raise LiveTestFailure(
                    f"{backend.id} chat completion returned HTTP {response.status}: {text[:1000]}"
                )
            try:
                result = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LiveTestFailure(f"{backend.id} returned invalid JSON: {text[:500]}") from exc
        choices = result.get("choices") if isinstance(result, dict) else None
        if not isinstance(choices, list) or not choices:
            raise LiveTestFailure(f"{backend.id} response has no choices: {result}")
        return {"model": model_id, "latency_seconds": latency, "response": result}

    async def _baseline(self, backend: BackendConfig) -> dict[str, Any]:
        await self._wake(backend.id)
        direct = await self._direct_sleep_state(backend)
        if direct is True:
            raise LiveTestFailure("admin reports awake but upstream /is_sleeping is true")
        model = await self._discover_model(backend)
        self.model_ids[backend.id] = model
        result = await self._chat(backend, model=model)
        return {"model": result["model"], "latency_seconds": result["latency_seconds"]}

    async def _manual_sleep(self, backend: BackendConfig) -> dict[str, Any]:
        before = await self._backend_status(backend.id)
        before_mib = extract_model_memory(before)
        before_log = count_log_event(self.proxy_log, backend.id, "sleep_requested")
        await self._sleep(backend.id)
        direct = await self._direct_sleep_state(backend)
        if direct is not True:
            raise LiveTestFailure(f"upstream /is_sleeping did not confirm sleep: {direct}")
        await asyncio.sleep(self.args.memory_settle_seconds)
        after = await self._backend_status(backend.id)
        after_mib = extract_model_memory(after)
        after_log = count_log_event(self.proxy_log, backend.id, "sleep_requested")
        detail: dict[str, Any] = {
            "memory_before_mib": before_mib,
            "memory_after_mib": after_mib,
            "memory_drop_mib": None,
            "sleep_log_delta": None,
        }
        if not self.args.skip_log_count_check and before_log is not None and after_log is not None:
            detail["sleep_log_delta"] = after_log - before_log
            if after_log - before_log != 1:
                raise LiveTestFailure(
                    f"expected one sleep_requested log event, observed {after_log - before_log}; "
                    "ensure no other admin client is controlling this backend during the test"
                )
        if before_mib is not None and after_mib is not None:
            drop = before_mib - after_mib
            detail["memory_drop_mib"] = drop
            if before_mib >= self.args.memory_check_minimum_mib and drop < self.args.minimum_memory_drop_mib:
                raise LiveTestFailure(
                    f"model memory only dropped {drop:.1f} MiB "
                    f"({before_mib:.1f} -> {after_mib:.1f} MiB)"
                )
        else:
            self.report.warnings.append(f"{backend.id}: NVML model-process memory was unavailable")
        return detail

    async def _request_wake(self, backend: BackendConfig) -> dict[str, Any]:
        before = count_log_event(self.proxy_log, backend.id, "wake_requested")
        started = time.monotonic()
        model = self.model_ids[backend.id]
        result = await self._chat(backend, model=model)
        total = time.monotonic() - started
        status = await self._wait_state(backend.id, {"awake"}, self.args.request_timeout)
        direct = await self._direct_sleep_state(backend)
        if direct is True:
            raise LiveTestFailure("request completed but upstream still reports sleeping")
        await asyncio.sleep(0.3)
        after = count_log_event(self.proxy_log, backend.id, "wake_requested")
        delta = None if before is None or after is None else after - before
        if not self.args.skip_log_count_check and delta is not None and delta != 1:
            raise LiveTestFailure(f"expected exactly one wake_requested event, observed {delta}")
        return {
            "model": result["model"],
            "total_latency_seconds": total,
            "reported_wake_duration_ms": status.get("last_wake_duration_ms"),
            "wake_log_delta": delta,
        }

    async def _burst_wake(self, backend: BackendConfig) -> dict[str, Any]:
        await self._sleep(backend.id)
        before = count_log_event(self.proxy_log, backend.id, "wake_requested")
        model = self.model_ids[backend.id]
        start_gate = asyncio.Event()

        async def one_request(index: int) -> float:
            await start_gate.wait()
            result = await self._chat(
                backend,
                model=model,
                prompt=f"Reply with exactly BURST_{index}.",
                max_tokens=self.args.max_tokens,
            )
            return float(result["latency_seconds"])

        tasks = [asyncio.create_task(one_request(index)) for index in range(self.args.concurrency)]
        start_gate.set()
        outputs = await asyncio.gather(*tasks)
        await self._wait_state(backend.id, {"awake"}, self.args.request_timeout)
        await asyncio.sleep(0.5)
        after = count_log_event(self.proxy_log, backend.id, "wake_requested")
        delta = None if before is None or after is None else after - before
        if not self.args.skip_log_count_check and delta is not None and delta != 1:
            raise LiveTestFailure(
                f"{self.args.concurrency} concurrent requests produced {delta} wake_requested events"
            )
        return {
            "concurrency": self.args.concurrency,
            "wake_log_delta": delta,
            "latency_seconds": {
                "minimum": min(outputs),
                "maximum": max(outputs),
                "average": sum(outputs) / len(outputs),
            },
        }

    async def _stream_guard(self, backend: BackendConfig) -> dict[str, Any]:
        assert self.proxy_session
        await self._wake(backend.id)
        model = self.model_ids[backend.id]
        base = listener_url(choose_listener(backend))
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Produce a long numbered list from 1 through 300, one item per line, "
                        "without summarizing or stopping early."
                    ),
                }
            ],
            "temperature": 0,
            "max_tokens": max(256, self.args.stream_max_tokens),
            "stream": True,
        }
        response = await self.proxy_session.post(
            f"{base}/v1/chat/completions",
            headers=self.api_headers,
            json=payload,
        )
        try:
            if response.status != 200:
                text = await response.text()
                raise LiveTestFailure(f"stream request returned HTTP {response.status}: {text[:1000]}")

            await self._wait_active_requests(backend.id, minimum=1, timeout=10)
            status_code, sleep_payload = await self._admin_json(
                "POST",
                f"/api/backends/{backend.id}/sleep",
                expected={409},
            )
            if status_code != 409:
                raise LiveTestFailure(f"sleep during active stream was not rejected: {sleep_payload}")

            chunks = 0
            deadline = time.monotonic() + min(30, self.args.request_timeout)
            async for chunk in response.content.iter_any():
                if chunk:
                    chunks += 1
                if chunks >= self.args.stream_chunks_to_read or time.monotonic() >= deadline:
                    break
        finally:
            response.close()

        await self._wait_active_requests(backend.id, minimum=0, timeout=15, exact=True)
        return {"sleep_rejected_status": 409, "stream_chunks_read": chunks}

    async def _wait_active_requests(
        self,
        backend_id: str,
        *,
        minimum: int,
        timeout: float,
        exact: bool = False,
    ) -> int:
        deadline = time.monotonic() + timeout
        last = -1
        while time.monotonic() < deadline:
            status = await self._backend_status(backend_id)
            last = int(status.get("active_requests", 0))
            if (exact and last == minimum) or (not exact and last >= minimum):
                return last
            await asyncio.sleep(0.1)
        comparator = "==" if exact else ">="
        raise LiveTestFailure(f"active_requests did not become {comparator} {minimum}; last={last}")

    async def _test_auto_idle(self, backend: BackendConfig) -> dict[str, Any]:
        assert self.original_config_json is not None
        temporary = copy.deepcopy(self.original_config_json)
        target = next(item for item in temporary["backends"] if item["id"] == backend.id)
        target["idle_timeout_seconds"] = self.args.auto_idle_seconds
        await self._put_config_json(temporary)

        refreshed = await load_config(self.config_path)
        active_backend = next(item for item in refreshed.backends if item.id == backend.id)
        await self._wake(backend.id)
        await self._chat(active_backend, model=self.model_ids[backend.id])
        started = time.monotonic()
        sleeping = await self._wait_state(
            backend.id,
            {"sleeping"},
            self.args.auto_idle_seconds
            + max(self.args.auto_idle_grace_seconds, active_backend.wake_timeout_seconds + 10),
        )
        elapsed = time.monotonic() - started
        if elapsed + 1 < self.args.auto_idle_seconds:
            raise LiveTestFailure(
                f"backend slept too early after {elapsed:.2f}s with idle timeout {self.args.auto_idle_seconds}s"
            )
        wake_result = await self._chat(active_backend, model=self.model_ids[backend.id])
        await self._wait_state(backend.id, {"awake"}, self.args.request_timeout)
        return {
            "configured_idle_seconds": self.args.auto_idle_seconds,
            "observed_sleep_after_seconds": elapsed,
            "sleep_duration_ms": sleeping.get("last_sleep_duration_ms"),
            "wake_request_latency_seconds": wake_result["latency_seconds"],
        }

    async def _test_independent_timers(
        self,
        short_backend: BackendConfig,
        long_backend: BackendConfig,
    ) -> dict[str, Any]:
        assert self.original_config_json is not None
        temporary = copy.deepcopy(self.original_config_json)
        for item in temporary["backends"]:
            if item["id"] == short_backend.id:
                item["idle_timeout_seconds"] = self.args.independent_short_seconds
            elif item["id"] == long_backend.id:
                item["idle_timeout_seconds"] = self.args.independent_long_seconds
        await self._put_config_json(temporary)

        refreshed = await load_config(self.config_path)
        by_id = {item.id: item for item in refreshed.backends}
        short_backend = by_id[short_backend.id]
        long_backend = by_id[long_backend.id]

        await asyncio.gather(self._wake(short_backend.id), self._wake(long_backend.id))
        await asyncio.gather(
            self._chat(short_backend, model=self.model_ids[short_backend.id]),
            self._chat(long_backend, model=self.model_ids[long_backend.id]),
        )

        short_status = await self._wait_state(
            short_backend.id,
            {"sleeping"},
            self.args.independent_short_seconds
            + max(self.args.auto_idle_grace_seconds, short_backend.wake_timeout_seconds + 10),
        )
        long_status = await self._backend_status(long_backend.id)
        if long_status.get("state") != "awake":
            raise LiveTestFailure(
                f"{long_backend.id} should remain awake while {short_backend.id} sleeps; status={long_status}"
            )

        # Reset only the long backend's timer and verify that activity does not
        # wake the short backend or alter its timer.
        await self._chat(long_backend, model=self.model_ids[long_backend.id])
        await asyncio.sleep(2)
        short_after = await self._backend_status(short_backend.id)
        long_after = await self._backend_status(long_backend.id)
        if short_after.get("state") != "sleeping":
            raise LiveTestFailure(f"activity on {long_backend.id} changed {short_backend.id}: {short_after}")
        if long_after.get("state") != "awake":
            raise LiveTestFailure(f"{long_backend.id} did not remain awake after its own activity: {long_after}")

        return {
            "short_backend": short_backend.id,
            "short_idle_seconds": self.args.independent_short_seconds,
            "short_state": short_status.get("state"),
            "long_backend": long_backend.id,
            "long_idle_seconds": self.args.independent_long_seconds,
            "long_state": long_after.get("state"),
        }

    async def _restore_all(self) -> None:
        if self.admin_session is None:
            return
        if self.original_config_json is not None:
            try:
                await self._put_config_json(self.original_config_json)
                self.report.restored_config = True
            except Exception as exc:
                self.report.warnings.append(f"failed to restore original proxy configuration: {exc}")

        restored = True
        for backend_id, initial_state in self.initial_states.items():
            try:
                if initial_state == "sleeping":
                    await self._sleep(backend_id, force=True)
                elif initial_state == "awake":
                    await self._wake(backend_id)
                # Offline/error states are not actively recreated.
            except Exception as exc:
                restored = False
                self.report.warnings.append(
                    f"failed to restore {backend_id} to initial state {initial_state}: {exc}"
                )
        self.report.restored_backend_states = restored

    def _write_report(self) -> None:
        output = Path(self.args.report).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nReport: {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run destructive integration tests against currently running vLLM models and "
            "the smart sleep proxy. The test sleeps and wakes real models."
        )
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "proxy_config.json"))
    parser.add_argument("--admin-url", help="Override the admin URL, e.g. http://127.0.0.1:5100")
    parser.add_argument("--backend", action="append", help="Backend id to test; repeatable. Default: all")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        help="Bearer token for vLLM if configured",
    )
    parser.add_argument("--confirm-live", action="store_true", help="Required safety acknowledgement")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--stream-max-tokens", type=int, default=512)
    parser.add_argument("--stream-chunks-to-read", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--memory-settle-seconds", type=float, default=2)
    parser.add_argument("--memory-check-minimum-mib", type=float, default=1024)
    parser.add_argument("--minimum-memory-drop-mib", type=float, default=512)
    parser.add_argument("--auto-idle-seconds", type=int, default=8)
    parser.add_argument("--auto-idle-grace-seconds", type=int, default=30)
    parser.add_argument("--independent-short-seconds", type=int, default=8)
    parser.add_argument("--independent-long-seconds", type=int, default=60)
    parser.add_argument("--skip-stream", action="store_true")
    parser.add_argument("--skip-auto-idle", action="store_true")
    parser.add_argument("--skip-independent-timers", action="store_true")
    parser.add_argument(
        "--skip-log-count-check",
        action="store_true",
        help="Do not require exactly one sleep/wake log event; useful when other clients are active",
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / f"live_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.confirm_live:
        raise SystemExit(
            "Refusing to operate real models without --confirm-live. "
            "This test sends inference requests and puts selected models to sleep."
        )
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")
    if args.max_tokens < 1 or args.stream_max_tokens < 1:
        raise SystemExit("token limits must be positive")
    if args.auto_idle_seconds < 2:
        raise SystemExit("--auto-idle-seconds must be at least 2")
    if args.independent_short_seconds < 2:
        raise SystemExit("--independent-short-seconds must be at least 2")
    if args.independent_long_seconds <= args.independent_short_seconds + 5:
        raise SystemExit("--independent-long-seconds must exceed short seconds by more than 5")


async def async_main(args: argparse.Namespace) -> int:
    tester = LiveModelTester(args)
    try:
        report = await tester.run()
    except Exception as exc:
        print(f"\nLIVE TEST FAILED: {exc}", file=sys.stderr)
        return 1

    passed = sum(item.status == "passed" for item in report.results)
    failed = sum(item.status == "failed" for item in report.results)
    print(f"\nSummary: {passed} passed, {failed} failed", flush=True)
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    return 0 if report.success else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    raise SystemExit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
