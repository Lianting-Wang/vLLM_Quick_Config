from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_CONFIG_PATH = Path(os.environ.get("PROXY_CONFIG_FILE", "proxy_config.json"))


@dataclass(slots=True)
class ListenerConfig:
    host: str
    port: int
    mode: str = "passthrough"
    enabled: bool = True
    name: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ListenerConfig":
        return cls(
            host=str(value.get("host", "0.0.0.0")),
            port=int(value["port"]),
            mode=str(value.get("mode", "passthrough")),
            enabled=bool(value.get("enabled", True)),
            name=str(value.get("name", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "mode": self.mode,
            "enabled": self.enabled,
            "name": self.name,
        }


@dataclass(slots=True)
class BackendConfig:
    id: str
    name: str
    upstream_url: str
    profile: str = ""
    pidfile: str = ""
    idle_timeout_seconds: int = 1800
    wake_timeout_seconds: int = 180
    sleep_level: int = 1
    listeners: list[ListenerConfig] = field(default_factory=list)
    idle_ignore_paths: list[str] = field(
        default_factory=lambda: ["/health", "/ping", "/version", "/v1/models"]
    )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BackendConfig":
        return cls(
            id=str(value["id"]),
            name=str(value.get("name", value["id"])),
            upstream_url=str(value["upstream_url"]).rstrip("/"),
            profile=str(value.get("profile", "")),
            pidfile=str(value.get("pidfile", "")),
            idle_timeout_seconds=int(value.get("idle_timeout_seconds", 1800)),
            wake_timeout_seconds=int(value.get("wake_timeout_seconds", 180)),
            sleep_level=int(value.get("sleep_level", 1)),
            listeners=[ListenerConfig.from_dict(item) for item in value.get("listeners", [])],
            idle_ignore_paths=list(value.get("idle_ignore_paths", ["/health", "/ping", "/version", "/v1/models"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "upstream_url": self.upstream_url,
            "profile": self.profile,
            "pidfile": self.pidfile,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "wake_timeout_seconds": self.wake_timeout_seconds,
            "sleep_level": self.sleep_level,
            "listeners": [item.to_dict() for item in self.listeners],
            "idle_ignore_paths": self.idle_ignore_paths,
        }


@dataclass(slots=True)
class AdminConfig:
    host: str = "127.0.0.1"
    port: int = 8070
    token: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AdminConfig":
        return cls(
            host=str(value.get("host", "127.0.0.1")),
            port=int(value.get("port", 8070)),
            token=str(value.get("token", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port, "token": self.token}


@dataclass(slots=True)
class ProxyConfig:
    admin: AdminConfig
    backends: list[BackendConfig]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProxyConfig":
        return cls(
            admin=AdminConfig.from_dict(value.get("admin", {})),
            backends=[BackendConfig.from_dict(item) for item in value.get("backends", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {"admin": self.admin.to_dict(), "backends": [item.to_dict() for item in self.backends]}


def _validate_host(host: str, field_name: str) -> None:
    if not host:
        raise ValueError(f"{field_name} cannot be empty")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if any(char.isspace() for char in host):
            raise ValueError(f"{field_name} is invalid")


def _validate_port(port: int, field_name: str) -> None:
    if not 1 <= port <= 65535:
        raise ValueError(f"{field_name} must be between 1 and 65535")


def validate_config(config: ProxyConfig) -> None:
    _validate_host(config.admin.host, "admin.host")
    _validate_port(config.admin.port, "admin.port")
    if config.admin.host not in {"127.0.0.1", "::1", "localhost"} and not config.admin.token:
        raise ValueError("admin.token is required when the admin interface is not localhost-only")

    ids: set[str] = set()
    endpoints: set[tuple[str, int]] = {(config.admin.host, config.admin.port)}
    any_host_ports: set[int] = {config.admin.port} if config.admin.host in {"0.0.0.0", "::"} else set()

    for backend in config.backends:
        if not backend.id or backend.id in ids:
            raise ValueError(f"backend id must be unique and non-empty: {backend.id!r}")
        ids.add(backend.id)
        parsed = urlparse(backend.upstream_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"invalid upstream_url for backend {backend.id}")
        if backend.idle_timeout_seconds < 0:
            raise ValueError("idle_timeout_seconds cannot be negative")
        if backend.wake_timeout_seconds <= 0:
            raise ValueError("wake_timeout_seconds must be positive")
        if backend.sleep_level not in {1, 2}:
            raise ValueError("sleep_level must be 1 or 2")
        for listener in backend.listeners:
            _validate_host(listener.host, f"listener host for {backend.id}")
            _validate_port(listener.port, f"listener port for {backend.id}")
            if listener.mode not in {"passthrough", "no_thinking"}:
                raise ValueError(f"unsupported listener mode: {listener.mode}")
            endpoint = (listener.host, listener.port)
            if endpoint in endpoints or listener.port in any_host_ports:
                raise ValueError(f"duplicate or conflicting listen endpoint: {listener.host}:{listener.port}")
            if listener.host in {"0.0.0.0", "::"}:
                if any(port == listener.port for _, port in endpoints):
                    raise ValueError(f"wildcard listener conflicts on port {listener.port}")
                any_host_ports.add(listener.port)
            endpoints.add(endpoint)


def default_config() -> ProxyConfig:
    return ProxyConfig.from_dict(
        {
            "admin": {"host": "127.0.0.1", "port": 8070, "token": ""},
            "backends": [
                {
                    "id": "qwen36",
                    "name": "Qwen 3.6",
                    "upstream_url": "http://127.0.0.1:5000",
                    "profile": "qwen3.6_uncensored",
                    "pidfile": "vllm_server.qwen3.6_uncensored_.pid",
                    "idle_timeout_seconds": 1800,
                    "wake_timeout_seconds": 180,
                    "sleep_level": 1,
                    "listeners": [
                        {"host": "0.0.0.0", "port": 8005, "mode": "passthrough", "enabled": True, "name": "Normal API"},
                        {"host": "0.0.0.0", "port": 8006, "mode": "no_thinking", "enabled": True, "name": "No Thinking API"},
                    ],
                    "idle_ignore_paths": ["/health", "/ping", "/version", "/v1/models"],
                }
            ],
        }
    )


async def load_config(path: Path = DEFAULT_CONFIG_PATH) -> ProxyConfig:
    if not path.exists():
        config = default_config()
        await save_config(config, path)
        return config
    data = await asyncio.to_thread(path.read_text, encoding="utf-8")
    config = ProxyConfig.from_dict(json.loads(data))
    validate_config(config)
    return config


async def save_config(config: ProxyConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    validate_config(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n"

    def _write() -> None:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    await asyncio.to_thread(_write)
