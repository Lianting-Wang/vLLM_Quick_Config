from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

COOKIE_NAME = "vllm_proxy_session"
ENV_FILE_NAME = ".env"
MIN_SESSION_SECRET_LENGTH = 32


@dataclass(frozen=True, slots=True)
class AdminCredentials:
    """Validated credentials used by the administration web interface."""

    password: str
    session_secret: str
    env_path: Path


def env_file_for(config_path: Path) -> Path:
    """Return the fixed .env path next to proxy_config.json."""

    return config_path.expanduser().resolve().parent / ENV_FILE_NAME


def load_admin_credentials(config_path: Path) -> AdminCredentials:
    """Load and validate admin credentials.

    Values from the process environment override values in the local .env file.
    Authentication is always enabled, regardless of the configured listen host.
    """

    env_path = env_file_for(config_path)
    if not env_path.is_file():
        raise RuntimeError(
            f"Missing {env_path}. Copy .env.example to .env, then set "
            "ADMIN_PASSWORD and SESSION_SECRET."
        )

    try:
        file_values = dotenv_values(env_path, encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Could not read {env_path}: {exc}") from exc

    password_value = os.environ.get("ADMIN_PASSWORD")
    if password_value is None:
        password_value = file_values.get("ADMIN_PASSWORD")

    secret_value = os.environ.get("SESSION_SECRET")
    if secret_value is None:
        secret_value = file_values.get("SESSION_SECRET")

    password = "" if password_value is None else str(password_value)
    session_secret = "" if secret_value is None else str(secret_value).strip()

    if not password or not password.strip():
        raise RuntimeError(f"ADMIN_PASSWORD is missing or empty in {env_path}")
    if not session_secret:
        raise RuntimeError(f"SESSION_SECRET is missing or empty in {env_path}")
    if len(session_secret) < MIN_SESSION_SECRET_LENGTH:
        raise RuntimeError(
            f"SESSION_SECRET in {env_path} must be at least "
            f"{MIN_SESSION_SECRET_LENGTH} characters"
        )

    return AdminCredentials(
        password=password,
        session_secret=session_secret,
        env_path=env_path,
    )


def _signature(secret: str, payload: bytes) -> bytes:
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).digest()


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def create_session_token(secret: str, ttl_seconds: int, now: int | None = None) -> str:
    issued_at = int(time.time() if now is None else now)
    expires_at = issued_at + ttl_seconds
    payload = f"{expires_at}:{secrets.token_urlsafe(18)}".encode("utf-8")
    return f"{_b64encode(payload)}.{_b64encode(_signature(secret, payload))}"


def verify_session_token(secret: str, token: str, now: int | None = None) -> bool:
    if not secret or not token:
        return False
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        payload = _b64decode(encoded_payload)
        supplied_signature = _b64decode(encoded_signature)
        expires_text, _nonce = payload.decode("utf-8").split(":", 1)
        expires_at = int(expires_text)
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return False

    current_time = int(time.time() if now is None else now)
    if expires_at < current_time:
        return False
    return hmac.compare_digest(supplied_signature, _signature(secret, payload))


def password_matches(expected: str, supplied: str) -> bool:
    if not expected or not supplied:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), supplied.encode("utf-8"))
