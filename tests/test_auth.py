from pathlib import Path

import pytest

from vllm_proxy.auth import (
    AdminCredentials,
    create_session_token,
    env_file_for,
    load_admin_credentials,
    password_matches,
    verify_session_token,
)


def test_session_token_round_trip():
    token = create_session_token("session-signing-secret", 3600, now=100)
    assert verify_session_token("session-signing-secret", token, now=101)
    assert not verify_session_token("wrong", token, now=101)


def test_expired_session_token_is_rejected():
    token = create_session_token("secret", 10, now=100)
    assert not verify_session_token("secret", token, now=111)


def test_env_file_is_fixed_next_to_config(tmp_path: Path):
    config_path = tmp_path / "nested" / "proxy_config.json"
    assert env_file_for(config_path) == (tmp_path / "nested" / ".env").resolve()


def test_credentials_are_loaded_from_env_next_to_config(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "proxy_config.json"
    (tmp_path / ".env").write_text(
        'ADMIN_PASSWORD="中文 管理密码"\n'
        'SESSION_SECRET="0123456789abcdef0123456789abcdef"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("SESSION_SECRET", raising=False)

    credentials = load_admin_credentials(config_path)

    assert credentials == AdminCredentials(
        password="中文 管理密码",
        session_secret="0123456789abcdef0123456789abcdef",
        env_path=(tmp_path / ".env").resolve(),
    )


def test_process_environment_overrides_dotenv(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "proxy_config.json"
    (tmp_path / ".env").write_text(
        "ADMIN_PASSWORD=file-value\n"
        "SESSION_SECRET=0123456789abcdef0123456789abcdef\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ADMIN_PASSWORD", "process-value")
    monkeypatch.setenv("SESSION_SECRET", "fedcba9876543210fedcba9876543210")

    credentials = load_admin_credentials(config_path)

    assert credentials.password == "process-value"
    assert credentials.session_secret == "fedcba9876543210fedcba9876543210"


def test_missing_dotenv_is_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="Missing .*\\.env"):
        load_admin_credentials(tmp_path / "proxy_config.json")


def test_empty_password_is_rejected(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "proxy_config.json"
    (tmp_path / ".env").write_text(
        "ADMIN_PASSWORD=\n"
        "SESSION_SECRET=0123456789abcdef0123456789abcdef\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD"):
        load_admin_credentials(config_path)


def test_short_session_secret_is_rejected(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "proxy_config.json"
    (tmp_path / ".env").write_text(
        "ADMIN_PASSWORD=password\nSESSION_SECRET=too-short\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="at least 32"):
        load_admin_credentials(config_path)


def test_password_comparison():
    assert password_matches("secret", "secret")
    assert not password_matches("secret", "different")
