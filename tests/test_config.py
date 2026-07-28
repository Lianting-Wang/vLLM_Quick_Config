import pytest

from vllm_proxy.config import default_config, validate_config


def test_default_config_is_valid():
    validate_config(default_config())


def test_duplicate_listener_port_is_rejected():
    config = default_config()
    config.backends[0].listeners[1].port = config.backends[0].listeners[0].port
    with pytest.raises(ValueError):
        validate_config(config)


def test_remote_admin_config_is_valid_without_secret_fields():
    config = default_config()
    config.admin.host = "0.0.0.0"
    validate_config(config)


def test_admin_config_does_not_serialize_a_secret():
    data = default_config().to_dict()
    assert "token" not in data["admin"]
    assert "password" not in data["admin"]
    assert "password_file" not in data["admin"]
    assert data["admin"] == {"host": "127.0.0.1", "port": 8070, "session_ttl_hours": 168}
