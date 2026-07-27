import pytest

from vllm_proxy.config import ProxyConfig, default_config, validate_config


def test_default_config_is_valid():
    validate_config(default_config())


def test_duplicate_listener_port_is_rejected():
    config = default_config()
    config.backends[0].listeners[1].port = config.backends[0].listeners[0].port
    with pytest.raises(ValueError):
        validate_config(config)


def test_remote_admin_requires_token():
    config = default_config()
    config.admin.host = "0.0.0.0"
    with pytest.raises(ValueError):
        validate_config(config)
