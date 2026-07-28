from pathlib import Path

from tools.live_model_sleep_wake_test import (
    choose_listener,
    count_log_event,
    extract_model_memory,
    listener_url,
    local_url,
    transition_counter,
)
from vllm_proxy.config import BackendConfig, ListenerConfig


def test_local_url_maps_wildcard_hosts_to_loopback() -> None:
    assert local_url("0.0.0.0", 5100) == "http://127.0.0.1:5100"
    assert local_url("::", 5100) == "http://[::1]:5100"


def test_choose_listener_prefers_passthrough() -> None:
    backend = BackendConfig(
        id="model",
        name="Model",
        upstream_url="http://127.0.0.1:4000",
        listeners=[
            ListenerConfig(host="0.0.0.0", port=5010, mode="no_thinking"),
            ListenerConfig(host="0.0.0.0", port=5000, mode="passthrough"),
        ],
    )
    selected = choose_listener(backend)
    assert selected.port == 5000
    assert listener_url(selected) == "http://127.0.0.1:5000"


def test_extract_model_memory_requires_available_nvml() -> None:
    assert extract_model_memory({"gpu": {"available": False}}) is None
    assert extract_model_memory(
        {"gpu": {"available": True, "model_process_memory_mib": 1234.5}}
    ) == 1234.5


def test_count_log_event(tmp_path: Path) -> None:
    log = tmp_path / "proxy.log"
    log.write_text(
        "backend=qwen action=wake_requested\n"
        "backend=qwen action=wake_requested\n"
        "backend=minicpm action=wake_requested\n",
        encoding="utf-8",
    )
    assert count_log_event(log, "qwen", "wake_requested") == 2
    assert count_log_event(log, "minicpm", "wake_requested") == 1


def test_transition_counter_reads_running_controller_counter() -> None:
    status = {"transition_counters": {"sleep_commands": 3, "wake_commands": 4}}
    assert transition_counter(status, "sleep_commands") == 3
    assert transition_counter(status, "wake_commands") == 4
    assert transition_counter(status, "missing") is None
    assert transition_counter({}, "sleep_commands") is None
