import json

from vllm_proxy.proxy import inject_no_thinking


def test_passthrough_path_is_unchanged():
    body = b'{"messages":[]}'
    assert inject_no_thinking(body, "application/json", "/v1/completions") == body


def test_no_thinking_is_injected():
    body = b'{"model":"qwen3.6","messages":[]}'
    result = json.loads(inject_no_thinking(body, "application/json", "/v1/chat/completions"))
    assert result["chat_template_kwargs"]["enable_thinking"] is False


def test_existing_kwargs_are_preserved_and_overridden():
    body = json.dumps(
        {
            "messages": [],
            "chat_template_kwargs": {"enable_thinking": True, "custom": "value"},
        }
    ).encode()
    result = json.loads(inject_no_thinking(body, "application/json", "/v1/chat/completions"))
    assert result["chat_template_kwargs"] == {"enable_thinking": False, "custom": "value"}


def test_invalid_json_is_forwarded_unchanged():
    body = b'{broken'
    assert inject_no_thinking(body, "application/json", "/v1/chat/completions") == body
