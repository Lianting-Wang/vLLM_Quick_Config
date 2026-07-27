# vLLM Quick Config + Smart Sleep Proxy

This project starts vLLM profiles and provides an optional multi-port proxy that automatically sleeps an idle model and wakes it when a request arrives.

## 1. Install

```bash
uv venv --python 3.12
uv sync --extra test
```

## 2. Start vLLM

```bash
./run.sh --list
./run.sh qwen3.6_uncensored
```

The default Qwen profile listens only on `127.0.0.1:5000`, enables vLLM sleep mode, and sets `VLLM_SERVER_DEV_MODE=1`. Do not expose this source port publicly: vLLM development mode includes privileged management endpoints.

Existing commands remain available:

```bash
./stop.sh qwen3.6_uncensored
./stop.sh --all
```

Model settings live in `models.conf`. A profile can override `host`, `port`, `enable_sleep_mode`, and `server_dev_mode` as well as existing vLLM arguments.

## 3. Start the smart proxy

```bash
./run_proxy.sh
```

Default endpoints:

- `http://HOST:8005/v1/...` — normal transparent API proxy.
- `http://HOST:8006/v1/...` — injects `chat_template_kwargs.enable_thinking=false` into JSON `POST /v1/chat/completions` requests.
- `http://127.0.0.1:8070` — administration page.

Stop it with:

```bash
./stop_proxy.sh
```

The proxy does not start or terminate vLLM. Start the selected vLLM profile first.

## 4. Configuration

Edit `proxy_config.json` directly or use the web page. Each backend supports:

- source URL and model profile metadata;
- idle timeout and wake timeout;
- any number of enabled listener ports;
- `passthrough` or `no_thinking` listener mode;
- manual sleep, wake, and health probe;
- GPU and model-process memory monitoring through NVML.

Changing listener ports from the web page takes effect immediately. Changing the administration host or port requires restarting the proxy.

If the administration interface is changed from localhost to a remote bind address, `admin.token` is mandatory. Open the page with `?token=YOUR_TOKEN`; API calls and the event stream will use that token.

## 5. Sleep behavior

The proxy calls vLLM's development endpoints internally:

```text
POST /sleep?level=1
POST /wake_up
GET  /is_sleeping
```

Health and model-list probes do not reset the idle timer. A streaming request remains active until the stream finishes or disconnects. Concurrent requests arriving during sleep share one wake operation.

The proxy blocks external access to privileged vLLM paths such as `/sleep`, `/wake_up`, `/collective_rpc`, and cache-reset endpoints.

Level 1 sleep offloads model weights to CPU RAM and discards KV cache. It substantially reduces GPU memory use but may leave CUDA/NCCL context memory allocated.

## 6. Test

```bash
uv run pytest
python -m compileall vllm_proxy
bash -n run.sh run_proxy.sh stop_proxy.sh stop.sh
```
