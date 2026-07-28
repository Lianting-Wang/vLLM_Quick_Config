# vLLM Quick Config + Smart Sleep Proxy

This project starts multiple vLLM profiles and provides a multi-port proxy that independently sleeps each idle model and wakes it when a request arrives.

## 1. Install

```bash
uv venv --python 3.12
uv sync --extra test
```

## 2. Start and stop vLLM profiles

Start a profile directly:

```bash
./run.sh qwen3.6_uncensored
./run.sh minicpm5_1b_fast
```

Run `./run.sh` without arguments to select a profile interactively.

Stop a profile directly:

```bash
./stop.sh qwen3.6_uncensored
```

Run `./stop.sh` without arguments to select one of the currently running models or stop all running models. Other commands:

```bash
./stop.sh --list
./stop.sh --all
```

The source vLLM ports listen only on `127.0.0.1`, use `--enable-sleep-mode`, and set `VLLM_SERVER_DEV_MODE=1`. Do not expose source ports publicly because vLLM development mode includes privileged management endpoints.

## 3. Configure administration authentication

`proxy_config.json` contains only non-secret settings:

```json
{
  "admin": {
    "host": "0.0.0.0",
    "port": 5100,
    "session_ttl_hours": 168
  }
}
```

Copy the environment template and edit it:

```bash
cp .env.example .env
chmod 600 .env
```

`.env`:

```dotenv
ADMIN_PASSWORD=your-admin-password
SESSION_SECRET=replace-with-a-long-random-value
```

Generate the session secret with:

```bash
openssl rand -hex 32
```

The real `.env` is ignored by Git; `.env.example` remains tracked as a template. The `.env` file is always located beside `proxy_config.json`. Process environment variables named `ADMIN_PASSWORD` and `SESSION_SECRET` override values from the file.

Open the administration page normally:

```text
http://SERVER_IP:5100
```

The browser shows a password login page. After login, authentication uses an HttpOnly, SameSite cookie. Neither the password nor the session token appears in the URL, browser history, JavaScript storage, or `proxy_config.json`.

Administration authentication is always enabled, including when the page listens only on localhost. Both `ADMIN_PASSWORD` and `SESSION_SECRET` are required. The Python application performs this validation once during startup, so the same security rules apply whether it is started through `run_proxy.sh` or directly with `uv run python -m vllm_proxy.app`.

## 4. Start the smart proxy

```bash
./run_proxy.sh
```

`run_proxy.sh` only manages the proxy process, PID file, and log file. It does not parse JSON or credentials. If `.env` is missing or invalid, the Python application exits and the startup script prints the relevant log message.

The supplied configuration exposes:

- `http://HOST:5000/v1/...` — Qwen normal transparent API.
- `http://HOST:5010/v1/...` — Qwen API with `chat_template_kwargs.enable_thinking=false`.
- `http://HOST:5001/v1/...` — MiniCPM transparent API.
- `http://ADMIN_HOST:5100` — administration page.

Stop the proxy with:

```bash
./stop_proxy.sh
```

The proxy does not start or terminate vLLM. Start the required profiles separately.

## 5. Proxy configuration

Edit `proxy_config.json` directly or use the web page. Each backend has its own source URL, idle timer, active request count, sleep/wake state machine, wake timeout, listener ports, manual controls, and GPU-memory status.

Qwen and MiniCPM idle timers are completely independent. A request sent to one backend does not reset the other backend's timer.

Changing listener ports from the web page takes effect immediately. Changing the administration host or port requires restarting the proxy.

## 6. Sleep behavior

The proxy calls vLLM's local development endpoints internally:

```text
POST /sleep?level=1
POST /wake_up
GET  /is_sleeping
```

Health and model-list probes do not reset the idle timer. A streaming request remains active until the stream finishes or disconnects. Concurrent requests arriving while one model is sleeping share one wake operation for that model.

The public proxy blocks privileged paths such as `/sleep`, `/wake_up`, `/collective_rpc`, and cache-reset endpoints.

Level 1 sleep offloads model weights to CPU RAM and discards KV cache. It releases most model GPU memory but can leave CUDA/NCCL context memory allocated.

## 7. Test

```bash
uv run pytest
python -m compileall vllm_proxy
bash -n run.sh run_proxy.sh stop_proxy.sh stop.sh
```
