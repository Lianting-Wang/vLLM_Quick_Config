# Live vLLM sleep/wake integration test

`tools/live_model_sleep_wake_test.py` operates the real vLLM processes described by
`proxy_config.json`. Unlike the deterministic unit tests, it sends actual inference
requests through the proxy and calls the administration sleep/wake endpoints.

## Safety

Run this only during a maintenance window. Selected models are deliberately put to
sleep several times. The script rejects manual sleep while a stream is active, but
other applications using the same model can still affect timing and log-count checks.

The script records the original proxy configuration and initial backend states. In a
`finally` block it restores the original configuration and returns each initially
awake/sleeping backend to that state. A machine crash or `kill -9` can prevent this
cleanup, so retain a copy of `proxy_config.json` before testing.

## Basic usage

Test all configured backends:

```bash
./run_live_model_tests.sh
```

Test only Qwen:

```bash
./run_live_model_tests.sh --backend qwen36
```

Test only MiniCPM:

```bash
./run_live_model_tests.sh --backend minicpm5
```

Non-interactive execution:

```bash
VLLM_LIVE_TEST_CONFIRM=YES ./run_live_model_tests.sh --backend qwen36
```

If vLLM requires an API key:

```bash
VLLM_API_KEY='your-key' ./run_live_model_tests.sh
```

## Test coverage

For every selected backend, the script verifies:

1. Admin login and backend health.
2. Manual wake and a real chat-completion request.
3. Manual Level 1 sleep and direct `/is_sleeping` confirmation.
4. Model-process GPU memory decreases after sleeping, when NVML/PID attribution is available.
5. A chat request sent while sleeping wakes the model and completes normally.
6. A concurrent burst sent while sleeping completes successfully and produces one
   `wake_requested` log event when the current proxy log is available.
7. An active streaming request causes manual non-force sleep to return HTTP 409.
8. Automatic idle sleep occurs after a temporary short timeout, followed by a real
   request-triggered wake.
9. When at least two backends are selected, short and long idle timers operate
   independently.

## Useful options

Skip tests that temporarily change idle timeouts:

```bash
./run_live_model_tests.sh --skip-auto-idle --skip-independent-timers
```

Skip streaming protection:

```bash
./run_live_model_tests.sh --skip-stream
```

Change burst concurrency:

```bash
./run_live_model_tests.sh --concurrency 8
```

Use an explicit admin URL:

```bash
./run_live_model_tests.sh --admin-url http://127.0.0.1:5100
```

## Output

The terminal shows each test step. A JSON report is written to the repository root:

```text
live_test_report_YYYYMMDD_HHMMSS.json
```

A zero exit code means every enabled live test passed. A nonzero exit code means at
least one test failed or cleanup could not be completed.


## Transition verification

Live tests verify exactly one upstream `/sleep` or `/wake_up` command using in-process counters exposed by `/api/status`. Text logs are retained only as diagnostics. Restart the proxy after updating project files so the running proxy version matches the test code.
