# Sleep/Wake State-Machine Tests

This suite exercises the proxy controller against an in-process fake vLLM server. It does not require a GPU, does not load a model, and does not alter running vLLM instances.

## Run

```bash
./run_sleep_wake_tests.sh
```

Or:

```bash
uv run --extra test pytest -vv tests/test_sleep_wake_state_machine.py
```

Run the complete project suite with:

```bash
uv run --extra test pytest -q
```

## State-machine guarantees covered

1. A burst of concurrent requests shares exactly one `/wake_up` operation.
2. Concurrent manual and automatic sleep triggers share exactly one `/sleep` operation.
3. A request arriving after `/sleep` starts waits for sleep to settle, wakes the model, and completes without leaving a false `ERROR` state.
4. A request arriving before the sleep command is sent cancels that sleep cleanly.
5. Active streaming responses prevent automatic sleep until the stream closes.
6. Different backends maintain independent activity counters and idle timers.
7. A failed sleep or wake operation can be retried.
8. Cancelling one waiting HTTP client does not cancel a shared sleep or wake transition needed by other clients.
9. Admin probes cannot overwrite `SLEEP_PENDING` or `WAKING` with a transient observation.
10. Forced sleep waits for active requests to drain without terminating them.
11. Repeated sleep/wake cycles do not retain stale transition tasks.

## Implementation model

The controller uses two synchronization layers:

- `_state_lock` protects local state, request counters, and shared task references.
- `_transition_lock` serializes the actual upstream `/sleep` and `/wake_up` operations.

Concurrent callers share `_sleep_task` or `_wake_task`. A wake operation cannot overtake a sleep already sent to vLLM. It waits for that transition to complete and then wakes the backend. Shared tasks are shielded so one disconnected client cannot cancel the operation for all other waiters.

## Verified result

```text
30 passed
```

The race-focused subset was also run repeatedly, and the full suite was run with `PYTHONASYNCIODEBUG=1`.
