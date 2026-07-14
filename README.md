# vLLM Quick Config

Create the Python environment:

```bash
uv venv --python 3.12
```

List configured model profiles:

```bash
./run.sh --list
```

Start a model profile:

```bash
./run.sh qwen3.6_uncensored
./run.sh minicpm5_1b_fast
```

Run `./run.sh` without arguments to select a profile from the interactive menu. In non-interactive shells, `./run.sh` uses `default_profile` from `models.conf`.

Each profile runs as an independent service and can be started alongside other profiles. Stop one profile with `./stop.sh PROFILE`, or stop every profile started by this launcher with `./stop.sh --all`.

The configured Qwen service is available on `http://localhost:5000/v1`; `minicpm5_1b_fast` is available on `http://localhost:5001/v1` with the served model name `minicpm5`.

Model settings live in `models.conf`. The `[defaults]` section supplies shared vLLM settings, and each model profile can override or add fields such as `model`, `served_model_name`, `reasoning_parser`, `enable_thinking`, `tool_call_parser`, `speculative_config`, `cudagraph_mode`, `max_num_batched_tokens`, and `tensor_parallel_size`.

`max_num_batched_tokens` maps to vLLM `--max-num-batched-tokens`. `tensor_parallel_size` maps to vLLM `--tensor-parallel-size`; set `cuda_visible_devices` to the GPUs you want vLLM to see and set `tensor_parallel_size` to the number of GPUs used for tensor parallelism.

For Qwen thinking mode, set `enable_thinking=true` or `enable_thinking=false` in the model profile. The launcher passes this through as vLLM `--default-chat-template-kwargs`.
