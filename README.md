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
./run.sh gptq
./run.sh fp8
```

Run `./run.sh` without arguments to select a profile from the interactive menu. In non-interactive shells, `./run.sh` uses `default_profile` from `models.conf`.

Model settings live in `models.conf`. The `[defaults]` section supplies shared vLLM settings, and each model profile can override or add fields such as `model`, `served_model_name`, `reasoning_parser`, `enable_thinking`, `tool_call_parser`, `speculative_config`, `cudagraph_mode`, `max_num_batched_tokens`, and `tensor_parallel_size`.

`max_num_batched_tokens` maps to vLLM `--max-num-batched-tokens`. `tensor_parallel_size` maps to vLLM `--tensor-parallel-size`; set `cuda_visible_devices` to the GPUs you want vLLM to see and set `tensor_parallel_size` to the number of GPUs used for tensor parallelism.

For Qwen thinking mode, set `enable_thinking=true` or `enable_thinking=false` in the model profile. The launcher passes this through as vLLM `--default-chat-template-kwargs`.
