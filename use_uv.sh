#!/usr/bin/env bash

# 强制退出 conda 环境（如果存在）
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    conda deactivate
fi

# 激活本地 venv / uv 环境
source .venv/bin/activate

# 显示当前 Python 路径用于确认
echo "Using Python:"
which python
python --version