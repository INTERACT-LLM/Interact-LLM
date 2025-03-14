#!/bin/bash

models=("mlx-community/meta-Llama-3.1-8B-Instruct-4bit" "mlx-community/Qwen2.5-7B-Instruct-1M-4bit")
prompt_ids=("A1" "B1" "C1")

for model in "${models[@]}"; do
    for prompt_id in "${prompt_ids[@]}"; do
            uv run python src/scripts/simulate.py --model "$model" --prompt_id "$prompt_id"
    done
done
