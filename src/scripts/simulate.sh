#!/bin/bash

models=("bartowski/Qwen2.5-7B-Instruct-1M-GGUF")
prompt_ids=("A1" "B1" "C1")

for model in "${models[@]}"; do
    for prompt_id in "${prompt_ids[@]}"; do
            uv run python src/scripts/simulate.py --model "$model" --prompt_id "$prompt_id"
    done
done
