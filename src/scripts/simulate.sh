#!/bin/bash

models=("llama3.1:8b" "qwen2.5:7b")
prompt_ids=("A1" "B1" "C1")
backend="hf"

for model in "${models[@]}"; do
    for prompt_id in "${prompt_ids[@]}"; do
            uv run python src/scripts/simulate.py --model_name "$model" --prompt_id "$prompt_id" --backend "$backend"
    done
done
