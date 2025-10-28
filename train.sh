#!/bin/bash
# Quick training script

python local_llm_training.py \
    --model gpt2 \
    --dataset eli5 \
    --num-samples 5000 \
    --epochs 2 \
    --batch-size 4 \
    --output-dir ./my_trained_model
