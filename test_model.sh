#!/bin/bash
# Test trained model

python inference_server.py \
    --model-path ./my_trained_model/final_model \
    --mode interactive
