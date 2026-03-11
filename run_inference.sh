#!/bin/bash
export RUN_NAME="inference"

python scripts/lots/inference_lots.py \
    --base_model_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --dataset_root="data/sketchy" \
    --out_dir="outputs/inference/$RUN_NAME" \
    --seed=3407 \
    --ckpt_path="ckpts/lots/lots.bin"
