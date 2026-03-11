#!/bin/bash
export RUN_NAME="train"

accelerate launch --mixed_precision "bf16" --num_processes 4 --multi-gpu \
    scripts/lots/train_lots.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --dataset_root="data/sketchy" \
    --output_dir="outputs/checkpoints/$RUN_NAME" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --num_train_epochs=80 \
    --dataloader_num_workers=8 \
    --save_steps=5000 \
    --train_batch_size=4 \
    --dinov2_model="vits14" \
    --num_cls_tokens=64 \
    --fusion_strategy="deferred" \
    --seed=3407 \
    --gradient_accumulation_steps=16
