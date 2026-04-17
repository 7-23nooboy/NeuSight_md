#!/bin/bash
# DeepMD-kit 推理性能预测：水系统，含 force 计算 @ NVIDIA H100
cd "$(dirname "$0")"

python3 ../pred_deepmd.py \
    --predictor_path ../asplos/data/predictor/MLP_WAVE \
    --device_config_path '../asplos/data/device_configs/NVIDIA_H100_80GB_HBM3.json' \
    --deepmd_config_path ../asplos/data/deepmd_configs/water_se_e2_a.json \
    --num_atoms 192 \
    --tile_dataset_dir ../asplos/data/dataset/train \
    --result_dir ./out_deepmd \
    --compute_force
