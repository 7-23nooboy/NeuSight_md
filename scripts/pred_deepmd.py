"""
DeepMD-kit 推理性能预测 CLI 入口

用法 1 — 直接传入 DeepMD 训练用的 input.json（推荐，自动提取参数）:
    python scripts/pred_deepmd.py \
        --predictor_path scripts/asplos/data/predictor/MLP_WAVE \
        --device_config_path scripts/asplos/data/device_configs/NVIDIA_H100_80GB_HBM3.json \
        --deepmd_config_path /path/to/your/deepmd/input.json \
        --num_atoms 192 \
        --tile_dataset_dir scripts/asplos/data/dataset/train \
        --result_dir results/deepmd/

用法 2 — 跨平台预测（加入 host config 进行 CPU/GPU 缩放）:
    python scripts/pred_deepmd.py \
        --predictor_path scripts/asplos/data/predictor/MLP_WAVE \
        --device_config_path scripts/asplos/data/device_configs/Tesla_T4.json \
        --deepmd_config_path /path/to/your/deepmd/input.json \
        --host_config_path scripts/asplos/data/host_configs/T4_estimated.json \
        --num_atoms 192 \
        --tile_dataset_dir scripts/asplos/data/dataset/train \
        --result_dir results/deepmd/

用法 3 — 传入已转换的 NeuSight config:
    python scripts/pred_deepmd.py \
        --predictor_path scripts/asplos/data/predictor/MLP_WAVE \
        --device_config_path scripts/asplos/data/device_configs/NVIDIA_H100_80GB_HBM3.json \
        --deepmd_config_path scripts/asplos/data/deepmd_configs/water_se_e2_a.json \
        --num_atoms 192 \
        --tile_dataset_dir scripts/asplos/data/dataset/train \
        --result_dir results/deepmd/

转换工具 — 批量转换 input.json 为 NeuSight config:
    python -m neusight.Tracing.parse_deepmd_input /path/to/input.json -o output.json
"""

import argparse
import sys
import os

# 将项目根目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import neusight


def main():
    parser = argparse.ArgumentParser(
        description="Predict DeepMD-kit inference latency on GPU"
    )
    parser.add_argument(
        "--predictor_path",
        type=str,
        required=True,
        help="Path to NeuSight predictor directory (e.g. data/predictor/MLP_WAVE)",
    )
    parser.add_argument(
        "--device_config_path",
        type=str,
        required=True,
        help="Path to GPU device config JSON",
    )
    parser.add_argument(
        "--deepmd_config_path",
        type=str,
        required=True,
        help="Path to DeepMD config JSON — supports both DeepMD input.json and NeuSight config format",
    )
    parser.add_argument(
        "--num_atoms",
        type=int,
        required=True,
        help="Number of atoms in the system",
    )
    parser.add_argument(
        "--tile_dataset_dir",
        type=str,
        default="",
        help="Path to tile dataset directory (for meta table lookup)",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results/deepmd",
        help="Output directory for prediction results",
    )
    parser.add_argument(
        "--compute_force",
        action="store_true",
        help="Include force computation overhead (autograd backward)",
    )
    parser.add_argument(
        "--host_config_path",
        type=str,
        default=None,
        help="Path to host CPU config JSON (for cross-platform overhead scaling). "
             "Contains Kernel_Launch_Chain_us and/or CPU_SingleThread_Score. "
             "If not provided, uses H100 NVL reference values (cpu_scale=1.0).",
    )
    parser.add_argument(
        "--box_size",
        type=float,
        default=None,
        help="Simulation box side length in Angstrom. "
             "Used for density correction of unmodeled compute. "
             "When provided, low-density systems get lower predicted latency. "
             "If not provided, no density correction is applied.",
    )

    args = parser.parse_args()

    predictor = neusight.DeepMDPredictor(
        predictor_path=args.predictor_path,
        tile_dataset_dir=args.tile_dataset_dir,
    )

    result = predictor.predict(
        device_config_path=args.device_config_path,
        deepmd_config_path=args.deepmd_config_path,
        num_atoms=args.num_atoms,
        result_dir=args.result_dir,
        compute_force=args.compute_force,
        host_config_path=args.host_config_path,
        box_size=args.box_size,
    )


if __name__ == "__main__":
    main()
