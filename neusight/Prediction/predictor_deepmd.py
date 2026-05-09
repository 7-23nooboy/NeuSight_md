"""
DeepMD-kit 推理性能预测器

复用 NeuSight 的 OperatorPredictor 后端（MLP_WAVE），
前端使用 trace_deepmd.build_deepmd_opgraph() 生成算子图。

支持直接传入 DeepMD-kit 的 input.json，自动提取模型参数。
"""

from pathlib import Path
import pandas as pd
import json
import ast
import numpy as np
import os

from .predictor import OperatorPredictor, dump_df
from .aggregator import aggregate_deepmd
from .overhead_model import DeepMDOverheadModel, load_calibration
from ..Tracing.trace_deepmd import build_deepmd_opgraph
from ..Tracing.parse_deepmd_input import parse_deepmd_input


class DeepMDPredictor:
    """
    DeepMD-kit 推理 latency 预测器

    用法与 NeusightPredictor 平行，但:
    - 不调用 trace_graph() / parse_trace()
    - 改用 build_deepmd_opgraph() 解析式生成算子图
    - 聚合使用简单求和（无 Transformer 层复制）
    """

    def __init__(self, predictor_path, tile_dataset_dir, calibration_path=None):
        if tile_dataset_dir != "":
            tile_dataset_dir = Path(tile_dataset_dir)

        self.predictor = OperatorPredictor(
            predictor_path=predictor_path,
            tile_dataset_dir=tile_dataset_dir,
        )

        # Overhead 模型 — 预测 kernel launch / CPU dispatch / autograd 开销
        # 优先级: 显式传入 calibration_path > NEUSIGHT_DEEPMD_CALIBRATION 环境变量
        #         > load_calibration() 默认搜索 (kernel_launch_cost.json)
        if calibration_path is None:
            import os
            calibration_path = os.environ.get("NEUSIGHT_DEEPMD_CALIBRATION")
        if calibration_path is None:
            calibration_path = load_calibration("")
        self.overhead_model = DeepMDOverheadModel(calibration_path=calibration_path)

    def predict(
        self,
        device_config_path,
        deepmd_config_path,
        num_atoms,
        result_dir,
        compute_force=False,
        host_config_path=None,
        box_size=None,
    ):
        """
        预测 DeepMD-kit 模型在指定 GPU 上的推理 latency

        Parameters
        ----------
        device_config_path : str
            GPU 配置文件路径 (JSON)
        deepmd_config_path : str
            DeepMD 模型配置文件路径，支持两种格式:
            - DeepMD-kit 训练用的 input.json（自动从 model.descriptor 等提取）
            - 已转换的 NeuSight deepmd config JSON
        num_atoms : int
            原子数量
        result_dir : str
            结果输出目录
        compute_force : bool
            是否包含 force 计算开销
        host_config_path : str, optional
            主机 CPU 配置文件路径 (JSON)。包含 Kernel_Launch_Chain_us
            和/或 CPU_SingleThread_Score，用于跨平台 overhead 缩放。
            如果为 None，使用 H100 NVL 参考值 (cpu_scale=1.0)。
        box_size : float, optional
            模拟 box 边长 (Å)。用于密度修正 power law。
            密度低于校准条件时降低 unmodeled compute 预测。
            如果为 None，不做密度修正。
        """
        result_dir = Path(result_dir)

        # ---- 1. 读取 GPU 配置 ----
        device_config_path = Path(device_config_path).absolute()
        with open(device_config_path, "r") as f:
            device_config = json.load(f)

        # ---- 2. 读取 DeepMD 模型配置（自动识别 input.json 或已转换格式）----
        deepmd_config_path = Path(deepmd_config_path).absolute()
        deepmd_config = parse_deepmd_input(str(deepmd_config_path))

        model_type = deepmd_config.get(
            "model_type",
            deepmd_config.get("descriptor", {}).get("type", "se_e2_a"),
        )
        model_tag = f"deepmd_{model_type}_n{num_atoms}"
        if compute_force:
            model_tag += "_force"

        # ---- 3. 构建算子图 ----
        opgraph_path = result_dir / f"opgraph/{model_tag}.csv"
        if os.path.exists(opgraph_path):
            print("already exists:", os.path.realpath(opgraph_path))
            df = pd.read_csv(
                opgraph_path,
                converters={
                    "FwOps": ast.literal_eval,
                    "BwOps": ast.literal_eval,
                    "AccOps": ast.literal_eval,
                    "InputShapes": ast.literal_eval,
                    "OutputShape": ast.literal_eval,
                },
            )
        else:
            df = build_deepmd_opgraph(deepmd_config, num_atoms, compute_force)
            dump_df(df, opgraph_path)
            # 重新读取以确保类型一致（CSV 序列化/反序列化）
            df = pd.read_csv(
                opgraph_path,
                converters={
                    "FwOps": ast.literal_eval,
                    "BwOps": ast.literal_eval,
                    "AccOps": ast.literal_eval,
                    "InputShapes": ast.literal_eval,
                    "OutputShape": ast.literal_eval,
                },
            )

        # ---- 4. 逐节点预测 latency ----
        df[["fw_latency", "bw_latency", "acc_latency"]] = df.apply(
            lambda x: self.predictor.predict(device_config, x), axis=1
        )
        df["bwall_latency"] = df["bw_latency"] + df["acc_latency"]
        df["e2e_latency"] = (
            df["fw_latency"] + df["bw_latency"] + df["acc_latency"]
        )

        # ---- 5. 保存逐节点预测结果 ----
        pred_path = (
            result_dir
            / f"prediction/{device_config['Device'].replace(' ', '_')}/{model_tag}.csv"
        )
        pred_path = Path(pred_path)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(pred_path, index=False)

        # ---- 6. 聚合纯计算 latency ----
        e2e, fw, bw, bwall, acc = aggregate_deepmd(df)

        # ---- 7. 加载 host config (如果提供) ----
        host_config = None
        if host_config_path is not None:
            host_config_path = Path(host_config_path).absolute()
            if host_config_path.is_file():
                with open(host_config_path, "r") as f:
                    host_config = json.load(f)
                print(f"  Host config: {host_config.get('Host', 'unknown')} "
                      f"(chain_us={host_config.get('Kernel_Launch_Chain_us', 'N/A')}, "
                      f"cpu_score={host_config.get('CPU_SingleThread_Score', 'N/A')})")
            else:
                print(f"  Warning: host_config_path not found: {host_config_path}")

        # ---- 8. 估算 overhead (kernel launch + CPU dispatch + autograd) ----
        overhead = self.overhead_model.estimate(
            device_config=device_config,
            deepmd_config=deepmd_config,
            op_df=df,
            compute_force=compute_force,
            compute_latency_ms=e2e,
            host_config=host_config,
            box_size=box_size,
        )

        # ---- 9. 最终 latency = 计算 + overhead ----
        e2e_total = e2e + overhead["total_overhead_ms"]

        json_dict = {
            "model_type": model_type,
            "num_atoms": num_atoms,
            "compute_force": compute_force,
            "e2e_latency": float(round(e2e_total, 4)),
            "compute_latency": float(round(e2e, 4)),
            "fw_latency": float(fw),
            "bw_latency": float(bw),
            "bwall_latency": float(bwall),
            "acc_latency": float(acc),
            "overhead": {
                "total_ms": overhead["total_overhead_ms"],
                "kernel_launch_ms": overhead["kernel_launch_ms"],
                "cpu_dispatch_ms": overhead["cpu_dispatch_ms"],
                "fixed_overhead_ms": overhead.get("fixed_overhead_ms", overhead["cpu_dispatch_ms"]),
                "gpu_overhead_roofline_ms": overhead.get("unmodeled_compute_ms", 0.0),
                "autograd_ms": overhead["autograd_ms"],
                "cpu_scale": overhead.get("cpu_scale", 1.0),
                "gpu_scale": overhead.get("gpu_scale", 1.0),
                "density_correction": overhead.get("density_correction", 1.0),
            },
            "kernel_count": overhead["kernel_count"],
            "confidence": {
                "level": overhead.get("confidence", "high"),
                "regime": overhead.get("regime", "unknown"),
                "transition_ratio": overhead.get("transition_ratio", 0.0),
                "e2e_lower_ms": overhead.get("e2e_lower_ms", e2e_total),
                "e2e_upper_ms": overhead.get("e2e_upper_ms", e2e_total),
            },
        }

        with open(pred_path.with_suffix(".json"), "w") as f:
            json.dump(json_dict, f, indent=4)

        confidence_info = overhead.get("confidence", "high")
        confidence_suffix = ""
        if confidence_info == "low":
            lo = overhead.get("e2e_lower_ms", e2e_total)
            hi = overhead.get("e2e_upper_ms", e2e_total)
            confidence_suffix = f" ⚠️ transition zone [{lo:.2f}, {hi:.2f}]ms"

        print(
            f"DeepMD E2E latency for {model_tag} on "
            f"{device_config_path.stem}: {round(e2e_total, 4)} ms "
            f"(compute={round(e2e, 4)}, overhead={overhead['total_overhead_ms']})"
            f"{confidence_suffix}"
        )

        return json_dict
