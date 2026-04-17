"""
DeepMD-kit input.json 解析器

自动从 DeepMD-kit 训练配置文件 (input.json) 中提取模型架构参数，
生成 NeuSight DeepMD predictor 所需的配置格式。

支持的输入来源:
  1. DeepMD-kit 标准 input.json（训练配置）
  2. 已转换的 NeuSight deepmd config JSON
  3. 命令行参数覆盖

典型的 DeepMD input.json 结构:
  {
    "model": {
      "type_map": ["O", "H"],
      "descriptor": {
        "type": "se_e2_a",
        "sel": [46, 92],
        "rcut": 6.0,
        "rcut_smth": 0.5,
        "neuron": [25, 50, 100],
        "axis_neuron": 16
      },
      "fitting_net": {
        "neuron": [240, 240, 240],
        "activation_function": "tanh"
      }
    },
    "training": { ... },
    "learning_rate": { ... }
  }
"""

import json
import os


# DeepMD 各描述子类型的默认参数
_DESCRIPTOR_DEFAULTS = {
    "se_e2_a": {
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "activation_function": "tanh",
        "rcut": 6.0,
        "rcut_smth": 0.5,
    },
    "se_e2_r": {
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "activation_function": "tanh",
        "rcut": 6.0,
        "rcut_smth": 0.5,
    },
    "se_atten": {
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "activation_function": "tanh",
        "rcut": 6.0,
        "rcut_smth": 0.5,
        "attn_heads": 1,
    },
    "dpa1": {
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "activation_function": "tanh",
        "rcut": 6.0,
        "rcut_smth": 0.5,
        "attn_heads": 1,
    },
}

_FITTING_DEFAULTS = {
    "neuron": [240, 240, 240],
    "activation_function": "tanh",
}


def parse_deepmd_input(input_path):
    """
    从 DeepMD-kit 的 input.json（或已转换的 config）中提取模型配置。

    自动识别两种格式:
      - DeepMD 标准 input.json: 参数在 "model" 键下
      - NeuSight deepmd config: 参数在顶层

    Parameters
    ----------
    input_path : str
        JSON 文件路径

    Returns
    -------
    dict
        标准化后的配置，格式为:
        {
            "model_type": "se_e2_a",
            "type_map": ["O", "H"],
            "num_types": 2,
            "descriptor": { "type", "sel", "rcut", "neuron", "axis_neuron", ... },
            "fitting_net": { "neuron", "activation_function", ... },
        }
    """
    with open(input_path, "r") as f:
        raw = json.load(f)

    # ---- 判断格式：DeepMD input.json 还是已转换格式 ----
    if "model" in raw:
        # DeepMD 标准 input.json
        model_section = raw["model"]
    elif "descriptor" in raw and "fitting_net" in raw:
        # 已是 NeuSight deepmd config 格式
        model_section = raw
    else:
        raise ValueError(
            f"无法识别配置格式: {input_path}\n"
            f"期望包含 'model' 键 (DeepMD input.json) 或 "
            f"'descriptor' + 'fitting_net' 键 (NeuSight config)"
        )

    # ---- 提取 descriptor ----
    desc_raw = model_section.get("descriptor", {})
    desc_type = desc_raw.get("type", model_section.get("model_type", "se_e2_a"))

    # DPA-1 在新版 DeepMD 中可能表示为 "dpa1" 或 "se_atten"
    desc_type_normalized = desc_type.lower().replace("-", "").replace("_", "")
    if desc_type_normalized in ("dpa1", "seatten"):
        desc_type_key = "se_atten"
    else:
        desc_type_key = desc_type

    defaults = _DESCRIPTOR_DEFAULTS.get(desc_type_key, _DESCRIPTOR_DEFAULTS["se_e2_a"])

    descriptor = {
        "type": desc_type,
        "sel": desc_raw.get("sel", []),
        "rcut": desc_raw.get("rcut", defaults["rcut"]),
        "rcut_smth": desc_raw.get("rcut_smth", defaults["rcut_smth"]),
        "neuron": desc_raw.get("neuron", defaults["neuron"]),
        "axis_neuron": desc_raw.get("axis_neuron", defaults["axis_neuron"]),
        "activation_function": desc_raw.get(
            "activation_function", defaults["activation_function"]
        ),
    }

    # se_atten / DPA-1 特有参数
    if desc_type_key in ("se_atten", "dpa1"):
        descriptor["attn_heads"] = desc_raw.get(
            "attn_heads", defaults.get("attn_heads", 1)
        )

    # ---- 提取 fitting_net ----
    fit_raw = model_section.get("fitting_net", {})
    fitting_net = {
        "neuron": fit_raw.get("neuron", _FITTING_DEFAULTS["neuron"]),
        "activation_function": fit_raw.get(
            "activation_function", _FITTING_DEFAULTS["activation_function"]
        ),
    }

    # ---- 提取 type_map ----
    type_map = model_section.get("type_map", raw.get("type_map", []))

    # ---- 校验必要字段 ----
    if not descriptor["sel"]:
        raise ValueError(
            f"缺少 descriptor.sel 参数 (每种原子类型的最大邻居数)，"
            f"请在 {input_path} 中指定"
        )

    # ---- 组装标准配置 ----
    config = {
        "model_type": desc_type,
        "type_map": type_map,
        "num_types": len(type_map) if type_map else len(descriptor["sel"]),
        "descriptor": descriptor,
        "fitting_net": fitting_net,
    }

    return config


def convert_deepmd_input(input_path, output_path=None):
    """
    将 DeepMD input.json 转换为 NeuSight deepmd config 并保存。

    Parameters
    ----------
    input_path : str
        DeepMD input.json 路径
    output_path : str, optional
        输出路径。如果不指定，自动生成为同目录下
        {model_type}_{descriptor_type}.json

    Returns
    -------
    str
        保存的文件路径
    """
    config = parse_deepmd_input(input_path)

    if output_path is None:
        dirname = os.path.dirname(input_path)
        model_type = config["model_type"]
        name = f"neusight_config_{model_type}.json"
        output_path = os.path.join(dirname, name)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"已转换: {input_path} -> {output_path}")
    _print_config_summary(config)

    return output_path


def _print_config_summary(config):
    """打印配置摘要"""
    desc = config["descriptor"]
    fit = config["fitting_net"]

    print(f"  模型类型:     {config['model_type']}")
    print(f"  原子类型:     {config['type_map']} ({config['num_types']} 种)")
    print(f"  邻居选择 sel: {desc['sel']} (总计 {sum(desc['sel'])} 邻居/atom)")
    print(f"  截断半径:     {desc['rcut']} Å")
    print(f"  Embedding 网络: {desc['neuron']}")
    print(f"  axis_neuron:  {desc['axis_neuron']}")
    print(f"  Fitting 网络:  {fit['neuron']}")
    print(f"  Descriptor dim: {4 * desc['axis_neuron']} = 4 × {desc['axis_neuron']}")


# ---- CLI 入口 ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="将 DeepMD-kit input.json 转换为 NeuSight 预测器配置"
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="DeepMD-kit 的 input.json 文件路径",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="输出配置文件路径 (默认: 同目录下自动命名)",
    )
    args = parser.parse_args()

    convert_deepmd_input(args.input_json, args.output)
