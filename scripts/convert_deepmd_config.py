#!/usr/bin/env python3
"""
将 DeepMD-kit 的 input.json 转换为 NeuSight 预测器所需的配置文件。

无需 PyTorch 或 DeepMD-kit 依赖，仅需 Python 标准库。

用法:
    python scripts/convert_deepmd_config.py input.json -o output.json
    python scripts/convert_deepmd_config.py input.json  # 自动命名输出
"""

import sys
import os

# 直接导入解析模块，绕过 neusight.__init__ 中的 torch import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "parse_deepmd_input",
    os.path.join(os.path.dirname(__file__), "..", "neusight", "Tracing", "parse_deepmd_input.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
convert_deepmd_input = _mod.convert_deepmd_input
parse_deepmd_input = _mod.parse_deepmd_input

import argparse


def main():
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
        help="输出文件路径 (默认: 同目录下自动命名)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="仅打印提取结果，不保存文件",
    )
    args = parser.parse_args()

    if args.print_only:
        import json
        config = parse_deepmd_input(args.input_json)
        print(json.dumps(config, indent=4, ensure_ascii=False))
    else:
        convert_deepmd_input(args.input_json, args.output)


if __name__ == "__main__":
    main()
