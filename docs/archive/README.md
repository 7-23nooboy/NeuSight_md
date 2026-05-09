# 归档说明

本目录里的 5 份报告是 v1-v4 时期 (2026-03 ~ 2026-04) 的中间产物，已被根目录的
[`DeepMD_Experiment_Report.md`](../../DeepMD_Experiment_Report.md) (v6 + P0a/P0b/P2/P1, 2026-05-09)
完全覆盖、并补全了所有验证表。这里仅作历史归档。

| 文件 | 时期 | 内容 |
|:---|:---:|:---|
| `NeuSight_代码解读与DeepMD改造建议.md` | v1 (3-31) | 最初的 NeuSight 源码解读 + 改造路线图 |
| `TODO_DeepMD_Predictor.md` | v4 (4-3) | v4 时期的开发 TODO + power law 拟合结果 |
| `Overhead_Modeling_Deep_Dive.md` | v4 (4-20) | v4 overhead model 物理动机 deep dive |
| `NeuSight_DeepMD_Technical_Report.md` | v4 (4-21) | v4 完整技术报告 (descriptor 解析 + tracer + overhead model) |
| `DeepMD_Performance_Prediction_Technical_Report.md` | v4 (4-21) | 上一份的精简稿 |

**当前唯一活跃的报告**：根目录的 `DeepMD_Experiment_Report.md`，包含：

- §1-6: 前 5 个体系的实验设计 + v6 模型推导
- §7: HE6 / DPA-1 stress test
- §8: fix / cmp / roof 三段拆解 + 责任划分
- §9: P0a (roofline LSQ) + P0b (transition bubble) + P2 (type_one_side 自适应) + P1 (MLP_WAVE retrain scaffolding) 的实装与最终验证表
