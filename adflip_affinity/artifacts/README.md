# Curated Artifacts

这个目录只保留准备随仓库长期发布的核心结果，避免把大量临时 `runs/`、日志和重复 checkpoint 一起带进 GitHub。

## 选择原则

- 单数据集模型：按各自训练日志中的最佳 `val_spearman` 保留 1 个 checkpoint。
- 联合训练模型：按 `macro_val_spearman` 保留 1 个最佳 run。
- zero-shot：只保留最终汇总结果，不保留逐数据集的中间打分文件。

## 保留内容

### 单数据集最佳模型

| 数据集 | 目录 | 最佳验证指标 | 对应测试 Spearman |
| --- | --- | ---: | ---: |
| `Shanehsazzadeh2023_trastuzumab_zero_kd` | `single_dataset/Shanehsazzadeh2023_trastuzumab_zero_kd/` | `0.369048` | `-0.063492` |
| `Warszawski2019_d44_Kd` | `single_dataset/Warszawski2019_d44_Kd/` | `0.083505` | `0.221047` |
| `Koenig2017_g6_Kd` | `single_dataset/Koenig2017_g6_Kd/` | `0.170522` | `0.236976` |

每个目录中包含：

- `best.ckpt`
- `metrics.csv`
- `run_meta.json`
- `training_curves.png`
- 与该 checkpoint 相关的 `eval_*.json`

### 联合训练最佳模型

保留目录：`joint/SWK_joint_20260323_134723/`

- 最佳 `macro_val_spearman`: `0.315204`
- 关键文件：
- `best_macro_spearman.ckpt`
- `config.yaml`
- `metrics.csv`
- `run_meta.json`
- `training_curves.png`
- 各数据集评测摘要 `best_macro_spearman+*.json`
- 与 zero-shot 对比的 `comparison/` 图表和汇总表

### Zero-shot 汇总

保留目录：`zero_shot/summary/`

- `all_tests_summary.csv`
- `all_tests_ranked.csv`
- `all_tests_report.md`
- `all_tests_dashboard.png`

## 说明

- 这些结果是发布时的人类可读产物。
- 重新训练或重新评测时，脚本仍然默认写入各模块自己的 `outputs/` 目录。
- `not_shipped` 表示原始中间文件在本次清理中未随仓库保留。
