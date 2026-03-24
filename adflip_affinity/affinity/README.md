# affinity

`affinity/` 是单数据集亲和力建模环境，负责把原始数据整理成 ADFLIP 可读格式，并在单个数据集上完成训练、验证和独立评测。

## 适用场景

- 为单个数据集生成标准化的 `train/val/test` 划分。
- 在保持原始 ADFLIP 主体结构的前提下训练 affinity head。
- 做 few-shot 子采样实验。
- 用指定 checkpoint 在当前数据集或外部数据集上做独立评测。

## 工作流

1. 从 `../../data/<dataset>/` 读取原始 CSV。
2. 到 `../../structure/<dataset>/` 匹配对应结构文件。
3. 生成标准化 CSV，并按照 8:1:1 划分 train/val/test。
4. 通过 manifest 选择数据集并训练。
5. 训练结束后对任意数据集单独评测。

## 目录说明

| 路径 | 作用 |
| --- | --- |
| `prepare_affinity_datasets.py` | 整理原始数据并生成划分与 manifest |
| `train_affinity_adflip_fewshot.py` | 单数据集训练主脚本 |
| `eval_affinity_adflip_fewshot.py` | 独立评测脚本 |
| `adflip_affinity_dataset.py` | 数据读取与 ADFLIP 输入构建 |
| `configs/base.yaml` | 当前本机可运行配置 |
| `configs/base.example.yaml` | 公开仓库建议使用的配置模板 |
| `scripts/prepare.sh` | 数据整理快捷脚本 |
| `scripts/run_all.sh` | 训练快捷脚本 |
| `scripts/eval.sh` | 评测快捷脚本 |
| `outputs/` | 默认输出目录，包含 prepared 数据、日志、checkpoint 与图表 |

## 输入要求

默认依赖以下目录：

- 原始数据：`../../data`
- 结构文件：`../../structure`
- 上游工程：自行准备的 `ADFLIP`
- 上游权重：`ADFLIP_ICML_camera_ready.pt`

原始表格至少应包含：

- 重链序列
- 轻链序列
- `negative log Kd` 或兼容列名
- 可选的样本名列

## 使用方式

在 `affinity/` 目录下执行：

### 1. 整理数据

```bash
bash scripts/prepare.sh
```

这一步会在 `outputs/prepared/` 下生成：

- `all.csv`
- `train.csv`
- `val.csv`
- `test.csv`
- `manifest.csv`

### 2. 训练单个数据集

先修改 `configs/base.yaml` 中的 `data.dataset_name`，再执行：

```bash
bash scripts/run_all.sh configs/base.yaml 0
```

常用配置项：

- `data.dataset_name`
- `training.list_size`
- `training.few_shot_enabled`
- `training.few_shot_frac`
- `training.few_shot_n_samples`
- `training.checkpoint_monitor`

### 3. 独立评测

```bash
bash scripts/eval.sh configs/base.yaml /path/to/best.ckpt /path/to/eval_dataset.csv 0
```

如果不显式传入 `eval_csv`，评测脚本会根据配置里的 manifest、数据集名和 split 自动推断测试集。

## 输出内容

- `outputs/prepared/`：数据整理结果和 manifest
- `outputs/checkpoints/`：训练得到的模型权重与评测结果
- `outputs/logs/`：日志、`metrics.csv`、`hparams.yaml`
- `outputs/figures/`：汇总图表

## 仓库内保留的筛选结果

为了公开仓库时更易读，最终保留的 3 个单数据集最佳模型被整理到了 `../artifacts/single_dataset/`。`outputs/` 仍然是脚本默认写入位置，但更适合作为重新运行后的生成物。

## GitHub 建议

- 公开仓库时优先保留 `configs/base.example.yaml` 作为示例。
- `configs/base.yaml` 更适合作为本地配置文件。
- 不建议直接提交 `outputs/` 下的大体积实验产物。
