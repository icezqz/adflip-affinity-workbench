# affinity_all

`affinity_all/` 是多数据集联合训练环境，用于把多个抗体亲和力数据集整理到统一格式后，在同一套 ADFLIP backbone + affinity head 上进行联合建模。

## 适用问题

- 多个来源的数据能否通过统一训练获得更稳健的排序能力。
- 跨数据集场景下应如何选择 checkpoint。

## 主要能力

- 将多个数据集整理为统一字段格式。
- 生成 `all.csv`、`train.csv`、`val.csv`、`test.csv` 和总 manifest。
- 将结构文件路径统一为 `pdb_abs`。
- 将标签统一为 `-log10(Kd[M])`。
- 训练时按数据集动态采样，并在数据集内部构建 listwise 单元。
- 默认按 `macro_spearman` 选择最佳 checkpoint。

## 默认数据集

- `Shanehsazzadeh2023_trastuzumab_zero_kd`
- `Warszawski2019_d44_Kd`
- `Koenig2017_g6_Kd`

## 目录说明

| 路径 | 作用 |
| --- | --- |
| `scripts/prepare_joint_training_data.py` | 联合训练数据整理主脚本 |
| `scripts/prepare.sh` | 数据整理入口 |
| `train_joint_affinity.py` | 联合训练主脚本 |
| `scripts/train.sh` | 训练入口 |
| `scripts/eval.sh` | 对联合训练模型做单 CSV 评测的快捷脚本 |
| `configs/base.yaml` | 当前本机可运行配置 |
| `configs/base.example.yaml` | 公开仓库建议使用的配置模板 |
| `data/` | 数据缓存与整理后的 CSV |
| `outputs/` | 训练结果、checkpoint、曲线和配置快照 |

## 训练策略

- 每个 epoch 运行固定 `steps_per_epoch` 个 step。
- 每个 step 先根据 `sampling_mode` 选取一个数据集。
- 再从该数据集中抽样并构建 listwise 单元。
- 验证阶段分别统计各数据集指标，再聚合为 `macro_spearman`。

支持的采样模式：

- `uniform`
- `inverse`
- `sqrt_inverse`

也可以通过 `dataset_probs` 手动覆盖采样概率。

## 使用方式

在 `affinity_all/` 目录下执行：

### 1. 整理联合训练数据

```bash
bash scripts/prepare.sh
```

如果只处理指定数据集：

```bash
bash scripts/prepare.sh \
  Shanehsazzadeh2023_trastuzumab_zero_kd \
  Warszawski2019_d44_Kd \
  Koenig2017_g6_Kd
```

脚本会优先读取 `data/<dataset>/all.csv` 作为缓存；如果缓存不存在，则回到 `../affinity/outputs/prepared/manifest.csv` 指向的数据源重新生成。

### 2. 启动训练

```bash
bash scripts/train.sh configs/base.yaml 0
```

### 3. 评测一个 checkpoint

```bash
bash scripts/eval.sh \
  outputs/runs/<run_name>/config.yaml \
  outputs/runs/<run_name>/checkpoints/best_macro_spearman.ckpt \
  data/prepared/Koenig2017_g6_Kd/test.csv \
  outputs/runs/<run_name>
```

## 关键配置项

- `data.datasets`
- `training.steps_per_epoch`
- `training.list_size`
- `training.sampling_mode`
- `training.dataset_probs`
- `training.output_root`

## 输出内容

每次训练默认输出到 `outputs/runs/<run_name>/`，其中包括：

- `config.yaml`
- `metrics.csv`
- `run_meta.json`
- `checkpoints/best_macro_spearman.ckpt`
- `training_curves.png`

## 仓库内保留的筛选结果

为了公开仓库时更易读，最终保留的联合训练最佳 run 已整理到 `../artifacts/joint/`。`outputs/runs/` 仍然是脚本默认输出位置，但默认视为可再生中间产物。

## GitHub 建议

- 保留 `configs/base.example.yaml` 作为示例配置。
- `data/prepared/` 和 `outputs/` 更适合作为生成物处理。
- 如果要公开仓库，建议在 README 中注明原始数据与结构文件的来源，而不是直接提交大文件。
