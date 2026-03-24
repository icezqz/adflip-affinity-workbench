# Antibody Affinity Workbench

这是一个围绕 `ADFLIP` 搭建的抗体亲和力复现实验仓库，主要包含：

- 单数据集 affinity 训练与评测
- 多数据集联合训练
- 基于预训练 backbone 的 zero-shot 基线
- 已筛选出的最佳模型与对比结果

当前仓库已经整理为一个可直接上传到 GitHub 的普通代码仓库；其中 `ADFLIP/` 已作为普通目录并入，不再带独立 Git 元数据。

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `adflip_affinity/` | 本项目主目录，包含训练、评测、zero-shot 与整理后的发布产物 |
| `adflip_affinity/affinity/` | 单数据集 few-shot 训练、评测与数据整理 |
| `adflip_affinity/affinity_all/` | 多数据集联合训练与跨数据集评测 |
| `adflip_affinity/zero_shot/` | 预训练 ADFLIP backbone 的 zero-shot NLL 基线 |
| `adflip_affinity/artifacts/` | 清理后保留的最佳模型、指标摘要和图表 |
| `ADFLIP/` | 上游 ADFLIP 代码副本，作为本仓库依赖一并保留 |
| `data/` | 数据表与中间整理输入 |
| `structure/` | 本地结构文件目录，默认不建议公开上传 |

## 当前保留的核心结果

为避免仓库中充满重复 `runs/` 和临时 checkpoint，当前仅保留：

- 1 个多数据集联合训练最佳模型
- 3 个单数据集最佳模型
- 1 份 zero-shot 汇总结果

它们都集中放在 `adflip_affinity/artifacts/`，具体说明见 `adflip_affinity/artifacts/README.md`。

## 快速开始

### 1. 准备环境

先按 `ADFLIP/README.md` 安装上游依赖，再根据需要补充本项目的数据与结构文件。

### 2. 单数据集训练

```bash
cd adflip_affinity/affinity
bash scripts/prepare.sh
bash scripts/run_all.sh configs/base.yaml 0
```

### 3. 联合训练

```bash
cd ../affinity_all
bash scripts/prepare.sh
bash scripts/train.sh configs/base.yaml 0
```

### 4. Zero-shot 基线

```bash
cd ../zero_shot
bash scripts/score_all_tests.sh configs/base.yaml 0
bash scripts/summarize_all_tests.sh
```

## 复现所需资源

运行本仓库通常需要：

- `ADFLIP` 预训练权重 `ADFLIP_ICML_camera_ready.pt`
- `data/` 下的数据表
- `structure/` 下的结构文件

这些路径在本地配置文件 `configs/base.yaml` 中指定；公开复现时，建议优先参考各子目录下的 `base.example.yaml`。

## 发布说明

- `outputs/` 默认视为可再生中间产物，因此已从仓库主内容中清理。
- `structure/` 体积较大，且通常依赖外部来源，建议只在 README 中说明获取方式。
- `ADFLIP/` 现在是本仓库中的普通目录，可以与其它代码一起提交到同一个 GitHub 仓库。

## 子模块文档

- `adflip_affinity/README.md`
- `adflip_affinity/affinity/README.md`
- `adflip_affinity/affinity_all/README.md`
- `adflip_affinity/zero_shot/README.md`
