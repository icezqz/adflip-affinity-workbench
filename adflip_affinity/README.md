# ADFLIP Affinity Workbench

这个目录是围绕原始 `ADFLIP` 工程搭建的实验工作台，目标是在尽量少改上游代码的前提下，完成抗体亲和力相关的三类工作：

- `affinity/`：单数据集 few-shot 训练、验证与独立评测。
- `affinity_all/`：多数据集联合训练。
- `zero_shot/`：只使用预训练 backbone 的 zero-shot NLL 基线。

## 仓库定位

这里更适合作为 `ADFLIP` 的实验封装层，而不是一个完全独立的新模型仓库。它主要补齐了以下内容：

- 将 `../data` 与 `../structure` 下的原始输入整理成统一 CSV 和 manifest。
- 为单数据集训练、联合训练、zero-shot 打分提供独立脚本与默认配置。
- 保留上游模型加载方式，便于复现实验并继续扩展数据集。

## 项目结构

| 路径 | 作用 |
| --- | --- |
| `affinity/` | 单数据集数据整理、few-shot 训练、独立 checkpoint 评测 |
| `affinity_all/` | 多数据集联合训练与跨数据集比较 |
| `zero_shot/` | 预训练 backbone 的 zero-shot 打分、汇总与报告 |

## 依赖关系

运行前通常需要准备：

- 原始 `ADFLIP` 工程目录。
- 预训练权重 `ADFLIP_ICML_camera_ready.pt`。
- `../data/` 中的数据表。
- `../structure/` 中的结构文件。

当前保留了你本机可直接运行的 `configs/base.yaml`，同时新增了 `configs/base.example.yaml` 作为公开仓库模板。准备上传 GitHub 时，建议：

- 保留 `base.example.yaml` 作为示例配置。
- 将 `base.yaml` 视为本地配置，按需改成你自己的路径。
- 不直接提交 `outputs/`、checkpoint 和大体积结构文件。

## 推荐顺序

1. 在 `affinity/` 中完成原始数据整理并生成 `manifest.csv`。
2. 先跑单数据集训练，确认流程和指标稳定。
3. 再用 `affinity_all/` 做联合训练。
4. 最后用 `zero_shot/` 生成不依赖 affinity head 的基线。

## 快速开始

建议从 `adflip_affinity/` 根目录进入各子项目执行：

```bash
cd affinity
bash scripts/prepare.sh
bash scripts/run_all.sh configs/base.yaml 0
```

```bash
cd ../affinity_all
bash scripts/prepare.sh
bash scripts/train.sh configs/base.yaml 0
```

```bash
cd ../zero_shot
bash scripts/score.sh configs/base.yaml "" 0
bash scripts/score_all_tests.sh configs/base.yaml 0
bash scripts/summarize_all_tests.sh
```

## 保留的发布产物

本次整理后，筛选出来准备长期保留的结果统一放在：

- `artifacts/single_dataset/`
- `artifacts/joint/`
- `artifacts/zero_shot/`

这些目录只保留最关键的 checkpoint、指标摘要和图表；常规训练输出仍然会写到各子项目自己的 `outputs/` 下，但默认视为可再生中间产物。

## GitHub 整理建议

- 将实验输出、日志、缓存和结构文件放在 Git 跟踪之外。
- 在 README 中明确说明 `data/` 与 `structure/` 的来源，而不是直接把大文件入库。
- 公开仓库时优先让示例命令使用相对路径，避免写死 `/home/wyy/...`。

## 进一步说明

每个子项目的输入格式、输出文件与命令细节见：

- `affinity/README.md`
- `affinity_all/README.md`
- `zero_shot/README.md`
