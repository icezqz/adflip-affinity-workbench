# zero_shot

`zero_shot/` 用于生成 ADFLIP backbone 的零样本基线分数。这里不加载 affinity head checkpoint，只使用预训练 denoiser 对样本做前向计算，并把结构-序列配对的困惑度转换成可比较的打分结果。

## 适用场景

- 为亲和力建模任务提供不依赖监督训练的 baseline。
- 快速比较不同数据集上预训练 backbone 的原生判别能力。
- 在启动 `affinity/` 或 `affinity_all/` 训练前先建立参考线。

## 评分逻辑

每个样本会输出：

- `nll`：平均 token negative log-likelihood
- `ppl`：`exp(nll)`
- `score`：`-nll`

随后还会和真实标签计算：

- `avg_nll`
- `avg_ppl`
- `spearman`
- `pearson`
- `top_hit_overall`

## 目录说明

| 路径 | 作用 |
| --- | --- |
| `score_zero_shot_nll.py` | 对单个数据集或单个 CSV 进行 zero-shot 打分 |
| `score_all_tests.py` | 从 manifest 读取所有数据集并批量执行打分 |
| `summarize_all_tests.py` | 汇总批量结果并生成表格、图和 Markdown 报告 |
| `configs/base.yaml` | 当前本机可运行配置 |
| `configs/base.example.yaml` | 公开仓库建议使用的配置模板 |
| `scripts/score.sh` | 单次打分快捷脚本 |
| `scripts/score_all_tests.sh` | 批量打分快捷脚本 |
| `scripts/summarize_all_tests.sh` | 汇总与可视化快捷脚本 |

## 使用方式

在 `zero_shot/` 目录下执行：

### 1. 对单个数据集打分

```bash
bash scripts/score.sh configs/base.yaml "" 0
```

第二个参数可以显式传入 `eval_csv`。如果留空，脚本会根据配置中的 `manifest + dataset_name + eval_split` 自动推断评测文件。

### 2. 批量跑所有数据集

```bash
bash scripts/score_all_tests.sh configs/base.yaml 0
```

该脚本会读取 manifest 中的所有数据集，并默认对每个数据集的评测 CSV 逐个执行打分。

### 3. 生成汇总结果

```bash
bash scripts/summarize_all_tests.sh
```

## 输出内容

单数据集打分默认写到 `outputs/scores/`：

- `*.csv`
- `*.metrics.json`
- `*.neglogkd_vs_nll.png`

批量汇总默认写到 `outputs/summary/`：

- `all_tests_summary.csv`
- `all_tests_ranked.csv`
- `all_tests_dashboard.png`
- `all_tests_report.md`

## GitHub 建议

- `all_tests_report.md` 很适合作为结果摘要展示。
- `outputs/` 建议作为生成物处理，不要默认全部提交。
- 公开仓库时优先保留 `configs/base.example.yaml`，本地机器再维护自己的 `base.yaml`。

## 仓库内保留的筛选结果

本次整理后，长期保留的 zero-shot 汇总结果被放到了 `../artifacts/zero_shot/summary/`，方便直接引用；`outputs/` 目录仍保留为重新打分时的默认输出位置。
