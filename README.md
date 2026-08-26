# RobustToM-RL

面向高阶 Theory of Mind（ToM）的反事实数据构造、结构化过程监督与 GRPO 训练项目。
本项目以 `Qwen2.5-3B-Instruct` 为基础模型，通过 RFT 和 GRPO 学习显式的嵌套信念链，
并使用反事实观测干预降低 last-mention、world-state 等 shortcut。

## 方法概览

训练流程分为三个阶段：

1. **Counterfactual data**：基于 Hi-ToM 和 ExploreToM 风格构造
   `observed` / `hidden` 成对样本，尽量避免shortcut数据出现。
2. **RFT**：对每个训练 prompt 采样 `K=16` 个候选，只保留严格 JSON、正常 EOS 且
   process reward 为 `1.0` 的完整轨迹，进行 response-only fine-tuning。
3. **GRPO**：从 RFT checkpoint 继续训练，每个 prompt 使用 16 个 rollout，按照结构化
   process reward 优化完整 belief trace 和最终答案。

模型输出格式如下：

```json
{
  "tom_order": 4,
  "belief_chain": ["William", "Alice", "Phoebe", "Jacob"],
  "object": "passport",
  "reasoning_mode": "nested_belief",
  "belief_trace": [
    {"belief_chain": ["Jacob"], "location": "filing_cabinet"},
    {"belief_chain": ["Phoebe", "Jacob"], "location": "glass_case"},
    {"belief_chain": ["Alice", "Phoebe", "Jacob"], "location": "blue_canvas_bag"},
    {"belief_chain": ["William", "Alice", "Phoebe", "Jacob"], "location": "cedar_chest"}
  ],
  "answer": "cedar_chest"
}
```

## 数据

| 数据 | 数量 | ToM 阶数 | 用途 |
| --- | ---: | --- | --- |
| Symbolic counterfactual train | 3,200 | 1–3 | RFT / GRPO 训练 |
| Symbolic counterfactual dev | 400 | 1–3 | 训练阶数范围内评测 |
| Hi-ToM 4-order OOD | 600 | 4 | 未见阶数外推与反事实过程评测 |
| Hi-ToM benchmark | 600 | 4 | 最终答案 Accuracy |
| ExploreToM benchmark | 1,053 | 1–2 | 跨数据生成机制的最终答案 Accuracy |

Symbolic 数据中的 `observed` 和 `hidden` 样本按 pair 组织。训练集、dev 和四阶 OOD
分别包含 1,600、200 和 300 个 pair。固定 few-shot 只包含 1–3 阶示例。

主要数据目录：

- `data/counterfactual_process_reward_v3_fewshot/`：结构化 JSONL 与固定 3-shot。
- `data/grpo/counterfactual_process_reward_v3_fewshot/`：GRPO 使用的 parquet。
- `data/rft/derived_v3_fewshot/`：RFT 使用的固定 split。
- `data/rft/hitom_order4/`：Hi-ToM 四阶 answer-only benchmark。
- `data/ExploreToM/`：处理后的 ExploreToM benchmark。

## 最终评测结果

评测产物位于 [`runs/20260821-qwen25-3b-k16`](runs/20260821-qwen25-3b-k16)。
表中 `Base`、`RFT` 和 `GRPO` 分别表示基础模型、rejection-sampling fine-tuning
checkpoint，以及从 RFT checkpoint 继续训练得到的 GRPO checkpoint。

### Answer Accuracy

| 数据集 | 样本数 | Base | RFT | GRPO |
| --- | ---: | ---: | ---: | ---: |
| dev（1–3 阶） | 400 | 4.50% | 10.25% | **66.50%** |
| Hi-ToM 4-order OOD | 600 | 6.17% | 10.33% | **47.67%** |
| Hi-ToM benchmark | 600 | 36.33% | 38.00% | **44.50%** |
| ExploreToM benchmark | 1,053 | 38.22% | 41.31% | **46.06%** |

Hi-ToM 和 ExploreToM benchmark 只比较模型 JSON 中的最终 `answer` 与
`gold_answer`；不对官方数据未提供的中间 belief trace 计分。

### Process Metrics

| 数据 | 模型 | Mean reward | 完整 belief trace | Full reward | Pair accuracy | Parse rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dev | Base | 0.274 | 0.25% | 0.00% | 0.00% | 88.25% |
| dev | RFT | 0.327 | 2.25% | 1.50% | 1.50% | 95.00% |
| dev | GRPO | **0.812** | **64.50%** | **61.00%** | **40.50%** | **100.00%** |
| Hi-ToM 4-order OOD | Base | 0.227 | 0.00% | 0.00% | 0.00% | 95.00% |
| Hi-ToM 4-order OOD | RFT | 0.256 | 0.33% | 0.00% | 0.33% | 97.33% |
| Hi-ToM 4-order OOD | GRPO | **0.634** | **27.00%** | **15.83%** | **16.67%** | **99.00%** |

### Shortcut 与输出稳定性

| 数据 | 模型 | Shortcut copy | Last-mention copy | EOS rate | Length P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| dev | Base | 12.67% | 28.50% | 88.25% | 256 |
| dev | RFT | 9.00% | 14.50% | 95.00% | 255.05 |
| dev | GRPO | **0.00%** | **0.25%** | **100.00%** | **89** |
| Hi-ToM 4-order OOD | Base | 7.50% | 15.83% | 94.67% | 256 |
| Hi-ToM 4-order OOD | RFT | 7.50% | 7.83% | 97.33% | 199.15 |
| Hi-ToM 4-order OOD | GRPO | **0.00%** | **0.17%** | **99.00%** | **132** |

对应指标文件：

- [`base_eval`](runs/20260821-qwen25-3b-k16/base_eval)
- [`rft_eval`](runs/20260821-qwen25-3b-k16/rft_eval)
- [`grpo_eval`](runs/20260821-qwen25-3b-k16/grpo_eval)

## Process Reward

对于 `nested_belief` 输出，reward 由以下部分组成：

| 组件 | 权重 |
| --- | ---: |
| 严格 JSON 与 schema | 0.05 |
| `tom_order` | 0.05 |
| `belief_chain` | 0.10 |
| `object` | 0.05 |
| `belief_trace` | 0.55 |
| 最终 `answer` | 0.20 |

`belief_trace` 按正确 step 比例给分；`answer` 分只在完整 trace 正确且最终答案正确时发放。

## 环境安装

在项目根目录运行：

```bash
conda create -n robusttom-grpo python=3.10 -y
conda activate robusttom-grpo

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.0
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.7.0.post2 --no-build-isolation
python -m pip install -e . --no-deps
```

RFT 可以使用独立环境：

```bash
conda create -n robusttom-rft python=3.10 -y
conda activate robusttom-rft
python -m pip install -r rft/requirements.txt
python -m pip install flash-attn==2.6.3 --no-build-isolation
```

## RFT

### 1. 候选采样

```bash
RUN_ID=20260821-qwen25-3b-k16
MODEL=Qwen/Qwen2.5-3B-Instruct

CUDA_VISIBLE_DEVICES=0 python -m rft.sample \
  --data data/rft/derived_v3_fewshot/train.jsonl \
  --model "${MODEL}" \
  --output "runs/rft_sampling/${RUN_ID}/candidates.jsonl" \
  --num-samples 16 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 256 \
  --seed 2026
```

### 2. 筛选与训练

```bash
python -m rft.score_candidates \
  --candidates "runs/rft_sampling/${RUN_ID}/candidates.jsonl" \
  --data data/rft/derived_v3_fewshot/train.jsonl \
  --output "runs/rft_sampling/${RUN_ID}/scored.jsonl"

python -m rft.build_dataset \
  --scored "runs/rft_sampling/${RUN_ID}/scored.jsonl" \
  --output "data/rft/accepted/${RUN_ID}/train.jsonl" \
  --min-samples 0 \
  --max-samples 3000 \
  --seed 2026

CUDA_VISIBLE_DEVICES=0 python -m rft.train \
  --model "${MODEL}" \
  --train-file "data/rft/accepted/${RUN_ID}/train.jsonl" \
  --output-dir "runs/rft_train/${RUN_ID}" \
  --max-seq-length 2048 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --num-train-epochs 1 \
  --learning-rate 1e-5 \
  --seed 2026
```

详细说明见 [`rft/README.md`](rft/README.md)。

## GRPO

设置 RFT checkpoint 和运行目录：

```bash
export RFT_MODEL_PATH=/path/to/rft_checkpoint/final
export GRPO_RUN_DIR=/path/to/grpo_run
export GRPO_DATA_DIR=/path/to/grpo_data
export GRPO_LOG_DIR="${GRPO_RUN_DIR}/logs"
```

依次执行数据构建、配置验证、smoke test、pilot 和正式训练：

```bash
bash grpo/run_grpo_v3.sh build
bash grpo/run_grpo_v3.sh validate
bash grpo/run_grpo_v3.sh smoke
bash grpo/run_grpo_v3.sh pilot
bash grpo/run_grpo_v3.sh train
```

正式训练配置为：8 prompts/step、16 rollouts/prompt、800 optimizer steps、
2 epochs、learning rate `5e-7`、temperature `1.0`，PPO ratio clip 为 `[0.8, 1.3]`。

详细环境变量、恢复训练方式和 A800 运行流程见 [`grpo/README.md`](grpo/README.md)。

## 评测

### 过程指标

```bash
python -m rft.generate \
  --data data/rft/derived_v3_fewshot/dev.jsonl \
  --model /path/to/checkpoint \
  --output runs/eval/dev_predictions.jsonl \
  --max-new-tokens 256 \
  --seed 2026

python -m rft.evaluate \
  --predictions runs/eval/dev_predictions.jsonl \
  --data data/rft/derived_v3_fewshot/dev.jsonl \
  --output runs/eval/dev_metrics.json
```

### 最终答案 Accuracy

```bash
python -m rft.generate \
  --data data/rft/hitom_order4/test.jsonl \
  --model /path/to/checkpoint \
  --output runs/eval/hitom_predictions.jsonl \
  --max-new-tokens 384 \
  --seed 2026

python -m rft.evaluate \
  --predictions runs/eval/hitom_predictions.jsonl \
  --data data/rft/hitom_order4/test.jsonl \
  --output runs/eval/hitom_metrics.json \
  --answer-only
```

## 项目结构

```text
RobustToM-RL/
├── data/       # 原始、反事实、RFT、GRPO 与 benchmark 数据
├── grpo/       # verl GRPO 适配、配置、reward 与运行脚本
├── rft/        # 候选采样、筛选、response-only 训练与评测
├── scripts/    # 数据生成、few-shot 和 process-target 工具
├── runs/       # 训练与评测产物
└── tests/      # 项目测试
```

## 数据来源与依赖

- [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
- [Hi-ToM](https://github.com/ying-hui-he/Hi-ToM_dataset)
- [ExploreToM](https://github.com/facebookresearch/ExploreToM)
- [verl](https://github.com/volcengine/verl)

项目代码许可证见 [`LICENSE`](LICENSE)。外部数据集继续受各自许可证约束；
ExploreToM 官方样本使用 CC BY-NC 4.0。
