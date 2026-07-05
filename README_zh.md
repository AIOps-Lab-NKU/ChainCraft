[English](README.md) | **中文**

# ChainCraft

基于大语言模型的 AIOps 故障预防框架。它结合时间序列异常检测、因果图构建和 LLM 推理，自动分析系统风险指标并精准发现潜在故障。

> **说明：** 当前开源版本已移除数据采集模块（data collection module），用户无法通过代码直接拉取监控数据。项目在 `./data/collected_data/demo/` 下提供了 demo 数据和演示案例，可直接使用这些数据体验和测试完整的分析流程。

## 快速开始

### 环境要求

- Python 3.9+
- 兼容 OpenAI 的 LLM API 端点

### 安装

```bash
git clone <repository-url>
cd chaincraft
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 编辑 .env 文件，填入 LLM API 配置
```

关键环境变量：

- `OPENAI_API_BASE` — API 基础 URL
- `OPENAI_API_KEY` — API 密钥
- `LLM_MODEL` — LLM 模型名称
- `EMBEDDING_MODEL` — Embedding 模型名称

### 运行

```bash
python main.py
```

默认情况下，`main.py` 会按顺序执行全部四个工作流，使用 demo 案例。你可以在文件底部注释/取消注释对应的 `_timed_run(...)` 调用来启用或禁用单个工作流。

## 使用方法

### 工作流 1：批量分析历史案例

对历史案例执行完整的分析流水线——异常检测（Prophet）、指标分析（LLM）和因果分析（PCMCI）。

**适用场景：** 预处理历史故障案例，生成分析结果，为构建知识库提供基础。

```python
from main import batch_analyze_cases

batch_analyze_cases(['case1'], collect_data=False, enable_iteration=True)
```

### 工作流 2：批量推理预测案例

对新案例或正在运行的案例进行推理分析，识别潜在故障风险。

**适用场景：** 分析当前系统状态，检测新兴故障模式并生成推理结果。

```python
from main import batch_inference_cases

batch_inference_cases(['case2'], collect_data=False, enable_iteration=True)
```

### 工作流 3：构建历史案例库

将已分析案例的故障报告处理后存入 ChromaDB 向量知识库，供后续检索使用。

**适用场景：** 基于历史故障案例构建和扩展 RAG 知识库。需在工作流 1 之后运行。

```python
from main import batch_deal_fault_reports

batch_deal_fault_reports(['case1'])
```

### 工作流 4：批量预测案例

使用基于 RAG 的相似案例检索和传播链重排序来处理推理报告，生成风险预测。

**适用场景：** 将当前推理结果与历史知识库进行匹配，生成可操作的风险预测。需在工作流 2 和 3 之后运行。

```python
from main import batch_deal_inference_reports

batch_deal_inference_reports(['case2'], use_structure_rag=True, use_chain_rerank=True)
```

### 复用已有分析结果

每个分析步骤（异常检测、指标分析、因果分析）均可跳过，复用之前已计算的结果：

```python
from main import batch_analyze_cases

batch_analyze_cases(
    ['case1'],
    run_anomaly_detection=False,  # 复用 ANOMALY_DETECTION_READ_PATH 下的结果
    run_causal_analysis=False,    # 复用 ANALYSIS_READ_PATH 下的结果
)
```

## Demo 数据

Demo 数据位于 `./data/collected_data/demo/`，包含两个案例：

- `case1` — 历史故障案例，用于构建知识库
- `case2` — 预测案例，用于验证推理能力

每个案例包含时间序列指标数据（`all_metrics.csv`）和原始指标 CSV 文件，按 `{app}_{app_group}/metric/` 结构组织。

## 配置说明

所有配置通过环境变量管理，从 `.env` 文件或系统环境变量加载。完整配置项参见 `.env.example`。

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。
