**English** | [中文](README_zh.md)

# ChainCraft

An LLM-based AIOps fault prevention framework. It combines time-series anomaly detection, causal graph construction, and LLM reasoning to automatically analyze system risk metrics and identify potential failures.

> **Note:** The open-source version has removed the data collection module. Users cannot pull monitoring data directly via code. Demo data and demonstration cases are provided under `./data/collected_data/demo/` — you can use them to experience and test the full analysis pipeline out of the box.

## Quick Start

### Prerequisites

- Python 3.9+
- An OpenAI-compatible LLM API endpoint

### Installation

```bash
git clone <repository-url>
cd chaincraft
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env and fill in your LLM API configuration
```

Key environment variables:

- `OPENAI_API_BASE` — API base URL
- `OPENAI_API_KEY` — API key
- `LLM_MODEL` — LLM model name
- `EMBEDDING_MODEL` — Embedding model name

### Run

```bash
python main.py
```

By default, `main.py` runs all four workflows sequentially using the demo cases. You can enable or disable individual workflows by commenting/uncommenting the corresponding `_timed_run(...)` calls at the bottom of the file.

## Usage

### Workflow 1: Batch Analyze Historical Cases

Runs the full analysis pipeline — anomaly detection (Prophet), metric analysis (LLM), and causal analysis (PCMCI) — on historical cases.

**Use case:** Preprocess historical fault cases to generate analysis results that serve as the foundation for building the knowledge base.

```python
from main import batch_analyze_cases

batch_analyze_cases(['case1'], collect_data=False, enable_iteration=True)
```

### Workflow 2: Batch Inference on Prediction Cases

Performs inference analysis on new or ongoing cases to identify potential fault risks.

**Use case:** Analyze current system states to detect emerging fault patterns and generate inference results.

```python
from main import batch_inference_cases

batch_inference_cases(['case2'], collect_data=False, enable_iteration=True)
```

### Workflow 3: Build Historical Case Database

Processes fault reports from analyzed cases and stores them into a ChromaDB vector knowledge base for future retrieval.

**Use case:** Build and expand the RAG knowledge base from historical fault cases. Run after Workflow 1.

```python
from main import batch_deal_fault_reports

batch_deal_fault_reports(['case1'])
```

### Workflow 4: Batch Predict Cases

Processes inference reports using RAG-based similar case retrieval and propagation chain reranking to produce risk predictions.

**Use case:** Generate actionable risk predictions by matching current inference results against the historical knowledge base. Run after Workflows 2 and 3.

```python
from main import batch_deal_inference_reports

batch_deal_inference_reports(['case2'], use_structure_rag=True, use_chain_rerank=True)
```

### Reusing Existing Results

Each analysis step (anomaly detection, metric analysis, causal analysis) can be skipped to reuse previously computed results:

```python
from main import batch_analyze_cases

batch_analyze_cases(
    ['case1'],
    run_anomaly_detection=False,  # reuse from ANOMALY_DETECTION_READ_PATH
    run_causal_analysis=False,    # reuse from ANALYSIS_READ_PATH
)
```

## Demo Data

Demo data is located in `./data/collected_data/demo/` and includes two cases:

- `case1` — A historical fault case for knowledge base building
- `case2` — A prediction case for inference validation

Each case contains time-series metrics (`all_metrics.csv`) and raw metric CSVs organized under `{app}_{app_group}/metric/`.

## Configuration

All configuration is managed through environment variables loaded from `.env` or system environment. See `.env.example` for the full list of available options.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
