# Psychiatric RAG Eval Engine

**Healthcare-grade clinical RAG system for mental health guidelines, with multi-layered evaluation, observability, and prompt management.**  

[![CI](https://github.com/julcambec/clinical-rag-eval-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/julcambec/clinical-rag-eval-engine/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<div align="center">

🚧 **Under construction - Big things are brewing.** 🚧

</div>

---

## Why This Service Exists

GenAI teams often build RAG chatbots. Few focus on ensuring they're safe to use in healthcare. Clinical AI hallucinations are dangerous.

This repo builds the **evaluation and observability infrastructure** that supports a deployable, reliable clinical AI service. It uses real clinical practice guidelines as its knowledge base, retrieves with hybrid dense+sparse search, generates answers with citations, and then subjects every response to a multi-layered evaluation gauntlet: standard RAGAS metrics, a custom clinical-safety scorer, and automated regression testing across prompt versions.

---

## Key Features

- **Hybrid Retrieval**: Dense embedding search (ChromaDB) + sparse keyword search (BM25) with reciprocal rank fusion
- **Clinical Faithfulness Scoring**: Custom LLM-as-judge with a domain-specific rubric (clinical accuracy, scope appropriateness, hedging quality, harmful omission risk)
- **RAGAS Evaluation Baselines**: Context precision, context recall, faithfulness, and answer relevancy on a curated gold dataset
- **Citation Accuracy Verification**: Automated checking that generated citations match retrieved source passages
- **Prompt-Version Regression Testing**: Measurable comparison across versioned prompts with documented rationale
- **Langfuse Observability**: Full LLM call tracing with cost, latency, and token breakdowns
- **MLflow Experiment Tracking**: Every eval run logged with metrics, parameters, and artifacts
- **Config-Driven Architecture**: YAML configuration with Pydantic validation throughout; no magic numbers
- **Docker Compose Deployment**: One-command stack: app + ChromaDB + Langfuse

---

## Quick Start

```bash
# Clone and enter
git clone https://github.com/julcambec/clinical-rag-eval-engine.git
cd clinical-rag-eval-engine

# Set up environment
python -m venv .venv
source .venv/Scripts/activate    # if using Git Bash for Windows
# source .venv/bin/activate      # macOS / Linux
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run
make ingest      # Ingest clinical guidelines into vector store
make serve       # Start the FastAPI service
make eval        # Run the evaluation suite
make dashboard   # Launch the eval results dashboard
```

---

## Tech Stack

| Layer | Tool | Role |
|---|---|---|
| Orchestration | LangChain | RAG chain, document loading, retrieval |
| Vector store | ChromaDB | Dense embedding storage + retrieval |
| Sparse retrieval | rank_bm25 | BM25 keyword search |
| Embeddings | *Under evaluation:* OpenAI vs. open source. | Document + query embedding |
| Generation | *Under evaluation:* OpenAI vs. open source. | Answer generation |
| Eval judge | *Under evaluation:* OpenAI vs. open source. | Clinical faithfulness scoring |
| Eval framework | RAGAS + custom modules | Standard + domain-specific evaluation |
| Observability | Langfuse | LLM tracing, cost/latency monitoring |
| Experiment tracking | MLflow | Eval run logging |
| API | FastAPI | Service endpoints |
| Dashboard | Streamlit | Eval results visualization |
| Config | YAML + Pydantic | Typed, validated configuration |
| Containerization | Docker Compose | Full-stack local deployment |
| CI | GitHub Actions | Lint, type-check, test |

---

## Knowledge Base

The system uses three publicly available clinical practice guidelines focused on mental health:

1. **WHO mhGAP Intervention Guide**: Global mental health clinical guidance
2. **CANMAT 2016 Guidelines for Major Depressive Disorder**: Canadian clinical practice guidelines
3. **NICE Depression in Adults (NG222)**: UK national clinical guidance

This GenAI service extends previous work on brain-based psychiatric risk subtyping (see [brain-risk-hybrid-ML-engine](https://github.com/julcambec/brain-risk-hybrid-ML-engine)).

---

## Project Structure

```
clinical-rag-eval-engine/
├── config/                     # YAML configuration files
├── data/
│   ├── guidelines/             # Clinical guideline PDFs
│   └── eval/                   # Gold eval dataset + results
├── prompts/                    # Versioned prompt templates
├── src/clinical_rag/
│   ├── ingest/                 # PDF loading, chunking, indexing
│   ├── retrieval/              # Dense, sparse, and hybrid retrieval
│   ├── generation/             # RAG chain, prompt loading, citations
│   ├── eval/                   # RAGAS, clinical judge, citation checker
│   ├── ops/                    # Langfuse tracing, logging, cost tracking
│   ├── api/                    # FastAPI service
│   └── dashboard/              # Streamlit eval dashboard
├── tests/                      # Unit + integration tests
├── notebooks/                  # Data exploration + eval analysis
├── docs/                       # Architecture, eval methodology, figures
├── docker-compose.yml          # Full-stack deployment
└── Makefile                    # CLI targets for all workflows
```

---

## License

[MIT](LICENSE)
