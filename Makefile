.PHONY: help install lint format type-check test test-cov ingest serve eval eval-offline dashboard clean

PYTHON ?= python

# Default target: show available commands
help:
	@echo ""
	@echo "  clinical-rag-eval-engine"
	@echo "  ========================"
	@echo ""
	@echo "  Setup:"
	@echo "    make install        Install project + dev dependencies"
	@echo ""
	@echo "  Quality:"
	@echo "    make lint           Run ruff linter"
	@echo "    make format         Auto-format code with ruff"
	@echo "    make type-check     Run mypy type checker"
	@echo "    make test           Run unit tests"
	@echo "    make test-cov       Run tests with coverage report"
	@echo ""
	@echo "  Pipeline:"
	@echo "    make ingest         Ingest clinical guidelines into vector store"
	@echo "    make serve          Start the FastAPI service"
	@echo "    make eval           Run the evaluation suite"
	@echo "    make eval-offline   Run evaluation fully offline"
	@echo "    make dashboard      Launch the Streamlit eval dashboard"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean          Remove generated artifacts"
	@echo ""

install:
	pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

type-check:
	$(PYTHON) -m mypy src/

test:
	$(PYTHON) -m pytest tests/unit/ -v

test-cov:
	$(PYTHON) -m pytest tests/unit/ -v --cov=clinical_rag --cov-report=term-missing

# --- Pipeline targets (stubs for now; I still need to wire them up later) ---

ingest:
	$(PYTHON) -m clinical_rag.ingest.indexer

serve:
	PYTHONPATH=src $(PYTHON) -m clinical_rag.ops.readiness serve "uvicorn clinical_rag.api.app:app --host 0.0.0.0 --port 8000 --reload"

eval:
	PYTHONPATH=src $(PYTHON) -m clinical_rag.ops.readiness eval "python -m clinical_rag.eval.runner"

eval-offline:
	PYTHONPATH=src $(PYTHON) -m clinical_rag.ops.readiness eval-offline "python -m clinical_rag.eval.runner --offline"

dashboard:
	PYTHONPATH=src $(PYTHON) -m clinical_rag.ops.readiness dashboard "streamlit run src/clinical_rag/dashboard/streamlit_app.py"

clean:
	rm -rf data/chroma_db/*
	rm -rf data/bm25_index/*
	rm -rf data/eval/results/*
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf mlruns
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned generated artifacts."
