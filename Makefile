.PHONY: help install lint format type-check test test-cov ingest serve eval dashboard clean

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
	@echo "    make dashboard      Launch the Streamlit eval dashboard"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean          Remove generated artifacts"
	@echo ""

install:
	pip install -e ".[dev]"

lint:
	python -m ruff check src/ tests/

format:
	python -m ruff format src/ tests/
	python -m ruff check --fix src/ tests/

type-check:
	python -m mypy src/

test:
	python -m pytest tests/unit/ -v

test-cov:
	python -m pytest tests/unit/ -v --cov=clinical_rag --cov-report=term-missing

# --- Pipeline targets (stubs for now; I still need to wire them up later) ---

ingest:
	python -m clinical_rag.ingest.indexer

serve:
	@echo "TODO: Wire up FastAPI service"
	@echo "Will run: uvicorn clinical_rag.api.app:app --host 0.0.0.0 --port 8000 --reload"

eval:
	@echo "TODO: Wire up evaluation suite"
	@echo "Will run: python -m clinical_rag.eval.runner"

dashboard:
	@echo "TODO: Wire up Streamlit dashboard"
	@echo "Will run: streamlit run src/clinical_rag/dashboard/streamlit_app.py"

clean:
	rm -rf data/chroma_db/*
	rm -rf data/bm25_index/*
	rm -rf data/eval/results/*
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf mlruns
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned generated artifacts."
