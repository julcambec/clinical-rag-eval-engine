"""Tests for the configuration system."""

from clinical_rag.config import (
    PROJECT_ROOT,
    Settings,
    load_settings,
)


class TestProjectRoot:
    """Verify project root detection works."""

    def test_project_root_contains_pyproject(self):
        """The detected project root must contain pyproject.toml."""
        assert (PROJECT_ROOT / "pyproject.toml").exists()

    def test_config_dir_exists(self):
        """The config directory must exist."""
        assert (PROJECT_ROOT / "config").is_dir()


class TestLoadSettings:
    """Verify that settings load correctly from YAML files."""

    def test_load_settings_returns_settings_object(self):
        """load_settings() should return a Settings instance."""
        settings = load_settings()
        assert isinstance(settings, Settings)

    def test_retrieval_defaults(self):
        """Retrieval config should reflect retrieval.yaml values."""
        settings = load_settings()
        assert settings.retrieval.chunking.chunk_size == 1000
        assert settings.retrieval.chunking.chunk_overlap == 200
        assert settings.retrieval.dense.embedding_model == "text-embedding-3-small"
        assert settings.retrieval.fusion.strategy == "rrf"

    def test_generation_defaults(self):
        """Generation config should reflect generation.yaml values."""
        settings = load_settings()
        assert settings.generation.llm.model == "gpt-4o-mini"
        assert settings.generation.llm.temperature == 0.1
        assert settings.generation.prompt.active_version == "v1_baseline"

    def test_eval_config_has_ragas_metrics(self):
        """Eval config should list the four RAGAS metrics."""
        settings = load_settings()
        assert len(settings.eval.ragas.metrics) == 4
        assert "faithfulness" in settings.eval.ragas.metrics

    def test_eval_config_has_clinical_rubric(self):
        """Eval config should define clinical judge rubric dimensions."""
        settings = load_settings()
        dims = settings.eval.clinical_judge.rubric_dimensions
        assert len(dims) == 4
        names = [d.name for d in dims]
        assert "clinical_accuracy" in names
        assert "harmful_omission_risk" in names

    def test_service_defaults(self):
        """Service config should reflect service.yaml values."""
        settings = load_settings()
        assert settings.service.api.port == 8000
        assert settings.service.langfuse.enabled is True
        assert settings.service.ollama.enabled is False


class TestSettingsDefaults:
    """Verify Pydantic defaults work when values are missing."""

    def test_settings_with_empty_dicts(self):
        """Creating Settings with no arguments should use all defaults."""
        settings = Settings()
        assert settings.retrieval.chunking.chunk_size == 1000
        assert settings.generation.llm.model == "gpt-4o-mini"
        assert settings.service.api.port == 8000
