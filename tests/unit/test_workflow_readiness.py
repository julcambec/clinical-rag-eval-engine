from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from clinical_rag.ops.readiness import not_implemented_message


ROOT = Path(__file__).resolve().parents[2]


def test_not_implemented_message_names_target_and_command() -> None:
    message = not_implemented_message("eval", "python -m clinical_rag.eval.runner")
    assert "make eval" in message
    assert "python -m clinical_rag.eval.runner" in message
    assert "exits non-zero" in message


def test_advertised_placeholder_targets_fail_loudly() -> None:
    for target in ("serve", "eval", "eval-offline", "dashboard"):
        result = subprocess.run(
            ["make", f"PYTHON={sys.executable}", target],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert f"NOT IMPLEMENTED: make {target}" in result.stderr
        assert "Intended command:" in result.stderr
