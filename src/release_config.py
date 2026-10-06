"""Paths and environment overrides shared by the publication scripts.

All relative paths are resolved from the release folder, regardless of the
working directory used to launch a script.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    """Resolve a project-relative path without depending on the shell's cwd."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


OUTPUT_DIR = project_path(os.environ.get("TB_OUTPUT_DIR", "results"))


def selected_drug(default: str = "TBAJ587") -> str:
    drug = os.environ.get("TB_DRUG", default)
    if drug not in {"BDQ", "TBAJ587"}:
        raise ValueError("TB_DRUG must be BDQ or TBAJ587")
    return drug

