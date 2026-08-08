"""Startup-time configuration failure handling (backend/app/main.py).

Settings() (and therefore importing backend.app.main) must run in a fresh
process for these tests, since Python caches module imports -- once
backend.app.main has been imported once in this test session by other
tests/fixtures, re-importing it here would be a no-op.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPORT_MAIN_SNIPPET = "from backend.app.main import app\nprint('IMPORTED_OK')"


def _run_in_clean_subprocess(tmp_path, env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "GEMINI_API_KEY"
    }
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", IMPORT_MAIN_SNIPPET],
        cwd=tmp_path,  # empty dir: no .env file to accidentally pick up
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_missing_gemini_api_key_fails_fast_with_clear_message(tmp_path):
    result = _run_in_clean_subprocess(tmp_path, {})

    assert result.returncode == 1
    assert "IMPORTED_OK" not in result.stdout
    assert "FATAL: missing required environment variable(s): GEMINI_API_KEY" in result.stderr
    # No raw pydantic traceback should reach the operator.
    assert "Traceback" not in result.stderr


def test_blank_gemini_api_key_fails_fast_without_raw_traceback(tmp_path):
    result = _run_in_clean_subprocess(tmp_path, {"GEMINI_API_KEY": ""})

    assert result.returncode == 1
    assert "IMPORTED_OK" not in result.stdout
    assert "FATAL: invalid configuration" in result.stderr
    assert "Traceback" not in result.stderr


def test_valid_gemini_api_key_starts_up_cleanly(tmp_path):
    result = _run_in_clean_subprocess(tmp_path, {"GEMINI_API_KEY": "a-real-looking-key"})

    assert result.returncode == 0
    assert "IMPORTED_OK" in result.stdout
    assert result.stderr == ""
