"""Install the vLLM stubs before any overlay import, then expose the
overlay package under its canonical import path."""
import os
import sys

import pytest

# A "coroutine was never awaited" RuntimeWarning means a startup hook was
# wired wrong (a coroutine object passed where a callable is expected).
# That bug hid from the offline tests once; make it fail the run. The
# warning fires during garbage collection, so also fail on pytest's
# unraisable-exception report.
def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings", "error:coroutine .* was never awaited:RuntimeWarning")
    config.addinivalue_line(
        "filterwarnings", "error::pytest.PytestUnraisableExceptionWarning")


HERE = os.path.dirname(__file__)
REPO = os.path.dirname(HERE)

from vllm_stubs import install  # noqa: E402

install()

# Expose overlay/.../decisions as vllm.entrypoints.generate.decisions
_overlay = os.path.join(
    REPO, "overlay", "vllm", "entrypoints", "generate")
# vllm.entrypoints.generate must resolve as a package whose path
# includes the overlay directory so "vllm.entrypoints.generate.decisions"
# imports the real overlay code.
import importlib

generate_mod = importlib.import_module("vllm.entrypoints.generate")
generate_mod.__path__ = [_overlay]
import vllm.entrypoints.generate.decisions  # noqa: F401,E402



@pytest.fixture()
def decisions_pkg():
    import vllm.entrypoints.generate.decisions as pkg
    return pkg


@pytest.fixture(autouse=True)
def _calibration_dir_isolated(tmp_path, monkeypatch):
    """No test may touch the user's real calibration cache: results_dir()
    falls back to $HF_HOME/decisions-calibration when the operator sets no
    dir, so every test gets a per-test tmp dir instead."""
    monkeypatch.setenv("VLLM_TYPED_DECISIONS_CALIBRATION_DIR",
                       str(tmp_path / "calibration"))


@pytest.fixture(autouse=True)
def _logit_reads_exact_in_older_suites(request, monkeypatch):
    """The logit backend reads `top-k` by default. The suites written
    before that default exercise the exact gather (their fake engines
    answer logprob_token_ids), so they keep `exact`; test_topk_logprobs
    tests the real default."""
    if request.module.__name__.endswith("test_topk_logprobs"):
        return
    from vllm.entrypoints.generate.decisions.backends.logit_backend import (
        LogitBackend)
    monkeypatch.setattr(LogitBackend.__init__, "__defaults__", ("exact",))
