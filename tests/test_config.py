"""Path-knob resolution — the same answer from wherever the process started.

The package is consumed by another project through an editable install, so
``chillrag`` is imported out of this tree by a process whose working directory
belongs to somebody else. `load_dotenv` resolves a relative value against that
working directory, which would point the corpus, the cache and the LLM config at
the importing project — silently, since all three are read later and separately.
"""

from __future__ import annotations

import importlib

import pytest

from chillrag import config


def test_a_relative_value_anchors_at_the_repo(monkeypatch):
    monkeypatch.setenv("CHILLRAG_MATERIALS_DIR", "./materials")
    assert config._path("CHILLRAG_MATERIALS_DIR", config.REPO_ROOT) == (
        config.REPO_ROOT / "materials"
    )


def test_an_absolute_value_is_left_alone(monkeypatch):
    """A deployment that points at its own volume must not get the repo prepended."""
    monkeypatch.setenv("CHILLRAG_MATERIALS_DIR", "/srv/corpora")
    resolved = config._path("CHILLRAG_MATERIALS_DIR", config.REPO_ROOT)
    assert resolved.as_posix() == "/srv/corpora"
    assert config.REPO_ROOT not in resolved.parents


def test_a_tilde_is_expanded(monkeypatch):
    monkeypatch.setenv("MODELS_DIR", "~/Models")
    resolved = config._path("MODELS_DIR", config.REPO_ROOT)
    assert resolved.is_absolute()
    assert "~" not in resolved.as_posix()


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_value_means_unset(monkeypatch, value):
    monkeypatch.setenv("CHILLRAG_CACHE_DIR", value)
    assert config._path("CHILLRAG_CACHE_DIR", config.REPO_ROOT / ".cache") == (
        config.REPO_ROOT / ".cache"
    )


def test_every_path_knob_goes_through_it(monkeypatch):
    """The wiring, not just the helper: a fresh import must pick the value up."""
    monkeypatch.setenv("CHILLRAG_LLM_CONFIG_DIR", "./conf/llm")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.LLM_CONFIG_DIR == reloaded.REPO_ROOT / "conf" / "llm"
    finally:
        # Reloading rebinds module globals, so every other test's view of
        # ``chillrag.config`` has to be restored before this one returns.
        monkeypatch.undo()
        importlib.reload(config)
