"""Verify that every registered AgentAdapter accepts the new seed_docs kwarg.

This is a forward-compatibility guard: any new adapter added to REGISTRY
must accept seed_docs or this test will catch it.
"""

from __future__ import annotations

import inspect

import pytest

from unison_evals.memory_evals.adapters import REGISTRY


@pytest.mark.parametrize("adapter_name", sorted(REGISTRY.keys()))
def test_adapter_accepts_seed_docs_kwarg(adapter_name: str) -> None:
    cls = REGISTRY[adapter_name]
    sig = inspect.signature(cls.answer)
    params = sig.parameters
    assert "seed_docs" in params, (
        f"Adapter '{adapter_name}' ({cls.__name__}.answer) does not accept "
        f"'seed_docs' kwarg — update it to match the AgentAdapter contract."
    )
    # Must be keyword-optional (has a default).
    param = params["seed_docs"]
    assert param.default is not inspect.Parameter.empty, (
        f"Adapter '{adapter_name}': seed_docs param must have a default value (None)"
    )


@pytest.mark.parametrize("adapter_name", sorted(REGISTRY.keys()))
def test_adapter_accepts_oracle_context_kwarg(adapter_name: str) -> None:
    cls = REGISTRY[adapter_name]
    sig = inspect.signature(cls.answer)
    params = sig.parameters
    assert "oracle_context" in params, (
        f"Adapter '{adapter_name}' does not accept 'oracle_context' kwarg — "
        f"backward-compat breakage."
    )
    param = params["oracle_context"]
    assert param.default is not inspect.Parameter.empty
