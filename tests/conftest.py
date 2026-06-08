"""Pytest fixtures + env hygiene."""

from __future__ import annotations

import os

# Stop config.py from reading the real .env when running tests.
os.environ.setdefault("UNISON_API_URL", "http://localhost:3001")
os.environ.setdefault("UNISON_JWT", "test-jwt")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
# Blank the eval secret so a real .env value can't leak in and silently flip the
# unison-agent adapter into per-question ephemeral-tenant mode during tests.
os.environ.setdefault("UNISON_EVAL_SECRET", "")
