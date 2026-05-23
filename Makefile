# One-line dev commands.

.PHONY: install test lint format server web smoke help

help:
	@echo "Common targets:"
	@echo "  make install   — uv sync --all-extras + bun install"
	@echo "  make test      — pytest"
	@echo "  make lint      — ruff check + format check"
	@echo "  make format    — ruff format"
	@echo "  make server    — start FastAPI on :8001"
	@echo "  make web       — start Next.js on :3000"
	@echo "  make smoke     — run 3-question smoke against unison-agent (requires UNISON_JWT)"

install:
	uv sync --all-extras
	cd web && bun install

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests

server:
	uv run unison-evals-server

web:
	cd web && bun run dev

smoke:
	uv run unison-evals run \
		--systems unison-agent \
		--dataset longmemeval \
		--limit 3 \
		--no-judge
