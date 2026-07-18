.PHONY: install fix lint type test check

install:           ## bootstrap: sync deps + install the git hook
	uv sync
	uv run pre-commit install

fix:               ## auto-fix + format
	uv run ruff check --fix .
	uv run ruff format .

lint:              ## check only, no writes
	uv run ruff check .
	uv run ruff format --check .

type:              ## strict mypy over the source packages (tests are runtime-verified by pytest)
	uv run mypy src/agent-fleet/src src/agent-fleet-api/src src/agent-fleet-mcp/src

test:
	uv run pytest

check: lint type test   ## the full gate
