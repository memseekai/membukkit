# Contributing to MemBukkit

Thanks for your interest. Small, focused PRs are the easiest to review.

## Setup

```bash
git clone https://github.com/memseekai/membukkit.git
cd membukkit
uv sync --extra all --group dev    # or: pip install -e ".[all,dev]"
```

Install details and extras: [docs/guide/install.md](docs/guide/install.md).

## Before you open a PR

```bash
uv run ruff check src tests
uv run pytest -x -q                # offline; no API keys needed
```

Smoke the product path you’re touching:

```bash
uv run membukkit ui --demo personal-assistant --no-browser
# or: uv run membukkit demo personal-assistant
```

If you change the React GUI:

```bash
cd ui && npm install && npm run build   # updates src/membukkit/ui_dist — commit it
```

Guidelines:

- Keep the core dependency-light. Provider- or UI-specific code belongs behind an optional extra.
- New retrieval / reading behavior needs a test under `tests/`.
- Benchmarks: if your change moves a claimed number, paste
  `membukkit bench --repro <recipe> --lite` output in the PR description
  (see [docs/guide/benchmarks.md](docs/guide/benchmarks.md)).

## Releasing

README image paths are absolute `raw.githubusercontent.com` URLs, so the same
README renders on GitHub and on the PyPI project page. Keep them that way;
relative paths break the PyPI page. `python scripts/pypi_readme.py --check`
reports the current state.

```bash
rm -rf dist && uv build
uvx twine check dist/*
uvx twine upload dist/*
```

Tag the release so the container image picks up `latest` and the version tag:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

## Reporting issues

Include your Python version, how you installed (`pip install -e ".[all]"` / git URL),
the provider spec (e.g. `openai:gpt-4o-mini`), and a minimal reproduction.
Never paste API keys.
