# Safe to publish on GitHub (allowlist)

Use this checklist before `git add` / opening a PR. When in doubt, treat content as **denied** until reviewed.

## Always OK (intended public surface)

| Path / pattern | Notes |
|----------------|--------|
| Root `README.md`, `REPOSITORY_LAYOUT.md`, `LICENSE`, `CONTRIBUTING.md` | Community-facing root docs |
| `pyproject.toml`, `requirements.txt` | Dependency manifests (no secrets) |
| `.env.example` | **Placeholder values only** — never real API keys |
| **`docs/**/*.md`** | Project documentation |
| **`docs/pdf/*.pdf`** | Replay exports — OK only after verifying **no personal information** |
| **`docs/assets/*.mp4`** | Demo videos |
| **`src/prmp_demo/**`** | Application source |
| **`app/streamlit_app.py`** | Dashboard |
| **`scripts/*.py`** | Helper scripts (includes `make_github_bundle.py`) |
| **`tests/**`** | Tests |
| **`samples/demo_*.jsonl`** | **Synthetic** fixtures only |

## Relationship to `.gitignore`

See **[`GITHUB_UPLOAD_DENYLIST.md`](./GITHUB_UPLOAD_DENYLIST.md)** for `private/` and other blocked paths.
