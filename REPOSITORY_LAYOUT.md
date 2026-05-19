# Repository layout (Multi-Agent Resonance Framework)

This repository is organized to separate the **public research framework** from **private session data**.

| Folder | GitHub | Purpose |
|--------|--------|---------|
| **`github/`** | Yes — push everything here except secrets | Full Python package (`src/prmp_demo`), Streamlit app (`app/`), docs (`docs/` including **`docs/pdf/`** after PII review), tests, scripts, synthetic `samples/`, `pyproject.toml`, `README.md`, `.env.example`. |
| **`private/`** | **No** — only `private/README.md` may be tracked from here | `.env`, live JSONL under `private/sessions/`, scratch files. See [.gitignore](../.gitignore) at workspace root and [docs/GITHUB_UPLOAD_DENYLIST.md](docs/GITHUB_UPLOAD_DENYLIST.md). |

Inside **`github/`** (this folder), paths are conventional:

- **`pyproject.toml`**, **`requirements.txt`**, **`README.md`**, **`.env.example`** live here so `pip install -e .` works when your shell’s cwd is **`github/`**.
- **`docs/pdf/`** holds replay PDFs intended for the repo — verify they contain **no personal information** before every push.

Typical commands — **`workspace/`** = parent of `github/` and `private/`:

```bash
pip install -e ./github
python -m prmp_demo.run_session --scenario silent_resonance --out private/sessions/run.jsonl
python -m prmp_demo.analyze_session --in private/sessions/run.jsonl --out private/sessions/run_analyzed.jsonl
streamlit run github/app/streamlit_app.py
python github/scripts/generate_fixture_sessions.py
```

### One-file bundle (manual upload)

From **`workspace/`**:

```bash
python github/scripts/make_github_bundle.py
```

Output: **`dist/not_named_yet_github_upload.zip`** under **`workspace/`** (ignored by git). Prefer **`git push`** for normal collaboration.
