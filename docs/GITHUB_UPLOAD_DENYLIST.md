# Do **not** upload to GitHub (denylist)

Publishing any of the following can compromise accounts, leak conversations, or violate policy. **Rotate keys** if something was ever committed by mistake.

## Secrets & credentials

| Item | Examples |
|------|-----------|
| API keys | `OPENROUTER_API_KEY`, bearer tokens in scripts or notebooks |
| OAuth / SSH private keys | `id_rsa`, `*.pem`, service-account JSON |

**Rule:** Real values belong in **`private/.env`** (preferred) or a **workspace-root `.env`** — both are gitignored — never commit them.

## `private/` workspace

Everything under **`private/`** except **`private/README.md`** is ignored. Typical contents:

| Path | Risk |
|------|------|
| `private/.env` | Live secrets |
| `private/sessions/**/*.jsonl` | Raw model outputs, rationales |

Do **not** `git add -f` files from here.

## Local environment & build artifacts

`.venv/`, `__pycache__`, `.pytest_cache/`, `*.egg-info/` — ignored; never force-add.

---

**Companion:** [`GITHUB_UPLOAD_ALLOWLIST.md`](./GITHUB_UPLOAD_ALLOWLIST.md).
