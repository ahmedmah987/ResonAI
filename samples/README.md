# Sample sessions

Synthetic JSONL fixtures for Streamlit replay (no secrets).

Run from **`github/`** (current directory is this repo folder):

```bash
python scripts/generate_fixture_sessions.py
```

Outputs under **`samples/`**:

- `demo_trivial_correlation.jsonl` — nearly identical embeddings each step (high \(\rho_{\text{text}}\)).
- `demo_silent_resonance.jsonl` — divergent wording but correlated latent drift (illustrative).
- `demo_conflicting_objectives.jsonl` — opposing mandated actions plus anti-correlated embeddings (expect lower `gamma_0` vs silent fixture — illustrative only).

Live OpenRouter exports belong under **`private/sessions/`** (gitignored); these bundled files are UI wiring demos only.
