# Why might this demo fail? (Falsification statement & limits)

> This file complements [`P_RMP_DEMO_PLAN.md`](./P_RMP_DEMO_PLAN.md). The goal is honesty with researchers and practitioners—not diminishing the work.

Operational aspects for this release (README, CLI, replay mode, step/token limits, Git hygiene) are detailed in **§10** of [`P_RMP_DEMO_PLAN.md`](./P_RMP_DEMO_PLAN.md). **Scenario contrast and the zero-resonance baseline** are documented in [`CONTRAST_AND_CALIBRATION.md`](./CONTRAST_AND_CALIBRATION.md).

## 1) What the current observer does *not* measure

- **Hidden layers** of chat models are not accessible via OpenRouter; measurement uses an **external trajectory** (fixed text embedding each step).
- Any resemblance between \(\vec{\Gamma}\) and models’ “internal intent” is **inference under the experimental protocol**, not a directly observable quantity.

## 2) Sensitivity of embedding model \(E\) and template \(h(\cdot)\)

- Changing `PRMP_EMBEDDING_MODEL` changes \(\mathcal{Z}\) entirely; comparing runs across different \(E\) without recalibration is invalid.
- How \(h\) is written determines whether the signal reflects “strategy” vs “surface wording”; a bad \(h\) yields confounded signals.

## 3) Dynamic geometry hyperparameters

- **PCA/UMAP:** Sensitive to hyperparameters and sample size; separate UMAP per agent without joint alignment breaks comparability.
- **Spectral / DMD proxy:** Needs adequate windows and \(k\); embedding noise can induce spurious spectral coupling.
- **Grassmann:** Window length \(W\) and subspace rank \(r\) change meaning entirely; there is no universal “correct” value.

## 4) TDA (when enabled)

- With few points per window, persistence diagrams can be **mostly noise**.
- Any \(W_p\) comparison should state window accumulation / subsampling policy in reports.

## 5) Network, cost, cross-provider variance

- Latency / billing variance across sessions can affect the qualitative time series (especially if retries occur silently).
- Rate limits or transient errors can distort trajectories; errors should be logged in JSONL.

## 6) How to empirically reject demo-level claims

Fill this in after an organized experiment batch:

- If scenario (a) fails to show high \(\rho_{\text{text}}\) with \(\vec{\Gamma}\) inconsistent with the programmatic definition → revisit \(\rho\) or verbal protocol.
- If scenario (b) fails to show low \(\rho_{\text{text}}\) with aligned mandated actions while \(\vec{\Gamma}\) disagrees → revisit \(h\), trajectory length, or DMD/Grassmann windows.
- If scenarios remain inseparable after tuning and repeated sessions → narrow public claims or redesign the task.

## 7) Privacy & compliance

- Do not commit API keys or sensitive session content to a public repo.
- If real user data is used later, update this section per retention/redaction policy.

## 8) What software v0.1 represents (post-implementation notes)

- Code under `src/prmp_demo/` implements: two chats via OpenRouter, one embedding observer, JSONL export (`session_meta` + steps), shared PCA/UMAP alignment into \(\mathcal{Z}\), spectral surrogate over sliding windows (DMD-inspired via singular-value spectra), Grassmann similarity on sub-windows, \(\vec{\Gamma}\) as three components **[\(R_{spec}\), \(W_p\) alignment, Grassmann]** per theory ordering, plus \(\rho_{\text{text}}\) as cosine similarity between raw embeddings at the same step.
- Near-constant trajectories trigger **PCA skip** with dimensional truncation fallback to avoid numerical failure; see `pca_meta.reason`.
- `--dry-parse-only` avoids network and uses **local pseudo-embeddings** for pipeline/protocol smoke tests—not a claim about “real resonance”.
- Files under `samples/demo_*.jsonl` are **synthetic** for Streamlit wiring only; substantive separation must be demonstrated with live OpenRouter sessions and documented here afterward.

For the research community next: run real sessions for `trivial_correlation`, `silent_resonance`, and `conflicting_objectives` under matched settings, and compare \(\rho_{\text{text}}\) vs `gamma_0` as in [`CONTRAST_AND_CALIBRATION.md`](./CONTRAST_AND_CALIBRATION.md) before generalizing claims.
