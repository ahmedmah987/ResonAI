# Contrast & calibration

This document complements [`P_RMP_DEMO_PLAN.md`](./P_RMP_DEMO_PLAN.md) and [`FALSIFICATION.md`](./FALSIFICATION.md). It documents the **scenario triangle** showing that \(\vec{\Gamma}(t)\)—especially **`gamma_0`**—should not be read as “high resonance” without a clear **negative baseline**.

---

## 1) What `gamma_0` represents in the current implementation

In code, \(\vec{\Gamma}(t) = [\texttt{gamma\_0}, \texttt{gamma\_1}, \texttt{gamma\_2}]\) follows the ordering in [`not-named-yet.md`](./not-named-yet.md):

- **`gamma_0`**: After normalization to roughly \([0,1]\), a **spectral overlap** proxy \(\mathcal{R}_{spec}\) between the two agents’ trajectories over a time window in \(\mathcal{Z}\) after **shared alignment** (**PCA** by default, or **UMAP** with `pip install .[full]`). The default path is an **SVD-on-snapshots** surrogate; enable an **EDMD / Koopman-style** surrogate via `PRMP_R_SPEC_METHOD=edmd`. See [`THEORY_TO_CODE.md`](./THEORY_TO_CODE.md).

- **`gamma_1`**: A **topological alignment** score derived from **Wasserstein \(W_p\)** between **H₀** persistence diagrams of time windows (requires `.[full]`; otherwise check `w_p_unavailable_reason`). The value stored in \(\Gamma\) is `w_p_norm = 1/(1+w_p)` (higher = better alignment in the UI, since raw \(W_p\) is a distance).

- **`gamma_2`**: **Grassmann** \(\sum\cos^2\theta_i\) similarity after normalization (time windows).

The **stability & falsification** layer (\(\lambda_\Gamma\) proxy, CSD, \(\kappa\), \(\Delta_{RG}\), and `analysis_summary`) is computed by **`python -m prmp_demo.analyze_session`**, not during the live session loop.

Definitions live in [`src/prmp_demo/gamma.py`](../src/prmp_demo/gamma.py), [`pipeline/dmd.py`](../src/prmp_demo/pipeline/dmd.py), [`pipeline/koopman_edmd.py`](../src/prmp_demo/pipeline/koopman_edmd.py), and [`pipeline/tda_optional.py`](../src/prmp_demo/pipeline/tda_optional.py).

---

## 2) Three poles of contrast (why three scenarios, not two?)

| Pole | Scenario id | Experimental role |
|------|-------------|---------------------|
| **High strategic resonance (positive reference)** | `silent_resonance` | Same mandated action sequence between agents with varied wording; dynamics in \(\mathcal{Z}\) should stay relatively aligned so **`gamma_0`** tends **higher** than the conflicting baseline (after windows stabilize). |
| **Surface mimicry** | `trivial_correlation` | \(\rho_{\text{text}}\) can be high while strategic meaning may not imply “deep resonance”; separates lexical similarity from \(\Gamma\). |
| **Targeted zero-resonance baseline (negative reference)** | `conflicting_objectives` | **Explicit conflict**: agent A follows a reference sequence while B is forced into an **opposing** action each step (fixed inverse map). Goal: deliberate divergence in the structured channel so **`gamma_0` drops sharply** vs `silent_resonance` under the same \(E\), \(h\), and windows. |

---

## 3) Core sanity hypothesis (complete the sentence)

**If the system shows a sharp drop in `gamma_0` under conflicting tasks** (`conflicting_objectives`) compared to matched `silent_resonance` runs (same models, same \(E\), same \(k,W,r\), enough steps for windows to stabilize), **that supports** the claim that the metric reacts to convergence/divergence of dynamics in \(\mathcal{Z}\) more than raw text similarity alone.

**If that does not happen:**

1. Review the **\(h(\cdot)\) template**—does what gets embedded hide conflict (e.g. collapsing fields)?
2. Review **window lengths**: `PRMP_DMD_WINDOW`, `PRMP_GRASSMANN_W`, **`PRMP_TDA_WINDOW`**, and **`PRMP_PCA_K`**.
3. Review **JSON adherence**; failures shred the structured channel without \(\Gamma\) reflecting the intended semantics.
4. Record the outcome in [`FALSIFICATION.md`](./FALSIFICATION.md) as a scope limit.

---

## 4) How to document replay exports (e.g. PDF from Streamlit)

When sharing a screenshot or PDF from **P-RMP Demo Replay**:

1. State the **session id** and source JSONL path.
2. Copy from the UI: curves **`gamma_0` (R_spec)**, **`gamma_1` (W_p alignment)**, **`gamma_2` (Grassmann)**, and **`rho_text`** vs \(t\); for derivatives run `analyze_session` and open secondary plots (\(\lambda_\Gamma\), \(\kappa\), \(\Delta_{RG}\)) when present.
3. Add a **comparison table** for three runs with matched settings:

   | Run | Scenario | Note on `gamma_0` after the window settles |
   |-----|----------|----------------------------------------------|
   | R1 | `silent_resonance` | Positive reference |
   | R2 | `conflicting_objectives` | Negative baseline (expect drop) |
   | R3 | `trivial_correlation` | Lexical mimicry reference |

4. Use the **same \(t\) axis range** across Streamlit captures to avoid unfair comparisons between shorter and longer sessions.

**Example PDF (optional artifact next to docs):** [P-RMP Demo Replay.pdf](./P-RMP%20Demo%20Replay.pdf). Metadata in the file indicates export via Chrome / Skia (save as PDF). Use it alongside the original JSONL when comparing “high resonance” vs other baselines.

---

## 5) Quick commands for contrast experiments

After configuring `.env` and OpenRouter:

```powershell
python -m prmp_demo.run_session --scenario silent_resonance --out private\sessions\ref_high.jsonl
python -m prmp_demo.run_session --scenario conflicting_objectives --out private\sessions\ref_zero.jsonl
python -m prmp_demo.run_session --scenario trivial_correlation --out private\sessions\ref_trivial.jsonl
streamlit run app\streamlit_app.py
```

Synthetic fixtures (no API), after updates:

```powershell
python scripts\generate_fixture_sessions.py
```

---

## 6) Honesty bound

The “zero resonance” baseline here is **protocol-relative**, not a theorem that \(\Gamma \equiv 0\). Stronger claims need statistical framing (means across sessions, confidence intervals, etc.) beyond this initial demo.
