# Mapping `not-named-yet.md` to this codebase

This demo observes **text embeddings** \(E(h(\cdot))\), not hidden LLM states. Shared latent \(\mathcal{Z}\) is built from **paired embedding trajectories**, which operationalizes the theory’s \(\psi_i: \mathcal{M}_i \to \mathcal{Z}\) at the level of observable channels.

## Four-stage pipeline

| Theory section | Implementation |
|----------------|----------------|
| **1. Manifold alignment** (UMAP / Diffusion idea) | [`preprocess.manifold_align`](../src/prmp_demo/pipeline/preprocess.py): joint **PCA** default; optional **`pip install .[full]`** **UMAP** on `vstack([X_a,X_b])` with PCA fallback when samples/deps are insufficient. |
| **2. Spectral resonance / Koopman** | [`dmd.spectral_overlap_at_end`](../src/prmp_demo/pipeline/dmd.py) (**SVD snapshot proxy**) or [`koopman_edmd.edmd_overlap_at_end`](../src/prmp_demo/pipeline/koopman_edmd.py) (**EDMD** \(Z_{t+1}\approx K Z_t\) + truncated eigenvalue magnitude overlap). Env `PRMP_R_SPEC_METHOD=svd_proxy|edmd`. |
| **3. Topological synchronization \(W_p\)** | [`tda_optional.persistence_wasserstein_h0`](../src/prmp_demo/pipeline/tda_optional.py): **H₀** diagrams via **ripser**, **Wasserstein** via **persim** on trailing windows in \(\mathcal{Z}\). Requires `.[full]`; otherwise `w_p_unavailable_reason` is set. |
| **4. Grassmann \(\sum\cos^2\theta_i\)** | [`grassmann.grassmann_similarity`](../src/prmp_demo/pipeline/grassmann.py): truncated SVD bases on paired windows → principal angles → \(\sum \sigma_i^2\) normalized to \([0,1]\). |

## Resonance vector \(\vec{\Gamma}(t)\)

Theory: \(\vec{\Gamma}(t)=[\mathcal{R}_{spec}(t), W_p(t), \sum\cos^2\theta_i]\).

Code [`gamma.compute_metrics_prefix`](../src/prmp_demo/gamma.py) publishes **`gamma`** as **`[R_spec_norm, W_p_alignment_norm, Grassmann_norm]`** — same ordering; \(W_p\) in theory is a **distance** (smaller is better); in **`gamma[1]`** we store **`w_p_norm = 1/(1+w_p)`** so larger aligns with “better” display alongside \(R_{spec}\).

## Predictive stability layer

| Concept | Implementation |
|---------|----------------|
| \(\lambda_\Gamma\) | [`stability.lambda_gamma_ftle_proxy`](../src/prmp_demo/pipeline/stability.py): slope of \(\log\|\Delta\Gamma\|\) over a trailing window (**FTLE-style proxy**, not full Jacobian Lyapunov). |
| Critical slowing down (CSD) | Rolling variance of \(\|\Gamma\|\); **`csd_spike`** when variance exceeds **`PRMP_CSD_VARIANCE_MULT`** × trailing median baseline. |
| Geodesic curvature \(\kappa\) | **`kappa_gamma`**: \(\|\Gamma_{t+1}-2\Gamma_t+\Gamma_{t-1}\|\) on the \(\Gamma\) trajectory in \(\mathbb{R}^3\) (**extrinsic curvature proxy**). |

Computed **offline** in [`analyze_session`](../src/prmp_demo/analyze_session.py), not during live `run_session`.

## Falsification metrics & phenomenology

| Metric | Module / field |
|--------|----------------|
| \(\Delta_{RG}=\|\Gamma\|-|\rho|\) | [`falsification_metrics.delta_resonance_gap`](../src/prmp_demo/falsification_metrics.py) → **`delta_rg`** |
| \(\tau_{lead}\) (session heuristic) | **`analysis_summary.tau_lead_steps`** — first **`unity_proxy`** minus first **`csd_spike`** |
| \(\eta_{topo}\approx \dot W_p/\kappa\) | **`eta_topo`** (discrete \(\dot W_p\) on raw **`w_p_raw`**) |
| Phenomenological regime table | **`phenomenological_state`** heuristic classifier |

Env thresholds: `PRMP_UNITY_*`, `PRMP_CSD_VARIANCE_*`.

## Optional install

```bash
pip install -e ".[full]"
```

Adds **umap-learn**, **ripser**, **persim** for nonlinear manifold alignment and persistent-homology distances.
