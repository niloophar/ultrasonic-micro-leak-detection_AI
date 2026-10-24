# Results Log

Concrete numbers, tables, and outcomes from experiments in this project. For methodology, architecture rationale, and problem-level interpretation, see `CLAUDE.md`.

Entries are chronological within each experiment family.

---

## 1. Prep-time Scaling Study (2026-04)

**Setup.** Train on 10.01/24.01/7.02/21.02 (4 winter series), test on 7.03 + 15.03, dual-channel supervised CNN, 18 features (`norm_interval_up/down_0..8`), adaptive Gaussian threshold from val NoFlow. Five `scaling_mode` variants tested end-to-end.

| Variant | 7.03 F1 (adapt) | 15.03 F1 (adapt) | 7.03 ROC-AUC | 15.03 ROC-AUC | NoFlow drift on 15.03 |
|---|---|---|---|---|---|
| 0 (none) | 0.981 | **0.359** | 0.998 | **0.925** | −10 logit |
| **A (noflow_global)** | **0.992** | 0.134 | 0.999 | 0.870 | −2 logit |
| B (noflow_per_series) | 0.470 | 0.000 | 0.541 | 0.109 | 0 (signal flipped) |
| C (segment_center) | 0.030 | 0.021 | 0.653 | 0.596 | logits compressed to ±1 |
| D (time_diff) | 0.010 | 0.016 | 0.564 | 0.520 | logits compressed to ±1 |

**Findings.**
- Variant A reduces NoFlow logit drift on 7.03 from −10 → −2 units, lifts F1 0.981 → 0.992.
- Variant A hurts 15.03: NoFlow↔Flow logit separation collapses from 16 → 4 because the low-flow signal lives in the absolute level that A normalizes away.
- B/C/D are unusable defaults. B sign-flips the flow signal at low flow; C/D destroy the absolute level the supervised CNN depends on.
- 15.03 baseline AUC=0.925 → ranking works, but the separation margin is too narrow for a fixed threshold. Not fixable by any prep-time scaler.

**Decision.** Variant A is the default for spring-drift deployments. For low-flow deployments (15.03-class) the path forward is the contrastive encoder, not further scaling experiments. The four feature groups (5/6/7/8) are exhaustive; no untested feature group remains.

---

## 2. Contrastive Encoder — Gruppe 5, Autoencoder variant

### 2.1 Optuna Study on 6 series (2026-04-11)

- Study name: `contrastive_gruppe5_dual`
- Storage: `sqlite:///data/contrastive/results/optuna_study.db`
- 212 trials (152 completed, 56 pruned), `MedianPruner(n_warmup_steps=15)` + `TPESampler(n_startup_trials=20, seed=42)`
- Objective: `mean(ROC-AUC_7.03, ROC-AUC_15.03)`; GMM per test series on held-out NoFlow latents.

**Search space (architecture fixed — only Contrastive/batch/latent params tuned).**

| Param | Range | Type |
|---|---|---|
| `latent_dim` | [8, 16, 32, 64, 128] | categorical |
| `k_per_series` | [4, 8, 16, 32] | categorical → `batch_size = 4 × k_per_series` |
| `projection_dim` | [32, 64, 128] | categorical |
| `lambda_supcon` | [0.05, 5.0] | log-uniform |
| `supcon_temperature` | [0.03, 0.5] | log-uniform |
| `lr` | [1e-4, 5e-3] | log-uniform |

**Best trial #58 vs. SCALER_A baseline.**

| Metric | SCALER_A Baseline (CNN) | Contrastive Best (#58) |
|---|---|---|
| Mean AUC (objective) | 0.935 | **0.981** |
| AUC 7.03 (Spring Drift) | 0.999 | 0.9999 |
| AUC 15.03 (Low Flow) | 0.870 | **0.962** |
| F1 15.03 | 0.134 / 0.359 | **0.739** |

Best params: `latent_dim=128, projection_dim=64, k_per_series=4, lambda_supcon=2.729, supcon_temperature=0.438, lr=0.00424, best_epoch=11, n_params=342914`.

**fANOVA parameter importance.**

`latent_dim` (41%) > `lr` (30%) > `lambda_supcon` (17%) > `supcon_temperature` (6%) > `projection_dim` (4%) > `k_per_series` (3%)

Patterns:
- `latent_dim = 128` dominates the top-20 trials. Smaller values show higher variance and lower medians.
- `lambda_supcon ≈ 2.7–4.6` sweet spot. Early trial with λ=2.3, τ=0.055 failed — low τ + high λ overpowers MSE. High τ (soft contrastive) lets strong λ work.
- `τ ≈ 0.39–0.49`. Trials with τ < 0.1 almost all fail.
- `lr ≈ 0.003–0.005`, best at 0.0042.
- `projection_dim` and `k_per_series` both < 4% importance — method robust to these.

Objective distribution (152 completed): min=0.681, median=0.958, mean=0.949, max=0.981. 113/152 trials > 0.95; 145/152 > 0.90.

### 2.2 Single Training Run (2026-04-12)

Trial #58 params, early stop at epoch 30 (best=15).

| Metric | Value |
|---|---|
| 7.03 ROC-AUC | 1.000 |
| 15.03 ROC-AUC | 0.891 |
| 15.03 F1 | 0.127 |
| Mean AUC | 0.946 |

Lower than Trial 58 objective (0.981) due to training variance (different seed/batch order). Confirms LOSO as authoritative eval, not single runs.

Diagnostics:
- Loss curves: MSE ~0.49 train / ~0.53 val (smooth). SupCon ~3.44 train / ~4.07 val (val gap expected — cross-series batches harder).
- t-SNE by label: clear NoFlow/Flow separation, mild overlap at boundary.
- t-SNE by series: series mix within label clusters — encoder ignores series identity. Core SupCon effect confirmed.
- Reconstructions: smooth block-level patterns recovered; point-level noise lost (expected).

### 2.3 LOSO on 6 series (2026-04-12)

| Held-out | ROC-AUC | F1 (Gauss) | n_NoFlow | n_Flow |
|---|---|---|---|---|
| 10.01.2026 | **1.0000** | 0.998 | 5050 | 8350 |
| 24.01.2026 | **1.0000** | 0.995 | 3397 | 4057 |
| 7.02.2026 | **0.9996** | 0.983 | 5741 | 1450 |
| 21.02.2026 | **0.9823** | 0.981 | 3341 | 4613 |
| 7.03.2026 | **1.0000** | 0.987 | 2217 | 2130 |
| 15.03.2026 | **0.9525** | 0.714 | 1972 | 1395 |
| **Mean** | **0.9891** | 0.943 |  |  |

vs. baselines:

| Method | Mean AUC (6-series LOSO) | 7.03 AUC | 15.03 AUC |
|---|---|---|---|
| SCALER_A CNN (supervised) | ~0.935* | 0.999 | 0.870 |
| **Contrastive Encoder + GMM** | **0.989** | **1.000** | **0.952** |

*SCALER_A is cross-series 4→2, not full LOSO; included for reference.

Notes:
- 5 of 6 series at ROC-AUC > 0.98.
- 15.03 F1 low (0.714) due to Gauss threshold being precision-oriented; ROC-AUC confirms per-patch discrimination is strong.
- 21.02 slight dip to 0.982 — transitional series (late winter, different from the other 4 training series).
- LOSO uses `val_ids=[]` — no early stopping signal. Known limitation with 6 series.

### 2.4 Retune + 8-series LOSO (2026-04-17)

With 5.04.2026 and 11.04.2026 added to data. Optuna run with same search space, test_series expanded to 4 spring-drift targets.

**Best trial #189.** Params: `latent_dim=128, projection_dim=32, k_per_series=8, lambda_supcon=2.388, supcon_temperature=0.030, lr=0.00153, best_epoch=7, n_params=338786`.

Per-series AUC/F1 at objective time:

| Series | ROC-AUC | F1 |
|---|---|---|
| 7.03.2026 | 1.0000 | 0.992 |
| 11.04.2026 | 0.994 | 0.934 |
| 15.03.2026 | 0.950 | 0.746 |
| 5.04.2026 | 0.957 | 0.731 |
| **Mean** | **0.9753** |  |

Notable shift vs. Trial #58: `τ = 0.030` (vs 0.438) with `λ = 2.4` — previously considered a failure regime, now winning. `k_per_series = 8` (vs 4) may compensate. Search-space lower bound on τ is now the actual optimum — keep the bound wide in future tunes.

**8-series LOSO with tuned params.**

| Held-out | ROC-AUC | F1 (Gauss) | n_NoFlow | n_Flow |
|---|---|---|---|---|
| 10.01.2026 | 1.0000 | 0.998 | 5050 | 8350 |
| 24.01.2026 | 1.0000 | 0.993 | 3397 | 4057 |
| 7.02.2026 | 1.0000 | 0.970 | 5741 | 1450 |
| 21.02.2026 | 0.9805 | 0.981 | 3341 | 4613 |
| 7.03.2026 | 1.0000 | 0.988 | 2217 | 2130 |
| 15.03.2026 | **0.9584** | 0.673 | 1972 | 1395 |
| 5.04.2026 | 0.9775 | 0.780 | 529 | 111 |
| 11.04.2026 | 0.9938 | 0.949 | 1045 | 693 |
| **Mean** | **0.9887** | 0.917 |  |  |

7 of 8 series > 0.97 AUC. 15.03 remains hardest but stable at 0.958 (vs 0.870 SCALER_A baseline).

### 2.5 NoFlow Cluster Compactness

Spread of NoFlow latents per series, same encoder used in 8-series LOSO.

Historical (static-default params, `latent_dim=32`, 6-series run 2026-04-12):
```
10.01=12.2, 24.01=12.4 (loose)
21.02=0.5, 15.03=1.2, 7.03=1.5 (tight)
```

Tuned params (`latent_dim=128`, 8-series run 2026-04-17, trial #189 params):
```
10.01=43.6, 24.01=42.7 (loosest)
7.02=23.3, 15.03=32.6, 7.03=29.6, 11.04=29.4
5.04=11.8, 21.02=1.8 (tight)
```

Absolute numbers scale with latent dimension. The *relative* ordering also shifted: the clean winter-loose / spring-tight split no longer holds.

Interpretation:
- Early-winter series (10.01, 24.01) still the loosest — supports "more raw data → more internal variation" (explanation a).
- 21.02 tight outlier despite mid-range data volume → likely a homogeneous NoFlow block between leakage events.
- "Winter physically spans wider temperature swings" (explanation b) is **not** supported — 15.03 (spring, low flow) is looser than 7.02 / 5.04.

Implication for Phase 3 drift modelling: cluster *radii* are not constant across series and are sensitive to encoder capacity. A centroid-only drift model will be systematically over- or under-confident. The drift function must model the *scale* of the NoFlow ellipsoid (covariance determinant or Mahalanobis whitening radius) as a second quantity.

### 2.6 Reconstruction Collapse under Tuned Params (2026-04-18)

Visual inspection of `reconstructions_val.png` from the tuned-params single training run showed reconstructions as pure horizontal stripes — pro-feature constants, zero time-axis variation, NoFlow and Flow each collapsing to a class prototype.

Loss breakdown at convergence:
```
total = mse + λ · supcon = 0.49 + 2.39 · 2.72 ≈ 7.0
SupCon contributes ~93% of the total loss.
```

With τ=0.030 (very peaky contrastive gradients) and λ=2.39, the optimizer finds a shortcut: emit two prototypes, minimise SupCon, accept residual MSE. Not a bug — the latent vector pre-decoder is clearly discriminative (LOSO AUC=0.989 confirms this), and Phase-3 drift modelling uses latents not reconstructions. But it means the "autoencoder" framing is misleading — the decoder is effectively a weak regulariser, not a reconstruction objective.

For thesis narrative: either document this honestly ("decoder as anti-collapse regulariser") or switch to the classifier variant (section 3 below), which has a per-sample gradient that prevents prototype collapse structurally.

---

## 3. Contrastive Encoder — Gruppe 5, Classifier variant

### 3.1 Optuna Study (2026-04-18)

Same architecture backbone, CE + SupCon loss replacing MSE + SupCon. Same 8-series setup, 200 Optuna trials.

**Best trial #143.** Params: `latent_dim=128, projection_dim=32, k_per_series=4, lambda_supcon=0.808, supcon_temperature=0.133, lr=0.00472, best_epoch=0, n_params=347172`.

| Series | ROC-AUC | F1 |
|---|---|---|
| 7.03.2026 | 1.0000 | 0.994 |
| 11.04.2026 | 0.975 | 0.831 |
| 15.03.2026 | 0.947 | 0.704 |
| 5.04.2026 | **0.987** | 0.796 |
| **Mean** | **0.9770** |  |

### 3.2 AE vs Classifier Comparison (Gruppe 5)

| Serie | AE (#189) | Classifier (#143) | Δ |
|---|---|---|---|
| Mean AUC | 0.9753 | **0.9770** | +0.002 |
| 7.03 AUC | 1.0000 | 1.0000 | 0 |
| **5.04 AUC** | 0.957 | **0.987** | **+0.030** |
| 15.03 AUC | 0.950 | 0.947 | −0.003 |
| 11.04 AUC | **0.994** | 0.975 | −0.019 |
| F1 15.03 | 0.746 | 0.704 | −0.042 |
| F1 5.04 | 0.731 | **0.796** | +0.065 |
| F1 11.04 | **0.934** | 0.831 | −0.103 |

Observations:
- Overall performance within noise (0.977 vs 0.975).
- **Different error profiles:** Classifier dominates 5.04, AE dominates 11.04. Opens the door for ensembling.
- Hyperparameter profiles are qualitatively different: Classifier λ=0.81 vs AE λ=2.39, τ=0.13 vs τ=0.03. Consistent with CE being a stronger per-sample signal that doesn't need SupCon to dominate.
- `best_epoch=0` for the Classifier (converges within first epoch) — not a bug, expected for a supervised 2-class task with 128-dim latent. Still worth monitoring on LOSO.

---

## 4. Gruppe 8 Studies (pending)

Configs set up, studies queued on the GPU server:

- `configs/contrastive_g8_ae/` — 18 × `norm_interval_up/down_*_tc` features, MSE + SupCon.
- `configs/contrastive_g8_clf/` — same features, CE + SupCon.

Expected ~12 h total for both Optuna runs. After completion, fill in the 2×2 matrix (feature group × loss type).

| | Gruppe 5 | Gruppe 8 |
|---|---|---|
| **AE** | Mean AUC 0.975 (Trial #189) | pending |
| **Classifier** | Mean AUC 0.977 (Trial #143) | pending |

---

## 4b. Eval A / Eval B / Eval C — Single Global GMM (2026-04-27 / 28)

**Setup.** v2 SupCon AE encoder, G5 dual + std + pw64 + noflow_filter. New
eval modes added: per-site GMM (legacy) plus single-global GMM fitted on
encoder-train NoFlow latents. Single training run with the v2-tuned
hyperparameters; no re-tune.

### Eval A — Winter-only encoder, single Winter-NoFlow GMM scored on Spring

| Serie | Per-Site (legacy) | **Global Winter-GMM** | Δ |
|-------|-------------------|------------------------|---|
| 7.03  | 1.000 | 0.868 | −0.13 |
| 5.04  | 0.999 | **0.003** ← inverted | −0.99 |
| 11.04 | 0.980 | 0.359 | −0.62 |
| 15.03 | 0.938 | 0.111 | −0.83 |
| 18.04 | 0.797 | 0.395 | −0.40 |
| **Mean** | **0.943** | **0.347** | **−0.60** |

F1 collapses to 0.0 across all five Spring series — the Gauss threshold
fitted on Spring-NoFlow scores under the Winter GMM ends up at 1.6 × 10¹⁸
(11.04, 18.04) or other absurd values (5.04 inverted). The Winter-GMM is
not "less accurate" on Spring, it is **functionally undefined** there.

5.04 ROC-AUC = 0.003 means perfect inversion: under the Winter GMM,
5.04-NoFlow latents land *further* outside the Winter cluster than
5.04-Flow latents do. Spring-NoFlow has lost its Winter-NoFlow identity
in latent space.

### Eval B — Encoder on 7-of-9 series, single global GMM scored on edge held-outs

Held-out edges: 21.02 (cool-Winter) and 18.04 (warm-Spring). Encoder
training set: 10.01, 24.01, 7.02, 7.03, 15.03, 5.04, 11.04 (7 series,
4037 NoFlow latents in the global GMM).

| Edge serie | Global GMM AUC | F1 | LOSO per-site reference |
|-----------|----------------|------|-------------------------|
| 21.02 (cool-Winter) | **0.987** | 0.979 | 0.989 |
| 18.04 (warm-Spring) | **0.878** | 0.894 | 0.960 |
| **Mean** | **0.932** | **0.937** | |

Trained 24 epochs, val_total min at epoch 8, train_aux 0.81 → 0.48
monotone — healthy convergence (no `best_epoch=0` pathology).

### Eval C — Temperature-conditioned global GMM, same encoder

Linear regression `z_i = a_i · T_pipe + b_i` per latent dim on the
training-NoFlow centroids; GMM fitted on residuals. Same encoder /
training as Eval B; eval-only difference.

| Eval | 21.02 AUC | 18.04 AUC | Mean AUC |
|------|-----------|-----------|----------|
| Eval B (Global GMM)                | 0.987 | 0.8776 | 0.9324 |
| **Eval C (T-conditional)**         | 0.987 | 0.8774 | 0.9322 |

Δ = −0.0002 — **no effect**. Decision rule (≥ 0.92 conditional →
linear-T-extrapolation suffices) clearly missed.

**Mechanistic reason for null result.** ROC-AUC is a within-series
ranking metric. Linear regression on centroids only adjusts the GMM
**mean** for each test series. NoFlow and Flow patches of the same
test series are then both shifted by the same amount → ranking within
that series is unchanged. The 18.04 gap (per-site 0.95 → global 0.88)
is a **covariance-structure** problem: the 7-series global GMM has a
covariance fitted to span all training NoFlow, which is too wide for
18.04's locally tighter NoFlow cluster. Per-site GMM adapts the
covariance locally; mean-conditional GMM does not. Polynomial / GP
mean-regression would produce the same null result — the issue is not
the functional form of the centroid model, it is that the centroid
isn't the bottleneck.

**Heteroscedastic conditioning** (covariance also a function of T)
would in principle attack the right problem but is under-determined:
fitting a 128×128 covariance trajectory from 7 series is not stable.
Not pursued.

### Findings

- **Per-site GMM has been the wrong headline metric for three months.** It
  reported 0.943 mean — but the deployment-equivalent number is 0.347.
  The 0.6 AUC gap between A's per-site and A's global is the optimistic
  bias of recalibration-per-site.
- **SupCon's invariance is real, but only inside the trained envelope.**
  Eval B reaches 0.987 AUC on the cool-Winter edge with one global GMM —
  practically identical to LOSO per-site. Cross-season unification works
  when the encoder has seen the season in training.
- **Warm-edge generalization is limited.** 18.04 at 0.878 (Eval B
  global) vs. 0.960 (LOSO per-site) shows: even with 11.04 in training,
  generalizing 16 days / 0.5 °C further out costs 0.082 AUC. The April
  cliff is partially mitigated, not removed.
- **A→B gap = 0.585 mean AUC (0.347 → 0.932).** Largest single lift in
  the entire study, larger than loss-family / patch-width / std / filter
  combined. Saison-diverse training data is the primary lever.

### Decisions

1. **Headline metric is the global-GMM number.** Per-site stays in the
   results as an upper bound under a recalibration assumption.
2. **Summer/autumn data collection promoted to priority 1.** Quantified
   estimate from the warm-edge gap: roughly one new training serie per
   ~0.5 °C envelope extension required to keep AUC ≥ 0.95 under global
   GMM.
3. **OOD detection still required.** Even under saison-diverse training
   the cliff moves rather than disappears. Mahalanobis-from-train-centroid
   threshold as hard-stop before classification.
4. **Per-site GMM not deprecated.** It remains the right metric for the
   "what-if recalibration is allowed" upper bound, useful for sensitivity
   analysis (e.g. "if a service technician calibrates monthly, what's
   the ceiling?" — answer is the per-site number).

---

## 5. Data / Series changes

- **2026-04-17.** Added 5.04.2026 and 11.04.2026. Old 6-series Optuna study deprecated; retune run on 8 series.
- **30.03.2026** remains excluded — estimated labels, not trusted.

---

## 6. Open Work Items

1. **Drift-Temperature Correlation** (high priority, low effort). Per-series mean `temp_diff` (Pipe − Ambient) from raw CSVs, plotted against latent centroid position from `drift_summary.json`. If monotonic, enables `centroid = f(temp_diff)` regression → drift prediction without on-site recalibration. No new training needed.

2. **Gruppe 6 / Gruppe 7 ablation** (medium priority, high effort). Neither tested with contrastive training. Gruppe 5 at Mean AUC 0.989 LOSO is already state-of-the-art for this project — speculative whether 6 or 7 beat it.

3. **LOSO val-split fix** (low priority). Current LOSO uses `val_ids=[]` — no early-stopping signal. Nested LOSO (hold out 1 for test, split remaining N−1 into train+val) is methodologically cleaner but unlikely to change conclusions.

4. **Balanced accuracy / per-class recall reporting.** F1 with Gauss threshold is precision-oriented and hides the majority-vote-relevant per-class recall. Downstream uses majority vote over many patches per sample, so per-class bacc ≥ 0.7 is the real success bar, not F1.

5. **Thesis figures.** LOSO table, Optuna parameter analysis, drift trajectory plots, deployment concept (initial NoFlow calibration + drift modelling). Core material ready; remaining is final figure styling.
