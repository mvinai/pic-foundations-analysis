---
title: "Stages 09–18: plan (PLAN ONLY, no build code)"
subtitle: "PIC Foundations. Plan v13 and contract v13. After Stages 06–08 v1.2.1 and EDA 06–08 v1.0.0"
date: "Nov 18, 2026 (America/Chicago)"
---

# 0. Sources read and what is (and is not) defined

Governing documents: contract v13 (`contract_v13/foundations_contract_v13.md`), plan v13 (`v13_build/revised_plan_v13.md`, sources `v13_build/src/p*.md`), preprocessing log (Step 5.14, "Open decisions (next)", "Then"), and `stage06_08/stage06_08_spec.md` v1.2.1 (output contract). Also checked: `scripts/`, `specs/`, `methods/`, `synthetic_svd_dataset/`, `team_guides_v2/`, `team_roles/`, `_work_tmp_guides/`, `static_vs_temporal_subspace_analysis_draft.md`.

**Existing stage 09+ code or specs: none.** There was no pre-existing `09_*.py` … `18_*.py` or stage 09–18 spec in the project tree when this plan was written. The team guides (v2) set teammate targets for synthetic packaging of 09/10/11 on Nov 18–19.

**Stage numbering 09–18 as the plan defines it** (plan v13 §13.2 stage map; contract "Execution workflow" steps 4–7). Every number from 09 to 18 is defined; 15 and 15i are both inside the range; there is no stage numbered for outcome labels (see §2, gap G1).

| Stage | Script (plan §13.2) | Objective | Owner (plan §15 / contract "Ownership") |
|---|---|---|---|
| 09 | `09_spectral_uncertainty.py` | Obj 1 | Vinay (code, synthetic); Vinai (real run) |
| 10 | `10_lowrank_cur_angles_recovery.py` | Obj 1 | Yashwant |
| 11 | `11_model_bakeoff.py` | Obj 2 | Habib |
| 12 | `12_tune_calibrate.py` | Obj 2 | Habib |
| 13 | `13_attribution_stability.py` | Obj 2 (criterion 5) | Habib |
| 14 | `14_test_once_report.py` | Obj 2 (A, B test) | Vinai |
| 15 | `15_build_ladder_layer_c.py` | build (C levels) | Vinai |
| 15i | `15i_imputation_arms.py` | build/sensitivity (C) | Vinai |
| 16 | `16_layer_c_tensor_svd.py` | Obj 3 (extension) | Vinai |
| 17 | `17_landmark_models.py` | Obj 2 co-primary | Habib |
| 18 | `18_sequence_attribution_test.py` | Obj 2 test + seq. attribution | Habib |

"Static vs temporal subspace analysis" (the Plan v9 draft) is **not a v13 stage**. In v13 its content lives in stage 10 principal angles (plan §9, H4) and the A+B vs A+B+C contrast (stage 17). It is not planned as a separate step here.

# 1. What Stages 06–08 hand over (input contract; spec v1.2.1 §1, §5)

Folder `data\processed\ladder\stage06_08\` (manifest `d8efd1fe…`, pinned by full SHA-256 on the command line, as Stage 3's was):

* `stage08_<level>.npz`, 8 tabular levels a0, a1, a2, b0, b1, b2s, b2d, b2p: `X_z` (primary; one reference category dropped per one-hot block) with `columns`; `X_full_z` (full one-hot set, H1 check) with `columns_full`; `X_imputed`; `kinds`; `split`; `encounter_id`; `eligible_24h`. Zero-variance columns are already dropped (train).
* `stage08_layer_c_L<L>.npz`, 14 landmarks L24…L336: `X_raw_z` (NaN where missing), `X_filled_z`, `missing`, `carried_forward`, `imputed` as (n_L × L/6 × q_L) arrays over **eligible encounters only** (no padding inside a prefix, plan §6.1), `channels` (q_L varies by landmark: D13 drops channels with no train variation at L).
* Team-internal, never opened by Copilot or pasted: `stage08_fitted_transforms_INTERNAL.json`, `stage06_08_category_report_INTERNAL.csv`.

**Not built by 06–08** (spec §7 and code check): the Layer C **levels** c0 / c1 / c2s / c2d / c2p (stage 15), the imputation arms (15i), microbiology M2 / M3, and **any outcome label file**. The Layer C npz is one standardized representation of the Stage 3 base channels, not the φ0/φ1/φ2 ladder of plan §6.5.

# 2. Gaps that block the plan as written (must be settled first)

* **G1, outcome labels (no stage number).** Stages 11–18 need Outcome 1 (K = 9, remaining ICU time from L, contract H4.1) per landmark, Outcome 2 (K = 5, H4.2), ordinal8 at 24/48/72 h and mortality. Nothing in Stage 3 or 06–08 writes them. The band-support audit (spec v2.3) computed them in memory for counts only. **Plan:** a label builder run as a preflight of stage 11 ("11p", not a new stage). It writes `labels_landmark.npz` (keys and labels only, team-internal), plus an aggregate audit containing check results only (assignable + unassignable = eligible, death precedence, keep_both) and no class counts.
* **G2, Layer C levels (stage 15) not built.** The 24 h C-level comparison (contract "Analysis questions", Objective 2 primary) and H15 (Obj 3 level ordering) need c0/c1/c2. See decision Q2.
* **G3, synthetic-frozen thresholds.** H1–H17 thresholds are to be "fixed from these [synthetic] runs and frozen before real data" (plan §12.2, §13.3). No synthetic stage 09–18 run exists. See decision Q3.
* **G4, sequence-model stack.** §6.7 needs GRU / TCN / Transformer. The laptop's PyTorch availability is unknown (the box has none). It needs a package check through the machine Shell.

# 3. Where D15 fits: Stage 09 preflight "09p" (dedup)

**D15 (Vinai, Nov 18):** collapse exactly identical columns before modeling, record which copies were dropped, keep near-constant columns.

* **Rule.** On each level's `X_z` (and on the AB / A+B concatenations that 11/17 use), group columns that are bit-identical **on train rows** (hash of the float32 column bytes, then exact `array_equal` confirmation). Keep the first column in canonical `columns` order and drop the rest. The map is train-fitted and applied unchanged to validation and test. A pair identical on train but not on validation/test is still collapsed, and only a yes/no flag per level is reported. Complements (x = 1 − y) and near-duplicates (|r| > 0.95) are **not** exact duplicates and stay (EDA v1.1 §9).
* **Outputs.** `stage09p_dedup_map.json` (team-shareable: level, kept column, dropped columns, by name; no counts of rows); `stage09p_dedup_<level>.npz` (the deduplicated `X_z` + `columns`); per-level width before/after in the audit.
* **Layer C.** The same rule applies per landmark on the channel axis of `X_filled_z` (all train valid cells). It is logged, never flattened.
* **Conflict to resolve (Q1).** φ2-dup (b2d) is duplicated **by design**: H1 requires "rank-deficiency count equals duplicated-column count; pruning restores rank", H7 tests the attribution fall from φ1 to φ2-dup, and §11 lists "the duplicated columns of φ2-dup" as a credit-split group. Collapsing b2d's `__dup__` copies would turn b2d back into b2s and erase H1/H7. EDA v1.1 §9 currently lists "65 phi2-dup copies" under the D15 collapse.

# 4. Stage-by-stage plan

Common to every stage. These rules hold throughout (contract "Copilot run rules", "Disclosure and release gates", "Documentation standards"; plan §13–§15):

* Hash-gated precheck of the script, tests and stage06_08 manifest.
* Build in memory, then publish atomically through a `<stage>.pending-<pid>` folder; exit 3 on a lock.
* A sidecar meta.json and manifest with every output's SHA-256.
* Train-only fitting.
* Subject-level bootstrap with seed 6310.
* Outputs are aggregate only: no row, ID, timestamp, min/max, or per-patient score/loading. Quantiles are limited to p10–p90.
* Any count is shown only if it is ≥ 10, with `<10` plus complementary suppression and an LP recoverability check (Stage 2 v1.3 gate). Stages 09, 10 and 16 publish no counts at all, so they get the stage06_08-style structural-whitelist gate. Stages 11–18 add a counts gate for class tables.
* No `*_INTERNAL` content goes into TEAM files, and no cohort, split or landmark N is printed. The script checks the landmark eligible counts internally and prints pass/fail only.
* `_raw_LOCAL` files are never read.
* A heartbeat line at least every 30 s.

## Stage 09: spectral diagnostics and uncertainty (Objective 1; plan §8; contract "Analysis questions: Objective 1"; H1, H2, H3)

* **Inputs.** The eight `stage08_<level>.npz`, train rows; both `X_z` and `X_full_z`; the 09p dedup map. Layer C at landmarks: the per-bin matrices (n_L × q, one per bin k) and the stacked (n_L·b_L) × q train matrix of `X_filled_z` (contract: Layer C Objective 1 is the **per-bin / stacked view only**, V9).
* **Math.**
  * SVD A = UΣVᵀ (never AᵀA).
  * κ₂ = σ₁/σ_p (∞ below tolerance) and κ₂ over the numerical rank.
  * Numerical rank at max(n,p)·ε·σ₁, 10⁻³σ₁ and 10⁻²σ₁.
  * Belsley indices η_j = σ₁/σ_j with variance-decomposition proportions. A near-dependency is η ≥ 30 with ≥ 2 proportions > 0.5; the columns are named.
  * Stable rank ‖A‖_F²/‖A‖₂².
  * Effective rank exp(H(w)) with w = σ²/Σσ² as the headline; the σ version is shown alongside.
  * k95 on σ² energy as the headline; σ energy alongside.
  * Permutation null: each column permuted independently, 200 reps; count σ_i above the null's 95th percentile.
  * MP edge √n(1+√(p/n)) as a reference line only.
  * Decay fits (power, exponential, rank-plus-noise) on the log scale, compared by BIC.
  * Subject bootstrap (B = 200) of all of the above.
  * Davis–Kahan/Wedin ‖sin α‖₂ ≤ ‖E‖₂/(σ_k − σ_{k+1}), with row-weight bootstrap and a vacuous-bound flag per k.
  * Relative gaps g_k.
  * Variants: with/without mask columns (plan §7 "Masks"), pre- and post-dedup, X_z vs X_full_z.
* **Outputs (TEAM).** `stage09_spectra_TEAM.csv` (σ_i/σ₁ per level × variant), `stage09_summary_TEAM.csv` (p, κ₂, ranks, stable/effective rank, k95, null count, BIC family + bootstrap intervals), `stage09_belsley_TEAM.csv` (named dependencies), `stage09_daviskahan_TEAM.csv`, `stage09_layer_c_perbin_TEAM.csv`, figures, `stage09_audit.md/json`, manifest.
* **Acceptance checks.**
  * Hash gates.
  * Train-only (no validation/test row enters).
  * Eckart–Young identities hold numerically.
  * The H1 identity: rank deficiency of b2d − b2s = retained `__dup__` count. This re-checks B10 and needs pre-dedup b2d.
  * The planted range column is named by Belsley (synthetic gate, §13.3).
  * Planted rank 10 is recovered on `synthetic_planted_v1` (§8 findings).
  * The bootstrap rows are subject-clustered.
  * Release gate (no counts).

## Stage 10: low-rank approximation, CUR, principal angles, recovery (Objective 1; plan §9; H4, H5)

* **Inputs.** As in stage 09: 09p-deduplicated `X_z` for A, B and AB (b2d per Q1), train rows.
* **Math.**
  * Eckart–Young checks: ‖A − A_k‖₂ = σ_{k+1} and ‖A − A_k‖_F² = Σ_{i>k}σ_i².
  * Randomized SVD with oversampling 10 and s ∈ {0, 1, 2}, against exact SVD.
  * CUR rules: uniform, leverage sampling, top-k leverage, CPQR and DEIM. Middle matrices U = C⁺AR⁺ and U = A(I,J)⁺.
  * k ∈ {5, 10, 20, 30, 40, 59}, k ≤ p. The ratio ‖A − CUR‖/‖A − A_k‖ is computed in Frobenius and spectral norms. Random rules use 50 repeats. Report the smallest c with ratio ≤ 1+ε for ε ∈ {0.1, 0.5} against the Drineas–Mahoney–Muthukrishnan bound.
  * Principal angles via thin QR then SVD(Q_Uᵀ Q_V): adjacent levels (a0↔a1↔a2, b0↔b1↔b2s/b2d/b2p) and A vs B. Nulls: Gaussian subspaces, column-permuted copies, split-half.
  * Linear vs nonlinear recovery of each added column from φ0 columns: ridge vs gradient boosting, 5-fold CV grouped by subject. R² for continuous columns, balanced accuracy for flags/stage; gap by family; a kNN mutual-information estimate.
  * It also **exports the train V_k and CPQR column sets** that 11 and 17 reuse (Shared guide interface I2). This export is team-internal (column names are shareable; V_k numbers stay team-internal).
* **Outputs (TEAM).** `stage10_cur_TEAM.csv`, `stage10_rsvd_TEAM.csv`, `stage10_angles_TEAM.csv`, `stage10_recovery_TEAM.csv`, `stage10_cpqr_columns_TEAM.csv` (names only), audit/manifest. Team-internal: `stage10_matched_width_bases_INTERNAL.npz`.
* **Acceptance checks.** CUR ratio ≥ 1 everywhere. Pooling steps give zero angles (H4). The rSVD error is ≥ σ_{k+1}. The planted-data gate (plan §13.2) passes. Recovery uses grouped folds, train only.

## Stage 11: model bake-off at matched width, 24 h landmark rows (Objective 2 primary; plan §10 steps 1–2; contract "Analysis questions"; H6)

* **Inputs.**
  * 11p labels (G1).
  * Deduplicated `X_z` for A, B and AB levels, restricted to `eligible_24h` rows.
  * Stage 10 bases: top-k SVD scores (train SVD) and k CPQR columns, k ∈ {5, 10, 20, 30, 40, 59}, plus full width.
  * Layer C at 24 h (bins 0–3), added to A+B: k CPQR channels of the stacked (4·n) × q train matrix (primary) or top-k channel-mode SVD scores (alternative), k ≤ q (V23; **no flattening**). This needs the §6.7 model (Q4) and C levels (Q2).
* **Math.**
  * Models:
    * Proportional-odds ridge LR plus an all-threshold variant and a multinomial reference.
    * RF and GB, Frank–Hall, with monotone repair.
    * MLP optional (default omit).
  * Decision rule: median of the predictive distribution.
  * Selection metric: **QWK** on validation, Outcome 1 (V6).
  * Also reported:
    * EMD = Σ_i π̂_i|i − y| as the secondary metric
    * RPS
    * MAE
    * macro and per-threshold AUROC/AUPRC
    * Somers' D
    * calibration by threshold
    * mortality AUROC
  * Winner: best validation QWK, with the simpler model winning if the paired-bootstrap CI includes 0 (LR < RF < GB).
  * Bootstrap: paired patient-level, B_eval = 1,000.
* **Outputs.** `stage11_validation_scores_TEAM.csv` (metrics only), `stage11_winner.json`, audit. Team-internal: fitted models and validation predictions.
* **Acceptance checks.** Nothing is fit on validation. The same rows and k grid apply at every level. Class weighting is declared. Validation class tables pass the counts gate (<10 suppressed). No test row is touched (hash of the split mask is logged).

## Stage 12: tuning and calibration (plan §10 steps 3–4)

* **Inputs.** The stage 11 winner, the same rows.
* **Math.** Deeper tuning of the winner only: validation-only tuning or nested CV grouped by subject (Q5), with a declared budget. Calibration per cumulative threshold (isotonic/Platt or ordinal temperature) **on validation only**, with monotone cumulative probabilities enforced.
* **Outputs.** The tuned model, calibrators (team-internal), `stage12_tuning_TEAM.csv`, and validation calibration summaries.
* **Acceptance checks.** The budget is logged. No test access. Monotonicity holds.

## Stage 13: attribution stability (criterion 5; plan §11; H7)

* **Inputs.** Ordinal LR and the tabular winner at fixed hyperparameters; Outcome 1 (Outcome 2 secondary); 24 h rows.
* **Math.**
  * Refit on B subject-bootstrap resamples of train (plan: B = 200).
  * Target: E[Y|x] = Σ_j P(Y > j|x), with Frank–Hall SHAP summed over thresholds.
  * Global importance: mean |SHAP| on a fixed validation sample.
  * Stability:
    * Kendall τ over refit pairs (median, IQR)
    * within-group credit split over the pre-registered groups (z-score / flag / organ block; min / median / max / last; Cr_max / AKI; the φ2-dup copies)
    * interventional vs conditional (tree path-dependent) SHAP gap
* **Outputs.** `stage13_tau_TEAM.csv`, `stage13_credit_split_TEAM.csv`, `stage13_shap_gap_TEAM.csv` (aggregates by column name; no per-row SHAP).
* **Acceptance checks.** The planted redundant-group gate (plan §13.2) passes. The validation sample is fixed and seeded.

## Stage 14: single test scoring, Layers A and B (plan §10 step 5; contract "Experimental principles": test once)

* **Inputs.** The frozen 24 h matched-width winners and their calibrators; test rows.
* **Math.** All §10 metrics with paired patient-level bootstrap 95% intervals (B_eval = 1,000). **Family (a)** Holm step-down on cluster-bootstrap p-values of the H6 adjacent-level comparisons at each k, Layer C at 24 h included, α = 0.05 (V25, V26). For equivalence, use the larger one-sided p against ±δ.
* **Outputs.** `stage14_test_TEAM.csv`, the Holm table, the prediction-file SHA-256 (the file itself stays team-internal), and a test-use log entry.
* **Acceptance checks.** It runs once; a second run refuses (a lock in the test-use log). Every choice was frozen before the run (hashes of 11/12 outputs are pinned). Late-landmark and class cells pass the counts gate.

## Stage 15: Layer C levels (build; plan §6.5, §6.8; contract "Layer C")

* **Inputs.** The Stage 3 v1.0.4 Layer C grid with `mask_valid`. Stage 08 Layer C npz files supply the base representation only.
* **Content.**
  * c0: per-bin native summaries, masks, carried-forward flags and time-since-last.
  * c1: bin z-scores and flags, past-only 4-bin rolling SD, AKI/KDIGO stage, 13 drug groups, M1, and surgery in the bin of T.
  * c2s / c2d: organ blocks.
  * c2p: per-bin pSOFA.
  * Then the per-landmark train-only fit, as in Stage 08 (D13).
* **Acceptance checks.** §6.8 checks 1–15: shape; padding rule; LOCF caps; started ⇒ active; landmark counts reproduce internally with pass/fail only; no post-L information; the as_of_hours check; and the C(12) surgery rule.
* **Status: not built** (G2). Real Layer C at present = the Stage 08 base tensor.

## Stage 15i: imputation arms (plan §6.6; contract B26)

* **Arms.**
  * (r): train median (exists).
  * (a): age-specific pediatric normal fill.
  * (b): low-rank completion on the train landmark prefix, value columns only, folded in for validation/test.
  * (c): optional bounded completion.
* **Evaluation.** Masked-known-value RMSE (reported, not a criterion) and the plausibility criterion (criterion 6).
* **Acceptance checks.** Train-only, prefix-only checks. The planted-missingness and planted look-ahead gates (§13.3) pass.
* **Open parameters.** Many remain open (plan §18.1 "Imputation arms"). **Recommend deferring** (§6).

## Stage 16: Layer C tensor SVD (Objective 3, extension; plan §12.3; contract "Objective 3"; H13, H15, H16, H17; criteria T1–T6)

* **Inputs.** Train landmark tensors X_L ∈ R^(n_L × L/6 × q_ℓ) from `stage08_layer_c_L<L>.npz`, at core landmarks 24/48/72/120/168/240/336 (all 14 for stability). The levels are raw (`X_raw_z` with NaN and masked weighted ALS), normalized (`X_filled_z` + imputed flags) and engineered. The engineered level needs stage 15; tonight it is limited to "normalized + masks/flags as extra feature-mode channels" from the Stage 08 arrays.
* **Math.**
  * Truncated HOSVD: mode-n spectra plus the core tensor.
  * Tucker by HOOI on the grid r₁ ∈ {5, 10, 20, 40}, r₂ ≤ b_L, r₃ ≤ q_ℓ.
  * CP-ALS for R = 1…15, with 10 random starts + an HOSVD start.
  * CORCONDIA, and Tucker congruence across starts and B = 50 subject bootstraps.
  * Projection of validation onto fixed train bin/feature factors; nothing is refit on validation or test.
  * Nulls: channel-wise patient permutation and per-bin patient shuffle.
  * Principal angles of mode-2/mode-3 subspaces across consecutive landmarks.
  * Organ-block alignment against a label-permutation null.
  * H17 padding probe on the full grid (train): zero-padded vs masked; R² for the number of valid bins. Descriptive only. **This needs the full 56-bin grid, which only Stage 3 has.**
  * Flattened baseline: the effective-rank curve of n_L × (b_L q) as bins are added, and CPQR/DEIM of (bin, channel) pairs. It is **labeled "flattened baseline"** and never chooses a rank, level or model.
* **Rules.** Never unfold C except inside HOSVD/HOOI. Honor `mask_valid`.
* **Outputs (TEAM).**
  * Mode spectra
  * Tucker error surfaces
  * CP error / CORCONDIA / congruence tables
  * Angle tables
  * Factor loadings over channels and bins
  * Probe R²
  * The labeled baseline
  * **No per-patient loadings.**
* **Acceptance checks (synthetic gate of §12.3, required before real).**
  * Planted CP/Tucker ranks are recovered.
  * The `4_answer_key` mode-2 angles reproduce.
  * Extreme padding values change nothing.

## Stage 17: landmark models, A+B vs A+B+C (Objective 2 co-primary; plan §10 analysis 2, §6.7; contract "Analysis questions"; H9)

* **Inputs.** 11p labels per landmark, deduplicated A+B (the winning tabular model on the chosen AB level, refit per landmark), and Layer C prefix bins 0…L/6 − 1 from `stage08_layer_c_L<L>.npz`.
* **Math.**
  * A+B+C model: the §6.7 static-plus-sequential model. Layer A context is given at **every time step**. The Layer A baseline risk (ordinal LR on A, cross-fitted on train) enters only as an offset in the cumulative-link head: thresholds γ_j − [s_static + g(x_≤t)]. PRISM III stays out of the sequence inputs.
  * Both models are refit at each of the 14 landmarks, one row per eligible encounter.
  * Validation QWK as the primary metric (EMD secondary) and Outcome 2 as the key secondary.
  * The stacked (encounter, landmark) model is a sensitivity analysis only (B6/B7).
  * Calibration on validation.
* **Outputs.** `stage17_validation_TEAM.csv` per landmark × model, plus frozen models (team-internal).
* **Acceptance checks.** Only prefix bins are visible. No padding enters. Train-only scalers per L. Per-landmark refit. Counts gate on any class table.

## Stage 18: sequence attribution and single test scoring of landmark models (plan §6.7 "Attribution", §10 step 5; H9 decision)

* **Inputs.** Frozen stage 17 models and test rows at each landmark.
* **Math.**
  * Test scoring once.
  * **Family (b):** Holm step-down over the seven core-landmark ΔQWK (A+B+C − A+B) cluster-bootstrap p-values for ΔQWK ≤ 0, α = 0.05.
  * The 14-landmark added-value curve (ΔQWK and Δmacro-AUROC) with bootstrap 95% intervals, **labeled descriptive, no p-values** (V25, V27).
  * Sequence attribution: TreeSHAP for A+B; GradientSHAP/DeepSHAP for sequence models with a permutation cross-check. Kendall τ at three granularities (bin × channel, channel, time block/organ system) over subject-bootstrap refits (reduced neural budget allowed by §6.7).
* **Outputs.** `stage18_test_landmarks_TEAM.csv`, `stage18_holm_core_TEAM.csv`, `stage18_curve_TEAM.csv`, `stage18_seq_tau_TEAM.csv`, and a test-use log entry.
* **Acceptance checks.** It runs once. Late-landmark test cells pass the counts gate; the plan notes small late-landmark test counts and a readmission cell under 10 at 336 h, so **Outcome 2 at late landmarks will need suppression**. Curve intervals are labeled descriptive.

# 5. Copilot run grouping for tonight (Nov 18)

The slip priority is Objective 2 primary, then Objective 1, then Objective 3 (V15; contract "Slip priority"). Every run follows the standing protocol:

* pause cloud file syncing first;
* run tests, then the precheck, then the build;
* every command starts with `Set-Location` to the project root;
* scripts run unedited; stop on the first failure;
* Copilot opens only the audit md.

I write and synthetic-test each package on the box (Stage 08 synthetic fixture kept at `stage09_18/_synthetic_stage08_fixture_v121/`).

| Run | Contents | Why grouped | Est. laptop time |
|---|---|---|---|
| **R1** | 09p dedup + **09** + **10** (A, B, AB, b2d per Q1; Layer C per-bin/stacked spectra) | Label-free linear algebra on the same npz files; produces the V_k / CPQR bases 11 and 17 need | 10–25 min (B = 200 bootstraps, 200 permutations, 50 CUR repeats on p ≤ ~400) |
| **R2** | 11p labels + **11** + **12** + **17** (validation only; no test access) | Shares labels, bases and the A+B winner; 17 reuses 11's tabular winner | 30–90 min (sequence models dominate; GRU only per Q4) |
| **R3** (after Vinai reviews R2 validation results) | **14** + **18** test-once (families (a) and (b), curve), then **13** (reduced B) and **16** (core landmarks, normalized + raw levels) | The test run must follow frozen choices. 13 and 16 are lowest priority and can slip to Nov 19–20 without hurting the primary results | 20–60 min |

**Deferred (recommendation):**

* stage 15 C levels and 15i arms, to after the report (the report names the Layer C "level" as the Stage 08 base representation);
* H11 microbiology (M2/M3 not built);
* ordinal8 and B-series sensitivities other than Outcome 2.

# 6. Open decisions for Vinai (short list; one question at a time, each with a default)

1. **Q1. D15 vs φ2-dup.** Should D15 exempt b2d's by-design `__dup__` copies? *Recommended:* yes. Apply D15 to every other level and to AB/A+B. Keep b2d as built (it exists to measure duplication; H1, H7). Stage 09 also reports every level pre-dedup for the H1 rank identity. Correct the EDA v1.1 §9 row accordingly.
2. **Q2. Layer C levels tonight.** *Recommended:* no stage 15 tonight. Use the Stage 08 base tensor as the single Layer C representation for 11 (C at 24 h) and 17. Report the C-level comparison (and H15) as deferred future work.
3. **Q3. Thresholds without a synthetic run.** *Recommended:* freeze the bracketed plan values (κ₂ < 100, p/3, +2, 80%, η ≥ 30 / 0.5, gap 0.05 / 0.3, congruence 0.85, CORCONDIA 80%) in the specs **before** R1, and set the unvalued ones as δ = δ_a = 0.02 (QWK / AUROC equivalence margin), δ_C = 0.01 (Layer C gain), ε = 0.01. Then log the deviation from §13.3 ("thresholds from the synthetic run") in the project log.
4. **Q4. Sequence model for A+B+C.** *Recommended:* one small GRU (static Layer A concatenated at every step, baseline-risk offset, CORAL head, fixed small grid), if the laptop has PyTorch (package check via machine Shell). TCN/Transformer are deferred and documented as a reduced bake-off (Shared guide risk row "use the simplest passing sequence model").
5. **Q5. Compute budgets.** *Recommended:*
   * validation-only tuning (no nested CV);
   * B_eval = 1,000;
   * spectral B = 200;
   * SHAP refits B = 50 (stage 13) and B = 20 for neural attribution (stage 18), logged as reduced from the plan's 200.
6. **Q6. Whose code runs tonight.** The plan has teammates writing 09–13, 17 and 18 (guides target Nov 18–20). *Recommended:* I write the scripts tonight to hit the deadline. Habib, Vinay and Yashwant cross-check them on the synthetic sandbox and keep their own implementations as independent replications (their write-up ownership is unchanged).

# 7. Teammates' synthetic-sandbox work (synthetic data only; never real data or `*_INTERNAL` files)

* **Vinay (09):** cross-check 09/09p on the synthetic fixture and `synthetic_planted_v1`. Confirm planted rank recovery, the Belsley planted-range test, the decay-family choice and Davis–Kahan.
* **Yashwant (10):** cross-check CUR/rSVD/angles/recovery on the planted data. Confirm the V_k/CPQR export matches his matched-width helper (I2). Cross-check 13.
* **Habib (11, 12, 13, 17, 18):** cross-check the ordinal metrics (QWK, EMD, RPS), Frank–Hall, Holm, and the GRU/offset head on the synthetic layers (`6_tensor_ready_layers`, dropping P/F, S/F, FiO2 and `psofa_resp`, V11). Also cross-check 09 decay/bootstrap.
* **Vinai (PI):** every real run (R1–R3); 14, 15, 15i, 16.

# 8. Contradictions and inconsistencies found

1. **D15 vs φ2-dup (plan §12.2 H1/H7, §11).** The EDA v1.1 §9 table lists the 65 φ2-dup copies under the D15 collapse; the plan needs them kept (Q1).
2. **Real runs tonight vs team guides and §13.3.** The guides package 09/10 for the real run on Nov 19 and 11–18 on Nov 19–20. Plan §13.3 requires synthetic gates and thresholds frozen from synthetic runs before any real run. Tonight's deadline makes both impossible as written (Q3, Q6).
3. **No outcome-label stage.** Plan §13.2 has no step that builds Outcome 1/2 per landmark, yet 11–18 need them (G1).
4. **Stage 15 listed as done-in-order but not built.** Contract "Execution workflow" step 4 and the Shared guide ("Wed Nov 18: 06/07/08/15/15i real") assume stage 15 and 15i ran with 06–08. The 06–08 spec builds no C levels; the Stage 08 Layer C npz is a single base representation (G2).
5. **Stage 08 vs plan §7 standardization note.** Plan §6.7/§12.3 leave "per-bin or pooled per channel" open. Stage 06–08 decided D13 (pooled per channel over observed train cells). This is consistent, but the plan text still shows it as open.
6. **Microbiology.** H11 (M1 vs M3 vs M2) is a stated hypothesis (plan §12.2), but Stage 3 has no organism fields (spec §7). H11 cannot be tested.
7. **Arm (b) flattening.** Plan §6.6 completes the "flattened n × (bq) landmark-prefix matrix" for imputation. This sits uneasily with the V9 no-flattening rule, which only exempts Objective 3's labeled baseline. It is moot if 15i is deferred.
8. **Cohort counts in the plan/contract.** Plan §7, §8, §10, §12.3 and the contract quote cohort/split/landmark N. Stage 09–18 outputs must not reprint them (current disclosure rule). Scripts check them internally and print pass/fail only.
9. **Plan §13.2 stage-map row "08x / 15".** It says 08x is absorbed by Stage 3 but keeps a `15_build_ladder_layer_c.py` row for C levels. The contract says the same. Not a contradiction, but readers may think C levels exist.

# 9. Decisions recorded (Vinai, Nov 18, 2026, 4:25 PM CT; all six recommendations accepted as proposed)

| # | Decision | Applies to |
|---|---|---|
| D16 | D15 dedup applies to every level and to the AB / A+B concatenations and the Layer C channel axis, **except** the by-design `__dup__` copies of φ2-dup (b2d), which are never dropped. Stage 09 reports every level both **before and after** dedup (the H1 rank identity uses both). EDA v1.1 §9 row to be corrected. | 09p, 09, 10, 11, 13, 17 |
| D17 | Stage 15 (Layer C levels c0/c1/c2) is skipped for the report. The Stage 08 base Layer C tensor (`stage08_layer_c_L<L>.npz`, `X_filled_z` / `X_raw_z`) is the single Layer C representation for 09 (per-bin/stacked), 11 (C at 24 h), 16 and 17. The C-level comparison and H15 are future work. | 09, 11, 16, 17 |
| D18 | Thresholds frozen now, before any real stage 09–18 run: the bracketed plan values (κ₂ < 100; null count < p/3 and rise ≤ 2; BIC family ≥ 80 %; Belsley η ≥ 30 with ≥ 2 proportions > 0.5; recovery gap < 0.05 for z-scores and ≥ 0.3 for flags/stage; congruence ≥ 0.85; CORCONDIA ≥ 80 %), plus δ = δ_a = 0.02 (QWK / AUROC equivalence), δ_C = 0.01 (Layer C gain), ε = 0.01 (frozen-rule). **Departure from plan §13.3** ("thresholds fixed from the synthetic runs"): logged in the project log. | all |
| D19 | A+B+C sequence model: one small GRU (Layer A static input concatenated at every step, Layer A baseline-risk offset, CORAL-style cumulative-link head, fixed small grid), PyTorch CPU (laptop has torch 2.10.0+cpu). TCN and Transformer deferred and documented as a reduced bake-off. | 17, 18 |
| D20 | Budgets: validation-only tuning (no nested CV); B_eval = 1,000; spectral subject bootstrap B = 200; SHAP refits B = 50 (stage 13) and B = 20 for neural attribution (stage 18), logged as reduced from the plan's 200. | 09–18 |
| D21 | Grok Bot writes the stage 09–18 scripts tonight; Habib, Vinay and Yashwant cross-check them on the synthetic sandbox and keep their own implementations as independent replications; write-up ownership unchanged. | all |

**Run 1 build notes (Nov 18).** Laptop packages confirmed by Vinai: Python 3.14, numpy 2.4.2, pandas 3.0.1, scipy 1.17.0, matplotlib 3.10.8, scikit-learn 1.8.0 (used only for the stage 10 recovery models), statsmodels 0.14.6, torch 2.10.0+cpu (Run 2). Run 1 runtime-driven reductions (logged in the Run 1 audit): permutation nulls 100 draws (plan 200); random CUR rules 50 repeats in Frobenius norm, spectral norm on the first 10; recovery targets capped per (level, family) on a train subsample; kNN mutual information deferred; the per-bin patient-shuffle null leaves the stacked channel spectrum unchanged and moves to stage 16.

**Run 1 build, final (Nov 18, 2026, v1.0.0, script SHA-256 5c501c50…6724f).** Two runtime changes were added after the first synthetic runs. (1) The between-level angle bootstrap uses the first 50 of the 200 shared replicates; spectra and Davis–Kahan use all 200, computed by QR then the SVD of R. (2) Residual spectral norms (Davis–Kahan ‖E‖₂, CUR, rSVD) come from the top eigenvalue of the small Gram matrix; every reported spectrum still comes from an SVD. BLAS threads are capped at min(4, cores/2); on the box, 8 threads made the bootstrap about 6× slower. The release gate now also catches sizes that sit in CSV cells next to commas. It skips structural integer columns, never matches inside a hex digest or the digits of a fraction, and matches identifiers only as whole tokens.

## Run 1 build note — v1.0.2 (Nov 18, 2026, release-gate fix)
Gate only: identifier scan uses the structural/non-structural column split (adds `weights`, `eta`); integer-valued floats are not identifier candidates; md/json exempt only audit-structural, script-fixed-text and timestamp integers; all problems collected then one refusal naming file / column / check, never values. Numerics byte-identical to v1.0.1 on the synthetic fixture. Tests 39 (Windows 38 + 1 skipped). Small-id synthetic e2e 375 s, gate ok. Package `stage09_10_r1_v1_0_2_pkg.zip`.
