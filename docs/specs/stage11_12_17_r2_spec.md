---
title: "Run 2 — label build 11p and stages 11, 12 and 17 (validation only)"
subtitle: "stage11_12_17_r2_spec v1.0.0 · script r2_11p_11_12_17_models v1.0.0 · plan v13 / contract v13 · PIC Foundations · Nov 18, 2026"
---

# 1. Summary

Run 2 of the stage 09–18 plan (`stage09_18_plan.md`, decisions D16–D21) builds the outcome labels (11p, gap G1), runs the stage 11 matched-width bake-off at the 24 h landmark, tunes and calibrates the winner (stage 12), and fits the stage 17 landmark models (A+B against A+B+C at all 14 landmarks). **Only the train and validation splits are used: test rows are dropped when the Stage 08 files are loaded and are never scored** (test scoring is Run 3, stages 14 and 18). It is one script, `r2_11p_11_12_17_models.py`, plus its tests, with the Run 1 v1.0.2 conventions:

* a hash-gated precheck, now pinned to the Stage 06–08 manifest, the canonical file **and the Run 1 manifest**; every Run 1 file it uses is verified against that manifest;
* a pending folder published atomically, a fail-closed release gate (v1.0.2 rules: structural-column whitelist, integer-cell identifier rule, collect-all reporting without values), an aggregate-only audit and a manifest;
* TEAM and INTERNAL outputs; library caches and temp files redirected into `data\processed\ladder\_cache_stage11_12_17_r2\`;
* guarded heavy imports at the start (scikit-learn, matplotlib, torch); unexpected errors print frames only (`basename:line:function`);
* a progress line or heartbeat at least every 30 s and a hard limit `--max-seconds 3600`;
* exit codes 0 ok, 2 input stop (nothing written), 3 build or gate failure (nothing published).

Seeds are deterministic (seed 6310, keyed per task); BLAS and torch threads are capped at min(4, cores/2). Two runs on the same inputs give byte-identical TEAM CSVs.

**Runtime.** estimated about 26–32 min on the laptop for the real data, so allow 30–40 min (Run 1 ratios applied: real/x3 = 2029/780 to the x3 fixture's 726 s, about 3× the base rows and close to the real size, gives about 32 min; real/base = 2029/335 to the base fixture's 256 s gives about 26 min); the hard limit is 60 min (`--max-seconds 3600`). Box verification: base fixture 256 s, x3 fixture 726 s, small-id fixture (ids 1..N) 297 s, all exit 0 with release gate ok; Python 3.14.10 venv with the laptop's package versions. Tests: 39 passed in 136 s on the box (Python 3.14.10, laptop package versions).

# 2. Inputs and hash gates (precheck; nothing is parsed before these pass)

| input | path (laptop, relative to the project root) | gate |
|---|---|---|
| Stage 06–08 manifest | `data\processed\ladder\stage06_08\stage06_08_manifest.json` | `--expect-stage06-08-manifest-sha256 d8efd1fe…71d9`; builder v1.2.1 `f55cdaf5…daec8e7`, gate ok, all checks pass, 42 files, each hashed |
| Run 1 manifest | `data\processed\ladder\stage09_10_r1\stage09_10_r1_manifest.json` | `--expect-run1-manifest-sha256` and the pin `703d7b46…7f61`; script `r1_09p_09_10_spectral_lowrank v1.0.2` with SHA-256 `707dafc8…fea2f`; gate ok; all checks pass; built from the same Stage 06–08 manifest; **every file it lists is hashed** |
| Run 1 bases | `stage10_matched_width_bases_INTERNAL.npz` | manifest entry must equal the pin `6b52a538…6933` |
| canonical | `data\interim\canonical_icu_encounters.csv` | pin `e5feeb22…6418` and the hash in the Stage 06–08 manifest |
| harmonization | `data\interim\encounter_harmonization_fields.csv` | pin `ffd1b8ae…d6d` (plan v13 Table 3.1) |
| subject split | `..\PIC_Analysis_3_0\data\processed\subject_splits.csv` | pin `60196792…e40e31` |
| raw ICUSTAYS | `..\PICDataset\ICUSTAYS.csv` (usecols only) | pin `449119e6…ea1b`; row count checked internally (never printed) |
| raw ADMISSIONS | `..\PICDataset\ADMISSIONS.csv` (usecols only) | pin `1451ae8e…e1ae`; row count checked internally |

The output folder `data\processed\ladder\stage11_12_17_r2\` must be absent or empty and no `stage11_12_17_r2.pending-*` folder may exist. Any path containing `_raw_LOCAL` is refused. The valid-stay count (OUTTIME > INTIME) is checked against Stage 1 internally and never printed.

# 3. Label build 11p (contract v13 §2.3; band-support audit spec v2.3; 3.0 contract §5.5)

All labels are built per encounter for the whole cohort (so the class tables and checks see every split) and are written only to `stage11p_labels_landmark_INTERNAL.npz`. No per-patient label reaches a TEAM file.

* **Eligibility at landmark L:** OUT > t0 + L (strict), with t0 and OUT from the harmonization file (OUT falls back to `OUTTIME`). It must reproduce the Stage 08 `eligible_24h` flag and every Layer C landmark set exactly.
* **Outcome 1 (primary, K = 9; remaining ICU time):** died (canonical `Died`) → 8. Survivors: r = OUT − (t0 + L) > 0, right-closed bands (0,24], (24,48], (48,72], (72,120], (120,168], (168,240], (240,336], >336 h → 0…7.
* **Outcome 2 (key secondary, K = 5):** died → 4. Index HADM: the valid raw stay whose OUTTIME equals OUT with HADM in `hadm_id_list` (tie: the latest DISCHTIME; none → unassignable `index_hadm_unmatched`). Readmission: the earliest valid stay of the same subject and index HADM with INTIME > OUT; a gap ≤ 24 h stops the run (`return_within_24h`, which harmonization must have merged); a gap > 24 h → 3. Otherwise p = DISCHTIME − OUT in days: < 7 → 0, 7 to < 14 → 1, ≥ 14 → 2 (`dischtime_missing` and `dischtime_before_out` are unassignable). Policy `keep_both` for a readmission that is itself an index encounter; both must sit in the same split.
* **ordinal8 (inherited, 24/48/72 h only):** 7 died; ICU < 72 h: hospital < 168 h → 0, < 336 h → 1, else 2; ICU 72 to < 168 h: hospital < 336 h → 3, else 4; ICU 168–336 h (336 inclusive) → 5; ICU > 336 h → 6; from canonical `icu_los_hours` and `hospital_los_hours`.
* **Mortality:** canonical `Died` (reported as mortality AUROC from P(Y = died)).
* **Class table** (`stage11p_label_classes_TEAM.csv`): within-row percentages (one decimal) per (outcome, landmark, split) for train and validation; **no counts, no row totals and no N anywhere**. Cells under 10 are written `<10`. Complementary suppression makes each row's suppressed set empty or at least two cells whose combined count is at least 10, and the death cells (Outcome 1 class 8 = Outcome 2 class 4 = ordinal8 class 7) are suppressed together across outcomes at the same landmark and split. Unassignable and ambiguous statuses are reported only as `none`, `<10` or `10 or more`.

# 4. Stage 11 (bake-off at matched width, 24 h rows)

* **Rows:** the eligible 24 h train and validation rows; the same rows for every matrix (checked).
* **Matrices (13):** the eight levels a0…b2p and the five A+B pairs (a0+b0, a1+b1, a2+b2s, a2+b2d, a2+b2p), deduplicated exactly as in Run 1: the columns are taken by name from the Run 1 bases file, which must resolve against the Stage 08 levels (D16).
* **Representations:** full width; top-k SVD scores X·V_k with the Run 1 train SVD; k CPQR columns in the Run 1 order; k ∈ {5, 10, 20, 30, 40, 59}, k < p.
* **Models:** proportional-odds ridge LR (λ = 1) and the all-threshold variant (own L-BFGS with analytic gradients), the multinomial reference (scikit-learn; cannot win), and Frank–Hall RF (100 trees, min leaf 5, sqrt features, half-sample bootstrap) and GB (HistGradientBoosting, 100 iterations, learning rate 0.1, 15 leaves) with monotone repair by rearrangement. Trees are fitted at k ∈ {20, 59} and full width; linear models at every k. No class weighting.
* **Decision rule and metrics:** the median of the predictive distribution; QWK (selection), EMD (secondary), RPS, MAE, macro and per-threshold AUROC / AUPRC, Somers' D, ECE by threshold, mortality AUROC.
* **Bootstrap:** paired subject-cluster bootstrap on validation, B = 1,000 (the same replicate weights for every model on the same rows).
* **Winner:** within each family (LR, RF, GB) the best validation QWK on the A+B matrices; the simplest family (LR < RF < GB) whose paired 95% interval of (best − family best) contains 0 wins. Its matrix and representation are used by stages 12 and 17.
* **H6 (descriptive on validation):** adjacent-level ΔQWK at matched width with 95% intervals and a verdict (equivalent within δ = 0.02, higher, lower, inconclusive).
* **Secondary at 24 h:** Outcome 2 and ordinal8 with the bake-off models on the chosen representation.
* **Layer C at 24 h (bins 0–3):** the D19 GRU on A+B (chosen representation as static context, cross-fitted Layer A offset) with all deduplicated channels, k CPQR channels (primary) or top-k channel-mode SVD scores (alternative), k ≤ q; ΔQWK against the tuned A+B model.

# 5. Stage 12 (tuning and calibration, validation only)

* Tuning of the winner only, on validation (D20; no nested CV): LR λ ∈ {0.01, 0.1, 1, 10, 100, 1000}; GB learning rate {0.05, 0.1} × leaves {7, 15, 31} × iterations {100, 200}; RF min leaf {1, 5, 20} × features {sqrt, 0.2}. The budget (grid size) is logged.
* Calibration: isotonic regression per cumulative threshold P(Y > j), with monotone cumulative probabilities by rearrangement (checked). The calibrators are fitted on validation; the reported post-calibration metrics are 5-fold subject-grouped cross-fits on validation.

# 6. Stage 17 (landmark models, A+B vs A+B+C; validation only)

At each of the 14 landmarks, on the eligible train and validation rows, every model is refit:

* **A:** ordinal LR on the A part of the chosen A+B matrix (the Layer A baseline).
* **A+B:** the tuned stage 12 winner on the chosen representation.
* **A+B+C:** the D19 GRU: Layer C prefix bins 0…L/6 − 1 (value, missing and carried-forward of the Run 1 deduplicated channels, plus the bin index), the A+B static vector at every step (linear embedding), the Layer A baseline score as an offset (cross-fitted on train with 5 subject-grouped folds) in a cumulative-link head γ_j − [s_static + g(x_≤t)], thresholds initialized at the ordinal LR thresholds; CORAL binary cross-entropy, Adam, early stopping on a 10% subject holdout; hidden size {32, 64} chosen on validation QWK.
* **C only:** the same GRU without static input or offset (hidden 32), Outcome 1 only.
* ΔQWK (A+B+C − A+B) with paired 95% intervals and P(Δ ≤ 0); H9 is descriptive on validation (δ_C = 0.01); inference is stage 14/18 on test.
* Outcome 2 at the core landmarks (24, 48, 72, 120, 168, 240, 336 h) and ordinal8 at 24/48/72 h with A, A+B and A+B+C.
* Calibration on validation for A+B and A+B+C (Outcome 1), as in stage 12.

Only prefix bins are read (checked); eligibility guarantees no padding; the Stage 08 tensors and levels are standardized with train-only scalers.

# 7. Outputs and disclosure

**TEAM (shareable after review):** `stage11p_label_classes_TEAM.csv`, `stage11_validation_scores_TEAM.csv`, `stage11_h6_adjacent_TEAM.csv`, `stage11_winner_TEAM.csv`, `stage11_layer_c_24h_TEAM.csv`, `stage11_secondary_TEAM.csv`, `stage12_tuning_TEAM.csv`, `stage12_calibration_TEAM.csv`, `stage17_validation_TEAM.csv`, `stage17_contrast_TEAM.csv`, `stage17_calibration_TEAM.csv`, `fig11_1_matched_width_qwk_TEAM.png`, `fig17_1_delta_qwk_by_landmark_TEAM.png`, `stage11_12_17_r2_audit_TEAM.json` / `.md`, and `stage11_12_17_r2_manifest.json`.

**INTERNAL (team-internal; never opened, pasted or shared):** `stage11p_labels_landmark_INTERNAL.npz`, `stage11_12_17_validation_predictions_INTERNAL.npz`, `stage12_tuned_model_calibrators_INTERNAL.pkl`, `stage17_models_INTERNAL.pt`, `stage17_tabular_models_INTERNAL.pkl`.

**Release gate (fail closed, Run 1 v1.0.2 rules).** Every TEAM text file is re-read from disk. Forbidden sizes (≥ 100): cohort, subjects, every (landmark, split) eligible set and its subjects, the assignable Outcome 2 / ordinal8 sets and **every class count**. Identifiers: all encounter and subject ids. The structural columns (`k`, `p`, `landmark`, `class`, `threshold`, `hidden`, `epochs_trained`, `config_id` and the fixed hyperparameter columns) are skipped by both scans; a numeric cell is an identifier candidate only when written as an integer; the exact suppression marker `<10` is the script's constant and is skipped. Audit numbers must sit under whitelisted keys. All problems are collected and the refusal lists file, location and check, never the value.

# 8. Decisions and logged deviations

* **D16:** D15 exact-duplicate collapse on every level and A+B concatenation (Run 1 dedup, reused by name), except the by-design __dup__ copies of b2d
* **D17:** the Stage 08 base Layer C tensor is the single Layer C representation (C at 24 h in stage 11; stage 17)
* **D18:** thresholds frozen before real runs: delta = delta_a = 0.02 (QWK equivalence), delta_C = 0.01 (Layer C gain), epsilon = 0.01
* **D19:** A+B+C sequence model: one small GRU, Layer A+B static context at every step, Layer A baseline-risk offset (ordinal LR on A, cross-fitted on train) in a CORAL-style cumulative-link head, fixed small grid; TCN and Transformer deferred (reduced bake-off)
* **D20:** validation-only tuning (no nested CV); B_eval = 1000 paired subject-cluster bootstrap replicates
* **D21:** scripts written by the assistant; teammates cross-check them on the synthetic sandbox

**Deviations (also in the audit):**

1. no lightgbm / xgboost / shap on the laptop: gradient boosting is scikit-learn HistGradientBoostingClassifier and random forest is RandomForestClassifier, both as Frank-Hall threshold models with monotone repair by rearrangement (sorting P(Y > j) in j); no SHAP is computed in Run 2 (attribution is stages 13 and 18, Run 3)
2. proportional-odds ridge LR and the all-threshold variant are implemented directly (L-BFGS with analytic gradients, ridge penalty on the slopes only); the multinomial reference is scikit-learn LogisticRegression
3. stage 11 tree models (RF and GB) are fitted at k in {20, 59} (top-k SVD scores and k CPQR columns) and at full width; the three linear models are fitted at every k of the grid; runtime
4. stage 11 hyperparameters are fixed defaults (light grid deferred to stage 12, which tunes the winner only)
5. the winner is chosen among LR (proportional odds or all-threshold), RF and GB on the A+B levels; the multinomial model is a reference and cannot win
6. ordinal8 is computed from the canonical icu_los_hours and hospital_los_hours with the 3.0 contract 5.5 rule (the inherited label file is not a v13 pinned input)
7. GRU inputs per bin: Stage 08 X_filled_z, missing and carried_forward of the Run 1 deduplicated channels plus the bin index; time since last measurement and the imputed flags are not used
8. GRU early stopping on a 10% subject holdout inside train; the grid (hidden 32 or 64) is chosen on validation QWK
9. stage 17 sequence-only comparator: GRU with hidden 32 and no static input or offset (Outcome 1 only); Outcome 2 at the core landmarks and ordinal8 at 24, 48 and 72 h use the A, A+B and A+B+C models, the GRU with the hidden size chosen for Outcome 1 at the same landmark
10. the stage 11 winner is written as stage11_winner_TEAM.csv (plan: stage11_winner.json) so the release gate scans every cell
11. the stacked (encounter, landmark) sensitivity model is deferred to Run 3
12. calibration: isotonic regression per cumulative threshold fitted on validation, monotone by rearrangement; the reported post-calibration metrics are 5-fold subject-grouped cross-fits on validation
13. class tables give within-row percentages (train and validation), never counts or row totals; cells under 10 and complementary cells are shown as <10
14. (spec note; the audit declares it as `class weighting: none`) no class weighting in the stage 11 light grids, although plan v13 §10 step 1 allowed class weighting for sparse classes; every likelihood and loss is unweighted
15. (spec note) SHAP refits (B = 50, and 20 for neural attribution, D20) belong to stages 13 and 18 in Run 3; without the shap package they will use home-made attributions

# 9. Acceptance checks (all must pass, otherwise exit 3)

* `11p_split_matches_split_file`
* `11p_eligible_sets_match_stage08`
* `11p_return_within_24h_zero`
* `11p_readmit_overlap_pairs_same_split`
* `11p_survivor_remaining_time_positive`
* `11p_death_precedence`
* `11p_outcome2_assignable_plus_unassignable_equals_eligible`
* `11p_label_boundary_selftest`
* `11p_class_table_no_suppressed_cell_recoverable`
* `11p_layer_c_rows_have_labels`
* `11_run1_dedup_columns_resolved`
* `11_no_test_rows_used`
* `11_same_rows_every_matrix`
* `12_calibrated_cumulative_probabilities_monotone`
* `12_no_test_access`
* `17_prefix_bins_only`
* `11_multinom_never_wins`
* `11_all_metrics_finite`
* `17_all_landmarks_scored`
* `17_probabilities_valid`

# 10. Copilot steps (Windows, project root `<PROJECT_ROOT>`)

**Rules.**

* Run every command exactly as written. Each one starts with `Set-Location` to the project root, so it can be pasted on its own.
* Do not edit any file. Do not change, add or remove options.
* Stop at the first failure and send back the full console text only. A failure is any `EXIT:` other than 0, a hash that differs from the value shown, or an expected line that does not appear.
* After a failure, do not patch, retry with other options, or delete anything.
* Do not open any input or output data file. The only output file you may open is `data\processed\ladder\stage11_12_17_r2\stage11_12_17_r2_audit_TEAM.md` (Step 4).
* Never open or paste a file whose name contains `_INTERNAL` or `_raw_LOCAL`, and never open a `*.pending-*` folder.
* Steps 1–3 set `PYTHONNOUSERSITE=1` and run `python -s` (the documented invocation since Run 1 v1.0.1; it keeps Python from loading packages from the per-user site-packages folder). Do not drop it.
* **Hang rule:** the script prints a progress line or a heartbeat at least every 30 s. If nothing new appears for 5 minutes, press Ctrl+C once and send the console text.

**Step 0: pause cloud file syncing and check the package.**

1. *Pause cloud file syncing first* (taskbar cloud-sync icon → Settings/gear → Pause syncing → 2 hours). Keep it paused until Step 3 has finished.
2. List leftovers. List only; do not open anything:

```
Set-Location '<PROJECT_ROOT>'; Get-ChildItem data\processed\ladder -Force -Filter 'stage11_12_17_r2*' | Select-Object Name,Mode; Write-Host "LISTED"
```

3. If anything named `stage11_12_17_r2` or `stage11_12_17_r2.pending-*` is listed above `LISTED`, stop and report it. The folder `_cache_stage11_12_17_r2` (library caches and temp files; no data) is not listed by this filter and is not a leftover.
4. **The assistant has already copied the package** into `scripts\stage11_18_r2\`, so do not copy anything. This step is a hash check only. Each hash must equal the value on the line below it; if either differs, stop.

```
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage11_18_r2\r2_11p_11_12_17_models.py -Algorithm SHA256
# b0af376745ee5e828e9fed041eae45918de13a7d33f8edddd36971452dc80a40
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage11_18_r2\test_r2_11p_11_12_17_models.py -Algorithm SHA256
# 3a69f8fe3225a792a274048baa37ec0826c64479572b978172a2e63ddb056fd1
```

**Step 1: tests (synthetic data only, about 3–8 min).** The tests build their own tiny synthetic data in a temporary folder. Expect `39 passed` and `EXIT:0`.

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s -m pytest -q -p no:cacheprovider scripts\stage11_18_r2\test_r2_11p_11_12_17_models.py 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

**Step 2: precheck (hashes only; writes only the cache folder; about 1 min).** Expect `EXIT:0` and these lines (the hashes are printed in this order):

* `result: PRECHECK OK (Stage 06-08 manifest and files, Run 1 manifest and files, canonical, harmonization, split, raw ICUSTAYS / ADMISSIONS, packages); nothing written (library cache folder only)`
* `ADMISSIONS sha256 1451ae8ea0ff463269b348522047a4bc99efd2665c770e9c5cef0ffe2647e1ae`
* `ICUSTAYS sha256 449119e698acaaf948921c34375db535b9fd3f6eb6e996a86069da95e90eea1b`
* `canonical sha256 e5feeb22b5457ab89c4c73235e6a4722d538aec5f88595b6f049004171286418`
* `harmonization sha256 ffd1b8ae6a13e491be2f7a6631eb4efc1f323656795b80b91e105a0644209d6d`
* `split sha256 60196792457af718155a878f25b3d502ae12dbc71ccabb4c2709518c62e40e31`
* `stage06_08_manifest.json sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9`
* `stage09_10_r1_manifest.json sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61`
* `stage10_matched_width_bases_INTERNAL.npz sha256 6b52a538da8d79dba84d5c66aa4f67a4d4395c87c533606968497425ddeb6933`

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s scripts\stage11_18_r2\r2_11p_11_12_17_models.py --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 --expect-run1-manifest-sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61 --precheck-only 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

**Step 3: real run (about 30–40 min; hard limit 60 min).** Cloud file syncing must still be paused. Expect progress lines and the audit text, then a last line `result: RUN 2 (11p, 11, 12, 17) OK; RELEASE GATE OK` and `EXIT:0`.

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s scripts\stage11_18_r2\r2_11p_11_12_17_models.py --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 --expect-run1-manifest-sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

* On `EXIT:2`, nothing was written.
* On `EXIT:3`, the console names any pending folder left behind. Do not open or delete it; report it. An unexpected error prints `frame <file>:<line>:<function>` lines (no data); send them back as they are.
* After Step 3 ends, whatever the result, resume cloud file syncing.

**Step 4: report.** Open only the TEAM audit:

```
Set-Location '<PROJECT_ROOT>'; Get-Content data\processed\ladder\stage11_12_17_r2\stage11_12_17_r2_audit_TEAM.md
```

Send back the full console text of Steps 0–3 and the contents of `stage11_12_17_r2_audit_TEAM.md` only. Do not open, paste or upload any `*_INTERNAL` file or any other output file.

