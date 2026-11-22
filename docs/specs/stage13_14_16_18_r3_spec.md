---
title: "Run 3a — stages 14 and 18: single test scoring of the frozen Run 2 models"
subtitle: "stage13_14_16_18_r3_spec v1.0.0 · script r3a_14_18_test_scoring v1.0.0 · plan v13 / contract v13 · PIC Foundations · Nov 19, 2026"
---

# 1. Summary

Run 3 of the stage 09–18 plan is split in two, as the plan allows when one run would exceed about 45 min:

* **Run 3a (this package):** stage 14 (test scoring of the frozen 24 h models, family (a), H6) and stage 18 test scoring (the frozen landmark models, family (b), H9), plus the stacked (encounter, landmark) sensitivity model.
* **Run 3b (separate package, not in this one):** stage 13 attribution stability (H7), stage 18 sequence attribution and stage 16 Layer C tensor SVD. None of these reads test rows, so Run 3b can run before or after Run 3a.

Run 3a is one script, `r3a_14_18_test_scoring.py`, plus its tests. **Test rows are scored exactly once, with frozen Run 2 choices only.** Nothing is tuned, selected or refitted on test rows. The Run 2 conventions carry over:

* a hash-gated precheck. It runs Run 2's own precheck (Stage 06–08 manifest, Run 1 manifest and bases, canonical, harmonization, split, raw ICUSTAYS / ADMISSIONS). It then checks the **Run 2 manifest** (`--expect-run2-manifest-sha256` and the pin `ec394345…28ed`), and hashes every file the Run 2 manifest lists;
* the Run 2 script is imported only after its SHA-256 equals the pin `b0af3767…0a40`, and no bytecode is written. Run 2 model files are read with a restricted unpickler;
* a pending folder published atomically, and the fail-closed release gate with the Run 2 rules: structural-column whitelist, integer-cell identifier rule, collect-all reporting by file and column without values, and a size scan. Outputs are an aggregate-only audit and a manifest;
* TEAM and INTERNAL outputs. Library caches and temp files go to `data\processed\ladder\_cache_stage14_18_r3a\`;
* a progress line or heartbeat at least every 30 s, and a hard limit `--max-seconds 3600`. Unexpected errors print frames only;
* exit codes: 0 ok; 2 input stop (nothing written); 3 build or gate failure (nothing published);
* `--split-file` and `--raw-dir` point at the copies under `data\interim\r2_inputs\` (the Copilot sandbox reads only inside the project folder). Their hashes must equal the pins and the Run 2 manifest's input hashes.

**Test-use log.** `data\processed\ladder\test_use_log_stage14_18_r3a.json` gets a "started" entry just before the first test row is read. When the run ends, that entry becomes "published" (with the manifest SHA-256) or "failed (nothing published)". A later run refuses to start if a "published" entry exists. Failed attempts do not block a rerun, but the audit counts them. If verification fails, no test row has been read and the log is not touched.

**Runtime.** estimated about 22–25 min on the laptop for the real data, so allow 25–35 min (Run 2 ratios applied: real/x3 = 2837/726 to this run's x3 fixture time of 383 s gives about 25 min; real/base = 2837/256 to the base fixture's 118 s gives about 22 min); the hard limit is 60 min (`--max-seconds 3600`). On the x3 fixture about half the time is the stacked sensitivity (two GRUs, at most 20 epochs each) and a third is stage 14 test scoring (B = 20,000 family (a) bootstrap).

# 2. Inputs and hash gates

| input | path (relative to the project root) | gate |
|---|---|---|
| Run 2 script | `scripts\stage11_18_r2\r2_11p_11_12_17_models.py` | pin `b0af376745ee5e828e9fed041eae45918de13a7d33f8edddd36971452dc80a40` before import |
| Run 2 manifest | `data\processed\ladder\stage11_12_17_r2\stage11_12_17_r2_manifest.json` | `--expect-run2-manifest-sha256` and pin `ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed`. It must come from the pinned script, with gate ok and all checks passing. Its input hashes must equal this run's hashes for Stage 06–08, Run 1, canonical, harmonization, split and raw tables. Every listed file is hashed, including the five INTERNAL files and the three TEAM tables plus the audit JSON that Run 3a reads |
| Stage 06–08, Run 1, canonical, harmonization | as in Run 2 | Run 2's precheck and pins (`d8efd1fe…71d9`, `703d7b46…7f61`, bases `6b52a538…6933`) |
| subject split | `data\interim\r2_inputs\subject_splits.csv` | pin `60196792…e40e31` |
| raw ICUSTAYS / ADMISSIONS | `data\interim\r2_inputs\raw\` | pins `449119e6…ea1b` / `1451ae8e…e1ae` |

The output folder `data\processed\ladder\stage14_18_r3a\` must be absent or empty, and no `stage14_18_r3a.pending-*` folder may exist. Any path containing `_raw_LOCAL` is refused.

# 3. Verification on validation (before any test row is read)

Every check below must pass. Otherwise the run stops with exit 3, no test row has been read, and the test-use log is unchanged.

* **Frozen stage 12 winner:** its validation probabilities reproduce Run 2's saved validation predictions (max abs difference ≤ 1e-4). The isotonic calibrators are monotone.
* **Ordinal LR at every width:** Run 2 saved no stage 11 bake-off models. So the proportional-odds and all-threshold ordinal LR are refitted on train at every (matrix, width) with the frozen Run 2 configuration and seeds. Their validation QWK must reproduce Run 2's `stage11_validation_scores_TEAM.csv` within 2e-6.
* **Layer C 24 h GRU:** Run 2 did not save this model. The validation-selected representation is refitted on train with Run 2's seed and recorded epoch cap (read from the hash-verified Run 2 audit). Its validation QWK must reproduce Run 2 within 0.005.
* **Stage 17 frozen landmark models:** A, A+B, A+B+C and C-only at all 14 landmarks, with both outcomes and ordinal8 where present. Their validation predictions must reproduce Run 2's saved predictions within 1e-4.
* A `TestGuard` object is the only path to test rows. In the verification phase it refuses any test access. Every fit is registered with the splits of its rows, and a fit on any test row is refused. In the scoring phase, each named scoring is allowed once.

# 4. Stage 14: test scoring at the 24 h landmark (family (a), H6)

* Scored on test: the frozen stage 12 winner (uncalibrated and calibrated), the ordinal LR refits at every width, and the Layer C 24 h GRU on A+B.
* Metrics are the §10 set: QWK, EMD, RPS, MAE, macro AUROC / AUPRC, Somers' D, ECE, mortality AUROC and per-threshold metrics. Each has a paired subject-cluster bootstrap 95% interval (B = 1000, seed 6310).
* **Family (a):** H6 TOST on the test ΔQWK for the ten adjacent steps (a0→a1, a1→a2, b0→b1, b1→b2s/b2d/b2p, a0+b0→a1+b1, a1+b1→a2+b2s/b2d/b2p), at every width (full, SVD k grid, CPQR k grid), for the proportional-odds ordinal LR (the Run 2 H6 headline model). Layer C at 24 h added to the frozen A+B winner is also in the family. The equivalence p-value is the larger one-sided bootstrap p against −δ and +δ (δ = 0.02). Holm step-down runs at α = 0.05 over all 131 tests.
* The family (a) p-values come from a dedicated paired subject-cluster bootstrap with **B = 20,000**, generated in chunks from a fixed seed. With 131 tests, Holm's first threshold (0.05/131 ≈ 0.00038) is below 1/1001, the smallest p-value that B = 1000 can give. With B = 1000, no equivalence could ever be declared.
* The all-threshold ordinal LR is test-scored at every width as a descriptive check (no Holm). The H6 AUROC parts are reported with intervals but are not in the Holm family.

# 5. Stage 18: test scoring of the frozen landmark models (family (b), H9)

* At all 14 landmarks: A, A+B, A+B+C and C-only (uncalibrated), plus A+B and A+B+C with the frozen Outcome 1 calibrators. Outcome 1 is scored at all 14 landmarks, Outcome 2 at the seven core landmarks and ordinal8 at 24, 48 and 72 h (descriptive).
* **Family (b), confirmatory (contract v13 "Multiplicity"; plan 12.2 H9):** the one-sided bootstrap p for ΔQWK = QWK(A+B+C) − QWK(A+B) ≤ 0 at the seven contract core landmarks (24, 48, 72, 120, 168, 240, 336 h), with Holm at α = 0.05. H9 is supported at a landmark where Holm rejects and ΔQWK ≥ δ_C = 0.01.
* **Reporting order (Vinai, Nov 19):** every H9 table leads with the later core landmarks: day 3, 5, 7, 10 and 14 (72, 120, 168, 240, 336 h). The 24 h and 48 h rows follow. The 24 h Layer C result was expected to be weak, and the real test of added sequence value is at the later landmarks.
* Outcome 2 (key secondary) uses the same contrast at the core landmarks, with Holm within Outcome 2.
* The 14-landmark added-value curve (ΔQWK, Δmacro AUROC, ΔEMD with bootstrap 95% intervals) is **labeled descriptive**. It has no p-values. In the contrast table, p-values appear only for the core-landmark A+B+C − A+B contrasts.
* **Stacked (encounter, landmark) sensitivity (B6/B7, descriptive only):** one GRU fitted on the train rows of all 14 landmarks. It uses hidden size 32, the Layer C channels shared by every landmark, at most 20 epochs with D19 early stopping, and batches that never mix landmarks. It is trained with per-row and with per-patient weights and scored on test. Each version is compared with the frozen per-landmark A+B, both pooled and per landmark, on the same test rows.

# 6. Outputs and disclosure

TEAM (shareable after the gate):

* `stage14_test_TEAM.csv`, `stage14_h6_test_TEAM.csv`, `stage14_holm_family_a_TEAM.csv`;
* `stage18_test_landmarks_TEAM.csv`, `stage18_contrast_TEAM.csv`, `stage18_holm_core_TEAM.csv`, `stage18_curve_TEAM.csv`, `stage18_stacked_sensitivity_TEAM.csv`;
* `fig14_1_test_h6_matched_width_TEAM.png`, `fig18_1_test_delta_qwk_by_landmark_TEAM.png`;
* `stage14_18_r3a_audit_TEAM.json` / `.md` and `stage14_18_r3a_manifest.json`.

INTERNAL (team-internal, never opened, pasted or shared): `stage14_18_test_predictions_INTERNAL.npz` holds the test probabilities and encounter ids. Only its SHA-256 appears in the manifest.

Counts gate: per-threshold metrics with fewer than 10 rows in either class, and the mortality AUROC with fewer than 10 deaths or survivors, are written as `<10`. No TEAM file contains a patient, row, encounter, subject or eligible count. The size scan refuses every cohort, split, landmark-eligible, class, threshold and death count of 100 or more anywhere in a TEAM file.

# 7. Decisions and logged deviations

Decisions applied: D16–D21 as in Run 2; test once (plan I4); family (a) and family (b) as above.

Deviations from the plan (also in the audit):

1. the Run 2 module is imported (after its SHA-256 matches the pin) for the models, metrics, bootstrap and GRU; frozen model files are read with a restricted unpickler
2. Run 2 saved no stage 11 bake-off models and no Layer C 24 h GRU: the ordinal LR (proportional odds and all-threshold) at every width and the validation-selected Layer C 24 h GRU are refitted on train with the frozen Run 2 configuration and seeds, and must reproduce the Run 2 validation QWK before any test row is read
3. family (a) uses the ordinal LR only (the model fitted at every width); random forest, gradient boosting and the multinomial reference are not refitted at every width for test (runtime); the frozen stage 12 winner is scored
4. family (a) includes one Layer C 24 h representation (the Run 2 validation choice), compared with the frozen uncalibrated A+B winner as in Run 2
5. bootstrap p-values on test use the (count + 1) / (B + 1) rule; Run 2 validation p-values used count / B
6. family (a) TOST p-values come from a dedicated paired subject-cluster bootstrap with B = 20000 (intervals stay at B = 1000): with 131 tests Holm's first threshold (0.05 / 131) is below 1 / 1001, the smallest p-value B = 1000 can give, so no equivalence could ever be declared
7. H6 AUROC parts (macro AUROC and death-threshold AUROC within delta_a) are reported with bootstrap intervals but are not in the Holm family (QWK only, as in the Run 2 verdicts)
8. the H9 frozen-rule check (landmark features under the frozen 3.0 whole-stay rule, epsilon) is not evaluated: no such feature build exists
9. stacked (encounter, landmark) sensitivity: one GRU over all 14 landmarks (hidden 32, batches never mix landmarks; the Layer C channels shared by every landmark's Run 1 channel list; at most 20 epochs with the D19 early stopping, a runtime bound), per-row and per-patient weighted training; compared with the frozen per-landmark A+B on the same test rows; descriptive only
10. counts gate: per-threshold metrics with fewer than 10 rows in either class, and the mortality AUROC with fewer than 10 deaths or survivors, are written as <10
11. (spec note) Run 3 is split into Run 3a (stages 14 and 18 test scoring, this package) and Run 3b (stages 13, 18 attribution and 16), as the plan allows when one run would exceed about 45 min
12. (spec note) the split file and raw ICUSTAYS / ADMISSIONS are read from the hash-identical copies under `data\interim\r2_inputs\` (the Copilot sandbox reads only inside the project folder), exactly as in the Run 2 real run
13. (spec note) no class weighting, as in Run 2: every refit (ordinal LR at every width, Layer C 24 h GRU, stacked sensitivity GRU) is unweighted except the stacked per-patient version, whose weights are per-patient, not per-class

# 8. Acceptance checks (all must pass, otherwise exit 3)

* `r3_labels_match_stage08_order_and_split`, `r3_split_matches_split_file`, `r3_eligible_sets_match_labels`, `11_run1_dedup_columns_resolved`
* `14_frozen_winner_reproduces_run2_validation`, `14_frozen_calibrators_monotone`, `14_ordinal_lr_refits_reproduce_run2_validation_qwk`, `14_layer_c_24h_refit_reproduces_run2_validation_qwk`
* `18_frozen_landmark_models_reproduce_run2_validation`, `18_all_landmarks_scored`, `18_probabilities_valid`
* `r3_no_fit_on_test_rows`, `r3_fits_only_before_scoring`, `r3_each_test_scoring_once`

# 9. Tests and box verification (synthetic data only)

Tests: 30 passed in 133 s on the box (Python 3.14.10 venv with the laptop's package versions), run from a copy of the laptop layout (`scripts\stage13_18_r3\` next to `scripts\stage11_18_r2\`). Expect about 4–8 min on the laptop (Run 2's tests took 136 s on the box and 363 s on the laptop).

The tests cover:

* the guard (no test access while verifying, no fit on test rows, each scoring once), plus an end-to-end record showing that test rows are touched only in the scoring phase;
* verification results that do not change when every test row of every Stage 08 level is replaced by noise;
* hash stops for a wrong Run 2 manifest SHA, the Run 2 pin, each tampered Run 2 INTERNAL file and the audit, a tampered Run 2 script, and a wrong Stage 06–08 SHA;
* test-use log refusal after a published scoring; a failed scoring that is logged and publishes nothing; a verification failure that reads no test row;
* the time limit; the precheck-only mode;
* the release gate: small integer ids pass, planted leaks fail (identifier in a text column, in the md and in a numeric row; a forbidden column name) without printing the value;
* the restricted unpickler (Run 2 classes pickled under `__main__`, HistGradientBoosting models, refusal of other modules);
* the family (a) high-B p-values (they can pass Holm over 131 tests, and chunking does not change them); Holm, bootstrap p and weighted AUROC units.

Fixtures (box, Python 3.14.10 with the laptop package versions):

* base fixture (proportional-odds winner): 118 s, exit 0, release gate ok;
* GB-winner fixture (Run 2 rerun with the winner restricted to GB, so the frozen Frank–Hall HistGradientBoosting path is exercised as on the real data): 113 s, exit 0, gate ok;
* x3 fixture (about 3× the base rows): 383 s, exit 0, gate ok;
* small-id fixture (ids 1..N; Run 2 rebuilt on it first): 109 s, exit 0, gate ok.

In every fixture the frozen and refitted models reproduced Run 2 on validation (max abs probability difference about 3e-8, ordinal LR QWK difference about 5e-7, Layer C 24 h QWK difference under 5e-7) before any test row was read.

# 10. Open decision for Vinai

* **Holm family for H9.** The Holm family is the contract's seven core landmarks (pre-specified; contract v13 "Multiplicity"; plan 12.2). The tables lead with Vinai's five later landmarks. The alternative is Holm over the five later landmarks only. That gives more power at 72–336 h, but it was chosen after seeing the Run 2 validation results, so it is post hoc. **Recommendation: keep the seven-landmark family**, and report the five-landmark reading as the focus of the narrative. Switching to five would need a script change before Step 3. After test scoring, the family cannot change.

# 11. Run 3b (next package; not part of this run)

* stage 13: attribution stability (H7) on the 24 h rows (ordinal LR and the frozen GB winner at fixed hyperparameters, B = 50 subject-bootstrap refits of train, a home-made SHAP-style attribution on a fixed seeded validation sample, Kendall τ, within-group credit split, conditional vs interventional gap);
* stage 18: sequence attribution (B = 20 neural refits at a reduced set of landmarks);
* stage 16: Layer C tensor SVD on the normalized level (HOSVD spectra, Tucker/HOOI grid, CP-ALS with reduced starts and bootstraps, the planted-rank synthetic gate).

All of these use train and validation rows only, so they do not touch the test-use log.

# 12. Copilot steps (Windows, project root `<PROJECT_ROOT>`)

The same text is in `COPILOT_INSTRUCTIONS_stage14_18_r3a.md`.

**This run scores the test rows. Test rows are scored once only.** After a published Step 3, the script refuses to run again.

**Rules.**

* Run every command exactly as written. Each one starts with `Set-Location` to the project root, so it can be pasted on its own.
* Do not edit any file. Do not change, add or remove options.
* Stop at the first failure and send back the full console text only. A failure is any `EXIT:` other than 0, a hash that differs from the value shown, or an expected line that does not appear.
* After a failure, do not patch, retry with other options, or delete anything. Never rerun Step 3 on your own.
* Do not open any input or output data file. The only output file you may open is `data\processed\ladder\stage14_18_r3a\stage14_18_r3a_audit_TEAM.md` (Step 4).
* Never open or paste a file whose name contains `_INTERNAL` or `_raw_LOCAL`, never open `test_use_log_stage14_18_r3a.json`, and never open a `*.pending-*` folder.
* Steps 1–3 set `PYTHONNOUSERSITE=1` and run `python -s`. Do not drop it.
* **Hang rule:** the script prints a progress line or a heartbeat at least every 30 s. If nothing new appears for 5 minutes, press Ctrl+C once and send the console text.

**Step 0: pause cloud file syncing and check the package.**

1. *Pause cloud file syncing first* (taskbar cloud-sync icon → Settings/gear → Pause syncing → 2 hours). Keep it paused until Step 3 has finished.
2. List leftovers. List only; do not open anything:

```
Set-Location '<PROJECT_ROOT>'; Get-ChildItem data\processed\ladder -Force -Filter '*stage14_18_r3a*' | Select-Object Name,Mode; Write-Host "LISTED"
```

3. Only `_cache_stage14_18_r3a` (library caches and temp files; no data) may appear above `LISTED`; it is not a leftover. If anything else is listed (`stage14_18_r3a`, `stage14_18_r3a.pending-*` or `test_use_log_stage14_18_r3a.json`), stop and report the names.
4. **The assistant has already copied the package** into `scripts\stage13_18_r3\`, so do not copy anything. This step is a hash check only. Each hash must equal the value on the line below it; if any differs, stop.

```
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage13_18_r3\r3a_14_18_test_scoring.py -Algorithm SHA256
# d7ba69805a0ca17cd52e0f1bd1dc9da020ca9f1a2e4cd85fad3156fca8e4026c
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage13_18_r3\test_r3a_14_18_test_scoring.py -Algorithm SHA256
# 85ce3fc635dc5f5d62039388f1178e259c04dff5926c1d0b4ae00a45b39ecee9
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage11_18_r2\r2_11p_11_12_17_models.py -Algorithm SHA256
# b0af376745ee5e828e9fed041eae45918de13a7d33f8edddd36971452dc80a40
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage11_18_r2\test_r2_11p_11_12_17_models.py -Algorithm SHA256
# 3a69f8fe3225a792a274048baa37ec0826c64479572b978172a2e63ddb056fd1
Set-Location '<PROJECT_ROOT>'; Get-FileHash data\processed\ladder\stage11_12_17_r2\stage11_12_17_r2_manifest.json -Algorithm SHA256
# ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed
```

**Step 1: tests (synthetic data only, about 4–8 min).** The tests build their own tiny synthetic data in a temporary folder; they also use the Run 2 test helper in `scripts\stage11_18_r2\`. Expect `30 passed` and `EXIT:0`.

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s -m pytest -q -p no:cacheprovider scripts\stage13_18_r3\test_r3a_14_18_test_scoring.py 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

**Step 2: precheck (hashes only; reads no test row; writes only the cache folder; about 1 min).** Expect `EXIT:0` and these lines (the hashes are printed in this order):

* `result: PRECHECK OK (Stage 06-08, Run 1, Run 2 manifests and files, Run 2 script, canonical, harmonization, split, raw ICUSTAYS / ADMISSIONS, test-use log, packages); nothing written (library cache folder only)`
* `ADMISSIONS sha256 1451ae8ea0ff463269b348522047a4bc99efd2665c770e9c5cef0ffe2647e1ae`
* `ICUSTAYS sha256 449119e698acaaf948921c34375db535b9fd3f6eb6e996a86069da95e90eea1b`
* `canonical sha256 e5feeb22b5457ab89c4c73235e6a4722d538aec5f88595b6f049004171286418`
* `harmonization sha256 ffd1b8ae6a13e491be2f7a6631eb4efc1f323656795b80b91e105a0644209d6d`
* `run2_script sha256 b0af376745ee5e828e9fed041eae45918de13a7d33f8edddd36971452dc80a40`
* `split sha256 60196792457af718155a878f25b3d502ae12dbc71ccabb4c2709518c62e40e31`
* `stage06_08_manifest.json sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9`
* `stage09_10_r1_manifest.json sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61`
* `stage10_matched_width_bases_INTERNAL.npz sha256 6b52a538da8d79dba84d5c66aa4f67a4d4395c87c533606968497425ddeb6933`
* `stage11_12_17_r2_manifest.json sha256 ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed`
* `stage11_12_17_validation_predictions_INTERNAL.npz sha256 3d4423f4dea695e9a712ac60b34238707814a676f633d48d7150c6eddf60fda0`
* `stage11p_labels_landmark_INTERNAL.npz sha256 6151530b99a4097d0c52f3a8b26f2d2b87c45f2e84ca1dfbb6667331376d730f`
* `stage12_tuned_model_calibrators_INTERNAL.pkl sha256 89da3c8ec2dd0d6db03686f0620bfc6b39061cf062b1546bf7fe77b35b9b0952`
* `stage17_models_INTERNAL.pt sha256 4adc72cad854e6de824543e7dc3728ad1677246ed228e4cd4e7251febbed6fdc`
* `stage17_tabular_models_INTERNAL.pkl sha256 9ddf6c8afaa7a6a4483e26dd8da7f87783b228ba802db4d3ba2103afc9ac3ea3`

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s scripts\stage13_18_r3\r3a_14_18_test_scoring.py --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 --expect-run1-manifest-sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61 --expect-run2-manifest-sha256 ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed --split-file data\interim\r2_inputs\subject_splits.csv --raw-dir data\interim\r2_inputs\raw --precheck-only 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

**Step 3: real run (about 25–35 min; hard limit 60 min).** Only run it if Step 2 passed. Cloud file syncing must still be paused. The script first checks on validation rows that the frozen and refitted models reproduce Run 2 (`verification (validation only)` lines); only then does it read test rows. Expect progress lines and the audit text, then a last line `result: RUN 3a (14, 18) OK; RELEASE GATE OK` and `EXIT:0`.

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s scripts\stage13_18_r3\r3a_14_18_test_scoring.py --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 --expect-run1-manifest-sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61 --expect-run2-manifest-sha256 ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed --split-file data\interim\r2_inputs\subject_splits.csv --raw-dir data\interim\r2_inputs\raw 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

* On `EXIT:2`, nothing was written and no test row was read.
* On `EXIT:3`, the console names any pending folder left behind; the progress lines show whether test scoring (`stage 14: test scoring`) had started. Do not open or delete anything; report it. An unexpected error prints `frame <file>:<line>:<function>` lines (no data); send them back as they are.
* After Step 3 ends, whatever the result, resume cloud file syncing.

**Step 4: report.** Open only the TEAM audit:

```
Set-Location '<PROJECT_ROOT>'; Get-Content data\processed\ladder\stage14_18_r3a\stage14_18_r3a_audit_TEAM.md
```

Send back the full console text of Steps 0–3 and the contents of `stage14_18_r3a_audit_TEAM.md` only. Do not open, paste or upload any `*_INTERNAL` file, the test-use log, or any other output file.
