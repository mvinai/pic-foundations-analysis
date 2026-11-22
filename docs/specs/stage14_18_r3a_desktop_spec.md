---
title: "Run 3a — stages 14 and 18: single test scoring of the frozen Run 2 models"
subtitle: "stage14_18_r3a_desktop_spec v2.0.1-desktop · script r3a_desktop_full v2.0.1-desktop · plan v13 / contract v13 · PIC Foundations · Nov 21, 2026"
---

# 1. Summary

**This is the desktop full-settings variant of Run 3a (v2.0.1-desktop).** It is a post-primary sensitivity analysis (a second look at the test set), not a replacement for the laptop Run 3a primary scoring. A published laptop Run 3a does not block this run. Every file the script writes goes under `data\\processed\\ladder\\DESKTOP_FULL_PLAN\\` only (published folder `stage14_18_r3a\\`, pending folders, cache, test-use log, internal files, and `README_DESKTOP.txt`). There is no 60-minute time cap. Training and the stacked sensitivity fit use CUDA if available, otherwise CPU (logged). Frozen-model reproduction forward passes, and test scoring of the frozen Run 2 landmark GRUs, run on CPU in float32, matching the laptop Run 2, even when CUDA is available. The stacked sensitivity GRU uses D19 early stopping with a maximum of 200 epochs.

Run 3 of the stage 09–18 plan is split in two, as the plan allows when one run would exceed about 45 min:

* **Run 3a (this package):** stage 14 (test scoring of the frozen 24 h models, family (a), H6) and stage 18 test scoring (the frozen landmark models, family (b), H9), plus the stacked (encounter, landmark) sensitivity model.
* **Run 3b (separate package, not in this one):** stage 13 attribution stability (H7), stage 18 sequence attribution and stage 16 Layer C tensor SVD. None of these reads test rows, so Run 3b can run before or after Run 3a.

Run 3a is one script, `scripts\\stage13_18_r3_desktop\\r3a_desktop_full.py` (`r3a_desktop_full v2.0.1-desktop`), plus its tests. **Test rows are scored exactly once, with frozen Run 2 choices only.** Nothing is tuned, selected or refitted on test rows. The Run 2 conventions carry over:

* a hash-gated precheck. It runs Run 2's own precheck (Stage 06–08 manifest, Run 1 manifest and bases, canonical, harmonization, split, raw ICUSTAYS / ADMISSIONS). It then checks the **Run 2 manifest** (`--expect-run2-manifest-sha256` and the pin `ec394345…28ed`), and hashes every file the Run 2 manifest lists;
* the Run 2 script is imported only after its SHA-256 equals the pin `b0af3767…0a40`, and no bytecode is written. Run 2 model files are read with a restricted unpickler;
* a pending folder published atomically, and the fail-closed release gate with the Run 2 rules: structural-column whitelist, integer-cell identifier rule, collect-all reporting by file and column without values, and a size scan. Outputs are an aggregate-only audit and a manifest;
* TEAM and INTERNAL outputs. Library caches and temp files go to `data\processed\ladder\DESKTOP_FULL_PLAN\_cache_stage14_18_r3a\`;
* a progress line or heartbeat at least every 30 s, and no time cap (`--max-seconds` defaults to 0; the flag is only for tests). Unexpected errors print frames only;
* exit codes: 0 ok; 2 input stop (nothing written); 3 build or gate failure (nothing published);
* `--split-file` and `--raw-dir` point at the copies under `data\interim\r2_inputs\` (the Copilot sandbox reads only inside the project folder). Their hashes must equal the pins and the Run 2 manifest's input hashes.

**Test-use log.** `data\processed\ladder\DESKTOP_FULL_PLAN\test_use_log_stage14_18_r3a.json` gets a "started" entry just before the first test row is read. When the run ends, that entry becomes "published" (with the manifest SHA-256) or "failed (nothing published)". A later run refuses to start if a "published" entry exists. Failed attempts do not block a rerun, but the audit counts them. If verification fails, no test row has been read and the log is not touched.

**Runtime.** Box synthetic full-settings runs of this desktop script (CPU torch): GB-winner fixture 119.9 s (stacked 17 and 22 epochs), x3 fixture 709.5 s, exit 0, release gate ok. On x3, the stacked sensitivity was 518 s (per-row GRU 234 s / 50 epochs, per-patient GRU 284 s / 61 epochs), stage 14 test scoring 106 s, and the rest about 86 s. Real data is about 4× the x3 wall time. The stacked GRUs are the part a desktop GPU can speed up; stage 14's B = 20,000 family (a) bootstrap is CPU-bound. Desktop estimate for the real run: about 25–45 min if CUDA is available (about 45–50 min on CPU only). There is no 60-minute time cap (`--max-seconds` defaults to 0).

# 2. Inputs and hash gates

| input | path (relative to the project root) | gate |
|---|---|---|
| Run 2 script | `scripts\stage11_18_r2\r2_11p_11_12_17_models.py` | pin `b0af376745ee5e828e9fed041eae45918de13a7d33f8edddd36971452dc80a40` before import |
| Run 2 manifest | `data\processed\ladder\stage11_12_17_r2\stage11_12_17_r2_manifest.json` | `--expect-run2-manifest-sha256` and pin `ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed`. It must come from the pinned script, with gate ok and all checks passing. Its input hashes must equal this run's hashes for Stage 06–08, Run 1, canonical, harmonization, split and raw tables. Every listed file is hashed, including the five INTERNAL files and the three TEAM tables plus the audit JSON that Run 3a reads |
| Stage 06–08, Run 1, canonical, harmonization | as in Run 2 | Run 2's precheck and pins (`d8efd1fe…71d9`, `703d7b46…7f61`, bases `6b52a538…6933`) |
| subject split | `data\interim\r2_inputs\subject_splits.csv` | pin `60196792…e40e31` |
| raw ICUSTAYS / ADMISSIONS | `data\interim\r2_inputs\raw\` | pins `449119e6…ea1b` / `1451ae8e…e1ae` |

The output folder `data\processed\ladder\DESKTOP_FULL_PLAN\stage14_18_r3a\` must be absent or empty, and no `DESKTOP_FULL_PLAN\stage14_18_r3a.pending-*` folder may exist. Any path containing `_raw_LOCAL` is refused.

# 3. Verification on validation (before any test row is read)

Every check below must pass. Otherwise the run stops with exit 3, no test row has been read, and the test-use log is unchanged.

* **Frozen stage 12 winner:** its validation probabilities reproduce Run 2's saved validation predictions (max abs difference ≤ 1e-4). The isotonic calibrators are monotone.
* **Ordinal LR at every width:** Run 2 saved no stage 11 bake-off models. So the proportional-odds and all-threshold ordinal LR are refitted on train at every (matrix, width) with the frozen Run 2 configuration and seeds. Their validation QWK must reproduce Run 2's `stage11_validation_scores_TEAM.csv` within 0.005 (an environment-reproducibility tolerance: desktop refits are not required to be bit-identical to the laptop Run 2). The measured max abs QWK difference is written in the audit and in the progress and failure lines.
* **Layer C 24 h GRU:** Run 2 did not save this model. The validation-selected representation is refitted on train with Run 2's seed and recorded epoch cap (read from the hash-verified Run 2 audit). Its validation QWK must reproduce Run 2 within 0.005.
* **Stage 17 frozen landmark models:** A, A+B, A+B+C and C-only at all 14 landmarks, with both outcomes and ordinal8 where present. Their validation predictions must reproduce Run 2's saved predictions within 1e-4, checked on a CPU float32 forward pass (matching the laptop Run 2) even when CUDA is available. The measured max abs probability difference is written in the audit and in the progress and failure lines.
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
* **Stacked (encounter, landmark) sensitivity (B6/B7, descriptive only):** one GRU fitted on the train rows of all 14 landmarks. It uses hidden size 32, the Layer C channels shared by every landmark, at most 200 epochs with D19 early stopping, and batches that never mix landmarks. It is trained with per-row and with per-patient weights and scored on test. Each version is compared with the frozen per-landmark A+B, both pooled and per landmark, on the same test rows.

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
2. Run 2 saved no stage 11 bake-off models and no Layer C 24 h GRU: the ordinal LR (proportional odds and all-threshold) at every width and the validation-selected Layer C 24 h GRU are refitted on train with the frozen Run 2 configuration and seeds, and must reproduce the Run 2 validation QWK before any test row is read (ordinal LR: within 0.005, an environment-reproducibility tolerance; Layer C 24 h GRU: within 0.005)
3. family (a) uses the ordinal LR only (the model fitted at every width); random forest, gradient boosting and the multinomial reference are not refitted at every width for test (runtime); the frozen stage 12 winner is scored
4. family (a) includes one Layer C 24 h representation (the Run 2 validation choice), compared with the frozen uncalibrated A+B winner as in Run 2
5. bootstrap p-values on test use the (count + 1) / (B + 1) rule; Run 2 validation p-values used count / B
6. family (a) TOST p-values come from a dedicated paired subject-cluster bootstrap with B = 20000 (intervals stay at B = 1000): with 131 tests Holm's first threshold (0.05 / 131) is below 1 / 1001, the smallest p-value B = 1000 can give, so no equivalence could ever be declared
7. H6 AUROC parts (macro AUROC and death-threshold AUROC within delta_a) are reported with bootstrap intervals but are not in the Holm family (QWK only, as in the Run 2 verdicts)
8. the H9 frozen-rule check (landmark features under the frozen 3.0 whole-stay rule, epsilon) is not evaluated: no such feature build exists
9. stacked (encounter, landmark) sensitivity: one GRU over all 14 landmarks (hidden 32, batches never mix landmarks; the Layer C channels shared by every landmark's Run 1 channel list; at most 200 epochs with the D19 early stopping), per-row and per-patient weighted training; compared with the frozen per-landmark A+B on the same test rows; descriptive only
10. desktop full-settings run (v2.0.1-desktop): no 60-minute time cap; training and the stacked sensitivity fit use CUDA if available, otherwise CPU (logged; seeds set on CPU and CUDA); frozen-model reproduction forward passes, and test scoring of the frozen Run 2 landmark GRUs, run on CPU in float32, matching the laptop Run 2, even when CUDA is available; every file this script writes is under `data\\processed\\ladder\\DESKTOP_FULL_PLAN\\` (published folder, pending folder, cache, test-use log, internal files and `README_DESKTOP.txt`), so a published laptop Run 3a does not block this sensitivity run
11. environment-reproducibility tolerance: desktop ordinal LR refits are not required to be bit-identical to the laptop Run 2; their validation QWK must fall within 0.005 of the Run 2 TEAM QWK. The measured max abs differences are written in the audit and in the progress and failure lines. Frozen landmark-model probabilities still use a 1e-4 tolerance, checked on a CPU float32 forward pass
12. the v2.0.0-desktop attempt on this desktop failed during validation-only verification (the 2e-6 ordinal LR QWK tolerance and frozen GRU forward passes on CUDA), before any test row was read; the desktop test-use log is unchanged, so this v2.0.1 run is not a second test scoring
13. counts gate: per-threshold metrics with fewer than 10 rows in either class, and the mortality AUROC with fewer than 10 deaths or survivors, are written as <10
16. (spec note) Run 3 is still two packages: Run 3a (stages 14 and 18 test scoring, this package) and Run 3b (stages 13, 16 and 18 attribution). That split is not a runtime reduction; the laptop `--part` split inside Run 3b is not used here
15. (spec note) the split file and raw ICUSTAYS / ADMISSIONS are read from the hash-identical copies under `data\interim\r2_inputs\` (the Copilot sandbox reads only inside the project folder), exactly as in the Run 2 real run
14. (spec note) no class weighting, as in Run 2: every refit (ordinal LR at every width, Layer C 24 h GRU, stacked sensitivity GRU) is unweighted except the stacked per-patient version, whose weights are per-patient, not per-class

# 8. Acceptance checks (all must pass, otherwise exit 3)

* `r3_labels_match_stage08_order_and_split`, `r3_split_matches_split_file`, `r3_eligible_sets_match_labels`, `11_run1_dedup_columns_resolved`
* `14_frozen_winner_reproduces_run2_validation`, `14_frozen_calibrators_monotone`, `14_ordinal_lr_refits_reproduce_run2_validation_qwk`, `14_layer_c_24h_refit_reproduces_run2_validation_qwk`
* `18_frozen_landmark_models_reproduce_run2_validation`, `18_all_landmarks_scored`, `18_probabilities_valid`
* `r3_no_fit_on_test_rows`, `r3_fits_only_before_scoring`, `r3_each_test_scoring_once`

# 9. Tests and box verification (synthetic data only)

Tests: 37 passed in 110 s on the box (Python 3.14.10, CPU torch; the same 37 tests are expected on a CUDA machine, with no skips), run from a copy of the laptop layout (`scripts\stage13_18_r3_desktop\` next to `scripts\stage11_18_r2\`). Expect about 4–8 min on the desktop. The tests include checks that frozen-model verification forward passes run on CPU, that the ordinal LR QWK tolerance is 0.005, that every write stays under `DESKTOP_FULL_PLAN`, and that a published laptop Run 3a does not block the desktop run.

The tests cover:

* the guard (no test access while verifying, no fit on test rows, each scoring once), plus an end-to-end record showing that test rows are touched only in the scoring phase;
* verification results that do not change when every test row of every Stage 08 level is replaced by noise;
* hash stops for a wrong Run 2 manifest SHA, the Run 2 pin, each tampered Run 2 INTERNAL file and the audit, a tampered Run 2 script, and a wrong Stage 06–08 SHA;
* test-use log refusal after a published scoring; a failed scoring that is logged and publishes nothing; a verification failure that reads no test row;
* the time limit; the precheck-only mode;
* the release gate: small integer ids pass, planted leaks fail (identifier in a text column, in the md and in a numeric row; a forbidden column name) without printing the value;
* the restricted unpickler (Run 2 classes pickled under `__main__`, HistGradientBoosting models, refusal of other modules);
* the family (a) high-B p-values (they can pass Holm over 131 tests, and chunking does not change them); Holm, bootstrap p and weighted AUROC units.

Synthetic full-settings runs of this desktop script on the box (CPU torch, default settings, exactly what the Copilot commands run; real input pins nulled as in the tests):

* GB-winner fixture (Run 2 rerun with the winner restricted to GB, so the frozen Frank–Hall HistGradientBoosting path is exercised as on the real data): 119.9 s, exit 0, release gate ok (stacked 17 and 22 epochs);
* x3 fixture (about 3× the base rows): 709.5 s, exit 0, release gate ok (stacked per-row GRU 234 s / 50 epochs, per-patient GRU 284 s / 61 epochs; stage 14 test scoring 106 s).

In both fixtures the frozen and refitted models reproduced Run 2 on validation before any test row was read, and every output file landed under `data\processed\ladder\DESKTOP_FULL_PLAN\`.

# 10. Status of earlier laptop deviations

The laptop 20-epoch cap on the stacked sensitivity GRU and the 60-minute time cap are removed in this desktop package (early stopping stays; the stacked maximum is 200 epochs). This run is labelled a post-primary sensitivity analysis (a second look at the test set) and does not replace the laptop Run 3a primary scoring. Remaining deviations that this package does not close: Stage 15 Layer C levels are still omitted (Layer C is the Stage 08 base tensor, D17); nested cross-validation is still omitted (D20); the sequence-model bake-off (LSTM, attention, transformer) is not in this package (Tier 2); family (a) still refits only the ordinal LR at every width; the H9 frozen-rule check is still not evaluated.

# 11. Open decision for Vinai

* **Holm family for H9.** The Holm family is the contract's seven core landmarks (pre-specified; contract v13 "Multiplicity"; plan 12.2). The tables lead with Vinai's five later landmarks. The alternative is Holm over the five later landmarks only. That gives more power at 72–336 h, but it was chosen after seeing the Run 2 validation results, so it is post hoc. **Recommendation: keep the seven-landmark family**, and report the five-landmark reading as the focus of the narrative. Switching to five would need a script change before Step 3. After test scoring, the family cannot change.

# 12. Companion package

Run 3b desktop full settings is a separate package (`r3b_desktop_full v2.0.1-desktop`): stages 13, 16 and 18 together, D20 refit counts, outputs also under `data\\processed\\ladder\\DESKTOP_FULL_PLAN\\`. It uses train and validation rows only and does not need this Run 3a desktop output.

# 13. Copilot steps

The paste-ready Copilot commands for the desktop sequence (Run 3a then Run 3b) are in `COPILOT_INSTRUCTIONS_stage13_18_r3_desktop.md`. Per-run commands are also in `COPILOT_INSTRUCTIONS_stage14_18_r3a_desktop.md`. Every command is prefixed with `Set-Location` to the project root and uses `$env:PYTHONNOUSERSITE='1'` and `<VENV_PYTHON> -s`. Step 0 unzips the v2.0.1 package into `scripts\` with `Expand-Archive -Force` (overwriting the v2.0.0 r3a/r3b files only; leave any r4 or r5 files already in `scripts\stage13_18_r3_desktop\` alone), pauses cloud file syncing, and checks Python and package versions against the desktop list (Python 3.14.3, numpy 2.4.2, pandas 3.0.1, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.8, torch 2.10.0+cu126) and `torch.cuda.is_available()`. The folders `DESKTOP_FULL_PLAN\stage17_seqmodels` (Tier 2, published) and, later, `stage15_layer_c_levels` (Tier 3) are expected in the leftover listing and are not leftovers. If a version other than this desktop list differs, report it and stop; do not upgrade or install anything. Scripts live in `scripts\\stage13_18_r3_desktop\\`. The only audit Copilot may open is `data\\processed\\ladder\\DESKTOP_FULL_PLAN\\stage14_18_r3a\\stage14_18_r3a_audit_TEAM.md`.
