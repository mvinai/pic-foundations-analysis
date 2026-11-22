---
title: "Run 3b — stages 13, 16 and 18 attribution (two parts): attribution stability, sequence attribution and the Layer C tensor analysis"
subtitle: "stage13_14_16_18_r3_spec v1.1.0, section 3b (addendum to the Run 3a spec) · script r3b_13_16_18_attribution v1.1.0 · plan v13 / contract v13 · PIC Foundations · Nov 19, 2026"
---

# 1. Summary

Run 3b finishes the attribution stages of Run 3. It reads **train and validation rows only** (plan v13 §11 and §6.7: global importance on a fixed seeded validation sample, refits on subject bootstraps of train; stage 16 fits on train and projects validation). Test rows are dropped when the inputs are loaded, so Run 3b never touches the Run 3a test-use log and does not need any Run 3a output.

Run 3b runs as **two independent parts**, each a separate run under the 60-minute limit:

* **part 1 (`--part 1`, output `stage13_r3b1`)**: stage 13, attribution stability (H7) on the 24 h rows;
* **part 2 (`--part 2`, output `stage16_18_r3b2`)**: stage 18 sequence attribution of the D19 GRU A+B+C at the five focus landmarks (72, 120, 168, 240 and 336 h, Vinai's day 3/5/7/10/14 framing), then stage 16, the Layer C tensor SVD.

Runtime (estimates scaled from the box with Run 2's measured ratios, real 2837 s vs synthetic x3 726 s and base 256 s; tabular stages about 3.2× x3, the GRU stages about 4.7× x3, laptop about 2.7× box): **part 1 about 33–39 min, part 2 about 41–45 min**, each under the 60-minute hard limit (`--max-seconds 3600`). Part 1 is dominated by the gradient boosting refits (Frank–Hall: one 200-iteration HistGradientBoosting per threshold; its fit time barely depends on the row count, so its laptop factor is close to the 2.7× machine factor); part 2 is about half GRU refits and half stage 16 (its HOOI and CP fits use a sub-sample of at most 2000 train patients, already reached on x3, so its factor is 2.7–3.2×). Factors used: ordinal LR 3.2–3.9×, gradient boosting 2.7–3.2×, GRU 4.4–4.7× (Run 2's stages 12 and 17 ran 4.7× their x3 time), stage 16 2.7–3.2×. Run 2 was underestimated in advance (26–32 min estimated, 47 min actual); these factors are calibrated on that actual run.

# 2. Inputs and hash gates

The same gates as Run 3a, before anything is parsed: the Stage 06-08 manifest (`d8efd1fe…`) and its files, the Run 1 v1.0.2 manifest (`703d7b46…`) and files, canonical, harmonization, the split and raw ICUSTAYS / ADMISSIONS copies under `data\interim\r2_inputs\` (hash-identical to the originals), the Run 2 script (`b0af3767…`, imported only after its hash matches) and the Run 2 manifest (`ec394345…`, pinned on the command line and in the script) with every Run 2 file it lists. Before any attribution, the frozen Run 2 stage 12 winner must reproduce the Run 2 validation probabilities (max abs difference ≤ 1e-4).

# 3. Stage 13 (part 1): attribution stability, H7

* Models: the ordinal LR (proportional odds, λ = 1) at every level a0–b2p and on the winner's matrix; the tabular winner (Frank–Hall gradient boosting, the frozen stage 12 hyperparameters: learning rate 0.1, 7 leaves, 200 iterations) on b1, b2d and the winner's matrix (a0+b0 at full width on the real data). Gradient boosting on b2s is read off the b2d refits when the b2d trees never split on a φ2-dup copy and the full-train b2s and b2d fits coincide (otherwise refitted). Outcome 2 (secondary): ordinal LR on b1, b2s and b2d.
* Refits: subject bootstraps of train; **B = 50 for the ordinal LR, B = 10 for gradient boosting** (see §5).
* Attribution target E[Y | x] = Σ_j P(Y > j | x) (plan §11) on a fixed seeded validation sample of 500 rows; ordinal LR: exact interventional linear SHAP; gradient boosting: exact path-dependent TreeSHAP of each threshold margin, carried to P(Y > j | x) by the rescale rule and summed. Home-made (no `shap` package on the laptop) and checked in the tests against brute-force Shapley values.
* Summaries: median Kendall τ of the mean-|SHAP| vectors between refit pairs (with IQR), split into grouped (summary statistics, z-flags, renal Cr/AKI, duplicated copies) and ungrouped columns; within-group credit split; the interventional vs conditional TreeSHAP gap on the full-train fit (200 validation rows, 50 train background rows).
* H7 (plan 12.2 bracketed values): the median τ falls from φ1 (b1) to φ2-dup (b2d) by at least 0.10, more in the grouped than the ungrouped columns, and φ2-single (b2s) stays within 0.05 of φ1.
* In-script synthetic gate: a planted redundant group must split its credit and lower τ.

# 4. Stages 18 and 16 (part 2)

**Stage 18, sequence attribution.** Expected-gradients SHAP (GradientSHAP, 16 baseline-and-α draws per row, 100 train baselines) of E[Y | x] for the D19 GRU A+B+C, Outcome 1, on 300 validation rows per landmark at 72, 120, 168, 240 and 336 h; **B = 10 subject-bootstrap refits of train** per landmark with the frozen Run 2 configuration (see §5). Kendall τ between refits at three granularities (bin × channel, channel, day block / organ block); permutation importance of the frozen Run 2 model as the cross-check; input-type shares (values, masks, carried-forward flags, bin index, static A+B context, Layer A offset); TreeSHAP of the frozen A+B landmark models for the A vs B share.

**Stage 16, Layer C tensor SVD (full settings).** Train tensors (patients × bins × channels) at the seven core landmarks (all 14 for the angle table), normalized, raw (masked EM fit) and engineered-lite levels: HOSVD mode spectra with 20 null draws; Tucker by HOOI on the r1 × r2 × r3 grid with validation projection error on fixed train factors; CP-ALS (10 random starts + the HOSVD start) with CORCONDIA (reported as a core-consistency fraction) and 50 subject bootstraps for congruence (D18 thresholds 0.85 and 80 %); principal angles between landmarks; organ-block alignment of the channel factors against a permutation null; the labeled flattened baseline. No patient-mode loadings are written. In-script synthetic gate: a planted rank-3 tensor must be recovered.

# 5. Runtime budget (deviation, logged for Vinai)

The plan-consistent full D20 settings (B = 50 refits for every stage 13 model, B = 20 GRU refits) were estimated at about 3 h on the laptop (box: about 12–33 s per gradient boosting refit and 5–8 s per GRU refit per landmark at the x3 size). The real run must fit under about 45 min (hard limit 60). Following the agreed order of preference, the reduction is: (1) **split Run 3b into two independent runs** (part 1 stage 13, part 2 stages 16 and 18); (2) **stage 13 gradient boosting B = 10** instead of 50 (the most expensive component; from its measured refit times, B = 15 gave about 45–55 min and B = 12 about 42–49 min for part 1, above the 45-minute target); (3) **stage 18 GRU B = 10** instead of 20, at the five focus landmarks 72, 120, 168, 240 and 336 h only (already the D19/H9 focus set); (4) stage 13 ordinal LR keeps **B = 50** and stage 16 runs with its **full** settings (10 CP starts + HOSVD, 50 bootstraps, 20 null draws). Kendall τ medians then rest on 45 refit pairs (gradient boosting and GRU, 10 refits each) instead of 1225 and 190, so their IQRs are wider; the H7 verdict uses the ordinal LR (B = 50) and gradient boosting (B = 10) side by side. The script defaults equal these settings, and the Copilot commands pass no budget options. Logged as a deviation in both audits and flagged for Vinai (§10).

# 6. Outputs and disclosure

Each part publishes atomically from a pending folder to its own folder in `data\processed\ladder\` (`stage13_r3b1\`, `stage16_18_r3b2\`): TEAM tables and figures (aggregate only: importance by column name, channel, bin or organ block; no per-row SHAP, no patient loadings, no patient, row, encounter or subject counts and no identifiers), a TEAM audit (json and md), the manifest, and one `*_INTERNAL.npz` (per-refit importance vectors and train factor matrices; never opened, pasted or shared). The Run 2 release gate re-reads every TEAM file from disk: forbidden column names, identifiers, every cohort, split, landmark, class and assignable size, and any number not on the structural whitelist stop the run with exit 3 and nothing published.

# 7. Decisions and logged deviations

1. runtime budget (Nov 19, logged for Vinai): the full D20 settings (B = 50 for every stage 13 model, B = 20 GRU refits) were estimated at about 3 h on the laptop, so Run 3b runs as two parts under the 60-minute limit (part 1 stage 13, part 2 stages 16 and 18) and the refit counts of the two most expensive components are reduced: stage 13 gradient boosting B = 10 (D20: 50) and stage 18 GRU B = 10 (D20: 20), at the five focus landmarks 72, 120, 168, 240 and 336 h only; stage 13 ordinal LR keeps B = 50 and stage 16 runs in full; Kendall tau medians then rest on 45 refit pairs (10 refits each, gradient boosting and GRU) instead of 1225 and 190, so their IQRs are wider
2. stage 13 ordinal LR at every level (a0-b2p) and on the winner's matrix; the tabular winner (gradient boosting, tuned stage 12 hyperparameters) on b1, b2d and the winner's matrix, all at full width; b2s gradient boosting is read off the b2d refits when the b2d trees never split on a phi2-dup copy and the full-train b2s and b2d fits coincide (otherwise refitted)
3. stage 13 Outcome 2 (secondary): ordinal LR on b1, b2s and b2d only
4. stage 13 interventional vs conditional gap on the full-train fit only (not per refit), on a sub-sample of the fixed validation sample against a fixed train background
5. stage 13 ordinal LR refits are warm-started at the full-train solution (same objective and optimum)
6. stage 18 credit split across temporally adjacent bins summarised as the share of the last bin within the last two bins for the ten most important channels (spread across refits), plus the last-day share
7. stage 18 attributions to masks, carried-forward flags, the bin index, the static A+B context and the Layer A offset are reported as input-type shares; the bin x channel, channel and organ granularities sum each channel's value, mask and carried-forward inputs
8. stage 16: HOOI grid r1 in {5, 10, 20, 40}, r2 in {1, 2, 4, 8, 16, b_L}, r3 in {5, 10, 20, 40, q} (capped), 5 sweeps from the HOSVD start (15 EM sweeps for the raw level); HOOI and CP fits on a seeded sub-sample of at most 2000 train patients (spectra, nulls and angles use all train patients); CP-ALS on the normalized level only, at most 60 sweeps
9. stage 16 CP rank: the largest R whose best fit has CORCONDIA >= 80 % and median congruence across starts >= 0.85 (both D18 thresholds); on large-n data with unstructured noise CORCONDIA alone stays high when the model is over-factored (synthetic experiments), the congruence across starts does not
10. stage 16 nulls use 20 draws; the H17 padding probe is deferred (it needs the full 56-bin Stage 3 grid; the Stage 08 prefix tensors contain no padding); the optional loading-based model is omitted
11. (spec note) the split file and raw ICUSTAYS / ADMISSIONS are read from the hash-identical copies under `data\interim\r2_inputs\` (the Copilot sandbox reads only inside the project folder), as in Run 2 and Run 3a
12. (spec note) no class weighting, as in Run 2: every refit is unweighted

# 8. Acceptance checks (all must pass, otherwise exit 3)

The frozen winner reproduces Run 2 on validation; no test row loaded and none in any fit or sample (fit log fails closed); Run 1 dedup columns resolved; all importances finite; the synthetic gates of the part pass; the release gate passes.

# 9. Tests and box verification (synthetic data only)

Tests: 45 passed in 109 s on the box (Python 3.14.10 venv with the laptop's package versions), run from a copy of the laptop layout (`scripts\stage13_18_r3\` next to `scripts\stage11_18_r2\`). Expect about 4–8 min on the laptop (Run 2's tests took 136 s on the box and 363 s on the laptop). The end-to-end tests run part 2 before part 1 on the same project, so part 2 is shown not to need part 1's output.

Synthetic full-setting runs on the box (default refit counts, exactly what the Copilot commands run; real input pins nulled as in the tests):

* GB-winner fixture (Run 2 rerun with the winner restricted to GB, as on the real data; base size): part 1 633 s, exit 0, gate ok (gradient boosting jobs 567 s); part 2 392 s, exit 0, gate ok;
* x3 fixture (about 3× the base rows; proportional-odds winner, so no gradient boosting job): part 1 126 s, exit 0, gate ok; part 2 693 s, exit 0, gate ok (stage 18 335 s, stage 16 358 s);
* base fixture: part 1 64 s, exit 0, gate ok; part 2 338 s, exit 0, gate ok;
* small-id fixture (ids 1..N): part 1 64 s, exit 0, gate ok; part 2 340 s, exit 0, gate ok.

The base and small-id runs ran side by side with 2 threads each (timings not used for the estimate); the x3 and GB-winner runs ran alone. In every run the frozen Run 2 models reproduced Run 2 on validation and all acceptance checks and synthetic gates passed.

# 10. Open decisions for Vinai

* **Runtime budget reduction (decided by the assistant, needs Vinai's sign-off).** Gradient boosting B = 10 (D20: 50) and GRU B = 10 (D20: 20) at the five focus landmarks, with Run 3b split into two runs. **Recommendation: accept.** The ordinal LR keeps B = 50, stage 16 is full, and the τ summaries remain usable with wider IQRs. Alternative: a later full-B sensitivity run on the GPU desktop, which would be a separate package (new defaults and pins), not an option change to this one.
* **When to run.** After Run 3a has finished (either result), never at the same time (CPU contention would slow both and risk the 60-minute limit). Part 2 can run in a later session. **Recommendation:** run part 1 and part 2 back to back once Run 3a is done.

# 11. Copilot steps (Windows, project root `<PROJECT_ROOT>`)

The same text is in `COPILOT_INSTRUCTIONS_stage13_16_18_r3b.md`.

**Run 3b reads no test row and does not depend on Run 3a.** It uses train and validation rows only (test rows are dropped when the inputs are loaded) and never touches the Run 3a test-use log. Do not start it while Run 3a is still running on this laptop; run it after Run 3a has finished (either result).

**Run 3b runs in two parts, each a separate run under the 60-minute limit:** part 1 is stage 13 (Step 3a), part 2 is stages 16 and 18 (Step 3b). The parts are independent: part 2 does not need part 1's output. They can run back to back, or part 2 can run in a later session (repeat Step 0 first).

**Rules.**

* Run every command exactly as written. Each one starts with `Set-Location` to the project root, so it can be pasted on its own.
* Do not edit any file. Do not change, add or remove options.
* Stop at the first failure and send back the full console text only. A failure is any `EXIT:` other than 0, a hash that differs from the value shown, or an expected line that does not appear.
* After a failure, do not patch, retry with other options, or delete anything. Never rerun Step 3a or Step 3b on your own.
* Do not open any input or output data file. The only output files you may open are the two TEAM audits named in Step 4.
* Never open or paste a file whose name contains `_INTERNAL` or `_raw_LOCAL`, and never open a `*.pending-*` folder.
* Steps 1–3b set `PYTHONNOUSERSITE=1` and run `python -s`. Do not drop it.
* **Hang rule:** the script prints a progress line or a heartbeat at least every 30 s. If nothing new appears for 5 minutes, press Ctrl+C once and send the console text.

**Step 0: pause cloud file syncing and check the package.**

1. *Pause cloud file syncing first* (taskbar cloud-sync icon → Settings/gear → Pause syncing → 8 hours). Keep it paused until Step 3b has finished (both parts take about 80–93 min with the tests).
2. List leftovers. List only; do not open anything:

```
Set-Location '<PROJECT_ROOT>'; Get-ChildItem data\processed\ladder -Force -Filter '*_r3b*' | Select-Object Name,Mode; Write-Host "LISTED"
```

3. Only `_cache_stage13_r3b1` and `_cache_stage16_18_r3b2` (library caches and temp files; no data) may appear above `LISTED`; they are not leftovers. If anything else is listed (`stage13_r3b1`, `stage16_18_r3b2`, or a `*.pending-*` folder), stop and report the names. (If part 1 was already published in an earlier session and you are now running part 2 only, `stage13_r3b1` is expected; say so and skip to Step 3b.)
4. **The assistant has already copied the package** into `scripts\stage13_18_r3\`, so do not copy anything. This step is a hash check only. Each hash must equal the value on the line below it; if any differs, stop.

```
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage13_18_r3\r3b_13_16_18_attribution.py -Algorithm SHA256
# 654f8718db69f770298516f863efe10b28b32562f0c398dc75292352f8faa36f
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage13_18_r3\test_r3b_13_16_18_attribution.py -Algorithm SHA256
# 02b179c472cebe2b74007f058e599c4165f9fee882753422b4984f7d2231a13b
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage11_18_r2\r2_11p_11_12_17_models.py -Algorithm SHA256
# b0af376745ee5e828e9fed041eae45918de13a7d33f8edddd36971452dc80a40
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage11_18_r2\test_r2_11p_11_12_17_models.py -Algorithm SHA256
# 3a69f8fe3225a792a274048baa37ec0826c64479572b978172a2e63ddb056fd1
Set-Location '<PROJECT_ROOT>'; Get-FileHash data\processed\ladder\stage11_12_17_r2\stage11_12_17_r2_manifest.json -Algorithm SHA256
# ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed
```

**Step 1: tests (synthetic data only, about 4–8 min).** The tests build their own tiny synthetic data in a temporary folder; they also use the Run 2 test helper in `scripts\stage11_18_r2\`. Expect `45 passed` and `EXIT:0`.

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s -m pytest -q -p no:cacheprovider scripts\stage13_18_r3\test_r3b_13_16_18_attribution.py 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

**Step 2: precheck (hashes only; reads no data row; writes only the cache folder; about 1 min).** Expect `EXIT:0` and these lines (the hashes are printed in this order):

* `result: PRECHECK OK (Stage 06-08, Run 1, Run 2 manifests and files, Run 2 script, canonical, harmonization, split, raw ICUSTAYS / ADMISSIONS, packages); nothing written (library cache folder only)`
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
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s scripts\stage13_18_r3\r3b_13_16_18_attribution.py --part 1 --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 --expect-run1-manifest-sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61 --expect-run2-manifest-sha256 ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed --split-file data\interim\r2_inputs\subject_splits.csv --raw-dir data\interim\r2_inputs\raw --precheck-only 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

**Step 3a: real run, part 1 = stage 13 (about 33–39 min; hard limit 60 min).** Only run it if Step 2 passed. Cloud file syncing must still be paused. The script first checks on validation rows that the frozen Run 2 winner reproduces Run 2. Expect progress lines (`stage 13: ... done`) and the audit text, then a last line `result: RUN 3b1 (13) OK; RELEASE GATE OK` and `EXIT:0`.

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s scripts\stage13_18_r3\r3b_13_16_18_attribution.py --part 1 --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 --expect-run1-manifest-sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61 --expect-run2-manifest-sha256 ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed --split-file data\interim\r2_inputs\subject_splits.csv --raw-dir data\interim\r2_inputs\raw 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

**Step 3b: real run, part 2 = stages 18 and 16 (about 41–45 min; hard limit 60 min).** Run it after Step 3a passed (or, if Step 3a failed, only when the assistant says so; the parts are independent). Cloud file syncing must still be paused. Expect progress lines (`stage 18: attribution at landmark ... done`, then `stage 16: landmark ... done`) and the audit text, then a last line `result: RUN 3b2 (16, 18 attribution) OK; RELEASE GATE OK` and `EXIT:0`.

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s scripts\stage13_18_r3\r3b_13_16_18_attribution.py --part 2 --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 --expect-run1-manifest-sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61 --expect-run2-manifest-sha256 ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed --split-file data\interim\r2_inputs\subject_splits.csv --raw-dir data\interim\r2_inputs\raw 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

* On `EXIT:2`, nothing was written.
* On `EXIT:3`, the console names any pending folder left behind. Do not open or delete anything; report it. An unexpected error prints `frame <file>:<line>:<function>` lines (no data); send them back as they are.
* After Step 3b ends (or after a failure that stops the session), whatever the result, resume cloud file syncing.

**Step 4: report.** Open only the two TEAM audits:

```
Set-Location '<PROJECT_ROOT>'; Get-Content data\processed\ladder\stage13_r3b1\stage13_r3b1_audit_TEAM.md
Set-Location '<PROJECT_ROOT>'; Get-Content data\processed\ladder\stage16_18_r3b2\stage16_18_r3b2_audit_TEAM.md
```

Send back the full console text of Steps 0–3b and the contents of the two `*_audit_TEAM.md` files only. Do not open, paste or upload any `*_INTERNAL` file or any other output file.
