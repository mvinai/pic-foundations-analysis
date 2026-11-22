# Copilot instructions — Run 3a (stages 14 and 18: single test scoring), r3a_14_18_test_scoring v1.0.0

Project root: `<PROJECT_ROOT>`. Paste this whole file to Copilot. Synthetic tests first; real data only through these unedited commands. Run 2 v1.0.0 must already be published in `data\processed\ladder\stage11_12_17_r2\` (unchanged since its real run of Nov 19), and the Run 2 script must still be in `scripts\stage11_18_r2\`. The split and raw copies must still be in `data\interim\r2_inputs\`.

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
