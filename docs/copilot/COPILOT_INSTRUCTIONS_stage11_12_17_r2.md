# Copilot instructions — Run 2 (label build 11p, stages 11, 12, 17), r2_11p_11_12_17_models v1.0.0

Project root: `<PROJECT_ROOT>`. Paste this whole file to Copilot. Synthetic tests first; real data only through these unedited commands. Run 1 v1.0.2 must already be published in `data\processed\ladder\stage09_10_r1\` (unchanged since its real run).

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
