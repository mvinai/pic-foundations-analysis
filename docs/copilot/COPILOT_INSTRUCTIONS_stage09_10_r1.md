# Copilot instructions — Run 1 (stages 09p, 09, 10), r1_09p_09_10_spectral_lowrank v1.0.2

Project root: `<PROJECT_ROOT>`. Paste this whole file to Copilot. Synthetic tests first; real data only through these unedited commands.

**Rules.**

* Run every command exactly as written. Each one starts with `Set-Location` to the project root, so it can be pasted on its own.
* Do not edit any file. Do not change, add or remove options.
* Stop at the first failure and send back the full console text only. A failure is any `EXIT:` other than 0, a hash that differs from the value shown, or an expected line that does not appear.
* After a failure, do not patch, retry with other options, or delete anything.
* Do not open any input or output data file. The only output file you may open is `data\processed\ladder\stage09_10_r1\stage09_10_r1_audit_TEAM.md` (Step 4).
* Never open or paste a file whose name contains `_INTERNAL` or `_raw_LOCAL`, and never open a `*.pending-*` folder.
* Steps 1–3 set `PYTHONNOUSERSITE=1` and run `python -s`. This is the documented invocation: it keeps Python from loading packages from the per-user site-packages folder (a psutil there, imported by scikit-learn's loky, caused the sandbox PermissionError). Do not drop it.
* **Hang rule:** the script prints a progress line or a heartbeat at least every 30 s. If nothing new appears for 5 minutes, press Ctrl+C once and send the console text.

**Step 0: pause cloud file syncing and check the package.**

1. *Pause cloud file syncing first* (taskbar cloud-sync icon → Settings/gear → Pause syncing → 2 hours). Keep it paused until Step 3 has finished.
2. List leftovers. List only; do not open anything:

```
Set-Location '<PROJECT_ROOT>'; Get-ChildItem data\processed\ladder -Force -Filter 'stage09_10_r1*' | Select-Object Name,Mode; Write-Host "LISTED"
```

3. If anything named `stage09_10_r1` or `stage09_10_r1.pending-*` is listed above `LISTED`, stop and report it. The folder `_cache_stage09_10_r1` (library caches and temp files, created since v1.0.1; no data) is not listed by this filter and is not a leftover.
4. **Vinai copies the package** into `scripts\stage09_18_r1\` (v1.0.2 replaces v1.0.1 there), so do not copy anything. This step is a hash check only. Each hash must equal the value on the line below it; if either differs, stop.

```
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage09_18_r1\r1_09p_09_10_spectral_lowrank.py -Algorithm SHA256
# 707dafc80827f6cd378d8ed18ec2db6c0e3ea68c9532d376b8ae6186e08fea2f
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage09_18_r1\test_r1_09p_09_10_spectral_lowrank.py -Algorithm SHA256
# fb9d442e77c86a3578202d08bede8d4bf96856d0f21b0bc9ef768d05bb8f69cf
```

**Step 1: tests (synthetic data only, about 1–2 min).** The tests build their own tiny synthetic data in a temporary folder. Expect `38 passed, 1 skipped` and `EXIT:0` (the skipped test simulates a read-only user profile and runs only on Linux).

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s -m pytest -q -p no:cacheprovider scripts\stage09_18_r1\test_r1_09p_09_10_spectral_lowrank.py 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

**Step 2: precheck (hashes only; writes only the cache folder; under 1 min).** Expect `EXIT:0` and these three lines:

* `result: PRECHECK OK (Stage 06-08 manifest, every file it lists, canonical file, packages); nothing written (library cache folder only)`
* `stage06_08_manifest.json sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9`
* `canonical sha256 e5feeb22b5457ab89c4c73235e6a4722d538aec5f88595b6f049004171286418`

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s scripts\stage09_18_r1\r1_09p_09_10_spectral_lowrank.py --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 --precheck-only 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

**Step 3: real run (about 33–35 min; hard limit 60 min).** Cloud file syncing must still be paused. Expect progress lines and the audit text, then a last line `result: RUN 1 (09p, 09, 10) OK; RELEASE GATE OK` and `EXIT:0`.

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "python -s scripts\stage09_18_r1\r1_09p_09_10_spectral_lowrank.py --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

* On `EXIT:2`, nothing was written.
* On `EXIT:3`, the console names any pending folder left behind. Do not open or delete it; report it. An unexpected error prints `frame <file>:<line>:<function>` lines (no data); send them back as they are.
* After Step 3 ends, whatever the result, resume cloud file syncing.

**Step 4: report.** Open only the TEAM audit:

```
Set-Location '<PROJECT_ROOT>'; Get-Content data\processed\ladder\stage09_10_r1\stage09_10_r1_audit_TEAM.md
```

Send back the full console text of Steps 0–3 and the contents of `stage09_10_r1_audit_TEAM.md` only. Do not open, paste or upload `stage10_matched_width_bases_INTERNAL.npz` or any other output file.
