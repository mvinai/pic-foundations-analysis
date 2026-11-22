# Copilot instructions — Desktop Tier 3 Stage 15 Layer C levels

**v1.0.1 (21 Nov 2026).** Same TEAM-file fix as desktop Tier 2: GPU name, GPU memory, and other host environment strings are kept out of TEAM files (INTERNAL environment record only). TEAM files may report device `cuda` or `cpu` and a coarse cuda yes/no flag. Use these v1.0.1 commands. Every Python command uses the desktop `ml-gpu` venv (numpy is now 2.4.2): `<VENV_PYTHON> -s`.

Paste-ready steps for `r5_desktop_layer_c_levels v1.0.1-desktop`. Run these in PowerShell in VS Code Copilot. Every command starts with `Set-Location` to the project root. Use `$env:PYTHONNOUSERSITE='1'` and `<VENV_PYTHON> -s`.

**Do not start this run while Tier 1 or Tier 2 is still running.** A pending folder under `data\processed\ladder\DESKTOP_FULL_PLAN\` from another desktop run will stop the precheck. Finish or clear those runs first, then start this one.

Project root:

`<PROJECT_ROOT>`

This is a labelled post-primary sensitivity analysis. It does not replace the laptop Run 2 GRU or the laptop Run 3a primary scoring. Test rows are scored once. Nested CV is not used. The default is the D19 GRU only; if you want `--second-family lstm`, `lstm_attn`, or `transformer`, add that flag in Step 3 **before** the single test scoring. You cannot add it later on the same test-use log.

Outputs go only under `data\processed\ladder\DESKTOP_FULL_PLAN\stage15_layer_c_levels\`. Open only TEAM files after the run. Never open, paste, or share `*_INTERNAL*` files.

## Step 0: pause cloud file syncing and check hashes

Pause cloud file syncing for the project folder (so a long run is not interrupted by sync). Then check that the unzipped files match these SHA-256 values exactly.

```
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage13_18_r3_desktop\r5_desktop_layer_c_levels.py -Algorithm SHA256
# b90ecfbc8e2b460346241d305d768d9410db9182b663c2e8a290919b5248d93f
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage13_18_r3_desktop\test_r5_desktop_layer_c_levels.py -Algorithm SHA256
# b70cd519bc46a0150b6e5e73ccf61fbcea0b264acf3f4cedffd022fbafb8620d
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage11_18_r2\r2_11p_11_12_17_models.py -Algorithm SHA256
# b0af376745ee5e828e9fed041eae45918de13a7d33f8edddd36971452dc80a40
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage11_18_r2\test_r2_11p_11_12_17_models.py -Algorithm SHA256
# 3a69f8fe3225a792a274048baa37ec0826c64479572b978172a2e63ddb056fd1
Set-Location '<PROJECT_ROOT>'; Get-FileHash data\processed\ladder\stage11_12_17_r2\stage11_12_17_r2_manifest.json -Algorithm SHA256
# ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed
```

If you plan to use `--second-family`, also confirm the Tier 2 script:

```
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage13_18_r3_desktop\r4_desktop_seq_models.py -Algorithm SHA256
# 7878e8bf9f161bf5b1091d72140a3b83aca6702d8b657ed017996f742e054dfb
```

Confirm there is no leftover `stage15_layer_c_levels` published or pending folder, and no other `*.pending-*` folder under `data\processed\ladder\DESKTOP_FULL_PLAN\`.

## Step 1: tests (synthetic data only, about 2–4 min)

The tests build their own tiny synthetic data in a temporary folder. They also use the Run 2 test helper in `scripts\stage11_18_r2\`. Expect `32 passed` and `EXIT:0`, or `30 passed, 2 skipped` if CUDA is available (the CUDA-requirement and AMP-on-CPU tests are skipped on a GPU machine).

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "<VENV_PYTHON> -s -m pytest -q -p no:cacheprovider scripts\stage13_18_r3_desktop\test_r5_desktop_layer_c_levels.py 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

Stop and paste the output if anything other than `32 passed` (or `30 passed, 2 skipped` on a CUDA machine) and `EXIT:0` appears.

## Step 2: precheck only

This hashes the frozen inputs and stops before any model is fit. Expect `PRECHECK OK` and `EXIT:0`.

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "<VENV_PYTHON> -s scripts\stage13_18_r3_desktop\r5_desktop_layer_c_levels.py --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 --expect-run1-manifest-sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61 --expect-run2-manifest-sha256 ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed --split-file data\interim\r2_inputs\subject_splits.csv --raw-dir data\interim\r2_inputs\raw --precheck-only 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

## Step 3: the desktop run

Keep cloud file syncing paused. Default (recommended for this first scoring): GRU only, all seven core landmarks, 40 epochs, CUDA if available. Typical desktop time on an RTX 3080: **45 minutes to 2 hours**; allow **4 hours**. Heartbeats print at least every 30 seconds. Expect `result: RUN stage15_layer_c_levels desktop (Tier 3) OK; RELEASE GATE OK; post-primary sensitivity analysis` and `EXIT:0`.

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "<VENV_PYTHON> -s scripts\stage13_18_r3_desktop\r5_desktop_layer_c_levels.py --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 --expect-run1-manifest-sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61 --expect-run2-manifest-sha256 ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed --split-file data\interim\r2_inputs\subject_splits.csv --raw-dir data\interim\r2_inputs\raw --require-cuda --amp 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

If you have already decided to add a second architecture, append `--second-family lstm` (or `lstm_attn` or `transformer`) and `--r4-script scripts\stage13_18_r3_desktop\r4_desktop_seq_models.py` to that same command. Do not start a second run after test has been scored.

If CUDA is not available, drop `--require-cuda` and `--amp`; the script will use CPU.

## Step 4: open only the TEAM audit

After `EXIT:0`, open only:

`data\processed\ladder\DESKTOP_FULL_PLAN\stage15_layer_c_levels\published\stage15_layer_c_levels_audit_TEAM.md`

You may also open the other `*_TEAM.csv` / `*_TEAM.json` / `*_TEAM.png` files in that published folder. **Do not open any `*_INTERNAL*` file.** Paste the TEAM audit here if you want it checked.

## Step 5: resume cloud file syncing

Resume cloud file syncing for the project folder.

