# Copilot instructions — Desktop Tier 2 sequence models (LSTM, LSTM-attention, transformer), v1.0.1-desktop

**v1.0.1 (21 Nov 2026).** The first real desktop run of v1.0.0 completed tuning and test scoring at all seven landmarks, then failed the release gate because the GPU name `NVIDIA GeForce RTX 3080` appeared in the TEAM audit (`EXIT:3`, nothing published). This package keeps the GPU name, GPU memory, and other host environment strings out of TEAM files (they go only to an INTERNAL environment record). Use these v1.0.1 commands. A leftover `test_use_log_stage17_seqmodels.json` from that failed unpublished attempt is expected; this script allows a rerun and will record that the test set was scored once before. Do not start Step 3 if a `published` folder already exists.

Project root: `<PROJECT_ROOT>`. Paste this whole file to Copilot. Synthetic tests first; real data only through these unedited commands. Run 2 v1.0.0 must already be published in `data\processed\ladder\stage11_12_17_r2\` (unchanged since its real run of Nov 19), and the Run 2 script must still be in `scripts\stage11_18_r2\`. The split and raw copies must still be in `data\interim\r2_inputs\`.

**This is a post-primary sensitivity analysis, not a replacement for the laptop Run 2 GRU or the laptop Run 3a primary results.** It is a second look at the test set. A published laptop or desktop Run 3a/3b does not block this run. All files this script writes go under `data\processed\ladder\DESKTOP_FULL_PLAN\stage17_seqmodels\` only (published folder `published\`, pending folders, cache, this run's test-use log, internal files, and `README_DESKTOP.txt`). After a published Tier 2 run, that test-use log refuses to run this script again.

**Desktop environment.** This run is meant for the desktop (128 GB RAM, RTX 3080), which already has the `ml-gpu` venv at `<VENV_DIR>` (numpy is now 2.4.2). Every Python command below uses that interpreter. Compare the installed versions with the laptop list: Python 3.14, numpy 2.4.2, pandas 3.0.1, scipy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8, and a CUDA build of PyTorch 2.10 (laptop CPU torch is 2.10.0). Step 0 prints the versions and `torch.cuda.is_available()`. If any analysis package's major.minor version differs, or if CUDA is not available, send the console text back and stop. Do not upgrade, downgrade, or install anything.

The project folder is the same project folder the laptop uses. Do not copy patient-level inputs. Pause cloud file syncing before any real run. Do not run the laptop 3a/3b scripts or the desktop Tier 1 scripts at the same time as this run.

This run does not depend on Tier 1 desktop outputs (`DESKTOP_FULL_PLAN\stage14_18_r3a\` or `stage13_16_18_r3b\`). It reads the already-published laptop Run 2 folder, Stage 06–08, Run 1, and `data\interim\r2_inputs\`. It can therefore run before or after the desktop Tier 1 package, but not at the same time.

**Default command and time.** The default is `--n-seeds 1` (official selection seed only), all seven core landmarks, the fixed grids in the spec, and a maximum of 40 epochs with early stopping. Estimated desktop wall time for the real tensors: about 1–3 hours, or 3–6 hours if the real Layer C tensors are much larger than the synthetic fixture used to time the box, or if the GPU stays under-utilized. Allow 6 hours, overnight if anything is slower. If time remains after the default run, `--n-seeds 5` is optional and is a stability check only; it does not change the selected configurations. Estimated wall time with `--n-seeds 5`: about 1.5–5 hours, or up to about 8 hours at the slow end; allow overnight. Do not add `--n-seeds 5` to the commands below unless the user has already asked for it in this session after seeing the default estimate.

**Rules.**

* Run every command exactly as written. Each one starts with `Set-Location` to the project root, so it can be pasted on its own.
* Do not edit any file. Do not change, add or remove options.
* Stop at the first failure and send back the full console text only. A failure is any `EXIT:` other than 0, a hash that differs from the value shown, or an expected line that does not appear.
* After a failure, do not patch, retry with other options, or delete anything. Never rerun a real-run step (Step 3) on your own.
* Do not open any input or output data file. The only output file you may open is the TEAM audit in Step 4.
* Never open or paste a file whose name contains `_INTERNAL` or `_raw_LOCAL`, never open `test_use_log_stage17_seqmodels.json` (or any other test-use log), and never open a `*.pending-*` folder.
* Every Python step sets `PYTHONNOUSERSITE=1` and runs `<VENV_PYTHON> -s`. Do not drop it.
* **Hang rule:** the script prints a progress line or a heartbeat at least every 30 s. If nothing new appears for 5 minutes, press Ctrl+C once and send the console text. There is no runtime cap.

**Step 0: pause cloud file syncing, check versions and CUDA, and check the package.**

1. *Pause cloud file syncing first* (taskbar cloud-sync icon → Settings/gear → Pause syncing → 8 hours, or overnight if the estimate above is larger). Keep it paused until Step 3 has finished.
2. Check Python, the analysis packages, and that CUDA torch is visible. Expect Python 3.14, the package versions named above, and `torch.cuda.is_available()` True. If a version differs or CUDA is False, stop and send the console text. Do not upgrade or install anything.

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; <VENV_PYTHON> -s -c "import sys,numpy,pandas,scipy,sklearn,matplotlib,torch; print('python', sys.version); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('scipy', scipy.__version__); print('sklearn', sklearn.__version__); print('matplotlib', matplotlib.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu')); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

3. List leftovers under the Tier 2 output folder only. List only; do not open anything. Existing laptop or desktop Run 3a/3b folders (for example `data\processed\ladder\stage14_18_r3a\` or `data\processed\ladder\DESKTOP_FULL_PLAN\stage14_18_r3a\`) are expected and must be left alone.

```
Set-Location '<PROJECT_ROOT>'; if (Test-Path data\processed\ladder\DESKTOP_FULL_PLAN\stage17_seqmodels) { Get-ChildItem data\processed\ladder\DESKTOP_FULL_PLAN\stage17_seqmodels -Force | Select-Object Name,Mode } else { Write-Host "stage17_seqmodels absent" }; Write-Host "LISTED"
```

4. Only `README_DESKTOP.txt`, `_cache_stage17_seqmodels` (library caches and temp files; no data), and possibly `test_use_log_stage17_seqmodels.json` from the failed unpublished v1.0.0 attempt may appear above `LISTED`; they are not leftovers. If `published` or a `*.pending-*` folder is listed, stop and report the names. Do not open the test-use log. A leftover log from the failed v1.0.0 run does not block this v1.0.1 run.
5. **The assistant has already copied the package** into `scripts\stage13_18_r3_desktop\`, so do not copy anything. This step is a hash check only. Each hash must equal the value on the line below it; if any differs, stop.

```
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage13_18_r3_desktop\r4_desktop_seq_models.py -Algorithm SHA256
# 7878e8bf9f161bf5b1091d72140a3b83aca6702d8b657ed017996f742e054dfb
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage13_18_r3_desktop\test_r4_desktop_seq_models.py -Algorithm SHA256
# 2266a9a1a37940bc95a81cc1c5d5b55e57a212e9ee182b6a4e43b1be534fbb1d
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage11_18_r2\r2_11p_11_12_17_models.py -Algorithm SHA256
# b0af376745ee5e828e9fed041eae45918de13a7d33f8edddd36971452dc80a40
Set-Location '<PROJECT_ROOT>'; Get-FileHash scripts\stage11_18_r2\test_r2_11p_11_12_17_models.py -Algorithm SHA256
# 3a69f8fe3225a792a274048baa37ec0826c64479572b978172a2e63ddb056fd1
Set-Location '<PROJECT_ROOT>'; Get-FileHash data\processed\ladder\stage11_12_17_r2\stage11_12_17_r2_manifest.json -Algorithm SHA256
# ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed
```

**Step 1: tests (synthetic data only, about 2–4 min).** The tests build their own tiny synthetic data in a temporary folder; they also use the Run 2 test helper in `scripts\stage11_18_r2\`. Expect `24 passed` and `EXIT:0`, or `23 passed, 1 skipped` if CUDA is available (the CUDA-requirement test is skipped on a GPU machine).

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "<VENV_PYTHON> -s -m pytest -q -p no:cacheprovider scripts\stage13_18_r3_desktop\test_r4_desktop_seq_models.py 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

**Step 2: precheck (hashes only; reads no test row; writes only under `DESKTOP_FULL_PLAN\stage17_seqmodels\`, the cache folder and README_DESKTOP.txt; about 1 min).** Expect `EXIT:0` and these lines (the hashes are printed in this order):

* `result: PRECHECK OK (Stage 06-08, Run 1, Run 2 manifests and files, Run 2 script, canonical, harmonization, split, raw ICUSTAYS / ADMISSIONS, test-use log, packages); nothing written (library cache folder and README only)`
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
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "<VENV_PYTHON> -s scripts\stage13_18_r3_desktop\r4_desktop_seq_models.py --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 --expect-run1-manifest-sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61 --expect-run2-manifest-sha256 ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed --split-file data\interim\r2_inputs\subject_splits.csv --raw-dir data\interim\r2_inputs\raw --require-cuda --precheck-only 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

**Step 3: full run (real data; post-primary test scoring once; about 1–3 hours, allow 6 hours or overnight if slower).** Expect `EXIT:0` and the line `result: RUN stage17_seqmodels desktop (Tier 2) OK; RELEASE GATE OK; post-primary sensitivity analysis`. Stop on any other result. Do not patch.

```
Set-Location '<PROJECT_ROOT>'; $env:PYTHONNOUSERSITE='1'; $ErrorActionPreference='Continue'; cmd /c "<VENV_PYTHON> -s scripts\stage13_18_r3_desktop\r4_desktop_seq_models.py --expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9 --expect-run1-manifest-sha256 703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61 --expect-run2-manifest-sha256 ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed --split-file data\interim\r2_inputs\subject_splits.csv --raw-dir data\interim\r2_inputs\raw --require-cuda 2>&1"; Write-Host "EXIT:$LASTEXITCODE"
```

* After Step 3 ends, whatever the result, resume cloud file syncing.

**Step 4: report.** Open only the TEAM audit:

```
Set-Location '<PROJECT_ROOT>'; Get-Content data\processed\ladder\DESKTOP_FULL_PLAN\stage17_seqmodels\published\stage17_seqmodels_audit_TEAM.md
```

Send back the full console text of Steps 0–3 and the contents of `stage17_seqmodels_audit_TEAM.md` only. Do not open, paste or upload any `*_INTERNAL` file, the test-use log, or any other output file.
