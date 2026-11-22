---
title: "Run 1 — stages 09p, 09 and 10 (spectral diagnostics, low rank, CUR, angles, recovery)"
subtitle: "stage09_10_r1_spec v1.0.2 · script r1_09p_09_10_spectral_lowrank v1.0.2 · plan v13 / contract v13 · PIC Foundations · Nov 18, 2026"
---

# 1. Summary

Run 1 of the stage 09–18 plan (`stage09_18_plan.md`, decisions D16–D21 accepted Nov 18) does stage 09p, stage 09 and stage 10 in one pass on the Stage 06–08 v1.2.1 release. It is one script, `r1_09p_09_10_spectral_lowrank.py`, plus its tests. The script follows the stage06_08 v1.2.1 builder conventions: hash-gated precheck, a pending folder published atomically, a release gate, an aggregate-only audit, a manifest, and TEAM / INTERNAL file split. Exit codes are 0 ok, 2 input stop (nothing written) and 3 build or release-gate failure (nothing published). It prints progress or a heartbeat at least every 30 s and has a hard limit of `--max-seconds 3600`. Seeds are deterministic (seed 6310, keyed per task).

* **Stage 09p (D15/D16).** Exactly identical columns on train rows (bit-identical in float32) are collapsed, keeping the first in canonical order. The by-design `__dup__` copies in b2d are never dropped. The same collapse is applied to the AB concatenations and the Layer C channels. Every dropped copy is listed in `stage09p_dedup_map_TEAM.csv`, and widths before and after are in `stage09p_widths_TEAM.csv`.
* **Stage 09 (plan §8).** For each of the eight levels it runs the SVD in four variants: dedup (headline), predup, full one-hot predup, and dedup without masks. It reports:
  * κ₂ and κ₂ over numerical rank;
  * numerical rank at max(n,p)·ε·σ₁, 1e-3 and 1e-2;
  * Belsley exact null directions and near-dependencies (η ≥ 30, two or more proportions > 0.5);
  * stable rank, σ² effective rank (headline) and σ effective rank;
  * k95 on σ² energy (headline) and σ energy;
  * a column-permutation null (95th percentile) and the count of σ above the MP edge;
  * power, exponential and rank-plus-noise decay fits compared by BIC;
  * a subject bootstrap (B = 200) of all of these, with Davis–Kahan / Wedin bounds against observed subspace angles and relative gaps.
* **Stage 09 Layer C (D17).** Uses the Stage 08 base tensor (`X_filled_z`, train only). Stacked channel spectra are computed at all 14 landmarks. The within-bin permutation null, the bootstrap and per-bin spectra (with top-5 angles to the stacked view) are computed at the seven core landmarks. The tensor is never flattened.
* **Stage 10 (plan §9).**
  * Eckart–Young identity checks.
  * Randomized SVD with oversampling 10 and s ∈ {0, 1, 2}.
  * CUR / CX for k ∈ {5, 10, 20, 30, 40, 59}, with uniform, leverage sampling, top leverage, CPQR and DEIM rules and both middles (C⁺AR⁺ and A(I,J)⁺). Results are reported as ratios to the rank-k optimum in the Frobenius and spectral norms.
  * The c needed to get within 1+ε, against the DMM 2008 order k ln k / ε².
  * CPQR column orders.
  * Principal angles between level column spaces (QR then SVD), with Gaussian, permutation and split-half nulls and bootstrap intervals.
  * Linear (ridge / logistic) against nonlinear (gradient boosting) recovery of the columns each level adds, using 5-fold CV grouped by subject.
* **Hypotheses H1–H5** are evaluated with the D18 thresholds frozen before any real run.

**Version history.**

* **v1.0.0 (Nov 18, 2026).** First build of stages 09p, 09 and 10.
* **v1.0.1 (Nov 18, 2026, sandbox fix).** Under Copilot's sandbox, v1.0.0 stopped with `BUILD FAILED (unexpected PermissionError at line 950)` right after hashing, at the first scikit-learn / matplotlib import. The cause was a psutil package in the per-user site folder (the per-user site-packages folder), which scikit-learn's loky imports. The documented invocation is now `$env:PYTHONNOUSERSITE='1'` with `python -s` (Steps 1–3), which skips the user site. v1.0.1 also made these changes:
  * Every library cache and temp location (`MPLCONFIGDIR`, `JOBLIB_TEMP_FOLDER`, `SCIKIT_LEARN_DATA`, `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `TMP` / `TEMP` / `TMPDIR` and Python's `tempfile`) points at `data\processed\ladder\_cache_stage09_10_r1\` inside the project (inside pytest's tmp folder in the tests). Matplotlib uses the Agg backend.
  * The heavy imports run at the start of `main` in a guard. A missing package is an input stop (exit 2) that names it; any other import error is exit 3 with frames.
  * Unexpected errors print every traceback frame as `basename:line:function` (with the module name for library frames) plus the chained cause, never messages. An `OSError` adds errno / winerror and a redacted file name.
  * Every `.npz` archive is read fully and closed at once; read-only files no longer block removal of a pending folder.
  * The cache folder is not a leftover and is never scanned or published; its `tmp` and `joblib` subfolders are emptied after each run.
  * On the laptop v1.0.1 passed the tests and the precheck and finished every stage (about 1,990 s), then the release gate refused to publish: real PIC identifiers are small integers, and a structural width in `stage09p_widths_TEAM.csv` equalled one of them.
* **v1.0.2 (Nov 18, 2026, release-gate fix).** Only the gate and its reporting changed; the numerical code is unchanged (on the synthetic fixture every TEAM CSV is byte-identical to v1.0.1). The identifier scan now uses the same structural / non-structural split as the size scan (Section 3), integer-valued floats are no longer identifier candidates, every problem is collected before one refusal, and the refusal names file, column (or `md text` / `json key <path>`) and check type, never the value.

**Runtime.** About 33–35 min on the laptop (v1.0.1 finished every stage in about 1,990 s on the real data before the gate refusal fixed in v1.0.2; the gate adds seconds; hard limit `--max-seconds 3600`). Box verification: v1.0.2 full end-to-end runs with the laptop package versions: on the synthetic 4,646-row Stage 06–08 v1.2.1 fixture with encounter identifiers remapped to 1..N and subject identifiers to 1..S (maximal collisions with widths, ranks and grids), 375 s (6.2 min), every acceptance check passes and the release gate is ok; on the original fixture, 329 s (5.5 min), and every TEAM CSV, figure and the INTERNAL npz are byte-identical to v1.0.1 (only version strings, script hash and timestamp differ in the audit). v1.0.0 scale test on the fixture tiled three times: 11.6 min (697 s), release gate ok. Tests: 39 passed in 33–35 s on the box: 39 of 39 pass with the box packages and with the laptop versions (Python 3.14.10, numpy 2.4.2, pandas 3.0.1, scipy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8) run with PYTHONNOUSERSITE=1 and python -s. On Windows the read-only-profile test is skipped.

# 2. Inputs and hash gates (precheck; nothing parsed before these pass)

* `data\processed\ladder\stage06_08\stage06_08_manifest.json`, pinned by its full SHA-256 on the command line (`--expect-stage06-08-manifest-sha256 d8efd1fedccc495ffb8adf51088d77e9298d8b9f46528034477cdc63a03671d9`). The manifest must name the builder `06_08_build_levels_preprocess v1.2.1` with SHA-256 `f55cdaf5…daec8e7`, release gate `ok` and every acceptance check `pass`, and it must list 42 files.
* The folder must hold exactly those 42 files plus the manifest (`desktop.ini` / `Thumbs.db` are ignored). Every file is hashed and compared with its manifest entry.
* `data\interim\canonical_icu_encounters.csv` must match both the hash recorded in the Stage 06–08 manifest and the plan pin `e5feeb22b5457ab89c4c73235e6a4722d538aec5f88595b6f049004171286418`. Only `encounter_id` and `SUBJECT_ID` are read, for subject-level resampling and grouped folds; the Stage 08 matrices carry no subject id.
* The output folder `data\processed\ladder\stage09_10_r1\` must be absent or empty, and no `stage09_10_r1.pending-*` folder may exist.
* `scikit-learn` and `matplotlib` must import.
* **Load checks:**
  * all levels share the encounter order and split;
  * each subject belongs to exactly one split;
  * every encounter appears in the canonical file;
  * the Layer C landmark encounters and splits match the tabular levels;
  * the train landmark sizes do not increase with L;
  * the eligible_24h train rows equal the L24 train rows.
* A failure on any of these returns **exit 2**.

# 3. Outputs and disclosure

TEAM files are aggregate only. They contain:

* relative singular values σᵢ/σ₁, never absolute σ, which would reveal n;
* ranks, widths, ratios, angles, R² / balanced accuracy, and percentile intervals of these;
* column names.

They contain no patient, row, encounter or subject counts, no identifiers, and no split or landmark sizes. The MP edge appears only as the count of σ above it.

| File | Content |
|---|---|
| `stage09p_dedup_map_TEAM.csv`, `stage09p_widths_TEAM.csv` | dedup map; widths before/after, dropped, `__dup__` kept |
| `stage09_summary_TEAM.csv`, `stage09_spectra_TEAM.csv` | spectral summary per level × variant; σᵢ/σ₁ and the permutation-null 95th percentile |
| `stage09_bootstrap_TEAM.csv`, `stage09_daviskahan_TEAM.csv` | bootstrap 2.5/97.5 percentiles, decay family stability; Davis–Kahan |
| `stage09_belsley_TEAM.csv` | exact null directions and near-dependencies (column names) |
| `stage09_layer_c_TEAM.csv`, `stage09_layer_c_bootstrap_TEAM.csv` | Layer C stacked and per-bin spectra; bootstrap |
| `stage10_eckart_young_TEAM.csv`, `stage10_rsvd_TEAM.csv` | identities; randomized SVD error ratios |
| `stage10_cur_TEAM.csv`, `stage10_cur_c_needed_TEAM.csv`, `stage10_cpqr_columns_TEAM.csv` | CUR/CX ratios; c needed vs DMM order; CPQR orders |
| `stage10_angles_TEAM.csv`, `stage10_angles_splithalf_TEAM.csv` | principal angles, nulls, bootstrap; split-half stability |
| `stage10_recovery_TEAM.csv`, `stage10_recovery_by_family_TEAM.csv` | linear vs nonlinear recovery |
| `stage09_10_hypotheses_TEAM.csv`, `fig09_1_spectra_TEAM.png`, `fig09_2_layer_c_stacked_TEAM.png` | H1–H5; figures |
| `stage09_10_r1_audit_TEAM.json` / `.md` | audit (the only file Copilot opens is the `.md`) |
| `stage10_matched_width_bases_INTERNAL.npz` | V (≤ 59 columns) and CPQR orders for stages 11 and 17 — **team-internal: never opened, pasted or shared** |
| `stage09_10_r1_manifest.json` | hashes, sizes, TEAM / INTERNAL lists |

**Release gate (v1.0.2).** Before publishing, the gate re-reads every TEAM text file from disk. It checks all files, collects every problem, and then refuses the release once (exit 3, nothing published). Each problem is reported as file, location (`column <name>`, `column names`, `md text` or `json key <path>`) and check type. The offending value is never printed.

1. **Column names.** A CSV column name looks like a count or an identifier (`rows`, `subject`, `encounter`, `patient`, `n`, `n_train`, `support`, `eligible`, `cohort`), or contains an identifier token.
2. **Structural columns are exempt from both value scans.** These are integer columns whose values are fixed by the feature widths, the k / c grids, configuration or the landmark / bin layout, so they cannot carry a patient count or an identifier: widths, `n_dropped`, `n_dup_kept`, `p`, ranks, k95, decay m, counts of σ above or below a threshold, indices, `direction`, `k`, `c`, `order`, `repeats`, `B`, `B_angles`, dimensions, zero-angle counts, c needed and its bounds, power iterations, oversampling, `landmark`, `bin`, `n_columns_evaluated`, `value`. v1.0.2 adds `weights` (Belsley variance-decomposition proportions, all in [0, 1]) and `eta` (Belsley condition indices σ₁/σⱼ, ratios of singular values rather than counts).
3. **Sizes.** In every other CSV column, any cohort, split, subject, eligible-24 h, landmark or stacked-row size ≥ 100 in plain or comma form, including a float such as `1234.0`. In the audit md / json the scan covers every number except integers held under the audit's structural whitelist, integers in the script's own fixed text, and the digits of the `generated_utc` timestamp.
4. **Identifiers.** In every other CSV column: a numeric cell is an identifier candidate only when it is written as an integer with no decimal point (`151`, not `151.0`), and it fails if it equals an encounter or subject identifier. Every non-numeric cell is token-scanned, and any whole token equal to an identifier fails (for example `S100001_0` inside text). In the audit md and json string values the same token scan runs on the free text; the only integers exempt are those listed in rule 3. A real leak (an identifier in a non-structural column, or in md / json free text) still fails.

The audit json must also pass a structural whitelist: every number sits under a width, rank, spectral, configuration or hypothesis key.

# 4. Decisions and logged deviations

* **D15/D16:** collapse exact duplicate columns everywhere except b2d `__dup__`, and report stage 09 before and after.
* **D17:** stage 15 is skipped; the Stage 08 base Layer C tensor is used.
* **D18:** the thresholds are frozen now (δ = 0.02 QWK/AUROC equivalence, δ_C = 0.01 Layer C gain, ε = 0.01 frozen rule; H1 κ₂ 100, H2 one third of p, H3 frequency 0.80, Belsley η 30 / proportion 0.5, H4 gap 0.05 / flag 0.30). The departure from plan §13.3 is logged.
* **D19:** one small GRU, in Run 2.
* **D20:** B = 200.
* **D21:** the assistant writes the scripts.

**Deviations, all printed in the audit:**

* The permutation nulls use 100 draws per matrix (plan: 200).
* Random CUR rules use 50 repeats (spectral norm on the first 10).
* CUR uses r = c = k, with row rules mirroring the column rules.
* The DMM bound is reported in order form.
* Recovery targets are the columns a level adds, capped at 6 per (level, family) and fitted on a subject-level subsample of about 3,000 train rows. kNN MI is deferred.
* The Layer C per-bin patient-shuffle null moves to stage 16, since a row permutation leaves the stacked spectrum unchanged.
* The Layer C bootstrap and null run at the core landmarks only.
* The between-level angle bootstrap uses the first 50 of the 200 shared replicates. Spectra and Davis–Kahan use all 200, computed by QR then the SVD of R.
* Largest singular values used only as residual norms come from the top eigenvalue of the small Gram matrix. Every reported spectrum comes from an SVD of the matrix itself.

# 5. Acceptance checks (all must pass, otherwise exit 3)

* Planted self-test: rank 10 is recovered by k95 and by the permutation null, and Belsley names a planted range column.
* `09p_b2d_dup_copies_never_dropped`, `09p_b2d_nondup_equals_b2s_after_dedup` and `09p_dropped_copies_identical_on_train`.
* `09_svd_reconstruction`, and the H1 rank identity both predup and dedup: the b2d minus b2s rank deficiency equals the number of `__dup__` copies.
* `09_bootstrap_subject_clustered`.
* `10_eckart_young_identities`, `10_cur_ratio_at_least_1`, `10_rsvd_rank_k_error_at_least_sigma_k1`, `10_angles_self_zero` and `10_recovery_folds_grouped_by_subject`.

# 6. Copilot steps (Windows, project root `<PROJECT_ROOT>`)

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


# 7. Verification on the box (synthetic data only)

* 39 pytest tests pass in 33–35 s on the box: 39 of 39 pass with the box packages and with the laptop versions (Python 3.14.10, numpy 2.4.2, pandas 3.0.1, scipy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8) run with PYTHONNOUSERSITE=1 and python -s. On Windows the read-only-profile test is skipped. They cover:
  * the hash gate (manifest SHA mismatch or missing, tampered listed file, extra file, desktop.ini ignored, canonical mismatch, plan pin);
  * refusal of a non-empty output folder or a pending folder;
  * dedup with the b2d exemption;
  * the planted self-test; Belsley on a range column;
  * Eckart–Young; CUR ratio ≥ 1; the Gram-based spectral norm and QR-then-SVD against a direct SVD; principal angles;
  * the release gate catching sizes and identifiers, including fractions, hex digests and structural columns;
  * v1.0.2: a full end-to-end run with encounter and subject identifiers remapped to 1..N (maximal collisions) that passes the gate; planted leaks (an identifier in a non-structural CSV column, in md free text and in a json string) reported by file and column / location without the value; integer cells against integer-valued floats; and all problems collected in one refusal;
  * the audit whitelist;
  * v1.0.1: frame printing (basenames, line, function and module only; no messages; chained cause; OSError file names redacted), the cache / temp environment setup and restore, the guarded import report, the cache folder not counting as a leftover, `.npz` handles closed before a rename, read-only file removal, and a precheck in a subprocess with a read-only HOME / USERPROFILE (Linux only; nothing is written to the profile);
  * an end-to-end run on a self-generated tiny synthetic release: it publishes, the manifest hashes match, every acceptance check passes, TEAM files carry no sizes or IDs, it is deterministic across two runs, and a re-run into the existing output is refused.
* v1.0.2 full end-to-end runs with the laptop package versions: on the synthetic 4,646-row Stage 06–08 v1.2.1 fixture with encounter identifiers remapped to 1..N and subject identifiers to 1..S (maximal collisions with widths, ranks and grids), 375 s (6.2 min), every acceptance check passes and the release gate is ok; on the original fixture, 329 s (5.5 min), and every TEAM CSV, figure and the INTERNAL npz are byte-identical to v1.0.1 (only version strings, script hash and timestamp differ in the audit). v1.0.0 scale test on the fixture tiled three times: 11.6 min (697 s), release gate ok.
