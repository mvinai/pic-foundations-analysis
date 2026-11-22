---
title: "Desktop Tier 2 — LSTM, LSTM with attention, and a small transformer for Layer C"
subtitle: "stage17_seqmodels_desktop_spec v1.0.1-desktop · script r4_desktop_seq_models v1.0.1-desktop · plan v13 / contract v13 · PIC Foundations · Nov 21, 2026"
---

# 1. Summary

**This is the desktop Tier 2 sequence-model sensitivity (v1.0.1-desktop).** It is a post-primary sensitivity analysis (a second look at the test set), not a replacement for the laptop Run 2 GRU or the laptop Run 3a primary scoring. A published laptop or desktop Run 3a/3b does not block this run. Every file the script writes goes under `data\\processed\\ladder\\DESKTOP_FULL_PLAN\\stage17_seqmodels\\` only (published folder `published\\`, pending folders, cache, test-use log, internal files, logs, and `README_DESKTOP.txt`). There is no runtime cap. Torch uses CUDA if available, otherwise CPU (logged). Mixed precision is off unless `--amp` is set and the device is CUDA.

**v1.0.1.** A real desktop run of v1.0.0 (21 Nov 2026) finished tuning and test scoring at all seven landmarks, then failed the release gate because the GPU name `NVIDIA GeForce RTX 3080` in the TEAM audit was tokenised as the identifier `3080`. v1.0.1 keeps the GPU name, GPU memory, the Python executable path, and other host environment strings out of TEAM files (they are written only to `stage17_seqmodels_environment_INTERNAL.json`). TEAM files may report a coarse cuda yes/no flag. The identifier gate itself is unchanged. The test-use log already allowed a rerun after a failed, unpublished attempt and still refuses after a published one; when a prior unpublished attempt exists, the audit logs that the test set was scored once before in that unpublished attempt.

The script, `scripts\\stage13_18_r3_desktop\\r4_desktop_seq_models.py` (`r4_desktop_seq_models v1.0.1-desktop`), plus its tests, fits three deeper A+B+C sequence models at each of the seven core landmarks, using the same inputs, Layer A+B static context at every step, Layer A baseline-risk offset and CORAL-style cumulative-link head as the Run 2 GRU (D19). The recurrent encoder is swapped for:

1. an LSTM (last valid step);
2. an LSTM with masked additive attention pooling over time steps;
3. a small transformer encoder (2 or 4 layers, `d_model` 64 or 128, 4 heads, sinusoidal or learned positional encoding, learned CLS token, padding mask).

A small fixed validation grid is used for each family. Early stopping uses a 10% train-subject holdout, as in Run 2. The official seed is 6310 plus the landmark and a family offset. The selected configuration of each family is the one with the highest validation QWK. Those selected models are then compared with the frozen Run 2 GRU and A+B on validation (paired subject-cluster bootstrap, `B_eval = 1000`, `delta_C = 0.01`). Test scoring of the selected configurations happens once, after selection, and is labelled a post-primary sensitivity analysis. `--n-seeds` (default 1) optionally refits the already-selected configuration with additional seeds as a stability check; it does not change the selected grid point.

The Run 2 conventions carry over:

* a hash-gated precheck. It runs Run 2's own precheck (Stage 06–08 manifest, Run 1 manifest and bases, canonical, harmonization, split, raw ICUSTAYS / ADMISSIONS). It then checks the **Run 2 manifest** (`--expect-run2-manifest-sha256` and the pin `ec394345…28ed`), and hashes every file the Run 2 manifest lists;
* the Run 2 script is imported only after its SHA-256 equals the pin `b0af3767…0a40`, and no bytecode is written. Run 2 model files are read with a restricted unpickler. The Run 2 file is never edited;
* a pending folder published atomically, and the fail-closed release gate with the Run 2 rules: structural-column whitelist, integer-cell identifier rule, collect-all reporting by file and column without values, and a size scan. Outputs are an aggregate-only audit and a manifest;
* TEAM and INTERNAL outputs. Library caches and temp files go to `data\\processed\\ladder\\DESKTOP_FULL_PLAN\\stage17_seqmodels\\_cache_stage17_seqmodels\\`;
* a progress line or heartbeat at least every 30 s, and no time cap (`--max-seconds` defaults to 0; the flag is only for tests);
* exit codes: 0 ok; 2 input stop (nothing published); 3 build or gate failure (nothing published);
* `--split-file` and `--raw-dir` point at the copies under `data\\interim\\r2_inputs\\`. Their hashes must equal the pins and the Run 2 manifest's input hashes.

**Test-use log.** `data\\processed\\ladder\\DESKTOP_FULL_PLAN\\stage17_seqmodels\\test_use_log_stage17_seqmodels.json` gets a "started" entry just before the first test row is read. When the run ends, that entry becomes "published" (with the manifest SHA-256) or "failed (nothing published)". A later run refuses to start if a "published" entry exists in this log. This log is separate from the Run 3a desktop log. Failed attempts do not block a rerun, but the audit counts them. If verification or tuning fails, no test row has been read and the log is not touched.

**Environment check.** The script checks Python 3.14 and the major.minor versions of numpy 2.4, pandas 3.0, scipy 1.17, scikit-learn 1.8 and torch 2.10, and stops on a mismatch. `--require-cuda` (used on the desktop) stops if CUDA is not available.

# 2. Models and grid

All three families share the Run 2 D19 head: a linear embedding (width 16) of the static A+B vector is concatenated at every time step; the Layer A ordinal-LR baseline score is an offset with coefficient 1; the cumulative logits are `gamma_j - (offset + g)`, with ordered `gamma`; training uses the CORAL binary cross-entropy, Adam, gradient clipping at 1, and early stopping (patience 5) on a 10% subject holdout of train. Default maximum epochs: 40. LSTM families use learning rate `2e-3` and no dropout, matching the Run 2 GRU. The transformer uses learning rate `1e-3`, dropout 0.1, GELU feed-forward width `4 * d_model`, and a learned CLS token.

**LSTM grid (4 points):** hidden in {32, 64}; 1 or 2 layers; last-step (masked) pooling.

**LSTM-attention grid (4 points):** the same hidden and layer grid; additive attention `e_t = v^T tanh(W h_t)`, softmax over valid time steps, then a weighted sum.

**Transformer grid (8 points):** 2 or 4 layers; `d_model` in {64, 128}; 4 heads; sinusoidal or learned positional encoding. A padding / key-padding mask is implemented; at a fixed landmark every eligible Stage 08 prefix has length `L/6`, so the mask is all-valid on this tensor.

Selection is per family and landmark, on validation QWK, official seed only. Outcome 1 only, at the seven core landmarks (24, 48, 72, 120, 168, 240, 336 h).

# 3. Comparisons and test scoring

After selection, each selected model is compared with the frozen Run 2 GRU and the frozen A+B landmark model on **validation**, using the same paired subject-cluster bootstrap as Run 2 (`B = 1000`) and the frozen threshold `delta_C = 0.01`. The frozen models must first reproduce the Run 2 validation predictions (within a probability tolerance of `1e-4`). That check, and all fitting, happens before any test row is read.

**Test scoring** then scores the selected configuration of each family, and the frozen GRU and A+B as reference comparators, once. Test contrasts are descriptive sensitivity only. A TestGuard refuses any fit on test rows and any test access outside the scoring phase.

`--n-seeds N` (default 1) refits each already-selected configuration with additional seeds as a stability check on validation. It does not rescore the test set for those extra seeds and does not change the selected grid point.

# 4. Outputs

Published under `data\\processed\\ladder\\DESKTOP_FULL_PLAN\\stage17_seqmodels\\published\\`:

* `stage17_seq_tuning_TEAM.csv` — validation metrics for every grid point, plus the frozen GRU and A+B;
* `stage17_seq_selected_TEAM.csv` — the selected configuration of each family, and the frozen comparators;
* `stage17_seq_val_contrast_TEAM.csv` — selected minus frozen GRU or A+B on validation;
* `stage17_seq_test_TEAM.csv` and `stage17_seq_test_contrast_TEAM.csv` — post-primary test scores and descriptive contrasts;
* `stage17_seq_multiseed_TEAM.csv` — only if `--n-seeds` is greater than 1;
* two TEAM figures (validation QWK by landmark; validation delta QWK vs the frozen GRU);
* `stage17_seqmodels_audit_TEAM.md` / `.json` and the manifest.

INTERNAL files (never opened, pasted or shared): selected states, validation/test predictions, and a host-environment record (GPU name, memory, executable path and package versions). Those host strings are not written to any TEAM file.

# 5. Deviations from the protocol (logged)

* This is a post-primary sensitivity analysis on the desktop, not a replacement for the laptop Run 2 GRU or the laptop Run 3a test scoring.
* Outcome 1 only, at the seven core landmarks; Outcome 2, ordinal8 and the non-core landmarks are not fit in this run.
* The Stage 15 Layer C feature levels and nested cross-validation remain deferred (Tier 3).
* Each family uses the small fixed grid in §2, not a full bake-off.
* Multi-seed refits, if requested, are applied only to the already-selected configuration.
* Validation is the primary comparison with the frozen Run 2 GRU and A+B; test contrasts are descriptive sensitivity only.
* A padding mask is implemented, but at a fixed landmark every eligible prefix has equal length.
* No runtime cap; mixed precision is off unless `--amp` is set on CUDA.
* All writes are confined under `data\\processed\\ladder\\DESKTOP_FULL_PLAN\\stage17_seqmodels\\`.

# 6. Tests and synthetic runtime

The pytest suite (`test_r4_desktop_seq_models.py`) covers write confinement under `stage17_seqmodels\\`, CPU determinism of the three trainers, the TestGuard (no test rows in tuning or in any fit), release-gate collect-all behaviour without leaking values, hash pins, precheck, the refuse-rerun lock, and a published laptop Run 3a not blocking this run. On the box, with the `/tmp/venv314` CPU torch build: **24 passed in 92.04 s, EXIT 0**. Script SHA-256 `7878e8bf…4dfb`; tests SHA-256 `2266a9a1…bb1d`. Neither the script nor the tests were changed after that pytest run. On a CUDA machine the CUDA-requirement unit test is skipped (23 passed, 1 skipped).

Synthetic end-to-end runs (pins nulled) on the existing x3 and gb fixtures, CPU only (`torch 2.10.0+cpu`). No synthetic jobs were run in parallel.

* x3 probe, landmark 24 h, 2 epochs: WALL 30.4 s, rc 0, release gate ok.
* x3 probe, landmark 336 h, 2 epochs: WALL 79.6 s, rc 0, release gate ok.
* x3 full default settings (7 core landmarks, 40 epochs, patience 5, `n_seeds = 1`, `B = 1000`): WALL 5079.3 s (1.41 h), rc 0, release gate ok. Test scoring and the gate took 11 s after tuning finished.
* gb reduced-epoch smoke only (`--max-epochs 2`, landmarks 24, 168 and 336 h): WALL 55.0 s, rc 0, release gate ok. A full-settings gb run was not repeated after the x3 full run.

Per-landmark wall time on the x3 full run (from the start of that landmark's tuning to the start of the next, or to the first test-scoring line):

* 24 h: 296.9 s (LSTM 16.1 s, LSTM-attention 16.9 s, transformer 258.6 s)
* 48 h: 666.4 s (16.0 s, 17.9 s, 628.5 s). One transformer configuration, `d128_L4_h4_learned`, took 354.0 s, an outlier relative to its 77.9 s sibling.
* 72 h: 485.6 s (18.5 s, 18.7 s, 444.5 s)
* 120 h: 775.1 s (27.9 s, 28.2 s, 716.0 s)
* 168 h: 809.5 s (29.2 s, 30.6 s, 747.1 s)
* 240 h: 915.9 s (29.9 s, 25.2 s, 858.1 s)
* 336 h: 1105.9 s (43.5 s, 18.0 s, 1041.4 s)

Family totals, summing the per-configuration training times: LSTM 181.1 s (28 fits, mean 6.5 s, every fit ran all 40 epochs); LSTM-attention 155.5 s (28 fits, mean 5.6 s, mean 36.0 epochs); transformer 4694.2 s (56 fits, mean 83.8 s, mean 38.4 epochs). The transformers accounted for about 92% of training time.

# 7. Desktop runtime estimate (RTX 3080)

These are estimates, not measured desktop times. The box runs were CPU-only (`torch 2.10.0+cpu`). The desktop has an NVIDIA RTX 3080 (10 or 12 GB) and 128 GB of RAM.

Assumptions: (1) these models are small, so a 3080 is not compute-bound at batch 256 and `T ≤ 56`; (2) GPU speedup for a single fit is about 3–10× over the box CPU, with the low end more likely if Python data loading leaves the GPU idle; (3) loading the Layer C tensor onto the GPU once, as this script does per landmark, and using `--amp` on CUDA, helps most; (4) early stopping (patience 5, max 40) may finish well before 40 epochs, as on the box; (5) `--n-seeds 1` is the default command; `--n-seeds 5` refits only the three selected configurations at each of 7 landmarks with 4 extra seeds (84 extra fits) and is estimated at about 1.5–1.8× the default wall time; (6) the 1.41 h figure is for a small synthetic fixture. The real Layer C tensors have substantially more training sequences, with the same maximum prefix length (56 bins of 6 h). Epoch time should scale roughly with the number of training sequences, so the real desktop run will take longer than a 3–10× speedup applied to the synthetic wall time.

**If the same synthetic x3 job were run on the 3080:** about 10–30 minutes (3–10× versus 1.41 h). That is a lower reference only; it is not a real-data estimate.

**Default desktop command on the real tensors (`--require-cuda`, `--n-seeds 1`, full grid, 7 landmarks, 40 epochs):** about 1–3 hours if the real tensors are only several times larger than this fixture and the GPU is well used, and about 3–6 hours if they are much larger or the GPU stays under-utilized. Allow 6 hours, overnight if anything is slower than this scaling suggests. Memory is not a constraint (well under 2 GB at these widths). `--n-seeds 5`, if time permits: about 1.5–5 hours, or up to about 8 hours at the slow end; allow overnight.

# 8. What this run does not do

It does not replace the laptop primary results. It does not build the Stage 15 Layer C levels (Tier 3). It does not run nested cross-validation. It does not fit Outcome 2 or ordinal8, or the non-core landmarks. It does not modify the pinned Run 2 module or the Tier 1 desktop package. It does not read Tier 1 desktop outputs, so it can run before or after the Tier 1 desktop 3a/3b package, as long as the two do not write at the same time.
