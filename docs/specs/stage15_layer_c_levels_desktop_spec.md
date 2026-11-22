---
title: "Desktop Tier 3 — Stage 15 Layer C levels (raw, normalized, clinically engineered)"
subtitle: "stage15_layer_c_levels_desktop_spec v1.0.1-desktop · script r5_desktop_layer_c_levels v1.0.1-desktop · plan v13 / contract v13 · PIC Foundations · Nov 21, 2026"
---

# 1. Summary

**This is the desktop Tier 3 Stage 15 Layer C level comparison (v1.0.1-desktop).** It is a labelled post-primary sensitivity analysis (a second look at the test set), not a replacement for the laptop Run 2 GRU or the laptop Run 3a primary scoring. Nested cross-validation is not used (D20: validation-only tuning). A published laptop or desktop Run 3a/3b or Tier 2 does not block this run, but **a pending folder from Tier 1 or Tier 2 under `DESKTOP_FULL_PLAN` does**: this run must not share the desktop with those runs.

Every file the script writes goes under `data\processed\ladder\DESKTOP_FULL_PLAN\stage15_layer_c_levels\` only (published folder `published\`, pending folders, cache, test-use log, internal files, logs, and `README_DESKTOP.txt`). There is no runtime cap. Torch uses CUDA if available, otherwise CPU (logged). Mixed precision is off unless `--amp` is set and the device is CUDA.

**v1.0.1.** The same TEAM-file issue as the Tier 2 v1.0.0 desktop run: a GPU name such as `NVIDIA GeForce RTX 3080` would trip the identifier gate. v1.0.1 keeps the GPU name, GPU memory, the Python executable path, and other host environment strings out of TEAM files (they are written only to `stage15_layer_c_levels_environment_INTERNAL.json`). TEAM files may report device `cuda` or `cpu` and a coarse cuda yes/no flag. The identifier gate itself is unchanged. A prior unpublished test-use attempt does not block a rerun; a published attempt still does. When a prior unpublished attempt exists, the audit logs that the test set was scored once before in that unpublished attempt.

The script, `scripts\stage13_18_r3_desktop\r5_desktop_layer_c_levels.py` (`r5_desktop_layer_c_levels v1.0.1-desktop`), plus its tests, builds three Layer C representations at each of the seven core landmarks (24, 48, 72, 120, 168, 240, 336 h) and compares them with the D19 GRU (Layer A+B static context at every step, Layer A baseline-risk offset, CORAL-style cumulative-link head). The three levels are:

1. **raw** — Stage 08 `X_raw_z` (already train-standardized, with NaN where missing), plus the missing mask, the carried-forward flag, and the scaled bin index. Missing values are filled with 0 for the network only; the mask is the model's way of seeing what was unobserved. Stage 08 does not publish native-unit Layer C arrays, so this is not an unstandardized raw matrix.
2. **normalized** — the Stage 08 base / D17 representation used by the Run 2 GRU: `X_filled_z`, missing, carried-forward, and the bin index. The script checks that this packed tensor is identical to `R2.c_inputs` (the Run 1 deduplicated channel list).
3. **engineered** — the normalized backbone plus a frozen clinical block built only from train-fit statistics and past-only windows (see §3).

The GRU hidden size is chosen from `{32, 64}` on validation QWK, as in Run 2. Early stopping uses a 10% train-subject holdout. The official seed is 6310 plus the landmark, a level offset, and a family offset. An optional `--second-family` (`lstm`, `lstm_attn`, or `transformer`) can add the matching small grid from the pinned Tier 2 module `r4_desktop_seq_models.py` (`85980e52…5c74`); the selected model at each level and landmark is then the highest validation QWK among the GRU grid and that second family. **This choice has to be made before the single test scoring.** The default is GRU only.

Level comparisons use the D18 thresholds: `delta = 0.02` (QWK equivalence, two one-sided tests) and `delta_C = 0.01` (engineering gain). Contrasts are paired subject-cluster bootstrap intervals (`B_eval = 1000`). Holm step-down at `alpha = 0.05` is applied to the three primary contrasts (engineered − normalized, normalized − raw, engineered − raw) across the seven landmarks. Validation contrasts are descriptive because the same validation set chose hyperparameters; the locked test look is the sensitivity analysis. Frozen Run 2 GRU and A+B scores are reported as context.

The Run 2 conventions carry over:

* a hash-gated precheck (Stage 06–08, Run 1, canonical, harmonization, split, raw ICUSTAYS / ADMISSIONS), then the **Run 2 manifest** (`--expect-run2-manifest-sha256` and the pin `ec394345…28ed`) and every file it lists;
* the Run 2 script is imported only after its SHA-256 equals the pin `b0af3767…0a40`, and no bytecode is written. Run 2 model files are read with a restricted unpickler. The Run 2 file is never edited;
* a pending folder published atomically, and the fail-closed release gate with the Run 2 rules (structural-column whitelist, integer-cell identifier rule, collect-all reporting by file and location without values, size scan). Outputs are an aggregate-only audit and a manifest;
* TEAM and INTERNAL outputs. Library caches go to `_cache_stage15_layer_c_levels\`;
* a progress line or heartbeat at least every 30 s, and no time cap (`--max-seconds` defaults to 0);
* exit codes: 0 ok; 2 input stop (nothing published); 3 build or gate failure (nothing published);
* `--split-file` and `--raw-dir` point at the copies under `data\interim\r2_inputs\`.

# 2. What is not in this run

* Nested cross-validation.
* Outcome 2, ordinal8, and the non-core landmarks.
* The full plan ladder `c0 / c1 / c2s / c2d / c2p` as five separate levels. Engineered is one level: the normalized backbone plus organ-block summaries and a pSOFA-like score.
* Per-bin vasoactive state, ventilation state, the thirteen drug groups, microbiology M1, and surgery in the bin of *T*. Stage 08 Layer C has no event channels, and plan §6.3 records that PIC has no ventilation source item. Layer B 24 h drug flags remain in the A+B static context only.
* Native-unit KDIGO creatinine cuts. Stage 08 publishes only z-scored Layer C values, and the fitted-transform file is team-internal and is not opened. The creatinine stage in this package is a labelled z-space surrogate (see §3).
* Imputation arms (Stage 15i), still deferred.

# 3. The engineered clinical block

Statistics used for age-band thresholds are fit on **train rows only**, over the whole landmark prefix (those bins are known at *L*). Per-encounter features at bin *k* never use that encounter's bins after *k*. A planted look-ahead check (perturbing a later bin with the train statistics frozen) is an acceptance gate.

**Age-band 2 SD flags.** Layer C z-scores are pooled per channel on the landmark prefix (Stage 08), not age-banded. The thresholds are therefore 2 SD around the train age-band mean of those already-pooled z-scores. The bands are contract §8.1: neonate (<28 d), infant (<1 y), toddler (1–2 y), preschool (3–5 y), school age (6–11 y), adolescent (12–17 y), plus `age_invalid` (D20). A band with fewer than 5 observed train cells falls back to the overall train mean and SD. Age comes from Layer A (`a0_age_months`, `a0_age_invalid`).

**Windows.** Rolling SD uses a strictly past 4-bin window (bins *k*−4 to *k*−1) and needs at least 3 observed, not-carried values. Slope, worst-in-window, and time-above/below use the current bin and the 3 before it (the past 24 h). Slope needs 2 observed points. Carried-forward values are excluded from slope, rolling SD, and time-above/below (they are not new observations). AKI staging uses available values, including carried creatinine, as the plan requires.

**Core channels** (a feature is emitted only when the channel exists in the Stage 08 tensor at that landmark): heart_rate, map, sbp, dbp, respiratory_rate, spo2, temperature, creatinine, urine_output, bun, cystatin_c, sodium, potassium, chloride, bicarbonate, calcium_total, calcium_ionized, magnesium, phosphate, ph, pco2, po2, lactate, base_excess, sao2_bloodgas, hemoglobin, wbc, platelets, anc, inr, ptt, bilirubin_total, bilirubin_direct, ast, albumin, ggt, glucose, glucose_bedside, fluid_intake, pain_score.

For each present core channel the block includes:

* `{ch}_slope_4bin` — OLS slope of observed z over bins *k*−3..*k*;
* `{ch}_rollsd_4bin_past` — SD of observed z over bins *k*−4..*k*−1;
* `{ch}_time_above_2sd_24h` and/or `{ch}_time_below_2sd_24h` — share of observed bins in the 24 h window on the worse side of the age-band 2 SD line (high-worse, low-worse, or both if two-sided);
* `{ch}_worst_24h` — direction-aware worst observed z in the 24 h window (high-worse: max; low-worse: min; two-sided: farthest from the age-band mean).

Hours since the last true observation, capped at 8 bins, are added for heart_rate, map, respiratory_rate, spo2, temperature, creatinine, urine_output, and lactate.

**Organ-state flags and block summaries** use Table 5.1 membership (channel map v2.3), for the ten systems cardiovascular, respiratory, renal, hepatic, gi, hematologic, infectious, neurologic, electrolytes, and endocrine. For each system that has at least one member in the tensor: an in-bin flag, a 24 h flag, the mean observed z of members, and the signed deviation of the farthest member.

**KDIGO / AKI.** `aki_cr_stage_zsurrogate` is a labelled z-space rule: baseline is the minimum observed creatinine z in bins 0..*k*; stages 1, 2, and 3 are frozen at baseline-deltas of 0.5, 1.0, and 1.5. `cr_delta_baseline_z` is the continuous delta. Stage 08 `kdigo_uo_stage` is used when the channel is present (it is dropped at 24 h for no train variation). `kdigo_combined_stage` is the max of the creatinine surrogate and an integer urine-output stage when that stage is recoverable (not when it is only a pooled z-score).

**pSOFA-like score.** Four available organ components (MAP low, platelets low, bilirubin high, creatinine high) plus lactate as a documented fifth (respiratory P/F is dropped by D29; per-bin vasoactive exposure is unavailable). Component points 0–4 are z-space surrogates (1, 2, 3, 4 age-band SDs on the worse side of the train mean), not official pSOFA cuts. `psofa_mod_total` is their sum.

**Fluid.** `fluid_net_cum` is the prefix cumulative sum of observed (fluid_intake z − urine_output z).

The TEAM file `stage15_engineered_channels_TEAM.csv` lists every catalogue name and its definition. Channels whose inputs are absent at a landmark are simply not emitted.

# 4. Synthetic checks and box timings

The tests are synthetic only. They include unit checks for age bands, slope recovery, strictly past rolling SD, look-ahead, train-only age-band statistics, the z-space AKI rule, carried-value handling, packing of raw vs normalized, the TestGuard, path confinement, hash gates, refuse-rerun, and a fail-closed release gate. A module-scoped end-to-end fixture builds a tiny Stage 06–08 / Run 1 / Run 2 project and runs this script at 24 h with 2 epochs.

* Tests on the box (`/tmp/venv314`, CPU): **32 passed in 81.97 s, EXIT 0**. Script SHA-256 `b90ecfbc…d93f`; tests SHA-256 `b70cd519…620d`. Neither the script nor the tests were changed after that pytest run. On a CUDA machine the CUDA-requirement and AMP-on-CPU unit tests are skipped (30 passed, 2 skipped).
* Full-settings synthetic run on the x3 fixture (7 core landmarks, 40 epochs, GRU only, CPU): **WALL 418.4 s (~7.0 min), rc=0, release gate ok.** Per-landmark GRU tuning: 24 h 25 s, 48 h 31 s, 72 h 34 s, 120 h 43 s, 168 h 43 s, 240 h 50 s, 336 h 79 s. Test scoring was about 2 s. Feature construction grows with the prefix (about 26 s at 336 h on this fixture).
* Desktop estimate on an RTX 3080 (GRU only, 7 landmarks, 40 epochs): **45 minutes to 2 hours typical, allow 4 hours.** Real PIC tensors are much larger than the synthetic fixture. Adding `--second-family transformer` would add several hours (see the Tier 2 package); LSTM or LSTM-attention would add less.

# 5. Deviations (logged)

The script logs the deviations in §1–§2 and in the TEAM audit: post-primary labelling; Outcome 1 and the seven core landmarks only; three levels rather than the five-rung plan ladder; raw = `X_raw_z` rather than native units; age-band thresholds on already-pooled z-scores; the frozen rolling-SD and 24 h windows; the z-space KDIGO surrogate and pSOFA-like components; omitted event channels; validation-only tuning; optional second architecture; writes confined under `DESKTOP_FULL_PLAN\stage15_layer_c_levels\`; no concurrent run with a Tier 1 or Tier 2 pending folder.

# 6. Decisions Vinai may still want to make

* Whether the first (and only) test-scoring run includes `--second-family` (`lstm`, `lstm_attn`, or `transformer`). The default is GRU only, which is the fair three-level comparison with the D19 model. The choice cannot be added later on the same test-use log.
* Whether a later revision should invert Stage 08 z-scores with the team-internal fitted transforms so that true native-unit KDIGO creatinine cuts can be used.
* Whether a later revision should rebuild event channels (thirteen drug groups, M1, surgery in the bin of *T*) from the Stage 3 builder / raw tables, which this package cannot do from the frozen Stage 08 Layer C tensor.
* Whether lactate as the fifth pSOFA-like component, and the z-space AKI cuts 0.5 / 1.0 / 1.5, should be kept or revised.

