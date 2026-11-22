---
title: "Foundations Analysis Feature-Matrix Contract, v13"
subtitle: "PIC Foundations project; aligned with revised plan v13 (revised November 16, 2026)"
date: "November 16, 2026"
---

# Status

Refined contract for the Foundations representation study, aligned with **revised plan v13** (as revised on November 16, 2026, decisions V1–V30). It replaces contract v12 (November 9, 2026). The v12 file is kept unchanged for provenance.

This version keeps the encounter-level cohort logic and the frozen cohort artifacts of the PIC_Analysis_3_0 contract exactly, rather than redefining the cohort. It absorbs the draft v12.1 amendment (A1 landmark-forward outcome; A2 harmonization, surgery, and map decisions), which is superseded by plan v13 and this contract (see `v12_1_amendment_landmark_forward_outcome_SUPERSEDED.md`). The representation hierarchy is the v13 one: Layers A, B, and C at levels φ0, φ1, φ2, with Layer C on a 56-bin grid and analyzed at landmarks. Where this contract and plan v13 disagree, plan v13 governs and the disagreement is flagged for resolution rather than silently resolved. Each changed rule cites its source in brackets: plan decision register IDs (V, D, C, O, T, P, DA), [TO CONFIRM] items (TC), and numbered decisions (decision 12–22).

v13 is the analysis plan (Team 1 / Team 2) [V2].

# Objective

Build and compare representation matrices and tensors for the PIC analysis using one fixed cohort definition and one evaluation protocol. The representation ladder runs from least processed (φ0) to age-normed and grouped (φ1) to organ-system engineered (φ2), at three time scales: admission context (Layer A), the first 24 hours (Layer B), and six-hour bins from t0 to 336 h (Layer C). φ0 is the minimally engineered benchmark within each layer. All matrices use the same encounters, split, targets, and validation rules.

Three objectives (plan v13 §1) [V3]:

- **Objective 1 (Team 1):** structure: spectra, conditioning, redundancy, low-rank and CUR diagnostics per level.
- **Objective 2 (Team 2):** prediction: matched-width level comparison and the Layer C gain, with the landmark-forward outcome.
- **Objective 3 (extension; owner Vinai; stage 16):** tensor SVD of Layer C (HOSVD, Tucker, CP). Layer C is never unfolded into a design matrix, and `mask_valid` is always honored. Objective 3 does not choose levels, does not change the Objective 2 winner, and does not enter the Objective 2 primary or co-primary analyses.

# Approval gate

This contract must be explicitly reviewed and approved (checklist `foundations_contract_v13_review_checklist.md`) before any feature-generation or model-fitting code is finalized or run on real data. Only read-only auditing, documentation, and synthetic-data development may proceed before approval.

# Data source and raw inventory

The source root is the sibling PICDataset directory, `../PICDataset`. Raw tables present: `PATIENTS`, `ADMISSIONS`, `ICUSTAYS`, `CHARTEVENTS`, `LABEVENTS`, `D_ITEMS`, `D_LABITEMS`, `DIAGNOSES_ICD`, `D_ICD_DIAGNOSES`, `PRESCRIPTIONS`, `INPUTEVENTS`, `OUTPUTEVENTS`, `MICROBIOLOGYEVENTS`, `SURGERY_INFO`, `SURGERY_VITAL_SIGNS`, `EMR_SYMPTOMS`, `OR_EXAM_REPORTS` (all `.csv`).

Raw inventory (schema audit, November 9, 2026; row counts now gated by full SHA-256, Table P below): `ADMISSIONS` [N_admissions omitted]; `ICUSTAYS` [N_icustays omitted]; `PATIENTS` [N_patients omitted]; `CHARTEVENTS` [N_chartevents omitted]; `LABEVENTS` [N_labevents omitted]; `PRESCRIPTIONS` [N_prescriptions omitted]; `MICROBIOLOGYEVENTS` [N_microbiologyevents omitted]; `SURGERY_INFO` 7,488.

**Not read:** `DIAGNOSES_ICD` (discharge codes; banned), `OR_EXAM_REPORTS` (excluded by the approved cleanup rules). `OUTPUTEVENTS` and `INPUTEVENTS` are inventoried for documentation only (decision 4). `EMR_SYMPTOMS` enters no v13 layer.

# Pinned inputs (Table P) [plan v13 §3, I14, Table 3.1; V10, V14, V30]

Every input below is checked against its **full 64-character SHA-256** (and, where listed, its row count) before it is read. Any mismatch stops the run. A changed input requires a new decision and a new pin, never an edit of the gate. Prefix-only gates are not allowed (the harmonization prefix `4b083d18` is retired).

| File | Role | SHA-256 | Gate |
|---|---|---|---|
| `canonical_icu_encounters.csv` | frozen 3.0 cohort (row count omitted from this public copy) | `e5feeb22b5457ab89c4c73235e6a4722d538aec5f88595b6f049004171286418` | hash; cohort, split, landmark counts reproduce |
| `encounter_harmonization_fields.csv` | reviewed version, accepted Nov 10, 2026 | `ffd1b8ae6a13e491be2f7a6631eb4efc1f323656795b80b91e105a0644209d6d` | exact full hash |
| `subject_splits.csv` | persisted split, seed 42 | `60196792457af718155a878f25b3d502ae12dbc71ccabb4c2709518c62e40e31` | hash |
| PATIENTS.csv | raw (D36) | `85b016fcc3050e5b5b6200103ed74ad1798720c881e26c60f1769be9f336ae0d` | hash; [N_patients omitted] rows |
| ADMISSIONS.csv | raw (D36) | `1451ae8ea0ff463269b348522047a4bc99efd2665c770e9c5cef0ffe2647e1ae` | hash; [N_admissions omitted] rows |
| ICUSTAYS.csv | raw (D36) | `449119e698acaaf948921c34375db535b9fd3f6eb6e996a86069da95e90eea1b` | hash; [N_icustays omitted] rows |
| SURGERY_INFO.csv | raw (D36) | `04450bff4592d676a7ebe2dad22d9e6c2a6f8edd07dff223b182eb0002a38d83` | hash; 7,488 rows |
| CHARTEVENTS.csv | raw (Stage 1) | `fe209d47d15825f180624f2cefb9bc38f8513e1ac0b3709d57aef37ae88f328e` | hash; [N_chartevents omitted] rows |
| LABEVENTS.csv | raw (Stage 1) | `784ecb929a358ee1a18fc9beafbdf67f30a91646b780d9983714a5e9ce0018c5` | hash; [N_labevents omitted] rows |
| PRESCRIPTIONS.csv | raw (Stage 1) | `61fa43dcb1ed977f4219c902b01977da2d898a16a2c33640997ad8337d2eb2c6` | hash; [N_prescriptions omitted] rows |
| MICROBIOLOGYEVENTS.csv | raw (Stage 1) | `e19cf9d41732f74911039acae6b2cd7bf5c5dd65179cbd31aa8634dd1e6bb45c` | hash; [N_microbiologyevents omitted] rows |
| OUTPUTEVENTS.csv | raw (documentation only) | `4c49f3c6910b4e8aaa817a13acf16d7120e5ecdd53cb6675bfc7760fda3d4d0d` | hash |
| INPUTEVENTS.csv | raw (documentation only) | `7d8aaebf51d82fdde32841a534f3d296911e92c3421885e40b379bc7c394f9c6` | hash |
| D_ITEMS.csv | dictionary | `a7b65cf5d61ca84f6ce5d07d3775b01116473176a980463d4fe9df1dd12e1591` | hash |
| D_LABITEMS.csv | dictionary | `d4557986fe9c5475d0db6506fdf12801ecfb0e2e7a3c01caf6ac86bbab69ed60` | hash |
| D_ICD_DIAGNOSES.csv | dictionary (Layer A diagnosis mapping) | **recorded by the Stage 3 precheck** [V14] | full hash; every later stage gates on it |
| SURGERY_VITAL_SIGNS.csv | raw (φ2 surgery clusters only) | **recorded by the Stage 3 precheck** [V14] | full hash; every later stage gates on it |
| `surgery_complexity_map_v1.csv` | MAP_v1 (1,495 names) | `5c993b88adfc32ae82c98de19d51c8295d07d34efe7ab29d83a131ca0ab9cfb5` | hash |
| `phi1_band_support_frozen_medians_MAP_v1.csv` | frozen φ1 duration medians | `e73c8b5a8d780b51edfd0935abdb05335f57e3a0cfc0c4c52eaaa85888db0047` | hash; never refit |
| **`feature_channel_map_v2_3.csv`** | **channel map v2.3 = v2.2 (v2.1 without aa_gradient 5210 [V10]) with the `derived:pf_ratio` note reworded without counts [V29] (686 rows; 94 channels)** | **`05c1f9171b375fe4feb1384c243c510a8d4c4b4e474c1ce5c65e3f070797b1fe`** | full hash; the gate from Stage 3 on; v2.2 and v2.1 refused |
| `drug_group_map_v1.csv` | 343 names | `6fa7d8a935f03ad71b4a7ecd8276f9a54f3a679f4cfc08c8b4504eda83455ead` | hash |
| `specimen_map_v1.csv` | 71 keys (D12) | `177b165ad8e5888a3999583a4949e93908be45af34cfa9c6e4937c40c08666f1` | hash |
| `organism_map_v1.csv` | 101 keys | `21f28448cb31c654c5312ca97883a34fe5cb6b269e8323973eb8b899bb3583d2` | hash |
| `neuro_admitting_dx_icd10_v1.csv` | neuro set (D3) | `3cc8f4acfe9df29037b60d903ad4204c40db5bc76923de8b64234ccb2553b9d9` | hash |
| `audit_surgery_complexity_duration.py` | v1.7 module (ID parsing, suppression, recoverability solver) | `ce894b20ad009369c2f21cd6db61a0379cd2db888dbf22309b21c393409de6d0` | hash; imported, never edited |
| `check_encounter_harmonization_consistency.py` | harmonization check (Nov 11 run) [V30] | `9b3e7e84a5a00ff6df10446a16ed67f0b2f393d574a256f97d9d8dd0514020c6` | hash; frozen, not re-run |

**Superseded pins (provenance only; never read by a new stage):** channel map v2.2 `1e7c6c98f20518a824483e690002e22a8e8c5c8d2c5b81233c2182deffa8077a` (old pf_ratio note; team-internal); channel map v2.1 `7d1c060a228ab0772f35ae27e43a30971bcc13c36c98fcce2e369640b91bc899`; channel map v2 `cbfdc68f…`; first v2 revision `9872e688…`; channel map v1 `56e8bf72…` (pin of Stage 2 v1.0.1); neuro draft `602e648a…`.

**Retired input [V8]:** the frozen 3.0 Layer C file `layer_c_features.csv` (28 bins, 168 h, 16 channels) is not an input of any stage, is not pinned, and is not a reconciliation control. No check compares bins 0–27 with it. The frozen 3.0 cohort file (`e5feeb22…`) and the `frozen3_` whole-stay medication-rule comparison build (rebuilt from the raw tables) are kept.

**Provenance of the audit scripts already run [V30].** Full SHA-256 and write times of the frozen audit scripts are in plan v13 Table 3.2 (Copilot file inspection, November 16, 2026; no script run, no data opened). Two hashes are permanently unrecoverable and are **not** open items: the stopped Stage 2 v1.0 copy, and the full hash of follow-up copy `a5619ebb…` (replaced by the `e955974d…` re-pin; the follow-up run stays valid, output JSON `9280ef6f…`). Run times exist only as file write times; there are no run logs.

# Non-negotiable pre-feature-engineering foundation

The encounter definition is inherited from the approved PIC_Analysis_3_0 contract and is not re-created.

## Canonical ICU encounter definition

A canonical ICU encounter is the maximal sequence of one subject's raw ICU stays in which each subsequent stay begins no later than 24 hours after the running episode end. For raw stays sorted by `SUBJECT_ID`, then `INTIME`: if `next.INTIME <= running_end + 24 h`, the stay is absorbed; otherwise a new encounter begins. Feature extraction is at the encounter level, never the raw ICUSTAY row.

## Inherited cohort and split (frozen; reproduced by Audit Stage 1) [plan v13 §2.1]

- Raw stays [N_icustays omitted]; invalid stays (`OUTTIME <= INTIME`) 33 removed; valid stays [N_valid_stays omitted], every `HADM_ID` found in ADMISSIONS. Merging gives **the frozen canonical cohort (encounter, subject and death counts omitted from this public copy)**.
- The canonical encounter file is byte-identical to the frozen 3.0 cohort `e5feeb22…`; the harmonization fields file is `ffd1b8ae…`; both gated by full SHA-256. The prefix `4b083d18` is retired.
- Split: `SUBJECT_ID`-grouped two-stage StratifiedShuffleSplit, seed 42, 70/15/15, persisted in `subject_splits.csv` (`60196792…`): **the frozen 70/15/15 split (sizes omitted from this public copy)** encounters. Never regenerated.
- All canonical encounters are kept in every Layer A and B matrix and on the full Layer C grid. Supervised analyses use the landmark rows of §H5; there is no whole-cohort tabular comparison [V16].

## Encounter-level fields

Each canonical encounter keeps at minimum `encounter_id`, `SUBJECT_ID`, `HADM_ID` (representative; display and audit only), `hadm_id_list`, `ICUSTAY_ID` (earliest stay), `INTIME`, `OUTTIME` (maximum stay OUTTIME), `encounter_n_stays`, `n_linked_admissions`, `hospital_admit_time`, `hospital_disch_time`, `hospital_los_hours`, `icu_los_hours`, and `Died`.

## Multi-admission policy

Mortality is positive if any linked admission has a nonmissing positive `HOSPITAL_EXPIRE_FLAG`, negative only if all linked admissions are observed and negative, otherwise missing. Hospital LOS spans the earliest linked admit to the latest discharge. `hadm_id_list` is used in every provenance-sensitive join; the representative `HADM_ID` is never sufficient.

# Encounter harmonization (v13-aligned)

This section fixes the event-assignment layer. It does not change the inherited cohort, split, or targets.

## H1. Encounter construction (inherited; resolved)

1. Sort raw ICU stays by `SUBJECT_ID` and `INTIME`.
2. **Running end (resolved; was [VERIFY]).** The running episode end is the latest discharge so far (maximum `OUTTIME`), so nested stays are absorbed. The real-data reproduction (November 10, 2026) found that the latest-discharge-so-far rule and the last-stay rule give identical encounters, and the harmonized `OUTTIME` is the maximum stay OUTTIME in every encounter (consistency check, November 11, 2026) [plan v13 §2.1; A2.6].
3. Absorb the next stay when `next.INTIME <= running_end + 24 h`.
4. Pre-merge validity checks were logged (Audit Stage 1): 33 invalid stays removed (the split between `OUTTIME < INTIME` and zero length is not quoted because one part is under 10); 200 merges; 88 overlapping stays, 40 of them nested; 112 within-encounter gaps (74 of 0–6 h, 38 of 6–24 h).
5. `encounter_id` is a deterministic string from `SUBJECT_ID` and the encounter sequence number.
6. **Reproduction check (done).** the frozen canonical cohort (counts omitted from this public copy) reproduce exactly. Every later stage re-asserts the cohort, split, and landmark counts before it runs.
7. **Coverage denominators.** Coverage is per encounter, never per admission (cohort and admission counts omitted from this public copy).

## H2. Time zero, windows, and landmarks [plan v13 §2.2, §6.1, I6; A1.4, A2.4]

- `t0` = earliest `INTIME` of the encounter (`INTIME − t0 = 0` in all rows). One continuous clock runs from t0; merge gaps get no separate gap feature and no `in_icu` indicator (decision of November 10, 2026); bins inside a gap receive the usual masks, and LOCF caps apply across gaps unchanged.
- Half-open windows:
  - Layer A: at t0, with the pre-ICU lookback `[t0 − 6 h, t0)` primary and `[t0 − 24 h, t0)` as a sensitivity analysis, capped by admission ID (`hadm_id_list`).
  - Layer B: `[t0, min(OUTTIME, t0 + 24 h))`; computed once and held fixed at every landmark.
  - **Layer C: 56 six-hour bins**, bin k = `[t0 + 6k, t0 + 6k + 6)` h, k = 0, …, 55 (0–336 h), truncated at OUTTIME, stored on the full grid of [N_encounters omitted] × 56 with **`mask_valid[e, k] = 1` iff `t0 + 6k < OUTTIME`**. An observation at exactly t0 + 24 h belongs to bin 4.
- **Landmarks:** L ∈ {24, 48, …, 336} h (**14 landmarks**). Eligible at L: `OUTTIME > t0 + L` (strict). **Core landmarks:** 24, 48, 72, 120, 168, 240, 336 h (prefix bins 4, 8, 12, 20, 28, 40, 56). At L the Layer C input is bins 0, …, L/6 − 1 with data available strictly before t0 + L; every bin of an eligible prefix is valid, so landmark tensors contain no padding.
- **Fixed 72 h window sensitivity:** the 12 bins immediately before L (L ≥ 72 h).
- Eligible counts (14 landmarks): [N_eligible omitted] / [N_eligible omitted] / [N_eligible omitted] / [N_eligible omitted] / [N_eligible omitted] / [N_eligible omitted] / [N_eligible omitted] / [N_eligible omitted] / [N_eligible omitted] / [N_eligible omitted] / [N_eligible omitted] / [N_eligible omitted] / [N_eligible omitted] / [N_eligible omitted]. Every stage reproduces the core counts before running.
- Short stays: encounters shorter than 24 h stay in the Layer A and B matrices (truncated Layer B window) but are not in any landmark row set.

## H3. Event linking to encounters [I13]

`ICUSTAY_ID` is not a universal join key (missing in 68% of CHARTEVENTS rows, 91% of OUTPUTEVENTS, 32% of PRESCRIPTIONS, 8% of INPUTEVENTS; absent from LABEVENTS and MICROBIOLOGYEVENTS).

1. Match on `SUBJECT_ID` and `HADM_ID ∈ hadm_id_list`.
2. Assign the event to the encounter whose window contains the event time.
3. If `ICUSTAY_ID` is present and disagrees, the time-based assignment wins; disagreements are counted per table.
4. **Event time per table** (`STORETIME` is never an event time):
   - CHARTEVENTS, INPUTEVENTS, OUTPUTEVENTS: `CHARTTIME`.
   - **LABEVENTS: available at `CHARTTIME + 1 h`** everywhere (Layer B window, Layer C bins, the landmark rule, the pre-ICU lookback); +0 h and +2 h sensitivities. PIC has only CHARTTIME (resolved [AUDIT]). Results are never backfilled to the order time; triggered tests keep the ordered flag at the order time [S9 as decided; A2.3, decision 15].
   - MICROBIOLOGYEVENTS: `CHARTTIME` as specimen collection; results only under M1 / M3 / M2 (S8).
   - PRESCRIPTIONS: `STARTDATE` / `ENDDATE` through the strict look-back state machine (S3).
   - **SURGERY_INFO: anesthesia end T** (`ANES_END_TIME`, fallback surgery end). Key `SURGERY_INFO.ROW_ID`. Recent surgery (Layer A): T ∈ `[t0 − 24 h, t0 + 6 h]`; remote surgery (Layer A): same HADM, T < t0 − 24 h; **Layer C surgery event: t0 + 6 h < T ≤ OUTTIME, placed in the bin containing T** (begin-time bin as sensitivity). A recent surgery never also appears as a Layer C event (S10). There are no Layer B surgery features [A2.1, A2.10, decisions 13, 18; replaces v12 H3.4].
5. **Pre-ICU lookback (resolved).** Events before t0 never enter a Layer B or C bin. Layer A carries a separate, labeled pre-ICU lab block (last released value in `[t0 − 6 h, t0)`, decision 19) and drugs active at t0. Medication `ever_received` uses pre-t0 documentation only within the same 6 h lookback [V20, TC13]. Pre-ICU microbiology collections count only for M3 availability.
6. **Gaps between merged stays (resolved).** Events in gaps are included on the continuous clock; no `in_icu` indicator and no gap feature. Gap bins count in empty-bin shares (primary) and are excluded as a sensitivity analysis (B22).
7. Events outside every window are excluded and counted per table.

## H4. Outcome labeling [plan v13 §2.3; A1.2, A1.3; V7]

1. **Primary target: Outcome 1, remaining ICU time from L (K = 9).** Level 8 died in hospital (any time after L); 7: remaining ICU time > 336 h; 6: > 240–336 h; 5: > 168–240 h; 4: > 120–168 h; 3: > 72–120 h; 2: > 48–72 h; 1: > 24–48 h; 0: ≤ 24 h.
2. **Key secondary: Outcome 2, post-ICU course (K = 5).** 4 died in hospital; 3 ICU readmission more than 24 h after OUTTIME; 2 no readmission and post-ICU stay ≥ 14 d; 1: 7 to < 14 d; 0: < 7 d.
3. **Rules.** Death takes precedence; Outcome 1 is measured to the index encounter's last OUTTIME; the earliest readmission sets the level; readmission precedes post-ICU stay levels; post-ICU stay = hospital discharge minus ICU OUTTIME. Index-HADM tie-break: when the last stays tie on OUTTIME with different HADMs, the HADM with the latest DISCHTIME supplies DISCHTIME and the readmission search. Unassignables are reported; assignable + unassignable = eligible is an acceptance check. Readmission overlap policy `keep_both` (S12). Outcome 1 and Outcome 2 are modeled and reported separately, never combined.
4. **Secondary:** inherited **ordinal8** (3.0 contract §5.5, death as class 7) at the 24, 48, and 72 h landmarks only; in-hospital mortality (binary) at every landmark as a sanity check. **Ordinal4 and the binary composite (`y_composite_unfavorable`) are removed entirely** [V7, TC1].
5. **Death attribution.** The inherited rule is primary (the recorded ICU deaths (count omitted from this public copy)). Sensitivity: the death is attributed only to the patient's last encounter (the last-encounter sensitivity, with `death_attribution_flag` on the differing encounters; counts omitted from this public copy) [B3].
6. Outcome consistency checks (flag vs DEATHTIME, DEATHTIME within the admission, agreement with PATIENTS) are reported once, aggregate only.
7. `DEATHTIME`, `DISCHTIME`, `OUTTIME`, hospital LOS, ICU LOS, and every other field on the banned list (H7) are descriptive or outcome fields, never features.

## H5. Row sets per objective and cohort composition [V5, V16; plan v13 §2.3]

- **Objective 2 primary:** the **24 h landmark rows**, one per encounter alive and in the ICU at t0 + 24 h: **[N_eligible omitted]** (train 6,556, validation 1,399, test 1,406), with Outcome 1 from 24 h.
- **Objective 2 co-primary:** at each core landmark, the encounters eligible at L (one row per encounter); the other seven landmarks are fitted and reported the same way as secondary results.
- **Stacked landmark model:** one row per (encounter, eligible landmark), all 14 landmarks; **sensitivity analysis for Layer C only** (per-row weighting, per-patient weighting as further sensitivity) [V5, B6, B7].
- **Objective 1:** all train encounters of a layer and, for C, the landmark-eligible train encounters.
- **Objective 3:** the landmark-eligible train encounters of Layer C.
- **No whole-cohort tabular comparison** (all canonical encounters, ordinal8) [V16, TC2]; ordinal8 stays secondary at 24/48/72 h.
- **`age_invalid` band** (D20, D34, D37): missing, unparseable, or negative age at t0 (negative binned < 1 d, 1–30 d, > 30 d); such encounters stay in the primary analysis inside the `age_invalid` band of every age partition. Sensitivities: exclude them; treat negative ages under 1 day as age 0 (B4). Safety gate: any run stops if invalid ages exceed 1% of the cohort (137 encounters). The 42 invalid DOBs of the first Stage 2 run are quoted only as the v1.0 run-stop event, never combined with v1.0.1 cells [V29].
- **Care unit** (D35): `first_careunit` kept as a Layer A feature; a unit-stratified sensitivity re-runs the main Objective 2 contrasts within unit (B5).
- Composition counts (care unit, age band, encounters per subject, landmark eligibility) are reported aggregate only, cells under 10 suppressed. No composition count changes the primary cohort.

## H6. Splits

Splits are by `SUBJECT_ID`; all encounters (and all landmark rows) of a patient stay in one partition. The persisted split is reused and never regenerated. Bootstraps resample subjects (patient-level cluster bootstrap), never rows.

## H7. Encounter fields, name table, and the banned list [plan v13 §2.1, I7; A2.5, A2.6; V12]

Contract name → real header: `t0` → `t0`; `OUT` → `OUTTIME` (maximum stay OUTTIME); `icu_span_hours` → `span_icu_hours` (OUTTIME − t0, gaps included; equals the inherited `icu_los_hours` in all canonical encounters, which resolves the v12 H7 [VERIFY]); `icu_time_hours` / `sum_icu_stay_hours` → `icu_time_hours` (span minus within-encounter gap hours; overlaps not double-counted; replaces v12's "sum of raw stay durations"); `n_gaps`, `total_gap_hours` → `gap_count_between_stays`, `gap_hours_total`; `age_at_t0` → `age_years_at_intime`; `prior_encounters_n` → `prior_encounters`; `death_attribution_flag`; landmark indicators `in_icu_at_24h/48h/72h` (eligibility only).

**Allowed as Layer A inputs:** `first_careunit`, age, and `prior_encounters` (candidate only; include or omit is open).

**Banned as features (I7):** DOD, EXPIRE_FLAG, DEATHTIME, HOSPITAL_EXPIRE_FLAG, DISCHTIME, DISCHARGE_DEPARTMENT, OUTTIME, LOS in any form (`icu_los_hours`, `span_icu_hours`, `icu_time_hours`, hospital LOS), gap counts and hours, LAST_CAREUNIT, LAST_WARDID, HAS_CHARTEVENTS_DATA, DIAGNOSES_ICD discharge codes, readmission fields, `death_attribution_flag`, landmark indicators, all target columns, all IDs (SUBJECT_ID, HADM_ID, ICUSTAY_ID, encounter_id, hadm_id_list), `mask_valid` and any count of open or padded bins, time to OUTTIME, **weight and the weight-source flag** (normalizer and audit use only) [V12, TC8], **per-bin and per-window measurement counts** [V22, TC16; S2], ENDDATE presence, STORETIME, any value at or after OUTTIME, and any value not yet available at its time step.

## H8. Table notes and audits [V13, V18; plan v13 §13.1, §14]

- `SURGERY_VITAL_SIGNS` are intraoperative. Rows are assigned to the `ROW_ID` whose anesthesia window contains MONITOR_TIME (strict window primary; ±30 min sensitivity, B11); they feed only the φ2 surgery clusters and never a Layer B or C vital channel (decision 14).
- `EDREGTIME` is used only in the admission-source rule; the share equal to a placeholder is checked before use [AUDIT, carried from v12; no v13 result recorded].
- `SURGERY_INFO.VISIT_ID` vs `HADM_ID` [AUDIT, carried from v12; surgery linking uses `ROW_ID`, `HADM_ID ∈ hadm_id_list`, and time].
- `OR_EXAM_REPORTS` are excluded from all matrices.
- Dictionary findings (resolved by Audit Stages 1 and 2): no charted FiO2, oxygen device, ventilation, GCS, or pupils exist in PIC; the mis-mapped urine sodium item and the "mechanical ventilation" item are never used.
- **Completed audits (frozen; not re-run):** encounter harmonization and cohort reproduction (Nov 10); landmark eligibility and band support (Nov 10); MAP_v1 review (Nov 10); surgery item-level audit and harmonization consistency check (Nov 11); complexity–duration audit v1.7 (Nov 11); φ1 band support (Nov 12); Audit Stage 1 item inventory (Nov 14); Audit Stage 2 sparsity and frequency v1.0.1 (Nov 15); Stage 2 follow-up checks (Nov 15).
- **Stage 3 audit:** reports out-of-gate counts per channel before the provisional † gates are locked [V13], and the value distributions of channels without a gate, each of which then gets a gate or a recorded "no gate" at the same lock [V18, TC9]. Aggregate only, cells under 10 suppressed.

## H9. Required harmonization outputs (produced; frozen)

`canonical_icu_encounters.csv` (reproduced, hash matched), `encounter_harmonization_fields.csv` (`ffd1b8ae…`), and the aggregate Stage 1 cohort-flow and consistency outputs. Each was reviewed and approved; none is regenerated by a later stage.

## H10. Decisions summary (v12 open items, now resolved)

| # | Decision | Resolution |
|---|---|---|
| H1.2 | Running-end rule | Latest discharge so far; identical to the last-stay rule on real data |
| H2 | Landmark set | 14 landmarks 24–336 h; core 24/48/72/120/168/240/336 h [A1.4, V5] |
| H3.5 | Pre-ICU lookback | None for B/C bins; Layer A pre-ICU lab block `[t0 − 6 h, t0)`; `ever_received` within 6 h [decision 19, V20] |
| H3.6 | Gap events | Included on the continuous clock; no `in_icu` indicator, no gap feature |
| H4.3 | Death attribution | Inherited rule primary; last-encounter rule sensitivity (B3) |
| H5 | Sensitivity subsets | `age_invalid` exclusion; negative age < 1 d as 0; unit-stratified (D34, D35); no other subset declared |
| H7 | ICU LOS definition | `icu_los_hours` = `span_icu_hours` in all encounters |

# Analysis questions

**Objective 2, primary [V5, V6, V23, V25, V26].** On the 24 h landmark rows with Outcome 1, which representation level of A, B, and AB (and the Layer C levels on the 24 h prefix, each added to A+B) gives the best balance of the six criteria of plan v13 §12.1, compared at matched width k (top-k SVD scores and k CPQR-selected columns, train-fitted; k ∈ {5, 10, 20, 30, 40, 59}, k ≤ p; also full width)? For Layer C at 24 h the width counts per-bin features: the primary representation is the k CPQR-selected channels of the stacked (nb) × q train matrix; the alternative is the top-k channel-mode SVD scores of that matrix; k ≤ q; no flattening.

**Objective 2, co-primary [V5, V24].** **A+B versus A+B+C**, Outcome 1 measured from L, both models refit separately at each core landmark on the encounters eligible at L; the contrast is the QWK difference (A+B+C minus A+B). All 14 landmarks are fitted and reported. Remove any "pooled across the stack" primary result: the stacked landmark model is a Layer C-only sensitivity analysis.

**Multiplicity [V25, V26, V27].** Family (a): the 24 h A/B level comparison (H6 comparisons, Layer C levels at 24 h included); family (b): the seven core-landmark A+B vs A+B+C contrasts. Both families use Holm's step-down correction on patient-level cluster-bootstrap p-values at family-wise α = 0.05; no adjustment across families. The 14-landmark added-value curve is descriptive only: no test or p-value; its bootstrap 95% intervals are kept and labeled descriptive.

**Metrics and model selection [V6].** **QWK** (on the decided class, the median of the predictive distribution) is the primary model-selection metric and the primary comparison metric. **EMD** (expected |i − j| cost) is secondary and reported beside it. RPS, macro AUROC across cutpoints, calibration, and mortality AUROC are reported. Sensitivity costs: (i − j)² and an optional clinically weighted matrix (B30).

**Objective 1.** Spectra, conditioning, redundancy, low-rank, CUR, and angle diagnostics per level (stages 09–10). For Layer C, Objective 1 keeps only the **per-bin / stacked view** (one n_k × q matrix per bin and the stacked (nb) × q matrix) [V9].

**Objective 3.** Whether Layer C has low multilinear structure (HOSVD, Tucker by HOOI, CP by ALS) on train landmark tensors per level, with hypotheses H13, H15–H17 and criteria T1–T6 (plan v13 §12.3).

Results are associations within this cohort; no causal claims.

# Comparison design set (v13 representation hierarchy)

Layers A and B: one row per canonical encounter (count omitted from this public copy). **Layer C: the full grid of encounters × 56 bins × q (encounter count omitted from this public copy)** with `mask_valid`; analyses use landmark prefixes of eligible encounters.

## Layer A: admission context (one row per encounter) [plan v13 §4]

| Level | Content | Prefix |
|---|---|---|
| φ0 | continuous age at INTIME (empty with an `age_invalid` indicator when invalid); hospital-to-ICU interval (hours); ICU-INTIME hour of day and day of week; three-character ICD-10 category from ADMISSIONS.ICD10_CODE_CN mapped through D_ICD_DIAGNOSES (rare categories pooled within (chapter, domain)); recent surgery `standardized_name`, occurrence, hours from T to t0; remote surgery occurrence and name; pre-ICU lab block (last released value in `[t0 − 6 h, t0)` plus mask); drug-group state at t0 | `a0_` |
| φ1 | age band plus `age_invalid`; four interval bands; three shifts and weekday/weekend; ICD-10 chapter; surgery type and system (MAP_v1) plus the type-specific φ1 duration band (3-level ordinal); inferred admission source; age-normed pre-ICU labs | `a1_` |
| φ2 | diagnosis-only admission domain (eight domains, plan Appendix C draft); train-only surgery clusters; admission source; pre-ICU labs in organ blocks | `a2_` |
| φ2-other | φ2 plus rare-cluster pooling to `other` | `a2o_` |

Fixed in every level: sex and `first_careunit` (D35). Removed: `sex_missing`, ED-present flag, hours from ED registration. **Admission source (φ1) [V17, TC7]:** first match of *floor* if the hospital-to-ICU interval ≥ X h; else ***OR* if a recent surgery has anesthesia end T ≤ t0**; else *ED* if EDREGTIME is before INTIME; else *other* (X from the grid 6, 12, 24, 48 h; open). **Weight [V12, decision 20]:** charted weight (first value from t0 − 24 h to the end of the Layer B window) or the train-median tiers (age band × sex, age band, sex, overall, each n ≥ 10; `age_invalid` never imputed from an age band); used **only** to normalize urine output and fluid intake, with a weight-source column for audits and sensitivity B38; neither is a feature. Height (1013) is Layer A / weight-model only and too sparse to be a feature.

## Layer B: first 24 hours (one row per encounter) [plan v13 §5]

Built on **channel map v2.3** (`05c1f917…`; 94 channels in ten organ systems; plan v13 Table 5.1) [V10, V29]. Layer B is held fixed at every landmark.

| Level | Content | Prefix |
|---|---|---|
| φ0 | vitals: median, min, max, range (no last value, **no count column**); labs: first and worst released value; triggered tests: ordered flag, worst released value, mask (S9); indicator-only channels: measured flag plus mask; shared panel masks (D33); drugs one-hot by name; microbiology M1; urine output (mL/kg/h) and fluid intake (mL/kg/day) | `b0_` |
| φ1 | age-normed z-scores and flags; SD of z-scores (≥ 3 values, else masked); age-appropriate lab cutoffs; derived: BUN/creatinine ratio, KDIGO urine-output stage, AKI stage; thirteen drug groups from the state machine; M1 microbiology | `b1_` |
| φ2-single | ten organ-system blocks, one primary home per column ("unstacked"; primary), plus modified PRISM III | `b2s_` |
| φ2-dup | same blocks with stacked "also in" duplicates, plus PRISM III | `b2d_` |
| φ2-pSOFA | **five-component** modified pSOFA (cardiovascular, coagulation, hepatic, renal, neurological; no respiratory component, D29), plus PRISM III | `b2p_` |

**Not built (any layer):** P/F, S/F, FiO2 5222, the analyser P/F 5243, **aa_gradient 5210** [V10], anion gap (D25), net fluid balance and percent fluid overload (D17), charted total fluid output (D10), serology (D11), lipase 6473 (D26), the urine-sodium item, the "mechanical ventilation" item. Microbiology result variants: `b1m3_` (availability at collection plus turnaround) and `b1m2_` (look-ahead; never used for primary results or winner selection).

**Channel rules carried from the decisions:** duplicates merged only when units convert and distributions agree (D6, D22, D23); troponin T pg/mL ÷ 1000 with a 0–50 ng/mL gate (D13, D32); hemoglobin gated 20–250 g/L, reported in g/dL, with the CBC / blood-gas source flag as a feature (D27); ionized calcium 5215 its own channel, gate 0.3–3.0 mmol/L, own mask (D21); bilirubin 5075 main, 5255 merged with a per-value source flag only when first_careunit = NICU (D24); MAP from same-time SBP/DBP pairs never straddling a bin (D8); HR 1003 preferred over 1002 at the same time (D9); urine output = volume ÷ weight ÷ hours since the previous entry (capped 24 h), never back-filled (D30); ALT and TSH measured indicator only (D26); magnesium "ug/dl" rows dropped (D31). **Shared panel masks (D33):** cytokines, TEG, urine dipstick, and **blood gas (15 items** — the map's `mask_group`; 5210 was never in it) [V10, TC10]; 5215 and 5257 keep their own masks. The map's `contract16` column is a provenance note only [V19, TC12].

**Plausibility gates [V13, V18].** Gates marked † in plan Table 5.1 are **provisional**: Stage 3 applies them and reports out-of-gate counts per channel; Vinai then locks each gate. Gates confirmed by decision: D21, D27, D32. Channels without a gate get a gate or a recorded "no gate" at the same lock. Values outside a gate are counted and set to missing before any other step.

## Combined design AB

AB_k = A_k ⊕ B_k. Prediction on AB [primary], A and B alone secondary.

## Layer C: sequences (encounters × 56 bins × q) [plan v13 §6]

| Level | Content | Prefix |
|---|---|---|
| φ0 | per-bin raw native-unit summaries per channel in its disposition class (median, min, max; range for vitals; median only for a vital failing bin-width adequacy); triggered tests as ordered flag, released value, mask; indicator-only as measured flag plus mask; `*_missing`, `*_carried_forward`, time since last; events by drug name; surgery in the bin of T by `standardized_name`; microbiology collected. **Per-bin measurement count is not a feature** [V22] | `c0_` |
| φ1 | bin-median z-scores and flags; past-only rolling 4-bin SD of z-scores (≥ 3 observed values); lab cutoffs; AKI stage from carried creatinine; KDIGO urine-output stage; thirteen drug groups (state machine at bin end); M1 microbiology; surgery type, system, and φ1 band in the bin of T | `c1_` |
| φ2-single, φ2-dup | per-bin organ-system blocks using the Layer B schema | `c2s_`, `c2d_` |
| φ2-pSOFA | per-bin five-component modified pSOFA from the worst observed values in bins k − 3 to k | `c2p_` |

The modified PRISM III scalar is computed once from Layer B and held fixed from bin 4 on (empty in bins 0–3). **Stage 3 is the one common builder** of the real A, B, and C and builds every channel of map v2.3 on 56 bins from the raw tables; it absorbs the v12 stage 08x (the Layer C lab extension). There is no separate extension file and no reconciliation with the retired frozen 3.0 Layer C file [V8, TC15, TC19].

**Masks, carry rules, and LOCF families [plan v13 §5.7, §6.2; V21, V28, TC14].** Masks are created before LOCF. The cap is in bin index; carried values cannot seed a later carry; nothing is carried into a bin with `mask_valid` = 0 or across the landmark.

| Family | LOCF | Cap |
|---|---|---|
| HR, MAP, RR, temperature, SpO2, SBP, DBP, pain score, bedside glucose | No | n/a (time since last capped at 24 h) |
| Creatinine, BUN, hemoglobin, platelets, WBC | last observed | 4 bins / 24 h |
| pH, lactate, bicarbonate, sodium, potassium, glucose | last observed | 1 bin / 6 h |
| Extension slow labs: liver panel (AST, bilirubin total and direct, albumin, GGT), INR, PTT, fibrinogen, D-dimer, ANC, absolute lymphocytes, cystatin C, amylase, lipase, LDH, CRP [V28] | last observed | 4 bins / 24 h |
| Extension: chloride, total and ionized calcium, magnesium, phosphate, pCO2, pO2, base excess, SaO2 [V28] | last observed | 1 bin / 6 h |
| Endocrine, CSF, and qualitative channels not [T] or indicator-only [V21] | last observed | 1 bin / 6 h |
| Urine output, fluid intake, KDIGO stage (D30) | at most 24 h | 4 bins / 24 h |
| Triggered tests; indicator-only channels; events [V21] | **No** | n/a |

Time since last measurement is a feature for value-class channels, capped at the channel's LOCF cap (24 h where there is no cap); omitted as a sensitivity (B21).

**Filled copy and imputation (reference arm (r)) [plan v13 §7].** Within valid bins, LOCF up to the cap; a cell still empty receives the **train median** of that column (valid train cells only, train prefixes at L), and an imputation flag (`*_imputed`) marks every filled cell, distinct from `*_carried_forward`. Padding is never filled. Empty-by-design fields and triggered / indicator-only channels are never imputed as if measured. Arms (a), (b), optional (c) replace the median step only (stage 15i).

**Layer C analysis at landmarks.** Inputs at L: Layer A plus Layer B (fixed) plus the prefix bins 0, …, L/6 − 1. Comparators: static only (A), **A+B**, sequence only, **A+B+C** (static plus sequential). Layer A context conditions every time step; the fusion head receives only the sequence contribution and the Layer A baseline risk (as an offset); PRISM III stays out of the sequence inputs. Sequence inputs are predictive signals, not causal effects.

**Flattened Layer C [V9, TC17].** The flattened n × (bq) view, its effective-rank curve (former H8), its CPQR/DEIM selection of (bin, channel) pairs, and the A+B plus flattened-prefix tabular model are **not** Objective 1 or Objective 2 analyses. They are kept only as the **labeled flattened baseline of Objective 3** (stage 16), which never chooses a rank, a level, or a model.

## Objective 3 (stage 16; extension) [V3, V9]

Train landmark tensors X_L ∈ R^(n_L × L/6 × q_l) per level (raw φ0 with masked weighted ALS; normalized φ1 from the filled copy with imputation flags beside it; engineered φ2 with masks and flags as feature-mode channels). Two binding rules: (1) **never unfold C** into a design matrix for any model, selection, or diagnostic; mode-n unfoldings appear only inside HOSVD/HOOI; (2) **honor `mask_valid`**: padding never enters a fit, a statistic, an imputation, or an error term (padding probe H17 uses weight 0 on padding). Validation and test are projected onto train factors; nothing is refit on them. Outputs are aggregate only; no per-patient loadings leave the secure environment.

## Pre-declared sensitivity and comparison builds [plan v13 Appendix B, B1–B38]

- `frozen3_`: medication features under the 3.0 whole-stay rule, rebuilt from the raw tables (B17).
- Layer C imputation arms (r), (a), (b), optional (c) (B26).
- Range included or excluded at Layer B φ0 (B35).
- Death attribution (B3); `age_invalid` (B4); unit-stratified (B5).
- **Stacked Layer C landmark model** with per-row weighting, per-patient weighting as further sensitivity (B6, B7).
- Fixed 72 h Layer C window (B8); lab release +0 / +2 h (B2); pre-ICU lookback 24 h (B1).
- Secondary targets: Outcome 2, ordinal8 at 24/48/72 h, mortality; **no ordinal4 or composite** (B10).
- **Weight source (B38):** weight-normalized channels restricted to charted weight or stratified by source.
- **No S5 whole-cohort run**: the 24 h landmark rows are the primary row set (B32).
- Objective 3 weighting and standardization (B36, B37).

# Experimental principles

- One canonical encounter definition and the frozen canonical cohort for every matrix and tensor.
- Targets fixed unless explicitly approved otherwise.
- Same raw source, split, and evaluation logic across all levels.
- Fit everything learned (vocabularies, pooling, imputation, scalers, SVD, CUR bases, tensor factors, hyperparameters, calibration, weight medians) on train only; validation and test receive the fitted transforms unchanged. At landmark L, Layer C statistics use only train prefixes at L.
- Score test **once** per analysis, after levels, hypotheses, thresholds, and winners are frozen; log it.
- Provenance for each feature family and each output; differences between matrices recorded explicitly.
- Never alter the encounter logic to improve one design.
- Respect the banned list (H7), release timing (S9), the state machine (S3), the landmark rule (S6), surgery timing (S10), train-only descriptors (S11), and readmission overlap (S12) at every level.

# Required deliverables

- The approved cohort specification and the frozen harmonization outputs (H9).
- Feature matrix and tensor specifications per layer, level, and variant (above), starting with the Stage 3 builder spec.
- Data dictionary with the plan v13 Table 5.1 organ-system membership (94 channels, map v2.3).
- Generation scripts with provenance, writing to `data/processed/ladder/`, never overwriting `layer_a/b/c_features.csv`.
- Spectral and redundancy diagnostics per level (Objective 1).
- Benchmark tables (Objective 2): the 24 h matched-width level comparison (Outcome 1, QWK primary, EMD secondary); **the Holm-corrected core-landmark A+B vs A+B+C contrast table**; **the descriptive 14-landmark added-value curve** with bootstrap intervals labeled descriptive; Outcome 2, ordinal8 (24/48/72 h), and mortality as secondary. No ordinal4 or composite outputs.
- Calibration and attribution-stability tables.
- Objective 3 tensor tables and figures, with the labeled flattened baseline.
- A recommendation summary per layer.

# Execution workflow and ownership

1. Pin channel map v2.3 (built and validated on Nov 16, 2026, `05c1f917…`; v2.2 with the `derived:pf_ratio` note reworded without counts; Vinai confirms the pin before Stage 3) [V10, V29].
2. **Stage 3, the one common builder** (`03_build_real_layers.py`): base Layer A fields, Layer B window arrays, Layer C 56-bin grid with `mask_valid`; full SHA-256 precheck including D_ICD_DIAGNOSES and SURGERY_VITAL_SIGNS; out-of-gate audit; absorbs 08x [V8, V13, V14].
3. Lock the † gates and the ungated channels' "gate / no gate" from the Stage 3 audit [V13, V18].
4. Level builders 06 (A), 07a / 07 (B), 15 (C), 15i (imputation arms); 08 preprocessing fit on train.
5. Synthetic gates (plan v13 §13.3) pass before any real-data fitting.
6. Objective 1 (09, 10); Objective 2 (11–13, 17, 18); Objective 3 (16).
7. Single logged test scoring (14, 18).
8. Recommendation, paper, and presentation.

**Ownership (person level; plan v13 §15).** Vinai (PI): Stage 3, 06, 07a / 07, 08, 15 (08x absorbed), 15i, 14, **16 (Objective 3)**, and **every real-data run**; cross-checks all specs. Habib: 11, 12, 13, 17, 18; cross-checks stage 09 decay-fit and bootstrap code. Vinay: 09; cross-checks classifier code (11–12). Yashwant: 10; cross-checks attribution code (13). Team 1 leads Objective 1, Team 2 leads Objective 2; Objective 3 is Vinai's extension.

**Slip priority [V15, TC20]:** the Objective 2 primary contrast (the 24 h matched-width level comparison and the co-primary core-landmark contrast) first, then Objective 1, then Objective 3.

# Copilot run rules (real data) [plan v13 §15]

- Copy the script into the project `scripts/` folder, verify its SHA-256, and **run it unedited**. Stop on any failure **without patching**. Report only the shared JSON/MD and the console summary.
- A stopped run is fixed in the spec and the script (new version, new hash) and re-run from the start; a stale copy is replaced, never edited.
- Every script stops on: any pinned-hash mismatch; any failure to reproduce the cohort, split, or landmark counts; any unit-test failure; any validation or test metric feeding a train threshold; any release-gate failure; and the 1% invalid-age safety gate.
- **Real-data runs are done by Vinai only.** Teammates and assistants develop and test on the synthetic dataset only.
- **`_raw_LOCAL` rule:** the unsuppressed `*_raw_LOCAL.*` copy stays on the credentialed machine, is never shared, and is **never used or accepted as an input** by any stage, teammate, reviewer, or assistant.
- Copilot and other assistants see code, specs, decision maps, and aggregate outputs only; they never execute analysis code on patient data and never open patient data.

# Synthetic development [V11, TC18; plan v13 §12.3, §13.3]

The synthetic dataset (`synthetic_svd_dataset/`, including `6_tensor_ready_layers`) is kept as is. It predates D29 and V10, so test code drops or ignores the P/F, S/F, FiO2-based, and `psofa_resp` channels; the schema-match requirement applies to the remaining channels. Known differences: synthetic maximum LOS is about 280 h (no synthetic encounter is eligible at 288–336 h), and synthetic imputation uses uncapped LOCF. Thresholds of H1–H17 are fixed from the synthetic runs and frozen before real data.

# Disclosure and release gates [plan v13 §14, I15; V29; Stage 2 v1.0.2]

**Suppression.** Real-data outputs leave the secure environment only as aggregates: counts 1–9 published as `<10`, with complementary `S` suppression; percentages and quantiles only where the underlying count is ≥ 10 and published; quantiles limited to p10–p90; no minima, maxima, identifiers, timestamps, or draw-level or patient-level rows; no SUBJECT_ID, HADM_ID, ICUSTAY_ID, or encounter identifiers. Shared files are written only after a disk-reload release gate passes.

**Complementary / differencing release gate (required for every future release gate: Stage 3 and later, and any re-release).** A single-cell recoverability check is not sufficient. Every release gate must include:

(a) **Residual check:** whenever a group total and some of its parts are published, total minus the published parts (the sum of the suppressed parts) must be at least 10 (or 0 when nothing is suppressed).

(b) **Interval (LP) audit** of every suppressed cell over all published identities, cross-table links, and bounds: a `<10` cell must keep its full 0–9 feasible range; any other suppressed cell must keep at least W = 10 feasible integers or straddle 9|10.

(c) **Declared links** across tables of the same release, against earlier releases, and against other shared outputs, with already-public cells frozen.

(d) **Fail closed:** the gate exits nonzero and leaves no release file when any check fails.

Reference implementation: `stage2_v1_0_2_suppress.py` v1.3 (`df77234b6c68e1ca9b542f6f387f861141649fdd32e88f5f23ea009acac46d4f`; it replaces v1.2 `c8e449b47158f321c8bc3d1e8824f72aaa0b98fb9c8efdad1fff0041a56caf7f`, which stopped on two cells that could not be protected, and earlier v1.0 `a341cdb0…` and v1.1 `c290ebce…`, which did not converge on the real input) with `stage2_v1_0_2_suppression_spec.md` v1.3, which also re-release the Stage 2 v1.0.1 output as v1.0.2. The v1.0.2 release uses the drop list `stage2_v1_0_2_drops.csv` (`960e2c6560d4902b6ffdb980fcbb82cacf9589002c50cbc294b3880ca932923a`), which withholds the subtree `/drug_groups/nmb/validation/strata/surgical_status` as a whole (2 count cells; pointers only, no values) by decision of November 16, 2026; the NMB validation total is kept.

**Rules for the contract text and its outputs [V29].**

- Never quote a total beside its published component cells when the difference would reveal suppressed cells (example: the FiO2 5222 numeric-row total is never quoted beside the count of its values in 21–100, because the remainder outside that range is a suppressed cell under 10).
- Quote only published counts; reasoning-based statements are labeled as reasoning (for example, that every out-of-gate P/F pair lies above 700 is the review's reasoning, log Step 5.11, not a published count; the published follow-up result is stated in words: about half of the in-gate pairs have P/F ≥ 400). Where a count could be combined with another published count to recover a suppressed cell, the text states the finding in words only.
- The 42 invalid DOBs are a v1.0 run-stop event only, never combined with v1.0.1 cells.
- Cohort-log counts under 10 are not quoted.

[DUA confirmation that aggregate outputs may enter an assistant's context: carried open from v12.]

# Documentation standards

- Each matrix or tensor records provenance for every feature family and transformation.
- Each script states its dependencies, cohort assumptions, pinned inputs, and output paths.
- **Publication (I9):** build in memory, validate, write a temp file, fsync, `os.replace`. Sidecar `meta.json`: plan version (v13), contract version (v13), SHA-256 of the output bytes, fieldnames / feature_columns, UTC timestamp, raw_summary flag, and the SHA-256 of every pinned input used; plus a manifest.
- Keys vs features (I8): `encounter_id` and split are keys; every check runs on the `feature_columns` list of meta.json.
- Audit outputs are aggregate-only, cells under 10 suppressed, released only through the gate above.

# Decision rule for finalization

No final matrix specification, matrix-generation script, or benchmark run proceeds on real data without explicit approval of this contract (checklist v13) by Vinai.
