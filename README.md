# PIC Foundations analysis (stages 09–18)

This repository holds the **Foundations analysis** that starts from the frozen Stage 06–08 layers and runs the spectral diagnostics, predictive models, attribution, and desktop sensitivity studies.

The preprocessing and three-layer A / B / C build live in the companion repository: [pic-preprocessing-layers](https://github.com/mvinai/pic-preprocessing-layers).

**Authors.** Vinai (principal investigator), with contributions from Habib, Vinay and Yashwant. Scripts were written with AI assistance and validated on synthetic data before any real PIC run.

This repository is released under the MIT License. Copyright (c) 2026 Vinai Modem. See `LICENSE`.

## What this is not

This is not a copy of the [Paediatric Intensive Care (PIC) database](https://physionet.org/content/picdb/) and it is not a copy of any real-run output. PIC is credentialed health data under a PhysioNet data-use agreement. To reproduce the real analysis you need approved access to PIC, the frozen Stage 06–08 release from the preprocessing repository (run locally), and the omitted cohort pins described in that repository. Do not commit PIC data or `*_INTERNAL` / `*_TEAM` real-run files here.

## How the analysis fits together

The analysis is organised as a sequence of **runs**. Each run has a script, a test module that builds its own tiny synthetic project, and a specification.

| Run | Stages | Script (under `scripts/`) | Package version |
|---|---|---|---|
| Run 1 | 09p, 09, 10 | `stage09_18_r1/r1_09p_09_10_spectral_lowrank.py` | v1.0.2 |
| Run 2 | 11p, 11, 12, 17 | `stage11_18_r2/r2_11p_11_12_17_models.py` | v1.0.0 |
| Run 3a | 14, 18 (test scoring) | `stage13_18_r3/r3a_14_18_test_scoring.py` | v1.0.0 |
| Run 3b | 13, 16, 18 (train/validation only) | `stage13_18_r3/r3b_13_16_18_attribution.py` | v1.1.0 |
| Desktop Tier 1 | 14 + 18 (3a) and 13 + 16 + 18 (3b), full settings | `stage13_18_r3_desktop/r3a_desktop_full.py`, `r3b_desktop_full.py` | **r3a v2.0.1-desktop**, **r3b v2.0.2-desktop** |
| Desktop Tier 2 | sequence-model family comparison | `stage13_18_r3_desktop/r4_desktop_seq_models.py` | v1.0.1-desktop |
| Desktop Tier 3 | Stage 15 Layer C levels (raw / normalized / engineered) | `stage13_18_r3_desktop/r5_desktop_layer_c_levels.py` | v1.0.1-desktop |

The folder layout under `scripts/` matches the project layout used in the Copilot instruction notes (`scripts\stage09_18_r1\`, `scripts\stage11_18_r2\`, `scripts\stage13_18_r3\`, `scripts\stage13_18_r3_desktop\`), so those notes still point at the same relative paths.

### Desktop Tier 1 package version

Desktop Tier 1 ships **r3a v2.0.1-desktop** and **r3b v2.0.2-desktop**. v2.0.2 keeps the v2.0.1 settings and disables cuDNN only around the expected-gradients backward pass (the cuDNN GRU backward does not run when the frozen network is in eval mode). These public copies may differ slightly from the private desktop packages; see `docs/PUBLIC_REDACTIONS.md`.

## Environment

The released runs used:

- Python 3.14
- numpy 2.4.2
- pandas 3.0.1
- scipy 1.17.x
- scikit-learn 1.8.0
- matplotlib 3.10.8
- torch 2.10 (CPU is enough for the tests; the desktop full runs used CUDA)

See `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the tests (synthetic data only)

Every test module builds a temporary synthetic Stage 06–08-shaped project. No PIC data and no pre-built fixture tree are required. Run from the repository root. CPU only is expected.

```bash
export PYTHONNOUSERSITE=1

python -s -m pytest -q -p no:cacheprovider scripts/stage09_18_r1/test_r1_09p_09_10_spectral_lowrank.py
python -s -m pytest -q -p no:cacheprovider scripts/stage11_18_r2/test_r2_11p_11_12_17_models.py
python -s -m pytest -q -p no:cacheprovider scripts/stage13_18_r3/test_r3a_14_18_test_scoring.py
python -s -m pytest -q -p no:cacheprovider scripts/stage13_18_r3/test_r3b_13_16_18_attribution.py
python -s -m pytest -q -p no:cacheprovider scripts/stage13_18_r3_desktop/test_r3a_desktop_full.py
python -s -m pytest -q -p no:cacheprovider scripts/stage13_18_r3_desktop/test_r3b_desktop_full.py
python -s -m pytest -q -p no:cacheprovider scripts/stage13_18_r3_desktop/test_r4_desktop_seq_models.py
python -s -m pytest -q -p no:cacheprovider scripts/stage13_18_r3_desktop/test_r5_desktop_layer_c_levels.py
```

Or, in one pass: `python -s -m pytest -q -p no:cacheprovider scripts`.

A small number of tests may be skipped on this platform (for example a read-only user-profile test that is Linux-only, or CUDA-only checks). That is expected.

## Documentation

Final specifications and the analysis protocol are under `docs/`:

- `docs/plan/revised_plan_v13.md` — analysis plan
- `docs/plan/foundations_contract_v13.md` — data and analysis contract
- `docs/plan/stage09_18_plan.md` — stages 09–18 run plan
- `docs/specs/` — per-run specifications
- `docs/copilot/` — the instruction notes used to launch the real runs (hashes are of the private packages; see `docs/PUBLIC_REDACTIONS.md`)

Cohort and split sizes have been omitted from the public copies of the plan and contract.

## Citation

See `CITATION.cff`.
