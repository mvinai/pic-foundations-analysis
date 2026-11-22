# What was omitted from this public copy

This repository is a public, DUA-safe copy of the analysis code. It is not a byte-for-byte dump of the private project folder.

## Not included

- Any PIC (Paediatric Intensive Care) row-level data, real-run outputs, `*_INTERNAL` files, `*_raw_LOCAL` files, or `*_TEAM` result tables from real runs.
- Cohort, split, landmark-eligibility, death, and raw-table row-count pins. In the original project these are hard-coded reproduction gates. Here the corresponding defaults are empty or `None`. Restore them locally from `configs/cohort_pins.example.json` (copy to `configs/cohort_pins.json`) before a real PIC run. SHA-256 hashes of pinned inputs are kept: they do not reveal counts.
- Per-item `train_encounters` / `train_rows` / `train_n` values in the channel, drug, organism, specimen, and φ1-median maps. The mapping columns themselves are kept. The φ1 map still includes `median_hours`, which is a frozen train-set timing parameter used by the builder (see the open questions in the README).
- The Stage 2 team-internal public-values file and the Stage 3 Stage-2-constants file. Tests that read the former are skipped when it is absent; Stage 3 tests build a synthetic constants file themselves.
- Internal planning notes and report drafts. Specs and the analysis plan/contract are included as cleaned markdown.

## How this differs from the laptop-run packages

The script SHA-256 values recorded in the Copilot instruction notes are the hashes of the **private** packages, which still contain the pins. The published `.py` files differ by this redaction (and, for one Stage 2 test, by a skip marker and a broader search path for the Stage 3 builder). Use the Copilot notes as a record of how the real runs were launched, not as a hash check against this public tree.

Source PDFs and spreadsheets of WHO / published vital-sign standards are not included (third-party copyright). The derived CSVs and their provenance notes are included.

## Script hash pins in this public tree

The published Run 2 script differs from the private v1.0.0 package by redaction of a course identifier in the module docstring, so its SHA-256 is `8b0cd7a14152e977fac7c6a078208f2835131ca24c0d9646f09700d5ce61feb4`. Downstream scripts in this tree (`r3a`, `r3b`, `r4`, `r5`, laptop and desktop) pin that published hash. The private-package hash `b0af376745ee5e828e9fed041eae45918de13a7d33f8edddd36971452dc80a40` remains the recorded hash of the real-run package and is still what the Copilot instruction notes and some specifications quote.

## Personal path redaction

Local Windows paths (the project folder, the raw PIC directory, the split-file path, the desktop venv interpreter, and the username they contained) were replaced with placeholders: `<PROJECT_ROOT>`, `<PIC_RAW_DIR>`, `<SPLIT_FILE>`, `<VENV_PYTHON>`, and `<VENV_DIR>`. CLI defaults that used to hard-code those paths now read `PIC_RAW_DIR` and `SPLIT_FILE_PATH` from the environment, with no local fallback. Downstream scripts in this tree that hash-pin a sibling public script were retargeted to the published hash of that script (the hashes quoted in Copilot notes and some specifications remain the recorded hashes of the private packages). Script hashes below are of the files in this public tree after that redaction.

Changed scripts and their SHA-256 in this tree:

- `scripts/stage09_18_r1/r1_09p_09_10_spectral_lowrank.py` — `c656ef85ea638cfc2f1074975952b5a6a8be7df8c6217b1556b00b532a65b1e3`
- `scripts/stage09_18_r1/test_r1_09p_09_10_spectral_lowrank.py` — `51db3ab8975d7594971ae2fcbcb88193bcaa7c4ddb89e0355c165b999bef3d54`
- `scripts/stage11_18_r2/r2_11p_11_12_17_models.py` — `0262df9482c360baeb091f9bfcb4107e19d45712fa9f1aa8fd9db8f5c270ffbb`
- `scripts/stage11_18_r2/test_r2_11p_11_12_17_models.py` — `a26ca8810994e50cc8bdaacfa4bded4d3ca7f63fc6675f52dcdf50b02f6d8d7b`
- `scripts/stage13_18_r3/r3a_14_18_test_scoring.py` — `b5709a919b90ef89cf02e6438c6a0fee12b8346e033c2093902a8811daf36328`
- `scripts/stage13_18_r3/r3b_13_16_18_attribution.py` — `1d2610a366382bcff65f39d4cc6e1242500000c33f5e1b4f08b38694fb07c787`
- `scripts/stage13_18_r3/test_r3a_14_18_test_scoring.py` — `5aa44ae210e678d3bc0e117a2c617c0662844a934d200ce48332df847f5d873b`
- `scripts/stage13_18_r3/test_r3b_13_16_18_attribution.py` — `832d22a3c63ce7fc1a934b7d8dccbac52471b6e3df283535ced6780f267c96fb`
- `scripts/stage13_18_r3_desktop/r3a_desktop_full.py` — `d6eee12b0e2ce5e4d2db58351cc033602a942b6838955dd784350364176f1093`
- `scripts/stage13_18_r3_desktop/r3b_desktop_full.py` — `ac47394f4b9308b07c072b25a09b985f58999bd3dbdcc23ca583fd08d697e312`
- `scripts/stage13_18_r3_desktop/r4_desktop_seq_models.py` — `6a98be6e6e9f1b51d2bb4e0cbacd3c20872706296bed2a8a17664a85ccb0cf48`
- `scripts/stage13_18_r3_desktop/r5_desktop_layer_c_levels.py` — `7a1b28df250b19c2b0d3030348646f8d8d5b22b23053e1076394963aafced710`
- `scripts/stage13_18_r3_desktop/test_r3a_desktop_full.py` — `24297119755343bd21d6a6f6258b2e7093d1536d806ea9fb66a03236895e50cc`
- `scripts/stage13_18_r3_desktop/test_r3b_desktop_full.py` — `d6e78e8812ab4deae518f6aade3c2c441a31a7c7bec809975f0db96a721cc2fd`
- `scripts/stage13_18_r3_desktop/test_r4_desktop_seq_models.py` — `669f757fff64b2775919c0f4568b08aa50bc961463485994a4f15b4301a7946b`
- `scripts/stage13_18_r3_desktop/test_r5_desktop_layer_c_levels.py` — `44aa45fe3e7765090a8abd7873d26838956aa50ba60ccf2e49a6223e8cccf9b5`
