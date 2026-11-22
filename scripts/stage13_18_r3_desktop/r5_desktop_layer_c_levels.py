"""Desktop Tier 3 Layer C levels (PIC Foundations, PIC; plan v13, contract v13):

Stage 15 builds three Layer C representations — raw, normalized, and clinically engineered — and
compares them with the D19 GRU (Layer A+B static context at every step, Layer A baseline-risk
offset, CORAL-style cumulative-link head) at the seven core landmarks (24, 48, 72, 120, 168, 240,
336 h). Tuning is validation-only (D20). Test is scored once under a refuse-rerun lock. This is a
labelled post-primary sensitivity analysis, not a replacement for the laptop Run 2 GRU or the
laptop Run 3a primary scoring. Nested cross-validation is not used. An optional second architecture
(LSTM, LSTM with attention, or transformer) can be taken from the pinned Tier 2 module
r4_desktop_seq_models.py and chosen on validation.

r5_desktop_layer_c_levels v1.0.1-desktop (Nov 21, 2026).

v1.0.1: GPU name, GPU memory, the Python executable path and other host environment strings are kept out of TEAM files (they are written only to an INTERNAL environment record). TEAM files may report a coarse cuda yes/no flag and device 'cuda' or 'cpu'. The identifier gate is unchanged. A prior unpublished test-use attempt does not block a rerun; a published attempt still does. If a prior unpublished attempt exists, the audit logs that the test set was scored once before in that unpublished attempt.

All files this script writes go under data/processed/ladder/DESKTOP_FULL_PLAN/stage15_layer_c_levels/
(published folder, pending folder, cache, test-use log, internal files, logs and README_DESKTOP.txt).
A published laptop or desktop Run 3a/3b or Tier 2 does not block this run, but a pending folder from
Tier 1 or Tier 2 under DESKTOP_FULL_PLAN does (this run must not share the desktop with those runs).

Frozen inputs (hash-gated before anything is parsed): everything Run 2 checked (Stage 06-08 manifest
and files, Run 1 v1.0.2 manifest and files, canonical, harmonization, split, raw ICUSTAYS /
ADMISSIONS; the same pins), the Run 2 script (scripts/stage11_18_r2/r2_11p_11_12_17_models.py,
imported only after its SHA-256 matches the pin), and the Run 2 manifest (pinned by its full
SHA-256 on the command line and in this script) with every file it lists. When --second-family is
set, the Tier 2 script is imported only after its SHA-256 matches the pin.

A test-use log (data/processed/ladder/DESKTOP_FULL_PLAN/stage15_layer_c_levels/test_use_log_stage15_layer_c_levels.json)
records each desktop attempt; a run refuses to start once a published desktop attempt exists in
that log. Device is CUDA if available, otherwise CPU, and is logged. Mixed precision is used only
when --amp is set and the device is CUDA.

Outputs are published atomically from a pending folder. TEAM files are aggregate only. INTERNAL
files (predictions and selected states) are never opened, pasted or shared. Release gate as in Run 2.
Exit codes: 0 ok; 2 stopped on an input check (nothing published); 3 build or release-gate failure
(nothing published).
Assistants never run this script on patient data; it runs on the real data only through Copilot, unedited.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import pickle
import re
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_THREADS = str(min(4, max(1, (os.cpu_count() or 2) // 2)))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_VERSION = "r5_desktop_layer_c_levels v1.0.1-desktop"
SPEC_DOC = "stage15_layer_c_levels_desktop_spec v1.0.1-desktop"
PLAN, CONTRACT = "v13", "v13"
OUT_NAME = "stage15_layer_c_levels"
DESKTOP_ROOT_NAME = "DESKTOP_FULL_PLAN"
README_NAME = "README_DESKTOP.txt"
README_TEXT = (
    "These are post-primary sensitivity outputs from the desktop Tier 3 Stage 15 Layer C level "
    "comparison (r5_desktop_layer_c_levels v1.0.1-desktop): raw, normalized, and clinically "
    "engineered Layer C tensors compared with the D19 GRU at the seven core landmarks, tuned on "
    "validation only and scored once on test. They are a second look, not a replacement for the "
    "laptop primary Run 2 / Run 3a results. Published folders, pending folders, caches, internal "
    "files and logs for this run belong only in this folder "
    "(data/processed/ladder/DESKTOP_FULL_PLAN/stage15_layer_c_levels/).\n"
)
RUN2_NAME = "stage11_12_17_r2"
RUN2_MODULE = "r2_11p_11_12_17_models"
R4_MODULE = "r4_desktop_seq_models"
TEST_LOG_NAME = f"test_use_log_{OUT_NAME}.json"
CACHE_NAME = f"_cache_{OUT_NAME}"
PUBLISHED_NAME = "published"

PINS3 = {
    "run2_script": "r2_11p_11_12_17_models v1.0.0",
    "run2_script_sha256": "0262df9482c360baeb091f9bfcb4107e19d45712fa9f1aa8fd9db8f5c270ffbb",
    "run2_manifest": "ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed",
    "r4_script": "r4_desktop_seq_models v1.0.1-desktop",
    "r4_script_sha256": "7878e8bf9f161bf5b1091d72140a3b83aca6702d8b657ed017996f742e054dfb",
}
RUN2_INTERNAL = ("stage11p_labels_landmark_INTERNAL.npz", "stage11_12_17_validation_predictions_INTERNAL.npz",
                 "stage12_tuned_model_calibrators_INTERNAL.pkl", "stage17_models_INTERNAL.pt",
                 "stage17_tabular_models_INTERNAL.pkl")
RUN2_AUDIT = "stage11_12_17_r2_audit_TEAM.json"
RUN2_TEAM_USED = ("stage11_validation_scores_TEAM.csv", "stage11_layer_c_24h_TEAM.csv", "stage11_winner_TEAM.csv")

EXPECTED_PY = (3, 14)
EXPECTED_PACKAGES = OrderedDict([
    ("numpy", "2.4"),
    ("pandas", "3.0"),
    ("scipy", "1.17"),
    ("sklearn", "1.8"),
    ("torch", "2.10"),
])

LEVELS = ("raw", "normalized", "engineered")
LEVEL_CODE = {"raw": 0, "normalized": 1, "engineered": 2}
FAMILY_CODE = {"gru": 0, "lstm": 1, "lstm_attn": 2, "transformer": 3}
SECOND_FAMILIES = ("lstm", "lstm_attn", "transformer")
CORE_LANDMARKS = (24, 48, 72, 120, 168, 240, 336)
DEFAULT_MAX_EPOCHS = 40
DEFAULT_N_SEEDS = 1
DEFAULT_N_BOOT = 1000
DEFAULT_PATIENCE = 5
DEFAULT_BATCH = 256
GRU_HIDDEN_GRID = (32, 64)
EMB = 16
ALPHA = 0.05
DELTA = 0.02
DELTA_C = 0.01
AGEBAND_MIN_CELLS = 5
ROLL_W = 4
SINCE_CAP = 8
AKI_Z_CUTS = (0.5, 1.0, 1.5)  # z-space surrogate, not native-unit KDIGO
DAYS_PER_MONTH = 365.25 / 12.0
R2 = None
R4 = None

# Contract §8.1 / D20 age bands, in months.
AGE_BANDS = (
    ("neonate", 0.0, 28.0 / DAYS_PER_MONTH),
    ("infant", 28.0 / DAYS_PER_MONTH, 12.0),
    ("toddler", 12.0, 36.0),
    ("preschool", 36.0, 72.0),
    ("school", 72.0, 144.0),
    ("adolescent", 144.0, 216.0),
)
AGE_INVALID_BAND = "age_invalid"

# Core channels that receive slope / rolling SD / threshold / worst-in-window features when present.
CORE_CHANNELS = (
    "heart_rate", "map", "sbp", "dbp", "respiratory_rate", "spo2", "temperature",
    "creatinine", "urine_output", "bun", "cystatin_c",
    "sodium", "potassium", "chloride", "bicarbonate", "calcium_total", "calcium_ionized",
    "magnesium", "phosphate",
    "ph", "pco2", "po2", "lactate", "base_excess", "sao2_bloodgas",
    "hemoglobin", "wbc", "platelets", "anc", "inr", "ptt",
    "bilirubin_total", "bilirubin_direct", "ast", "albumin", "ggt",
    "glucose", "glucose_bedside", "fluid_intake", "pain_score",
)
HOURS_SINCE_CHANNELS = (
    "heart_rate", "map", "respiratory_rate", "spo2", "temperature",
    "creatinine", "urine_output", "lactate",
)
HIGH_WORSE = frozenset({
    "creatinine", "bun", "cystatin_c", "lactate", "bilirubin_total", "bilirubin_direct",
    "ast", "ggt", "inr", "ptt", "d_dimer", "crp", "amylase", "lipase",
})
LOW_WORSE = frozenset({
    "map", "sbp", "dbp", "spo2", "po2", "ph", "bicarbonate", "hemoglobin", "platelets",
    "albumin", "urine_output", "fibrinogen", "anc",
})
# Table 5.1 (channel map v2.3), names only; a channel is used when present in the Stage 08 tensor.
ORGAN_MEMBERS = OrderedDict([
    ("cardiovascular", ("heart_rate", "sbp", "dbp", "map", "lactate", "bnp", "troponin_t", "troponin_i")),
    ("respiratory", ("respiratory_rate", "spo2", "ph", "pco2", "po2", "base_excess", "bicarbonate", "sao2_bloodgas")),
    ("renal", ("creatinine", "bun", "cystatin_c", "urine_output", "fluid_intake", "kdigo_uo_stage")),
    ("hepatic", ("bilirubin_total", "bilirubin_direct", "albumin", "ast", "ggt", "alt", "ammonia")),
    ("gi", ("amylase", "lipase", "stool_occult_blood")),
    ("hematologic", ("hemoglobin", "platelets", "lymphocytes_abs", "ldh", "inr", "ptt", "fibrinogen", "d_dimer")),
    ("infectious", (
        "temperature", "wbc", "anc", "crp",
        "urine_wbc", "urine_nitrite", "urine_leukocyte_esterase",
        "resp_panel_rsv", "resp_panel_parainfluenza", "resp_panel_influenza",
        "resp_panel_metapneumovirus", "resp_panel_mycoplasma",
        "systemic_viral_cmv", "systemic_viral_ebv", "systemic_viral_hsv",
        "systemic_viral_enterovirus", "bacteremia_pcr",
    )),
    ("neurologic", ("pain_score", "csf_wbc", "csf_rbc", "csf_protein_pandy", "nse")),
    ("electrolytes", ("sodium", "potassium", "chloride", "calcium_total", "calcium_ionized", "magnesium", "phosphate")),
    ("endocrine", ("glucose", "glucose_bedside", "cortisol", "free_t4", "insulin", "osmolality")),
])
# Fifth pSOFA-like component is lactate (perfusion); respiratory P/F and vasoactive are unavailable.
PSOFA_COMPONENTS = OrderedDict([
    ("cardio", ("map", "low")),
    ("coag", ("platelets", "low")),
    ("liver", ("bilirubin_total", "high")),
    ("renal", ("creatinine", "high")),
    ("lactate", ("lactate", "high")),
])

DECISIONS = OrderedDict([
    ("D17", "the Stage 08 base Layer C tensor was the single Layer C representation in earlier runs; "
            "this Tier 3 run builds the Stage 15 raw / normalized / engineered comparison on top of it"),
    ("D18", "thresholds frozen before real runs: delta = 0.02 (QWK equivalence), delta_C = 0.01 (engineering gain)"),
    ("D19", "A+B+C sequence model: one small GRU, Layer A+B static context at every step, Layer A baseline-risk "
            "offset in a CORAL-style cumulative-link head, early stopping on a 10% subject holdout inside train, "
            "hidden in {32, 64} chosen on validation QWK"),
    ("D20", "validation-only tuning (no nested CV); B_eval = 1000 paired subject-cluster bootstrap replicates"),
    ("D21", "scripts written by the assistant; teammates cross-check them on the synthetic sandbox"),
])
DEVIATIONS = [
    "this is a post-primary sensitivity analysis on the desktop, not a replacement for the laptop Run 2 GRU "
    "or the laptop Run 3a test scoring; test rows are scored once under a separate refuse-rerun log",
    "Outcome 1 only, at the seven core landmarks; Outcome 2, ordinal8 and the non-core landmarks are not fit",
    "the three levels are raw, normalized and engineered, not the full plan c0/c1/c2s/c2d/c2p ladder; "
    "engineered is the normalized backbone plus a frozen clinical block (organ-block summaries and a "
    "pSOFA-like score live inside that one engineered level)",
    "the raw level uses Stage 08 X_raw_z (already train-standardized, with NaN) plus missing and "
    "carried-forward flags, not native-unit values: Stage 08 does not publish unstandardized Layer C arrays, "
    "and the fitted-transform file is team-internal and is not opened",
    "Layer C z-scores are pooled per channel on the landmark prefix (Stage 08), not age-banded; age-band "
    "thresholds are therefore 2 SD flags around the train age-band mean of those already-pooled z-scores "
    f"(contract §8.1 bands, fallback to the overall train mean/SD when a band has fewer than {AGEBAND_MIN_CELLS} "
    "observed cells)",
    "rolling SD uses a strictly past window (bins k-4 to k-1, at least 3 observed not-carried values); "
    "slope, worst-in-window and time-above/below use the current bin and the 3 before it (past 24 h)",
    "KDIGO creatinine staging is a labelled z-space surrogate (stage 1/2/3 at baseline-delta z of "
    f"{AKI_Z_CUTS[0]}/{AKI_Z_CUTS[1]}/{AKI_Z_CUTS[2]}); true native-unit KDIGO cutoffs cannot be applied "
    "without inverting Stage 08 z-scores. Baseline is the minimum observed creatinine z in the prefix up to "
    "the current bin. Stage 08 kdigo_uo_stage is used when the channel is present (it is dropped at 24 h)",
    "per-bin vasoactive state, ventilation state, the thirteen drug groups, microbiology M1 and surgery in "
    "the bin of T are omitted: Stage 08 Layer C has no event channels, and plan §6.3 records that PIC has "
    "no ventilation source item. Layer B 24 h drug flags stay in the A+B static context only",
    "the modified pSOFA-like score uses four available organ components (MAP, platelets, bilirubin, "
    "creatinine) plus lactate as a documented fifth; the respiratory P/F component is dropped (D29) and "
    "vasoactive exposure is not available per bin. Component points are z-space surrogates, not official pSOFA cuts",
    "nested cross-validation remains deferred; hyperparameter choice and the optional second architecture "
    "are selected on validation QWK, so validation level contrasts are descriptive and the locked test look "
    "is the sensitivity analysis",
    "the optional second architecture (--second-family) uses the pinned Tier 2 module and that family's "
    "small fixed validation grid; the selected model at each level and landmark is the highest validation QWK "
    "among the GRU hidden grid and, when requested, that second family",
    "no runtime cap (max_seconds default 0); mixed precision is off unless --amp is set on CUDA",
    "all writes are confined under data/processed/ladder/DESKTOP_FULL_PLAN/stage15_layer_c_levels/",
    "this run must not be started while a Tier 1 or Tier 2 pending folder exists under DESKTOP_FULL_PLAN",
]

UNPUBLISHED_RESCORE_DEVIATION = (
    "the test set was scored once before in an unpublished attempt recorded in this run's test-use "
    "log; a rerun is allowed because that attempt did not publish; the published results are from "
    "this later attempt, and the earlier unpublished scoring is a logged deviation"
)


def deviations_for(pre: dict | None = None) -> list:
    """Module deviations, plus a logged rescoring deviation when a prior unpublished attempt exists."""
    out = list(DEVIATIONS)
    if pre and int(pre.get("prior_attempts") or 0) > 0:
        out.append(UNPUBLISHED_RESCORE_DEVIATION)
    return out


class InputError(Exception):
    pass


class BuildError(Exception):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load_module(name: str, path: Path, expect_sha: str | None):
    if not path.is_file():
        raise InputError(f"{name} script not found: {path.name}")
    h = sha256_file(path)
    if expect_sha and h != expect_sha:
        raise InputError(f"{name} script SHA-256 differs from the pin")
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) and \
            os.path.abspath(sys.modules[name].__file__) == os.path.abspath(path):
        return sys.modules[name], h
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod, h


def load_run2(path: Path, expect_sha: str | None):
    global R2
    mod, h = _load_module(RUN2_MODULE, path, expect_sha)
    R2 = mod
    return mod, h


def load_r4(path: Path, expect_sha: str | None):
    global R4
    mod, h = _load_module(R4_MODULE, path, expect_sha)
    R4 = mod
    return mod, h


def errs(kind: str) -> tuple:
    extra = ()
    if R2 is not None:
        extra = (R2.InputError,) if kind == "input" else (R2.BuildError, R2.TimeLimit)
    if kind == "input":
        return (InputError,) + extra
    return (BuildError,) + extra


def say(msg: str) -> None:
    R2.say(msg)


def check_time() -> None:
    R2.check_time()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def desktop_root(project_root: Path) -> Path:
    return Path(project_root) / "data" / "processed" / "ladder" / DESKTOP_ROOT_NAME


def work_root(project_root: Path) -> Path:
    return desktop_root(project_root) / OUT_NAME


def must_write_under_work(path: Path, project_root: Path, role: str, *, allow_root: bool = False) -> None:
    base = work_root(project_root).resolve()
    p = path.resolve()
    if p == base:
        if allow_root:
            return
        raise InputError(f"{role} must be a subfolder of data/processed/ladder/{DESKTOP_ROOT_NAME}/{OUT_NAME}")
    try:
        p.relative_to(base)
    except ValueError:
        raise InputError(f"{role} must be under data/processed/ladder/{DESKTOP_ROOT_NAME}/{OUT_NAME}") from None


def write_desktop_readme(project_root: Path) -> None:
    d = work_root(project_root)
    d.mkdir(parents=True, exist_ok=True)
    p = d / README_NAME
    data = README_TEXT.encode("utf-8")
    if p.is_file() and p.read_bytes() == data:
        return
    p.write_bytes(data)


def pkg_major_minor(ver: str) -> str:
    parts = re.split(r"[^\d]+", str(ver).lstrip())
    return ".".join(parts[:2]) if len(parts) >= 2 else str(ver)


def check_environment(require_cuda: bool = False) -> dict:
    py = sys.version_info
    info = OrderedDict(python=f"{py.major}.{py.minor}.{py.micro}", executable=sys.executable)
    if (py.major, py.minor) != EXPECTED_PY:
        raise InputError(f"Python {py.major}.{py.minor} found, expected {EXPECTED_PY[0]}.{EXPECTED_PY[1]}")
    import numpy, pandas, scipy, sklearn, torch  # noqa: F401
    found = {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
    }
    for name, expect in EXPECTED_PACKAGES.items():
        got = pkg_major_minor(found[name])
        if got != expect:
            raise InputError(f"{name} {found[name]} found, expected {expect}.x")
        info[name] = found[name]
    info["cuda_available"] = bool(torch.cuda.is_available())
    if require_cuda and not torch.cuda.is_available():
        raise InputError("CUDA is required (--require-cuda) but torch.cuda.is_available() is False")
    if torch.cuda.is_available():
        info["cuda_version"] = str(torch.version.cuda)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    else:
        info["cuda_version"] = None
        info["gpu_name"] = None
        info["gpu_memory_bytes"] = None
    return info


def torch_device():
    torch = R2._torch()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not getattr(torch_device, "_logged", False):
        extra = ""
        if dev.type == "cuda":
            extra = f" ({torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda})"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        say(f"torch device: {dev}{extra}")
        torch_device._logged = True
    return dev


def seed_torch(torch, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class R2Unpickler(pickle.Unpickler):
    R2_CLASSES = {"OrdinalLinear", "FrankHall", "Multinom"}
    OK_ROOTS = {"numpy", "sklearn", "scipy", "collections", "copyreg", "_codecs"}
    OK_BUILTINS = {"dict", "list", "tuple", "set", "frozenset", "float", "int", "str", "bool", "bytes", "slice",
                   "complex", "range", "object"}

    def find_class(self, module, name):
        if module in ("__main__", RUN2_MODULE):
            if name not in self.R2_CLASSES:
                raise BuildError("a Run 2 model file references an unexpected class")
            return getattr(R2, name)
        if module == "builtins":
            if name not in self.OK_BUILTINS:
                raise BuildError("a Run 2 model file references an unexpected builtin")
            return super().find_class(module, name)
        if module == "_loss" and re.fullmatch(r"Cy\w*Loss", name):
            from sklearn._loss import _loss as sk_loss
            if not hasattr(sk_loss, name):
                raise BuildError("a Run 2 model file references an unexpected scikit-learn loss class")
            return getattr(sk_loss, name)
        if module.split(".")[0] in self.OK_ROOTS:
            return super().find_class(module, name)
        raise BuildError("a Run 2 model file references an unexpected module")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return R2Unpickler(f).load()



# ------------------------------------------------------------------------------------------------- Stage 15 math
def age_band_codes(age_months: np.ndarray, age_invalid: np.ndarray | None = None) -> np.ndarray:
    """Integer band codes: 0..5 for the six contract §8.1 bands, -1 for invalid / 18+ / missing age."""
    a = np.asarray(age_months, dtype=float)
    inv = np.zeros(a.shape, dtype=bool) if age_invalid is None else np.asarray(age_invalid, dtype=bool)
    inv = inv | ~np.isfinite(a) | (a < 0)
    code = np.full(a.shape, -1, dtype=np.int8)
    for i, (_name, lo, hi) in enumerate(AGE_BANDS):
        code[(code < 0) & ~inv & (a >= lo) & (a < hi)] = i
    return np.where(inv, -1, code)


def observed_mask(missing: np.ndarray, carried_forward: np.ndarray) -> np.ndarray:
    """True only for a value observed in that bin (not missing and not carried forward)."""
    return (np.asarray(missing) == 0) & (np.asarray(carried_forward) == 0)


def _rolling_windows(X: np.ndarray, width: int) -> np.ndarray:
    """(n, b, q) -> (n, b, width, q) windows ending at the current bin, left-padded with NaN."""
    n, b, q = X.shape
    parts = []
    for lag in range(width - 1, -1, -1):
        if lag == 0:
            parts.append(X)
        else:
            pad = np.full((n, lag, q), np.nan, dtype=X.dtype)
            parts.append(np.concatenate([pad, X[:, :b - lag, :]], axis=1))
    return np.stack(parts, axis=2)


def _shift_prev(X: np.ndarray) -> np.ndarray:
    n, b, q = X.shape
    return np.concatenate([np.full((n, 1, q), np.nan, dtype=X.dtype), X[:, :-1, :]], axis=1)


def _ols_slope(windows: np.ndarray, mask: np.ndarray, min_obs: int = 2) -> np.ndarray:
    """OLS slope of observed values vs window index 0..W-1. windows/mask: (n, b, W, q)."""
    w = windows.shape[2]
    t = np.arange(w, dtype=np.float64)
    t4 = np.broadcast_to(t[None, None, :, None], windows.shape)
    tm = np.where(mask, t4, np.nan)
    ym = np.where(mask, windows, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_mean = np.nanmean(tm, axis=2)
        y_mean = np.nanmean(ym, axis=2)
        tc = np.where(mask, t4 - t_mean[..., None, :], np.nan)
        yc = np.where(mask, ym - y_mean[..., None, :], np.nan)
        num = np.nansum(tc * yc, axis=2)
        den = np.nansum(tc * tc, axis=2)
        slope = num / np.where(den > 1e-12, den, np.nan)
    n_obs = mask.sum(axis=2)
    return np.where(n_obs >= min_obs, slope, np.nan).astype(np.float32)


def _nan_sd(windows: np.ndarray, mask: np.ndarray, min_obs: int = 3) -> np.ndarray:
    ym = np.where(mask, windows, np.nan)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(ym, axis=2, ddof=1)
    n_obs = mask.sum(axis=2)
    return np.where(n_obs >= min_obs, sd, np.nan).astype(np.float32)


def hours_since_last(observed: np.ndarray, cap: int = SINCE_CAP) -> np.ndarray:
    """Bins since the last true observation, capped; NaN before the first observation."""
    n, b, q = observed.shape
    idx = np.where(observed, np.arange(b, dtype=np.int32)[None, :, None], -1)
    last = np.maximum.accumulate(idx, axis=1)
    age = np.arange(b, dtype=np.float32)[None, :, None] - last.astype(np.float32)
    out = np.where(last >= 0, np.minimum(age, float(cap)), np.nan)
    return out.astype(np.float32)


def fit_ageband_stats(z: np.ndarray, observed: np.ndarray, train: np.ndarray, bands: np.ndarray,
                      min_cells: int = AGEBAND_MIN_CELLS) -> dict:
    """Train-only mean and SD of observed z per channel, overall and by age band. No val/test cells."""
    # z, observed: (n, b, q); train: (n,); bands: (n,)
    n, b, q = z.shape
    tr = np.asarray(train, dtype=bool)[:, None, None]
    use = observed & tr
    zv = np.where(use, z, np.nan)
    with np.errstate(invalid="ignore"):
        overall_mean = np.nanmean(zv, axis=(0, 1))
        overall_sd = np.nanstd(zv, axis=(0, 1), ddof=1)
    overall_sd = np.where(np.isfinite(overall_sd) & (overall_sd > 1e-8), overall_sd, 1.0)
    overall_mean = np.where(np.isfinite(overall_mean), overall_mean, 0.0)
    n_bands = len(AGE_BANDS)
    bmean = np.tile(overall_mean, (n_bands, 1))
    bsd = np.tile(overall_sd, (n_bands, 1))
    for bi in range(n_bands):
        m = use & (bands[:, None, None] == bi)
        if not m.any():
            continue
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(np.where(m, z, np.nan), axis=(0, 1))
            sd = np.nanstd(np.where(m, z, np.nan), axis=(0, 1), ddof=1)
        cnt = m.reshape(-1, q).sum(axis=0)
        ok = (cnt >= min_cells) & np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-8)
        bmean[bi, ok] = mu[ok]
        bsd[bi, ok] = sd[ok]
    return dict(overall_mean=overall_mean.astype(np.float32), overall_sd=overall_sd.astype(np.float32),
                band_mean=bmean.astype(np.float32), band_sd=bsd.astype(np.float32))


def apply_ageband_thresholds(z: np.ndarray, observed: np.ndarray, bands: np.ndarray, stats: dict) -> tuple:
    """above / below 2 SD of the train age-band mean, only on observed cells."""
    n, b, q = z.shape
    mu = np.zeros((n, q), dtype=np.float32)
    sd = np.ones((n, q), dtype=np.float32)
    mu[:] = stats["overall_mean"]
    sd[:] = stats["overall_sd"]
    bm, bs = stats["band_mean"], stats["band_sd"]
    for bi in range(len(AGE_BANDS)):
        sel = bands == bi
        if sel.any():
            mu[sel] = bm[bi]
            sd[sel] = bs[bi]
    mu3 = mu[:, None, :]
    sd3 = sd[:, None, :]
    above = observed & np.isfinite(z) & (z > mu3 + 2.0 * sd3)
    below = observed & np.isfinite(z) & (z < mu3 - 2.0 * sd3)
    return above, below, mu3, sd3


def worst_in_window(windows: np.ndarray, mask: np.ndarray, names: list, mid: np.ndarray) -> np.ndarray:
    """Direction-aware worst value in the window. mid is (n, 1, q) age-band mean for two-sided channels."""
    ym = np.where(mask, windows, np.nan)
    n, b, w, q = ym.shape
    out = np.full((n, b, q), np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        for j, name in enumerate(names):
            col = ym[:, :, :, j]
            if name in HIGH_WORSE:
                out[:, :, j] = np.nanmax(col, axis=2)
            elif name in LOW_WORSE:
                out[:, :, j] = np.nanmin(col, axis=2)
            else:
                # farther from the age-band (or overall) mean
                m = mid[:, :, j][:, :, None]
                dev = np.where(mask[:, :, :, j], np.abs(col - m), np.nan)
                far = np.nanargmax(np.where(np.isfinite(dev), dev, -1.0), axis=2)
                ii, kk = np.indices((n, b))
                pick = col[ii, kk, far]
                out[:, :, j] = np.where(mask[:, :, :, j].any(axis=2), pick, np.nan)
    return out


def aki_cr_stage(cr_z: np.ndarray, available: np.ndarray) -> tuple:
    """Z-space AKI/KDIGO creatinine surrogate. Baseline = min available creatinine z in bins 0..k.
    available includes carried values (plan: stage from carried creatinine). Returns (stage, delta, miss)."""
    x = np.where(available, cr_z, np.nan).astype(np.float64)
    filled = np.where(np.isfinite(x), x, np.inf)
    baseline = np.minimum.accumulate(filled, axis=1)
    baseline = np.where(np.isfinite(baseline) & (baseline < np.inf), baseline, np.nan)
    # current uses the available (observed or carried) value
    cur = x
    delta = cur - baseline
    stage = np.zeros(cr_z.shape, dtype=np.float32)
    stage[np.isfinite(delta) & (delta >= AKI_Z_CUTS[0])] = 1
    stage[np.isfinite(delta) & (delta >= AKI_Z_CUTS[1])] = 2
    stage[np.isfinite(delta) & (delta >= AKI_Z_CUTS[2])] = 3
    miss = ~np.isfinite(delta)
    stage = np.where(miss, np.nan, stage)
    return stage.astype(np.float32), delta.astype(np.float32), miss


def psofa_component(z: np.ndarray, available: np.ndarray, mu: np.ndarray, sd: np.ndarray, direction: str) -> np.ndarray:
    """Z-space 0-4 points from how many age-band SDs the available value sits on the worse side of the mean."""
    with np.errstate(invalid="ignore", divide="ignore"):
        zz = (z - mu) / np.where(sd > 1e-8, sd, np.nan)
    if direction == "low":
        zz = -zz
    pts = np.zeros(z.shape, dtype=np.float32)
    pts[np.isfinite(zz) & (zz >= 1.0)] = 1
    pts[np.isfinite(zz) & (zz >= 2.0)] = 2
    pts[np.isfinite(zz) & (zz >= 3.0)] = 3
    pts[np.isfinite(zz) & (zz >= 4.0)] = 4
    return np.where(available & np.isfinite(zz), pts, np.nan).astype(np.float32)


def bin_index_feature(n: int, b: int) -> np.ndarray:
    t = (np.arange(b, dtype=np.float32) / max(55, b - 1))[None, :, None]
    return np.broadcast_to(t, (n, b, 1)).copy()


def pack_backbone(X_val: np.ndarray, missing: np.ndarray, carried: np.ndarray) -> np.ndarray:
    """(n, b, 3q+1) float32: value, missing, carried-forward, scaled bin index."""
    n, b, q = X_val.shape
    val = np.where(np.isfinite(X_val), X_val, 0.0).astype(np.float32)
    return np.concatenate([
        val,
        missing.astype(np.float32),
        carried.astype(np.float32),
        bin_index_feature(n, b),
    ], axis=2)


def _channel_index(channels: list, name: str) -> int | None:
    try:
        return channels.index(name)
    except ValueError:
        return None


def build_engineered_block(X_raw_z: np.ndarray, missing: np.ndarray, carried: np.ndarray,
                           channels: list, train: np.ndarray, bands: np.ndarray, stats: dict | None = None) -> tuple:
    """Clinical block (values + missing flags) from train-only, past-only rules. Returns (block, names, stats).
    Population age-band stats are fit once on the train prefix (all of its bins, which are known at L) and
    may be reused; per-row features at bin k never use that row's bins after k."""
    n, b, q = X_raw_z.shape
    ch = [str(c) for c in channels]
    obs = observed_mask(missing, carried)
    avail = (missing == 0)  # observed or carried
    z = np.where(np.isfinite(X_raw_z), X_raw_z, np.nan).astype(np.float64)
    if stats is None:
        stats = fit_ageband_stats(z, obs, train, bands)
    above, below, mu3, sd3 = apply_ageband_thresholds(z, obs, bands, stats)

    # Windows on observed values only (others NaN)
    z_obs = np.where(obs, z, np.nan)
    win24 = _rolling_windows(z_obs, ROLL_W)
    mask24 = np.isfinite(win24)
    z_past = _shift_prev(z_obs)
    win_past = _rolling_windows(z_past, ROLL_W)
    mask_past = np.isfinite(win_past)

    slope_all = _ols_slope(win24, mask24, min_obs=2)
    sd_all = _nan_sd(win_past, mask_past, min_obs=3)
    above_w = _rolling_windows(above.astype(np.float64), ROLL_W)
    below_w = _rolling_windows(below.astype(np.float64), ROLL_W)
    # time-above uses the same observed mask as the value windows
    with np.errstate(invalid="ignore"):
        time_above = np.nanmean(np.where(mask24, above_w, np.nan), axis=2)
        time_below = np.nanmean(np.where(mask24, below_w, np.nan), axis=2)
    worst = worst_in_window(win24, mask24, ch, mu3)
    hsl_all = hours_since_last(obs)

    parts, names = [], []

    def add(name: str, arr: np.ndarray):
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 2:
            a = a[:, :, None]
        miss = (~np.isfinite(a)).astype(np.float32)
        a = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
        parts.append(a)
        parts.append(miss)
        names.append(name)
        names.append(f"{name}_missing")

    core_present = [c for c in CORE_CHANNELS if c in ch]
    for c in core_present:
        j = ch.index(c)
        add(f"{c}_slope_4bin", slope_all[:, :, j])
        add(f"{c}_rollsd_4bin_past", sd_all[:, :, j])
        if c in LOW_WORSE:
            add(f"{c}_time_below_2sd_24h", time_below[:, :, j])
        elif c in HIGH_WORSE:
            add(f"{c}_time_above_2sd_24h", time_above[:, :, j])
        else:
            add(f"{c}_time_above_2sd_24h", time_above[:, :, j])
            add(f"{c}_time_below_2sd_24h", time_below[:, :, j])
        add(f"{c}_worst_24h", worst[:, :, j])

    for c in HOURS_SINCE_CHANNELS:
        if c in ch:
            add(f"{c}_hours_since_last", hsl_all[:, :, ch.index(c)])

    # Organ-state flags and block summaries (in-bin and 24 h)
    for organ, members in ORGAN_MEMBERS.items():
        js = [ch.index(c) for c in members if c in ch]
        if not js:
            continue
        ab = above[:, :, js]
        bl = below[:, :, js]
        inbin = (ab | bl).any(axis=2)
        # 24 h: any in the window
        in24 = _rolling_windows(inbin.astype(np.float64)[:, :, None], ROLL_W)
        flag24 = np.nanmax(np.where(np.isfinite(in24), in24, 0.0), axis=2)[:, :, 0]
        add(f"organ_flag_{organ}", inbin.astype(np.float32))
        add(f"organ_flag24_{organ}", flag24.astype(np.float32))
        z_m = z_obs[:, :, js]
        with np.errstate(invalid="ignore"):
            meanz = np.nanmean(z_m, axis=2)
        # organ worst: max |z - mu| among observed members, signed toward the worse side if known
        mid = mu3[:, :, js]
        dev = np.where(obs[:, :, js], z_m - mid, np.nan)
        with np.errstate(invalid="ignore"):
            far = np.nanmax(np.abs(dev), axis=2)
            # keep the signed deviation of the farthest member
            arg = np.nanargmax(np.where(np.isfinite(dev), np.abs(dev), -1.0), axis=2)
            ii, kk = np.indices((n, b))
            signed = dev[ii, kk, arg]
            signed = np.where(obs[:, :, js].any(axis=2), signed, np.nan)
        add(f"organ_meanz_{organ}", meanz)
        add(f"organ_worstz_{organ}", signed)

    # KDIGO / AKI
    j_cr = _channel_index(ch, "creatinine")
    if j_cr is not None:
        stg, delta, miss = aki_cr_stage(z[:, :, j_cr], avail[:, :, j_cr])
        add("aki_cr_stage_zsurrogate", stg)
        add("cr_delta_baseline_z", delta)
    else:
        stg = np.full((n, b), np.nan, dtype=np.float32)

    j_uo = _channel_index(ch, "kdigo_uo_stage")
    if j_uo is not None:
        uo = np.where(avail[:, :, j_uo], z[:, :, j_uo], np.nan)  # already a 0-3 stage, but z-scored in Stage 08
        # re-expose the filled/available value from X_raw_z (may be z-scored); also a missing flag
        add("kdigo_uo_stage_value", uo)
        uo_for_combo = uo
    else:
        uo_for_combo = np.full((n, b), np.nan, dtype=np.float32)

    # Combined stage: max of Cr surrogate and, when the UO channel is a raw 0-3 stage stored as z, we cannot
    # recover the integer stage from a pooled z without inversion. When the array looks like {0,1,2,3} (synthetic
    # or unstandardized stage), use it directly; otherwise omit UO from the combo and keep the value channel.
    uo_int = np.full((n, b), np.nan, dtype=np.float32)
    if j_uo is not None:
        raw_uo = X_raw_z[:, :, j_uo]
        finite = raw_uo[np.isfinite(raw_uo)]
        if finite.size and np.nanmax(np.abs(finite - np.round(finite))) < 1e-5 and np.nanmax(finite) <= 3 + 1e-6 and np.nanmin(finite) >= -1e-6:
            uo_int = np.where(avail[:, :, j_uo] & np.isfinite(raw_uo), np.round(raw_uo), np.nan).astype(np.float32)
    combo = np.nanmax(np.stack([stg if j_cr is not None else np.full((n, b), np.nan), uo_int], axis=0), axis=0)
    add("kdigo_combined_stage", combo)

    # pSOFA-like components on the 24 h worst available value (observed or carried in the window)
    z_avail = np.where(avail, z, np.nan)
    win_av = _rolling_windows(z_avail, ROLL_W)
    mask_av = np.isfinite(win_av)
    worst_av = np.full((n, b, q), np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        for j, name in enumerate(ch):
            col = win_av[:, :, :, j]
            if name in HIGH_WORSE or (name not in LOW_WORSE):
                # high or two-sided default to max for pSOFA-style "worst high" except LOW_WORSE
                if name in LOW_WORSE:
                    worst_av[:, :, j] = np.nanmin(col, axis=2)
                else:
                    worst_av[:, :, j] = np.nanmax(col, axis=2)
            else:
                worst_av[:, :, j] = np.nanmin(col, axis=2)
    comps = []
    for cname, (chname, direction) in PSOFA_COMPONENTS.items():
        j = _channel_index(ch, chname)
        if j is None:
            add(f"psofa_comp_{cname}", np.full((n, b), np.nan))
            comps.append(np.full((n, b), np.nan, dtype=np.float32))
            continue
        pts = psofa_component(worst_av[:, :, j], np.isfinite(worst_av[:, :, j]),
                              mu3[:, :, j], sd3[:, :, j], direction)
        add(f"psofa_comp_{cname}", pts)
        comps.append(pts)
    total = np.nansum(np.stack(comps, axis=0), axis=0)
    any_c = np.any(np.stack([np.isfinite(c) for c in comps], axis=0), axis=0)
    add("psofa_mod_total", np.where(any_c, total, np.nan))

    # Cumulative net fluid (intake - output), observed increments only
    j_in, j_out = _channel_index(ch, "fluid_intake"), _channel_index(ch, "urine_output")
    if j_in is not None or j_out is not None:
        inc = np.zeros((n, b), dtype=np.float64)
        if j_in is not None:
            inc += np.where(obs[:, :, j_in], z[:, :, j_in], 0.0)
        if j_out is not None:
            inc -= np.where(obs[:, :, j_out], z[:, :, j_out], 0.0)
        cum = np.cumsum(inc, axis=1)
        seen = np.zeros((n, b), dtype=bool)
        if j_in is not None:
            seen |= np.logical_or.accumulate(obs[:, :, j_in], axis=1)
        if j_out is not None:
            seen |= np.logical_or.accumulate(obs[:, :, j_out], axis=1)
        add("fluid_net_cum", np.where(seen, cum, np.nan))

    block = np.concatenate(parts, axis=2) if parts else np.zeros((n, b, 0), dtype=np.float32)
    return block.astype(np.float32), names, stats


def build_level_tensor(level: str, X_raw_z: np.ndarray, X_filled_z: np.ndarray, missing: np.ndarray,
                       carried: np.ndarray, channels: list, keep: list, train: np.ndarray,
                       bands: np.ndarray) -> tuple:
    """Build the (n, b, d) model input for one level, using only the named keep channels in the backbone."""
    ch = [str(c) for c in channels]
    pos = [ch.index(c) for c in keep if c in ch]
    if level == "raw":
        S = pack_backbone(X_raw_z[:, :, pos], missing[:, :, pos], carried[:, :, pos])
        names = [f"{c}_{s}" for s in ("value", "missing", "carried_forward") for c in keep if c in ch]
        names.append("bin_index")
        return S, names, None
    if level == "normalized":
        S = pack_backbone(X_filled_z[:, :, pos], missing[:, :, pos], carried[:, :, pos])
        names = [f"{c}_{s}" for s in ("value", "missing", "carried_forward") for c in keep if c in ch]
        names.append("bin_index")
        return S, names, None
    if level == "engineered":
        back = pack_backbone(X_filled_z[:, :, pos], missing[:, :, pos], carried[:, :, pos])
        # drop the backbone bin index; we append one at the end after the clinical block
        back, t = back[:, :, :-1], back[:, :, -1:]
        block, bnames, stats = build_engineered_block(X_raw_z, missing, carried, ch, train, bands)
        S = np.concatenate([back, block, t], axis=2)
        names = [f"{c}_{s}" for s in ("value", "missing", "carried_forward") for c in keep if c in ch]
        names.extend(bnames)
        names.append("bin_index")
        return S, names, stats
    raise ValueError(level)


def lookahead_is_clean(X_raw_z, missing, carried, channels, train, bands, n_probe: int = 8) -> bool:
    """Perturbing a later bin must leave features at earlier bins unchanged (sample of rows and a mid bin)."""
    n, b, q = X_raw_z.shape
    if b < 3 or n == 0:
        return True
    k = b // 2
    rng = np.random.default_rng(6310)
    take = rng.choice(n, size=min(n_probe, n), replace=False)
    sl = lambda a: np.ascontiguousarray(a[take])  # noqa: E731
    # freeze train population stats on the original tensor, then perturb only later bins of the probe rows
    stats = fit_ageband_stats(np.where(np.isfinite(X_raw_z), X_raw_z, np.nan),
                              observed_mask(missing, carried), train, bands)
    block0, names0, _ = build_engineered_block(sl(X_raw_z), sl(missing), sl(carried), channels,
                                               sl(train), sl(bands), stats=stats)
    X2, m2, c2 = sl(X_raw_z).copy(), sl(missing).copy(), sl(carried).copy()
    if k + 1 < b:
        X2[:, k + 1:, :] = rng.normal(size=X2[:, k + 1:, :].shape)
        m2[:, k + 1:, :] = 0
        c2[:, k + 1:, :] = 0
    block1, names1, _ = build_engineered_block(X2, m2, c2, channels, sl(train), sl(bands), stats=stats)
    if names0 != names1:
        return False
    return bool(np.allclose(block0[:, :k + 1, :], block1[:, :k + 1, :], equal_nan=True, atol=1e-6))


def engineered_channel_catalog() -> list:
    """Frozen catalogue of engineered-block names (a channel is emitted only when its inputs exist)."""
    rows = []
    for c in CORE_CHANNELS:
        rows.append((f"{c}_slope_4bin", "OLS slope of observed (not carried) z over bins k-3..k, need 2 points"))
        rows.append((f"{c}_rollsd_4bin_past", "SD of observed z over strictly past bins k-4..k-1, need 3 points"))
        if c in LOW_WORSE:
            rows.append((f"{c}_time_below_2sd_24h", "share of observed bins in k-3..k below the train age-band 2 SD line"))
        elif c in HIGH_WORSE:
            rows.append((f"{c}_time_above_2sd_24h", "share of observed bins in k-3..k above the train age-band 2 SD line"))
        else:
            rows.append((f"{c}_time_above_2sd_24h", "share of observed bins in k-3..k above the train age-band 2 SD line"))
            rows.append((f"{c}_time_below_2sd_24h", "share of observed bins in k-3..k below the train age-band 2 SD line"))
        rows.append((f"{c}_worst_24h", "direction-aware worst observed z in bins k-3..k"))
    for c in HOURS_SINCE_CHANNELS:
        rows.append((f"{c}_hours_since_last", f"bins since last true observation, capped at {SINCE_CAP}"))
    for organ in ORGAN_MEMBERS:
        rows.append((f"organ_flag_{organ}", "any member channel exceeds the age-band 2 SD line in this bin"))
        rows.append((f"organ_flag24_{organ}", "any member exceeds the age-band 2 SD line in bins k-3..k"))
        rows.append((f"organ_meanz_{organ}", "mean observed z of member channels in this bin"))
        rows.append((f"organ_worstz_{organ}", "signed deviation of the farthest member from its age-band mean"))
    rows.append(("aki_cr_stage_zsurrogate",
                 f"z-space Cr stage 0-3 from min-prefix baseline; cuts {AKI_Z_CUTS}"))
    rows.append(("cr_delta_baseline_z", "current available creatinine z minus the min observed z in bins 0..k"))
    rows.append(("kdigo_uo_stage_value", "Stage 08 kdigo_uo_stage when the channel is present"))
    rows.append(("kdigo_combined_stage", "max of the Cr z-surrogate and an integer UO stage when recoverable"))
    for cname, (chname, d) in PSOFA_COMPONENTS.items():
        rows.append((f"psofa_comp_{cname}", f"z-space 0-4 points on the 24 h worst {chname} ({d} is worse)"))
    rows.append(("psofa_mod_total", "sum of the five pSOFA-like components"))
    rows.append(("fluid_net_cum", "cumulative observed (fluid_intake z - urine_output z) in the prefix"))
    return rows



# ------------------------------------------------------------------------------------------------- config / I/O
@dataclass
class Config:
    project_root: Path
    in_dir: Path
    run1_dir: Path
    run2_dir: Path
    run2_script: Path
    r4_script: Path | None
    canonical: Path
    harmonization: Path
    split_file: Path
    raw_dir: Path
    work_dir: Path
    out_dir: Path
    test_log: Path
    cache_dir: Path
    expect_manifest_sha256: str | None
    expect_run1_manifest_sha256: str | None
    expect_run2_manifest_sha256: str | None
    precheck_only: bool
    n_boot: int = DEFAULT_N_BOOT
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    batch: int = DEFAULT_BATCH
    n_seeds: int = DEFAULT_N_SEEDS
    landmarks: tuple = CORE_LANDMARKS
    second_family: str | None = None
    amp: bool = False
    require_cuda: bool = False
    allow_concurrent: bool = False
    max_seconds: float = 0.0
    heartbeat_seconds: float = 30.0
    run2_script_sha: str | None = None
    r4_script_sha: str | None = None
    env_info: dict | None = None
    repro_tol_prob: float = 1e-4


def r2_config(cfg: Config):
    return R2.Config(project_root=cfg.project_root, in_dir=cfg.in_dir, run1_dir=cfg.run1_dir,
                     canonical=cfg.canonical, harmonization=cfg.harmonization, split_file=cfg.split_file,
                     raw_dir=cfg.raw_dir, out_dir=cfg.out_dir, expect_manifest_sha256=cfg.expect_manifest_sha256,
                     expect_run1_manifest_sha256=cfg.expect_run1_manifest_sha256, precheck_only=True,
                     max_seconds=cfg.max_seconds, heartbeat_seconds=cfg.heartbeat_seconds)


def read_test_log(p: Path) -> list:
    if not p.is_file():
        return []
    o = json.loads(p.read_text(encoding="utf-8"))
    return list(o.get("attempts") or [])


def write_test_log(p: Path, attempts: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"attempts": attempts}, indent=2), encoding="utf-8")


def parse_landmarks(s: str | None) -> tuple:
    if not s:
        return CORE_LANDMARKS
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v not in CORE_LANDMARKS:
            raise InputError(f"landmark {v} is not one of the seven core landmarks")
        out.append(v)
    if not out:
        raise InputError("no landmarks requested")
    return tuple(dict.fromkeys(out))


def other_desktop_pending(project_root: Path, work: Path) -> list:
    root = desktop_root(project_root)
    if not root.is_dir():
        return []
    hits = []
    for p in root.glob("*/*.pending-*"):
        try:
            p.resolve().relative_to(work.resolve())
            continue  # our own pending names are handled by Run 2 precheck
        except ValueError:
            hits.append(p)
    for p in root.glob("*.pending-*"):
        hits.append(p)
    return hits


def precheck(cfg: Config) -> dict:
    """Run 2's hash gates, the pinned Run 2 manifest and files, empty published folder, no own pending
folder, no published test-use attempt, and no other DESKTOP_FULL_PLAN pending folders. A prior 'failed (nothing published)' or 'started' attempt does not block a rerun."""
    for q in (cfg.run2_dir, cfg.out_dir, cfg.split_file, cfg.raw_dir, cfg.run2_script, cfg.work_dir, cfg.test_log):
        if R2.NEVER_READ_TOKEN in str(q):
            raise InputError(f"a path containing {R2.NEVER_READ_TOKEN} is never read or written")
    att_early = read_test_log(cfg.test_log)
    if any(a.get("status") == "published" for a in att_early):
        raise InputError("the test-use log records a published test scoring: test rows are scored once only")
    if not cfg.allow_concurrent:
        other = other_desktop_pending(cfg.project_root, cfg.work_dir)
        if other:
            raise InputError("a pending folder from another desktop run exists under DESKTOP_FULL_PLAN; "
                             "Tier 3 must not run concurrently with Tier 1 or Tier 2")
    pre = R2.precheck(r2_config(cfg))
    hashes = pre["hashes"]
    m2p = cfg.run2_dir / f"{RUN2_NAME}_manifest.json"
    if not m2p.is_file():
        raise InputError(f"Run 2 manifest not found: {m2p.name}")
    if cfg.expect_run2_manifest_sha256 is None:
        raise InputError("--expect-run2-manifest-sha256 is required")
    s2 = sha256_file(m2p)
    if s2 != cfg.expect_run2_manifest_sha256:
        raise InputError("stage11_12_17_r2_manifest.json SHA-256 differs from --expect-run2-manifest-sha256")
    if PINS3.get("run2_manifest") and s2 != PINS3["run2_manifest"]:
        raise InputError("stage11_12_17_r2_manifest.json SHA-256 differs from the Run 2 pin")
    m2 = json.loads(m2p.read_text(encoding="utf-8"))
    if m2.get("script") != PINS3["run2_script"] or m2.get("script_sha256") != PINS3["run2_script_sha256"]:
        raise InputError("Run 2 manifest does not come from the pinned Run 2 v1.0.0 script")
    if m2.get("release_gate") != "ok":
        raise InputError("Run 2 release gate is not ok")
    a2 = m2.get("acceptance_checks") or {}
    if not a2 or any(v != "pass" for v in a2.values()):
        raise InputError("Run 2 acceptance checks not all pass")
    ih = m2.get("input_hashes") or {}
    for k in ("stage06_08_manifest.json", "stage09_10_r1_manifest.json", "canonical", "harmonization", "split",
              "ICUSTAYS", "ADMISSIONS"):
        if ih.get(k) != hashes.get(k):
            raise InputError(f"Run 2 was built from a different input ({k})")
    f2 = m2.get("files", {})
    need = set(RUN2_INTERNAL) | set(RUN2_TEAM_USED) | {RUN2_AUDIT}
    if not need <= set(f2):
        raise InputError("Run 2 manifest lacks required files")
    say(f"hashing Run 2 files: {len(f2)}")
    for name, rec in sorted(f2.items()):
        q = cfg.run2_dir / name
        if not q.is_file() or sha256_file(q) != rec.get("sha256"):
            raise InputError(f"Run 2 file {name} is missing or its SHA-256 differs from the Run 2 manifest")
    hashes["stage11_12_17_r2_manifest.json"] = s2
    for n in RUN2_INTERNAL:
        hashes[n] = f2[n]["sha256"]
    hashes["run2_script"] = cfg.run2_script_sha
    if cfg.second_family:
        if cfg.r4_script is None or not Path(cfg.r4_script).is_file():
            raise InputError("--second-family requires --r4-script")
        hashes["r4_script"] = cfg.r4_script_sha
    att = read_test_log(cfg.test_log)
    if any(a.get("status") == "published" for a in att):
        raise InputError("the test-use log records a published test scoring: test rows are scored once only")
    return {"hashes": hashes, "run2_manifest": m2, "prior_attempts": len(att)}


def load_all(cfg: Config, checks: dict) -> dict:
    """Stage 08 levels on every 24 h-eligible split (test stays behind the TestGuard), Run 1 bases,
    Layer C index, Run 2 labels, subjects and frozen artefacts."""
    d = cfg.in_dir
    lv = OrderedDict()
    eid0 = split0 = elig24 = None
    for k in R2.LEVELS:
        z = R2.read_npz(d / f"stage08_{k}.npz", ("encounter_id", "split", "eligible_24h", "X_z", "columns"))
        eid, split = z["encounter_id"].astype(str), z["split"].astype(str)
        if eid0 is None:
            eid0, split0, elig24 = eid, split, z["eligible_24h"].astype(bool)
        elif not (np.array_equal(eid, eid0) and np.array_equal(split, split0)):
            raise InputError(f"level {k}: encounter order or split differs from level a0")
        X = z["X_z"][elig24].astype(np.float64)
        if not np.isfinite(X).all():
            raise InputError(f"level {k}: non-finite values in X_z")
        lv[k] = dict(X=X, cols=[str(c) for c in z["columns"]])
    if not set(np.unique(split0)) <= {"train", "validation", "test"}:
        raise InputError("unexpected split labels")
    rows = OrderedDict(eid=eid0[elig24], split=split0[elig24])
    emap = {e: i for i, e in enumerate(rows["eid"])}
    lab = R2.read_npz(cfg.run2_dir / RUN2_INTERNAL[0],
                      ["encounter_id", "split", "died", "outcome2", "ordinal8"] +
                      [f"outcome1_L{L}" for L in R2.LANDMARKS])
    checks["r5_labels_match_stage08_order_and_split"] = bool(
        np.array_equal(lab["encounter_id"].astype(str), eid0) and np.array_equal(lab["split"].astype(str), split0))
    if not checks["r5_labels_match_stage08_order_and_split"]:
        raise InputError("Run 2 labels: encounter order or split differs from Stage 08")
    can = R2.read_csv_str(cfg.canonical, usecols=["encounter_id", "SUBJECT_ID"])
    can["encounter_id"] = can["encounter_id"].str.strip()
    if can["encounter_id"].duplicated().any() or set(can["encounter_id"]) != set(eid0):
        raise InputError("the Stage 08 encounters differ from the canonical encounter set (count withheld)")
    subj_all = can.set_index("encounter_id").loc[eid0]["SUBJECT_ID"].map(R2.norm_id).to_numpy().astype(str)
    spl = R2.read_csv_str(cfg.split_file, usecols=["SUBJECT_ID", "split"])
    smap = dict(zip(spl["SUBJECT_ID"].map(R2.norm_id),
                    spl["split"].map(lambda v: R2.SPLIT_LABELS.get(str(v).strip().lower(), "?"))))
    checks["r5_split_matches_split_file"] = bool(np.array_equal(np.array([smap.get(s, "?") for s in subj_all]),
                                                                split0))
    if not checks["r5_split_matches_split_file"]:
        raise InputError("Stage 08 split differs from the persisted subject split (count withheld)")
    rows["subj"] = subj_all[elig24]
    rows["died"] = lab["died"].astype(bool)[elig24]
    rows["O1"] = OrderedDict((L, lab[f"outcome1_L{L}"].astype(np.int64)[elig24]) for L in R2.LANDMARKS)
    rows["O2"], rows["O8"] = lab["outcome2"].astype(np.int64)[elig24], lab["ordinal8"].astype(np.int64)[elig24]
    # age from Layer A (a0), train-aligned with the 24 h eligible rows
    a0_cols = lv["a0"]["cols"]
    a0X = lv["a0"]["X"]
    def _col(name, default):
        if name in a0_cols:
            return a0X[:, a0_cols.index(name)]
        return np.full(a0X.shape[0], default)
    rows["age_months"] = _col("a0_age_months", np.nan)
    rows["age_invalid"] = _col("a0_age_invalid", 1.0) > 0.5
    rows["bands"] = age_band_codes(rows["age_months"], rows["age_invalid"])
    lc = OrderedDict()
    elig_all = OrderedDict()
    ok_el = True
    for L in R2.LANDMARKS:
        z = R2.read_npz(d / f"stage08_layer_c_L{L}.npz", ("encounter_id", "split", "channels"))
        e, s = z["encounter_id"].astype(str), z["split"].astype(str)
        idx = np.array([emap.get(x, -1) for x in e])
        if (idx < 0).any() or not np.array_equal(rows["split"][idx], s):
            raise InputError(f"Layer C L{L}: encounters or split do not match the 24 h tabular rows")
        lc[L] = dict(idx=idx, mask=np.ones(len(e), dtype=bool), channels=[str(c) for c in z["channels"]],
                     path=d / f"stage08_layer_c_L{L}.npz")
        el = np.isin(eid0, e)
        elig_all[L] = el
        ok_el &= bool(np.array_equal(el, lab[f"outcome1_L{L}"] >= 0))
    checks["r5_eligible_sets_match_labels"] = ok_el
    bz = np.load(cfg.run1_dir / "stage10_matched_width_bases_INTERNAL.npz", allow_pickle=False)
    with bz:
        bases = {k: np.array(bz[k]) for k in bz.files}
    s12 = load_pickle(cfg.run2_dir / RUN2_INTERNAL[2])
    tab = load_pickle(cfg.run2_dir / RUN2_INTERNAL[4])
    torch = R2._torch()
    states = torch.load(io.BytesIO((cfg.run2_dir / RUN2_INTERNAL[3]).read_bytes()), weights_only=True)
    with np.load(cfg.run2_dir / RUN2_INTERNAL[1], allow_pickle=False) as z:
        vpred = {k: np.array(z[k]) for k in z.files}
    a2 = json.loads((cfg.run2_dir / RUN2_AUDIT).read_text(encoding="utf-8"))
    r2cfg = a2.get("config") or {}
    if not isinstance(r2cfg.get("gru_max_epochs"), int) or r2cfg.get("seed") != R2.SEED:
        raise InputError("Run 2 audit lacks its GRU configuration (gru_max_epochs, seed)")
    return dict(r2cfg=r2cfg, levels=lv, rows=rows, eid_all=eid0, split_all=split0, elig24_all=elig24, subj_all=subj_all,
                lab_all=lab, elig_all=elig_all, lc=lc, bases=bases, s12=s12, tab=tab, states=states, vpred=vpred)


class TestGuard:
    """The only path to test rows. Phase 'verify' refuses any test access; phase 'score' allows each named
    scoring exactly once. A fit on any test row is refused."""

    def __init__(self, split: np.ndarray):
        self.split = split
        self.phase = "verify"
        self.scored = OrderedDict()
        self.fits = []
        self.first_test_access = None

    def te(self) -> np.ndarray:
        if self.phase != "score":
            raise BuildError("test rows requested outside the scoring phase")
        if self.first_test_access is None:
            self.first_test_access = utc_now()
        return np.flatnonzero(self.split == "test")

    def fit_rows(self, idx: np.ndarray, what: str) -> np.ndarray:
        sp = set(np.unique(self.split[idx]).tolist())
        self.fits.append((what, self.phase, tuple(sorted(sp))))
        if "test" in sp:
            raise BuildError("a fit was given test rows")
        return idx

    def score(self, key: str, fn):
        if self.phase != "score":
            raise BuildError("test scoring outside the scoring phase")
        if key in self.scored:
            raise BuildError("a test scoring was requested twice")
        out = fn()
        self.scored[key] = True
        return out


def load_layer_c_arrays(path: Path) -> dict:
    z = R2.read_npz(path, ("X_filled_z", "X_raw_z", "missing", "carried_forward", "channels"))
    return dict(X_filled_z=np.asarray(z["X_filled_z"]), X_raw_z=np.asarray(z["X_raw_z"]),
                missing=np.asarray(z["missing"]), carried=np.asarray(z["carried_forward"]),
                channels=[str(c) for c in z["channels"]])


def keep_channels(D: dict, L: int, fallback: list) -> list:
    key = f"C_L{L}__channels"
    if key in D["bases"]:
        return [str(c) for c in D["bases"][key]]
    return list(fallback)


# ------------------------------------------------------------------------------------------------- models
def official_seed(family: str, level: str, L: int, seed0: int | None = None) -> int:
    return (R2.SEED if seed0 is None else int(seed0)) + int(L) + 1000 * LEVEL_CODE[level] + 10000 * FAMILY_CODE[family]


def gru_fit_predict(seq_tr, st_tr, off_tr, y_tr, g_tr, seq_va, st_va, off_va, K, hidden, seed,
                    gamma0=None, max_epochs=40, patience=5, batch=256, lr=2e-3, wd=1e-4, amp=False):
    """CORAL BCE, Adam, 10% subject holdout early stopping. CUDA when available; AMP only if requested."""
    torch = R2._torch()
    seed_torch(torch, seed)
    dev = torch_device()
    use_amp = bool(amp) and dev.type == "cuda"
    rng = np.random.default_rng(seed)
    Net = R2.make_gru_class()
    us = np.unique(g_tr)
    hold = set(rng.choice(us, size=max(1, int(round(0.1 * len(us)))), replace=False).tolist())
    hm = np.array([g in hold for g in g_tr])
    fit_i, hold_i = np.flatnonzero(~hm), np.flatnonzero(hm)
    if len(fit_i) == 0:
        fit_i = np.arange(len(y_tr))
        hold_i = fit_i[:1]
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    xs, xh = T(seq_tr), T(seq_va)
    st = T(st_tr) if st_tr is not None else torch.zeros((len(y_tr), 0), device=dev)
    sh = T(st_va) if st_va is not None else torch.zeros((seq_va.shape[0], 0), device=dev)
    ot, oh = T(off_tr), T(off_va)
    Y = torch.tensor((y_tr[:, None] > np.arange(K - 1)[None, :]).astype(np.float32), device=dev)
    net = Net(seq_tr.shape[2], int(st.shape[1]), hidden, K, emb=EMB, gamma0=gamma0).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    bce = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_amp)

    def hold_loss():
        net.eval()
        with torch.no_grad():
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                z = net(xs[hold_i], st[hold_i], ot[hold_i])
                return float(bce(-z, Y[hold_i]))

    best, best_state, bad, ep = math.inf, None, 0, 0
    for ep in range(1, max_epochs + 1):
        net.train()
        perm = rng.permutation(fit_i)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = bce(-net(xs[b], st[b], ot[b]), Y[b])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        hl = hold_loss()
        if hl < best - 1e-5:
            best, bad = hl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        check_time()
    net.load_state_dict(best_state)
    net.to(dev)
    net.eval()
    with torch.no_grad():
        with torch.autocast(device_type=dev.type, enabled=use_amp):
            F = torch.sigmoid(net(xh, sh, oh)).float().cpu().numpy().astype(np.float64)
    F = np.maximum.accumulate(F, axis=1)
    state = dict(state_dict=best_state, family="gru", hp=dict(hidden=int(hidden)), d_seq=int(seq_tr.shape[2]),
                 d_static=int(st.shape[1]), hidden=int(hidden), K=K, seed=int(seed))
    return state, R2.cum_to_probs(F), ep


def gru_predict(state: dict, seq: np.ndarray, st, off: np.ndarray, amp: bool = False) -> np.ndarray:
    torch = R2._torch()
    Net = R2.make_gru_class()
    hidden = int(state.get("hidden", state.get("hp", {}).get("hidden", 32)))
    net = Net(state["d_seq"], state["d_static"], hidden, state["K"])
    net.load_state_dict(state["state_dict"])
    dev = torch_device()
    use_amp = bool(amp) and dev.type == "cuda"
    net.to(dev)
    net.eval()
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    s = T(st) if st is not None and state["d_static"] > 0 else torch.zeros((seq.shape[0], 0), device=dev)
    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type=dev.type, enabled=True):
                F = torch.sigmoid(net(T(seq), s, T(off))).float().cpu().numpy().astype(np.float64)
        else:
            # match the Run 2 / Tier 2 frozen forward pass (no autocast wrapper)
            F = torch.sigmoid(net(T(seq), s, T(off))).cpu().numpy().astype(np.float64)
    F = np.maximum.accumulate(F, axis=1)
    return R2.cum_to_probs(F)


def second_fit_predict(family, hp, *args, **kw):
    if R4 is None:
        raise BuildError("the Tier 2 module was not loaded")
    return R4.seq_fit_predict(family, hp, *args, **kw)


def second_predict(state, seq, st, off, amp=False):
    return R4.seq_predict(state, seq, st, off, amp=amp)


def winner_X(D: dict, mats: OrderedDict) -> tuple:
    w = D["s12"]["winner"]
    kind, k = (w["rep"], None) if w["rep"] == "full" else (w["rep"].split("_")[0], int(w["rep"].split("_")[1]))
    return w, R2.rep_X(mats[w["matrix"]], kind, k)


def holm(p, alpha=ALPHA) -> tuple:
    p = np.asarray(p, dtype=float)
    m = int(np.isfinite(p).sum())
    adj = np.full(p.shape, np.nan)
    o = [i for i in np.argsort(np.where(np.isfinite(p), p, np.inf), kind="mergesort") if np.isfinite(p[i])]
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj, np.where(np.isfinite(adj), adj <= alpha, False)


def interval(v: np.ndarray) -> tuple:
    v = v[np.isfinite(v)]
    return (R2.pct(v, 2.5), R2.pct(v, 97.5)) if v.size else (float("nan"), float("nan"))


def p_two_sided(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def tost_p(d: np.ndarray, delta: float) -> float:
    """Larger one-sided TOST p, (count + 1) / (B + 1)."""
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    B = d.size
    lo = float(np.sum(d <= -delta))
    hi = float(np.sum(d >= delta))
    return float(max(lo + 1, hi + 1) / (B + 1))


def suppress_small(row: dict, y: np.ndarray, K: int, died: np.ndarray | None) -> dict:
    for j in range(K - 1):
        npos = int((y > j).sum())
        if min(npos, len(y) - npos) < R2.SUPPRESS_BELOW:
            for nm in ("auroc", "auprc", "ece"):
                if f"{nm}_t{j}" in row:
                    row[f"{nm}_t{j}"] = R2.SUPP
    if died is not None:
        nd = int(np.asarray(died, dtype=bool).sum())
        if min(nd, len(died) - nd) < R2.SUPPRESS_BELOW:
            row["mortality_auroc"] = R2.SUPP
    return row


def boot_macro_auc(W, P, y, K) -> np.ndarray:
    # reuse r4 if present; otherwise a conservative nan array
    if R4 is not None and hasattr(R4, "boot_macro_auc"):
        return R4.boot_macro_auc(W, P, y, K)
    return np.full(len(W), np.nan)


def score_split(P, y, K, died, W, extra: dict) -> tuple:
    row, kb = R2.score_row(P, y, K, died, W, extra)
    ka = boot_macro_auc(W, P, y, K)
    if np.isfinite(ka).any():
        row["auroc_macro_p2_5"], row["auroc_macro_p97_5"] = interval(ka)
    return suppress_small(row, y, K, died), kb, ka



# ------------------------------------------------------------------------------------------------- level build / tune
def build_landmark_levels(D, L, checks) -> dict:
    """Load the Stage 08 tensor at L and build the three level inputs. Train-only stats; no look-ahead."""
    rec = D["lc"][L]
    arr = load_layer_c_arrays(rec["path"])
    ch, idx = arr["channels"], rec["idx"]
    keep = keep_channels(D, L, ch)
    missing = [c for c in keep if c not in ch]
    if missing:
        raise InputError(f"Layer C L{L}: a Run 1 channel is missing from the Stage 08 tensor")
    split = D["rows"]["split"][idx]
    train = split == "train"
    bands = D["rows"]["bands"][idx]
    ok_la = lookahead_is_clean(arr["X_raw_z"], arr["missing"], arr["carried"], ch, train, bands)
    checks[f"r5_L{L}_engineered_no_lookahead"] = ok_la
    if not ok_la:
        raise BuildError(f"Layer C L{L}: engineered features changed when a later bin was perturbed")
    # train-only check: refitting stats on train+val must not be how we score; we just record that train is used
    levels = OrderedDict()
    meta = OrderedDict()
    for lev in LEVELS:
        S, names, stats = build_level_tensor(lev, arr["X_raw_z"], arr["X_filled_z"], arr["missing"], arr["carried"],
                                             ch, keep, train, bands)
        if not np.isfinite(S).all():
            raise BuildError(f"Layer C L{L} level {lev}: non-finite model inputs after packing")
        levels[lev] = S
        meta[lev] = dict(names=names, stats=stats, d=int(S.shape[2]), n_engineered=(len(names) - 1 - 3 * len(keep)))
    # normalized must match Run 2's packed input (channel order of keep)
    S_ref = R2.c_inputs(D, L, "all", None)
    same = bool(S_ref.shape == levels["normalized"].shape and np.allclose(S_ref, levels["normalized"], atol=1e-6))
    checks[f"r5_L{L}_normalized_matches_run2"] = same
    if not same:
        raise BuildError(f"Layer C L{L}: the normalized level does not match the Run 2 GRU input")
    return dict(levels=levels, meta=meta, keep=keep, channels=ch, idx=idx)


def fit_one(family, hp, S, Xab, off_tr, off_va, y, g, jtr, jva, K, seed, theta, cfg):
    if family == "gru":
        return gru_fit_predict(S[jtr], Xab[jtr], off_tr, y[jtr], g[jtr], S[jva], Xab[jva], off_va, K,
                               int(hp["hidden"]), seed, gamma0=theta, max_epochs=cfg.max_epochs,
                               patience=cfg.patience, batch=cfg.batch, amp=cfg.amp)
    return second_fit_predict(family, hp, S[jtr], Xab[jtr], off_tr, y[jtr], g[jtr], S[jva], Xab[jva], off_va, K,
                              seed, gamma0=theta, max_epochs=cfg.max_epochs, patience=cfg.patience,
                              batch=cfg.batch, amp=cfg.amp)


def predict_one(state, S, Xab, off, amp):
    if state.get("family", "gru") == "gru":
        return gru_predict(state, S, Xab, off, amp=amp)
    return second_predict(state, S, Xab, off, amp=amp)


def hp_label(family: str, hp: dict) -> str:
    if family == "gru":
        return f"hid{int(hp['hidden'])}"
    if R4 is not None:
        return R4.hp_label(hp)
    return str(hp)


def grid_for(family: str) -> list:
    if family == "gru":
        return [dict(hidden=h) for h in GRU_HIDDEN_GRID]
    if R4 is None:
        raise BuildError("the Tier 2 module was not loaded")
    return R4.default_grid(family)


def contrast_pair(kb1, kb0, q1, q0, dB, delta, delta_C) -> OrderedDict:
    d = dB  # already kb1 - kb0 if provided; else compute
    lo, hi = interval(d)
    p2 = p_two_sided(d)
    p_eq = tost_p(d, delta)
    # one-sided p that the gain exceeds delta_C (H0: mean <= delta_C)
    d_ok = d[np.isfinite(d)]
    p_gain = float((np.sum(d_ok <= delta_C) + 1) / (d_ok.size + 1)) if d_ok.size else float("nan")
    if lo > -delta and hi < delta:
        v = "equivalent"
    elif lo > 0:
        v = "higher"
    elif hi < 0:
        v = "lower"
    else:
        v = "inconclusive"
    return OrderedDict(delta_qwk=float(q1 - q0), delta_p2_5=lo, delta_p97_5=hi, p_two_sided=p2,
                       p_tost=p_eq, p_gain_gt_delta_C=p_gain, equivalent=bool(lo > -delta and hi < delta),
                       gain_gt_delta_C=bool(lo > delta_C), verdict=v)


def tune_landmark(D, mats, G, Vw, Xw, L, built, cfg, checks) -> dict:
    R, tab = D["rows"], D["tab"]
    sp = R["split"]
    idx = built["idx"]
    itr = np.flatnonzero(sp[idx] == "train")
    iva = np.flatnonzero(sp[idx] == "validation")
    yy = R["O1"][L][idx]
    K = 9
    ok = yy >= 0
    jtr, jva = itr[ok[itr]], iva[ok[iva]]
    G.fit_rows(idx[jtr], f"offset_L{L}")
    g, died = R["subj"][idx], R["died"][idx]
    A_name = Vw["matrix"].split("+")[0]
    XA, Xab = mats[A_name]["X"][idx], Xw[idx]
    S_norm = built["levels"]["normalized"]
    off_tr, off_va, th, mA = R2.crossfit_offset(XA[jtr], yy[jtr], g[jtr], XA[jva], K, ("r5", L, "outcome1"))
    W = R2.boot_weights(g[jva], cfg.n_boot, R2.rng_for("boot", "r5", L))
    yv, dv = yy[jva], died[jva]
    P_ab = tab[f"L{L}_outcome1_AB"].predict_proba(Xab[jva])
    P_gru = gru_predict(D["states"][f"L{L}_outcome1_ABC"], S_norm[jva], Xab[jva], mA.decision(XA[jva]))
    row_ab, kb_ab, ka_ab = score_split(P_ab, yv, K, dv, W,
                                       dict(outcome="outcome1", landmark=L, core=True, family="AB",
                                            model="AB", level="", config="frozen_run2", split="validation"))
    row_gru, kb_gru, ka_gru = score_split(P_gru, yv, K, dv, W,
                                          dict(outcome="outcome1", landmark=L, core=True, family="GRU",
                                               model="GRU", level="normalized", config="frozen_run2",
                                               split="validation"))
    frozen = dict(AB=(P_ab, row_ab, kb_ab, ka_ab), GRU=(P_gru, row_gru, kb_gru, ka_gru),
                  off_tr=off_tr, off_va=off_va, theta=th, mA=mA, Xab=Xab, XA=XA,
                  jtr=jtr, jva=jva, yy=yy, g=g, died=died, W=W, idx=idx, K=K)
    tune_rows, states, preds = [row_ab, row_gru], OrderedDict(), OrderedDict()
    selected = OrderedDict()
    families = ("gru",) + ((cfg.second_family,) if cfg.second_family else ())
    for lev in LEVELS:
        S = built["levels"][lev]
        best = None
        for fam in families:
            for hp in grid_for(fam):
                seed = official_seed(fam, lev, L)
                lab = f"{lev}_{fam}_{hp_label(fam, hp)}"
                G.fit_rows(idx[jtr], f"L{L}_{lab}")
                t0 = time.time()
                stt, P, ep = fit_one(fam, hp, S, Xab, off_tr, off_va, yy, g, jtr, jva, K, seed, th, cfg)
                elapsed = time.time() - t0
                row, kb, ka = score_split(P, yv, K, dv, W,
                                          dict(outcome="outcome1", landmark=L, core=True, family=fam, model=fam,
                                               level=lev, config=hp_label(fam, hp), hidden=hp.get("hidden", ""),
                                               epochs_trained=ep, split="validation"))
                row["fit_seconds"] = round(elapsed, 3)
                row["d_seq"] = int(S.shape[2])
                tune_rows.append(row)
                key = f"L{L}_{lab}"
                states[key] = stt
                preds[f"val_{key}"] = P.astype(np.float32)
                say(f"tune L{L} {lab}: val QWK {row['qwk']:.4f} epochs {ep} ({elapsed:.1f} s)")
                if best is None or row["qwk"] > best["qwk"]:
                    best = dict(hp=hp, family=fam, state=stt, P=P, row=row, kb=kb, ka=ka, epochs=ep,
                                key=key, qwk=row["qwk"], elapsed=elapsed, level=lev)
                check_time()
        selected[lev] = best
        preds[f"val_L{L}_{lev}_selected"] = best["P"].astype(np.float32)
    preds[f"val_L{L}__eid"] = R["eid"][idx][jva]
    preds[f"val_L{L}_AB"] = P_ab.astype(np.float32)
    preds[f"val_L{L}_GRU"] = P_gru.astype(np.float32)
    return dict(tune_rows=tune_rows, selected=selected, frozen=frozen, states=states, preds=preds,
                built=built)


PRIMARY_PAIRS = (("engineered", "normalized"), ("normalized", "raw"), ("engineered", "raw"))


def level_contrasts(L, pack, split_name, delta, delta_C) -> list:
    rows = []
    sel = pack["selected"]
    for a, b in PRIMARY_PAIRS:
        pa, pb = sel[a], sel[b]
        dB = pa["kb"] - pb["kb"]
        c = contrast_pair(pa["kb"], pb["kb"], pa["qwk"], pb["qwk"], dB, delta, delta_C)
        rows.append(OrderedDict(outcome="outcome1", landmark=L, core=True, contrast=f"{a}_minus_{b}",
                                split=split_name, family_a=pa["family"], family_b=pb["family"],
                                config_a=hp_label(pa["family"], pa["hp"]),
                                config_b=hp_label(pb["family"], pb["hp"]), **c))
    # also vs frozen A+B and frozen Run 2 GRU (normalized)
    for lev, best in sel.items():
        for ref_name in ("GRU", "AB"):
            _P0, r0, k0, _a0 = pack["frozen"][ref_name]
            dB = best["kb"] - k0
            c = contrast_pair(best["kb"], k0, best["qwk"], r0["qwk"], dB, delta, delta_C)
            rows.append(OrderedDict(outcome="outcome1", landmark=L, core=True,
                                    contrast=f"{lev}_minus_{ref_name}", split=split_name,
                                    family_a=best["family"], family_b=ref_name,
                                    config_a=hp_label(best["family"], best["hp"]), config_b="frozen_run2", **c))
    return rows


def apply_holm(rows: list, family_contrasts: tuple, split_name: str) -> list:
    """Holm over the named contrasts at every landmark in this split."""
    idx = [i for i, r in enumerate(rows) if r.get("split") == split_name and r.get("contrast") in family_contrasts]
    if not idx:
        return rows
    p = np.array([rows[i]["p_two_sided"] for i in idx], dtype=float)
    adj, rej = holm(p)
    for j, i in enumerate(idx):
        rows[i]["p_holm"] = float(adj[j])
        rows[i]["reject_holm"] = bool(rej[j])
    return rows


def score_test_landmark(D, G, L, pack, cfg) -> dict:
    R = D["rows"]
    fr = pack["frozen"]
    idx, Xab, XA, K = fr["idx"], fr["Xab"], fr["XA"], fr["K"]
    _ = G.te()
    jte_all = np.flatnonzero(R["split"][idx] == "test")
    yy = R["O1"][L][idx]
    jte = jte_all[(yy >= 0)[jte_all]]
    off_te = fr["mA"].decision(XA[jte])
    W = R2.boot_weights(R["subj"][idx][jte], cfg.n_boot, R2.rng_for("boot", "r5_te", L))
    y, died = yy[jte], R["died"][idx][jte]
    Pab = G.score(f"r5_L{L}_AB", lambda: D["tab"][f"L{L}_outcome1_AB"].predict_proba(Xab[jte]))
    S_norm = pack["built"]["levels"]["normalized"]
    Pgru = G.score(f"r5_L{L}_GRU",
                   lambda: gru_predict(D["states"][f"L{L}_outcome1_ABC"], S_norm[jte], Xab[jte], off_te, amp=cfg.amp))
    rows, preds, selected_te = [], OrderedDict(), OrderedDict()
    row, kb, ka = score_split(Pab, y, K, died, W,
                              dict(outcome="outcome1", landmark=L, core=True, family="AB", model="AB",
                                   level="", config="frozen_run2", split="test"))
    rows.append(row)
    row, kb_g, ka_g = score_split(Pgru, y, K, died, W,
                                  dict(outcome="outcome1", landmark=L, core=True, family="GRU", model="GRU",
                                       level="normalized", config="frozen_run2", split="test"))
    rows.append(row)
    frozen_te = dict(AB=(Pab, rows[0], kb, ka), GRU=(Pgru, rows[1], kb_g, ka_g))
    preds[f"te_L{L}__eid"] = R["eid"][idx][jte]
    preds[f"te_L{L}_AB"] = Pab.astype(np.float32)
    preds[f"te_L{L}_GRU"] = Pgru.astype(np.float32)
    for lev, best in pack["selected"].items():
        S = pack["built"]["levels"][lev]
        P = G.score(f"r5_L{L}_{lev}",
                    lambda S=S, st=best["state"]: predict_one(st, S[jte], Xab[jte], off_te, cfg.amp))
        row, kb, ka = score_split(P, y, K, died, W,
                                  dict(outcome="outcome1", landmark=L, core=True, family=best["family"],
                                       model=best["family"], level=lev,
                                       config=hp_label(best["family"], best["hp"]), split="test"))
        rows.append(row)
        preds[f"te_L{L}_{lev}_selected"] = P.astype(np.float32)
        selected_te[lev] = dict(P=P, row=row, kb=kb, ka=ka, qwk=row["qwk"], family=best["family"], hp=best["hp"])
    te_pack = dict(selected=selected_te, frozen=frozen_te)
    contrasts = level_contrasts(L, te_pack, "test", DELTA, DELTA_C)
    return dict(rows=rows, preds=preds, contrasts=contrasts)


# ------------------------------------------------------------------------------------------------- audit / gate / publish
_STRUCT_COL3 = re.compile(
    r"^(k|p|landmark|class|threshold|hidden|epochs_trained|n_layers|d_model|n_heads|n_seeds|seed_index|"
    r"fit_seconds|d_seq|n_boot|batch|patience|max_epochs)$")


def audit_num_keys() -> tuple:
    return tuple(getattr(R2, "AUDIT_NUM_KEYS", ())) + (
        "repro", "p_holm", "p_tost", "p_gain", "p_two", "alpha", "test_attempts", "epochs", "epsilon",
        "d_model", "n_layers", "n_heads", "n_seeds", "fit_seconds", "patience", "max_epochs", "seed_index",
        "qwk_sd", "d_seq", "delta", "batch", "heartbeat", "cfg_", "pe", "hidden",
    )


def script_text_ints(audit: dict | None) -> set:
    txt = " ".join(list(DECISIONS.values()) + list(DECISIONS) + list(DEVIATIONS) + [R2.SUPP, "10 or more"])
    if audit is not None:
        try:
            txt += " " + audit_md(audit)
        except Exception:  # noqa: BLE001
            pass
    return {int(t) for t in R2.id_tokens(txt) if t.isdigit()}


def gate_problems(pend: Path, team_files: list, forbidden: set, ids: set) -> list:
    """Run 2 release-gate rules on every TEAM file: collect-all, file and location only."""
    probs = []
    aj = pend / f"{OUT_NAME}_audit_TEAM.json"
    audit = json.loads(aj.read_text(encoding="utf-8")) if aj.exists() else None
    allowed = (R2._audit_ints(audit, set()) if audit is not None else set()) | script_text_ints(audit)
    if audit is not None:
        allowed |= {int(t) for t in re.findall(r"\d+", str(audit.get("generated_utc", "")))}
    for nm in team_files:
        q = pend / nm
        if not q.is_file():
            probs.append((nm, "file", "missing"))
            continue
        if nm.endswith(".png"):
            continue
        text = q.read_text(encoding="utf-8")
        if nm.endswith(".csv"):
            if not text.strip():
                continue
            df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
            cols = [str(c) for c in df.columns]
            if any(R2._FORBID_COL.search(c) for c in cols):
                probs.append((nm, "column names", "forbidden column name (count or identifier)"))
            if R2.team_text_problems(" ".join(cols), set(), ids):
                probs.append((nm, "column names", R2._ID_MSG))
            for c in cols:
                if _STRUCT_COL3.match(c):
                    continue
                size_hit = id_hit = False
                for cell in df[c].tolist():
                    cell = cell.strip()
                    if not cell or cell == R2._SUPP_CELL:
                        continue
                    if not size_hit and R2._size_hit(cell, forbidden):
                        size_hit = True
                    if not id_hit:
                        if R2._NUMERIC_CELL.match(cell):
                            id_hit = bool(re.fullmatch(r"[-+]?\d+", cell)) and cell.lstrip("+") in ids
                        else:
                            toks = R2.id_tokens(cell)
                            if c in ("item", "label", "family", "model", "config", "contrast", "split", "pe",
                                     "level", "family_a", "family_b", "config_a", "config_b", "verdict",
                                     "channel", "definition"):
                                toks = {t for t in toks if not t.isdigit()}
                            id_hit = bool(toks & ids)
                    if size_hit and id_hit:
                        break
                if size_hit:
                    probs.append((nm, f"column {c}", R2._SIZE_MSG))
                if id_hit:
                    probs.append((nm, f"column {c}", R2._ID_MSG))
        elif nm.endswith(".json"):
            try:
                obj = json.loads(text)
            except ValueError:
                probs.append((nm, "json", "unparseable"))
                continue
            for path, v in R2._json_leaves(obj):
                if isinstance(v, str):
                    for msg in R2.team_text_problems(v, forbidden - allowed, ids, allowed):
                        probs.append((nm, f"json key {path}", msg))
        else:
            for msg in R2.team_text_problems(text, forbidden - allowed, ids, allowed):
                probs.append((nm, "md text", msg))
    return probs


def release_gate(pend: Path, team_files: list, forbidden: set, ids: set) -> None:
    probs = gate_problems(pend, team_files, forbidden, ids)
    if probs:
        raise BuildError("release gate: " + str(len(probs)) + " problem(s): " +
                         "; ".join(f"{f} [{loc}]: {chk}" for f, loc, chk in probs))


def sensitive_numbers(D: dict) -> set:
    out = set()
    sp, subj, lab = D["split_all"], D["subj_all"], D["lab_all"]
    died = lab["died"].astype(bool)
    out.update({len(sp), len(np.unique(subj))})
    tot = {s: 0 for s in ("train", "validation", "test", "all")}
    for s in ("train", "validation", "test"):
        m = sp == s
        out.update({int(m.sum()), len(np.unique(subj[m]))})
    for L in R2.LANDMARKS:
        el = D["elig_all"][L]
        out.update({int(el.sum()), len(np.unique(subj[el])), int((el & (sp != "test")).sum())})
        tot["all"] += int(el.sum())
        for s in ("train", "validation", "test"):
            m = el & (sp == s)
            tot[s] += int(m.sum())
            out.update({int(m.sum()), len(np.unique(subj[m])), int((m & died).sum()), int((m & ~died).sum())})
            for lb, K in ((lab[f"outcome1_L{L}"], 9), (lab["outcome2"], 5), (lab["ordinal8"], 8)):
                v = lb[m & (lb >= 0)]
                out.add(int(v.size))
                bc = np.bincount(v, minlength=K)
                out.update(int(c) for c in bc)
                out.update(int(bc[j + 1:].sum()) for j in range(K - 1))
                out.update(int(bc[:j + 1].sum()) for j in range(K - 1))
    out.update(tot.values())
    return {x for x in out if x >= 100}


def _f(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    if x == R2.SUPP:
        return R2.SUPP
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


def make_audit(cfg, pre, checks, info, packs, val_ct, te_tab, te_ct, script_sha, n_written) -> dict:
    sel_rows = []
    for L, pack in packs.items():
        for lev, best in pack["selected"].items():
            sel_rows.append(OrderedDict(landmark=int(L), level=lev, family=best["family"],
                                        config=hp_label(best["family"], best["hp"]),
                                        val_qwk=_f(best["qwk"]), epochs=int(best["epochs"]),
                                        d_seq=int(pack["built"]["levels"][lev].shape[2])))
    return OrderedDict(
        script=SCRIPT_VERSION, spec=SPEC_DOC, plan=PLAN, contract=CONTRACT,
        seed=R2.SEED, alpha=ALPHA, delta=DELTA, delta_C=DELTA_C,
        landmarks=list(cfg.landmarks), levels=list(LEVELS),
        second_family=cfg.second_family, gru_hidden_grid=list(GRU_HIDDEN_GRID),
        max_epochs=cfg.max_epochs, patience=cfg.patience, batch=cfg.batch, n_boot=cfg.n_boot,
        n_seeds=cfg.n_seeds, amp=bool(cfg.amp),
        device="cuda" if (cfg.env_info or {}).get("cuda_available") else "cpu",
        cuda_available=bool((cfg.env_info or {}).get("cuda_available")),
        decisions=list(DECISIONS.keys()), deviations=deviations_for(pre),
        hashes=pre["hashes"], script_sha256=script_sha,
        test_attempts_before=int(pre.get("prior_attempts") or 0),
        repro_s17_max_abs_prob_diff=_f(info.get("repro_s17_max_abs_prob_diff")),
        selected=sel_rows,
        n_files=n_written, release_gate="ok",
        acceptance_checks={k: ("pass" if v else "fail") for k, v in checks.items()},
        labelled="post-primary sensitivity analysis",
    )


def audit_md(a: dict) -> str:
    L = [
        f"# Stage 15 Layer C levels (desktop Tier 3) — TEAM audit",
        "",
        f"- script: {a['script']}",
        f"- labelled: {a['labelled']}",
        f"- seed {a['seed']}; landmarks {a['landmarks']}; levels {a['levels']}",
        f"- second family: {a['second_family'] or 'none (GRU only)'}",
        f"- device: {a['device']}; cuda: {'yes' if a['cuda_available'] else 'no'}; AMP: {a['amp']}",
        f"- D18: delta={a['delta']}, delta_C={a['delta_C']}; Holm alpha={a['alpha']}",
        f"- max epochs {a['max_epochs']}, patience {a['patience']}, B_eval {a['n_boot']}",
        f"- earlier test-use attempts (failed, nothing published): {a['test_attempts_before']}",
        f"- frozen Run 2 GRU reproduction, max abs prob diff: {a['repro_s17_max_abs_prob_diff']}",
        "",
        "## Decisions and deviations",
        "",
    ]
    for k, v in DECISIONS.items():
        L.append(f"- {k}: {v}")
    L += ["", "Deviations from the plan / from earlier runs:", ""]
    for d in (a.get("deviations") or DEVIATIONS):
        L.append(f"- {d}")
    L += ["", "## Selected models (validation QWK)", "",
          "| landmark | level | family | config | val QWK | epochs | d_seq |",
          "|---|---|---|---|---|---|---|"]
    for r in a.get("selected") or []:
        L.append(f"| {r['landmark']} | {r['level']} | {r['family']} | {r['config']} | "
                 f"{r['val_qwk']} | {r['epochs']} | {r['d_seq']} |")
    L += ["", "## Acceptance checks", ""]
    for k, v in (a.get("acceptance_checks") or {}).items():
        L.append(f"- {k}: {v}")
    L += ["", "Release gate: ok. Detail tables: stage15_*_TEAM.csv; *_INTERNAL files are team-internal "
          "(never opened, pasted or shared).", ""]
    return "\n".join(L)


def figures(tune_df, val_ct) -> dict:
    figs = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return figs
    if len(val_ct):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        sub = val_ct[val_ct["contrast"].astype(str) == "engineered_minus_normalized"]
        if len(sub):
            xs = sub["landmark"].astype(int).to_numpy()
            ys = sub["delta_qwk"].astype(float).to_numpy()
            lo = sub["delta_p2_5"].astype(float).to_numpy()
            hi = sub["delta_p97_5"].astype(float).to_numpy()
            ax.axhline(0, color="k", lw=0.8)
            ax.axhline(DELTA_C, color="0.5", ls="--", lw=0.8, label=f"delta_C={DELTA_C}")
            ax.axhline(-DELTA, color="0.5", ls=":", lw=0.8)
            ax.axhline(DELTA, color="0.5", ls=":", lw=0.8, label=f"±delta={DELTA}")
            ax.errorbar(xs, ys, yerr=np.vstack([ys - lo, hi - ys]), fmt="o-", color="C0")
            ax.set_xlabel("landmark (h)")
            ax.set_ylabel("validation delta QWK")
            ax.set_title("Engineered minus normalized (validation)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            figs["stage15_engineered_minus_normalized_val_TEAM.png"] = fig
        plt.close("all")
    return figs


def publish(cfg, D, pre, T, internal, checks, info, packs, val_ct, te_tab, te_ct) -> dict:
    out = cfg.out_dir
    pend = out.parent / f"{out.name}.pending-{os.getpid()}"
    if pend.exists():
        raise BuildError(f"pending folder already exists: {pend.name}")
    pend.mkdir(parents=True)
    leftover_exc = None
    team = [
        f"{OUT_NAME}_audit_TEAM.md", f"{OUT_NAME}_audit_TEAM.json", f"{OUT_NAME}_manifest.json",
        "stage15_engineered_channels_TEAM.csv",
        "stage15_validation_scores_TEAM.csv", "stage15_validation_contrasts_TEAM.csv",
        "stage15_selected_TEAM.csv", "stage15_test_scores_TEAM.csv", "stage15_test_contrasts_TEAM.csv",
    ]
    try:
        cat = engineered_channel_catalog()
        pd.DataFrame(cat, columns=["channel", "definition"]).to_csv(pend / team[3], index=False)
        tune_df = T["tune"]
        sel_df = T["selected"]
        tune_df.to_csv(pend / "stage15_validation_scores_TEAM.csv", index=False)
        val_ct.to_csv(pend / "stage15_validation_contrasts_TEAM.csv", index=False)
        sel_df.to_csv(pend / "stage15_selected_TEAM.csv", index=False)
        te_tab.to_csv(pend / "stage15_test_scores_TEAM.csv", index=False)
        te_ct.to_csv(pend / "stage15_test_contrasts_TEAM.csv", index=False)
        for name, fig in figures(tune_df, val_ct).items():
            fig.savefig(pend / name, dpi=120)
            team.append(name)
        # INTERNAL
        pred_path = pend / "stage15_predictions_INTERNAL.npz"
        np.savez_compressed(pred_path, **internal["preds"])
        torch = R2._torch()
        torch.save(internal["states"], pend / "stage15_models_INTERNAL.pt")
        env_nm = f"{OUT_NAME}_environment_INTERNAL.json"
        env_payload = OrderedDict(
            script=SCRIPT_VERSION,
            generated_utc=utc_now(),
            environment=cfg.env_info or {},
            prior_test_use_attempts=read_test_log(cfg.test_log),
            prior_attempts=int((pre or {}).get("prior_attempts") or 0),
            note="GPU name, GPU memory, executable path and other host strings stay in this INTERNAL "
                 "record only; TEAM files carry device 'cuda' or 'cpu' and a coarse cuda yes/no flag.",
        )
        (pend / env_nm).write_text(json.dumps(env_payload, indent=2, default=str), encoding="utf-8")
        # audit
        script_sha = sha256_file(Path(__file__).resolve())
        n_written = 0  # filled after listing
        audit = make_audit(cfg, pre, checks, info, packs, val_ct, te_tab, te_ct, script_sha, 0)
        md = audit_md(audit)
        (pend / f"{OUT_NAME}_audit_TEAM.md").write_text(md, encoding="utf-8")
        (pend / f"{OUT_NAME}_audit_TEAM.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        files = {}
        for q in sorted(pend.iterdir(), key=lambda p: p.name):
            files[q.name] = {"sha256": sha256_file(q), "bytes": q.stat().st_size}
        audit["n_files"] = len(files) + 1
        man = OrderedDict(script=SCRIPT_VERSION, script_sha256=script_sha, spec=SPEC_DOC,
                          release_gate="ok",
                          acceptance_checks={k: ("pass" if v else "fail") for k, v in checks.items()},
                          input_hashes=pre["hashes"], files=files, labelled="post-primary sensitivity analysis")
        (pend / f"{OUT_NAME}_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        files[f"{OUT_NAME}_manifest.json"] = {"sha256": sha256_file(pend / f"{OUT_NAME}_manifest.json"),
                                              "bytes": (pend / f"{OUT_NAME}_manifest.json").stat().st_size}
        team_present = [n for n in team if (pend / n).is_file()]
        release_gate(pend, team_present, sensitive_numbers(D),
                     set(map(str, D["eid_all"])) | set(map(str, D["subj_all"])))
        if out.exists():
            if any(out.iterdir()):
                raise BuildError("output folder exists and is not empty")
            R2.retry(lambda: out.rmdir(), "remove empty output folder")
        R2.retry(lambda: os.replace(pend, out), "rename pending folder to the output folder")
        say(f"published {len(files)} files to {out.name}")
        return dict(audit=audit, audit_md=md, n_files=len(files))
    except Exception as e:  # noqa: BLE001
        leftover_exc = e
        try:
            e.leftover = str(R2.redact_path(pend, cfg.project_root))
        except Exception:  # noqa: BLE001
            e.leftover = pend.name
        raise
    finally:
        if leftover_exc is not None and pend.exists() and not out.exists():
            # leave pending for review; do not delete
            pass


def selected_table(packs: dict) -> pd.DataFrame:
    rows = []
    for L, pack in packs.items():
        for lev, best in pack["selected"].items():
            r = OrderedDict(outcome="outcome1", landmark=int(L), core=True, level=lev,
                            family=best["family"], config=hp_label(best["family"], best["hp"]),
                            val_qwk=best["qwk"], epochs_trained=best["epochs"],
                            d_seq=int(pack["built"]["levels"][lev].shape[2]))
            rows.append(r)
    return pd.DataFrame(rows)


def verify_frozen(D, mats, Vw, Xw, cfg, checks, info, landmarks) -> None:
    R, tab, st, vp = D["rows"], D["tab"], D["states"], D["vpred"]
    sp = R["split"]
    A_name = Vw["matrix"].split("+")[0]
    maxd, ok, missing = 0.0, True, False
    for L in landmarks:
        idx = D["lc"][L]["idx"]
        iva = np.flatnonzero(sp[idx] == "validation")
        S = R2.c_inputs(D, L, "all", None)
        XA, Xab = mats[A_name]["X"][idx], Xw[idx]
        yy = R["O1"][L][idx]
        jva = iva[(yy >= 0)[iva]]
        ek = f"s17_L{L}_outcome1__eid"
        ok &= ek in vp and bool(np.array_equal(vp[ek].astype(str), R["eid"][idx][jva]))
        mA = tab[f"L{L}_outcome1_A"]
        Pab = tab[f"L{L}_outcome1_AB"].predict_proba(Xab[jva])
        Pgru = gru_predict(st[f"L{L}_outcome1_ABC"], S[jva], Xab[jva], mA.decision(XA[jva]))
        for m, Pm in (("AB", Pab), ("ABC", Pgru)):
            ref = vp.get(f"s17_L{L}_outcome1_{m}")
            if ref is None or ref.shape != Pm.shape:
                missing = True
                continue
            maxd = max(maxd, float(np.abs(Pm - ref.astype(np.float64)).max()))
        check_time()
    checks["r5_frozen_gru_ab_reproduce_run2_validation"] = ok and not missing and maxd <= cfg.repro_tol_prob
    info["repro_s17_max_abs_prob_diff"] = maxd
    say("verify: frozen Run 2 GRU and A+B reproduce the Run 2 validation predictions")


def build(cfg: Config) -> dict:
    if cfg.max_seconds:
        R2._STATE["deadline"] = time.time() + cfg.max_seconds
    checks, info = OrderedDict(), OrderedDict()
    say("precheck: hashes")
    pre = precheck(cfg)
    if cfg.precheck_only:
        return {"mode": "precheck-only", "hashes": pre["hashes"]}
    write_desktop_readme(cfg.project_root)
    say("loading Stage 08 levels (24 h rows, all splits; test rows behind the guard), Layer C index, Run 1 bases, "
        "Run 2 labels and frozen models")
    D = load_all(cfg, checks)
    mats = R2.build_matrices(D, checks)
    Vw, Xw = winner_X(D, mats)
    G = TestGuard(D["rows"]["split"])
    say("verification (validation only): frozen Run 2 GRU and A+B")
    verify_frozen(D, mats, Vw, Xw, cfg, checks, info, cfg.landmarks)
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise BuildError("verification failed before any test row was read (test-use log unchanged): " +
                         "; ".join(failed))
    packs = OrderedDict()
    tune_rows, val_ct = [], []
    states, preds = OrderedDict(), OrderedDict()
    for L in cfg.landmarks:
        say(f"stage 15: build levels at {L} h")
        built = build_landmark_levels(D, L, checks)
        checks[f"r5_L{L}_prefix_bins"] = built["levels"]["normalized"].shape[1] == L // 6
        say("stage 15: tune GRU" + (f" and {cfg.second_family}" if cfg.second_family else "") + f" at {L} h")
        pack = tune_landmark(D, mats, G, Vw, Xw, L, built, cfg, checks)
        packs[L] = pack
        tune_rows.extend(pack["tune_rows"])
        val_ct.extend(level_contrasts(L, pack, "validation", DELTA, DELTA_C))
        states.update(pack["states"])
        for lev, best in pack["selected"].items():
            states[f"L{L}_{lev}_selected"] = best["state"]
        preds.update(pack["preds"])
    family = tuple(f"{a}_minus_{b}" for a, b in PRIMARY_PAIRS)
    val_ct = apply_holm(val_ct, family, "validation")
    checks["r5_no_test_rows_in_tuning"] = all("test" not in sp for _w, _ph, sp in G.fits) and G.phase == "verify"
    checks["r5_all_requested_landmarks_tuned"] = set(packs) == set(cfg.landmarks)
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise BuildError("acceptance checks failed before any test row was read (test-use log unchanged): " +
                         "; ".join(failed))
    attempts = read_test_log(cfg.test_log)
    script_sha = sha256_file(Path(__file__).resolve())
    attempts.append(dict(script=SCRIPT_VERSION, script_sha256=script_sha, started_utc=utc_now(), status="started"))
    write_test_log(cfg.test_log, attempts)
    G.phase = "score"
    try:
        te_rows, te_ct = [], []
        for L in cfg.landmarks:
            say(f"stage 15: test scoring at {L} h (once)")
            te = score_test_landmark(D, G, L, packs[L], cfg)
            te_rows.extend(te["rows"])
            te_ct.extend(te["contrasts"])
            preds.update(te["preds"])
        G.phase = "done"
        te_ct = apply_holm(te_ct, family, "test")
        checks["r5_no_fit_on_test_rows"] = all("test" not in sp for _w, _ph, sp in G.fits)
        checks["r5_fits_only_before_scoring"] = all(ph == "verify" for _w, ph, _sp in G.fits)
        n_expected = len(cfg.landmarks) * (2 + len(LEVELS))  # AB, frozen GRU, three levels
        checks["r5_each_test_scoring_once"] = len(G.scored) == n_expected
        failed = [k for k, v in checks.items() if not v]
        if failed:
            raise BuildError("acceptance checks failed: " + "; ".join(failed))
        T = dict(tune=pd.DataFrame(tune_rows), selected=selected_table(packs),
                 val_ct=pd.DataFrame(val_ct), te=pd.DataFrame(te_rows), te_ct=pd.DataFrame(te_ct))
        internal = dict(preds=preds, states=states)
        res = publish(cfg, D, pre, T, internal, checks, info, packs, T["val_ct"], T["te"], T["te_ct"])
    except BaseException:
        attempts[-1]["status"] = "failed (nothing published)"
        attempts[-1]["ended_utc"] = utc_now()
        write_test_log(cfg.test_log, attempts)
        raise
    attempts[-1]["status"] = "published"
    attempts[-1]["ended_utc"] = utc_now()
    attempts[-1]["manifest_sha256"] = sha256_file(cfg.out_dir / f"{OUT_NAME}_manifest.json")
    write_test_log(cfg.test_log, attempts)
    return dict(mode="full", audit=res["audit"], audit_md=res["audit_md"])


def describe_exception(e: BaseException, root, max_chain: int = 4) -> list:
    lines = [f"{type(e).__name__}: {e}"]
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    for ln in "".join(tb).splitlines():
        if root is not None:
            try:
                ln = ln.replace(str(Path(root).resolve()), "<project>")
            except Exception:  # noqa: BLE001
                pass
        lines.append(ln)
        if len(lines) > 80:
            break
    return lines


def build_config(argv=None) -> Config:
    ap = argparse.ArgumentParser(description=SCRIPT_VERSION)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--run1-dir", default=None)
    ap.add_argument("--run2-dir", default=None)
    ap.add_argument("--run2-script", default=None)
    ap.add_argument("--r4-script", default=None)
    ap.add_argument("--canonical", default=None)
    ap.add_argument("--harmonization", default=None)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--expect-stage06-08-manifest-sha256", default=None)
    ap.add_argument("--expect-run1-manifest-sha256", default=None)
    ap.add_argument("--expect-run2-manifest-sha256", default=None)
    ap.add_argument("--precheck-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    ap.add_argument("--landmarks", default=None, help="comma-separated core landmarks (default: all seven)")
    ap.add_argument("--second-family", default=None, help="optional lstm, lstm_attn, or transformer")
    ap.add_argument("--amp", action="store_true", help="mixed precision; ignored unless the device is CUDA")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="do not refuse when another DESKTOP_FULL_PLAN pending folder exists")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=30.0)
    a = ap.parse_args(argv)
    if a.n_seeds < 1:
        raise InputError("--n-seeds must be at least 1")
    if a.max_epochs < 1 or a.patience < 1 or a.n_boot < 1 or a.batch < 1:
        raise InputError("a run setting is out of range")
    second = (a.second_family or "").strip().lower() or None
    if second is not None and second not in SECOND_FAMILIES:
        raise InputError("--second-family must be lstm, lstm_attn, or transformer")
    root = Path(a.project_root)
    lad = root / "data" / "processed" / "ladder"
    here = Path(__file__).resolve().parent
    r2s = Path(a.run2_script) if a.run2_script else here.parent / "stage11_18_r2" / f"{RUN2_MODULE}.py"
    r4s = Path(a.r4_script) if a.r4_script else None
    work = work_root(root)
    cfg = Config(
        project_root=root,
        in_dir=Path(a.in_dir) if a.in_dir else lad / "stage06_08",
        run1_dir=Path(a.run1_dir) if a.run1_dir else lad / "stage09_10_r1",
        run2_dir=Path(a.run2_dir) if a.run2_dir else lad / RUN2_NAME,
        run2_script=r2s,
        r4_script=r4s,
        canonical=Path(a.canonical) if a.canonical else root / "data" / "interim" / "canonical_icu_encounters.csv",
        harmonization=Path(a.harmonization) if a.harmonization else root / "data" / "interim" /
        "encounter_harmonization_fields.csv",
        split_file=Path(a.split_file) if a.split_file else root / "data" / "interim" / "r2_inputs" /
        "subject_splits.csv",
        raw_dir=Path(a.raw_dir) if a.raw_dir else root / "data" / "interim" / "r2_inputs" / "raw",
        work_dir=work,
        out_dir=Path(a.out_dir) if a.out_dir else work / PUBLISHED_NAME,
        test_log=work / TEST_LOG_NAME,
        cache_dir=work / CACHE_NAME,
        expect_manifest_sha256=(a.expect_stage06_08_manifest_sha256 or "").strip().lower() or None,
        expect_run1_manifest_sha256=(a.expect_run1_manifest_sha256 or "").strip().lower() or None,
        expect_run2_manifest_sha256=(a.expect_run2_manifest_sha256 or "").strip().lower() or None,
        precheck_only=a.precheck_only, n_boot=a.n_boot, max_epochs=a.max_epochs, patience=a.patience,
        batch=a.batch, n_seeds=a.n_seeds, landmarks=parse_landmarks(a.landmarks),
        second_family=second, amp=bool(a.amp), require_cuda=bool(a.require_cuda),
        allow_concurrent=bool(a.allow_concurrent),
        max_seconds=a.max_seconds, heartbeat_seconds=a.heartbeat_seconds,
    )
    must_write_under_work(cfg.out_dir, root, "--out-dir")
    must_write_under_work(cfg.test_log, root, "test-use log")
    must_write_under_work(cfg.out_dir.parent / f"{cfg.out_dir.name}.pending-0", root, "pending folder")
    must_write_under_work(cfg.cache_dir, root, "cache folder")
    must_write_under_work(cfg.work_dir, root, "work root", allow_root=True)
    return cfg


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    cfg = prev = cache = None
    res = None
    try:
        cfg = build_config(argv)
        load_run2(cfg.run2_script, PINS3["run2_script_sha256"])
        cfg.run2_script_sha = sha256_file(cfg.run2_script)
        if cfg.second_family:
            if cfg.r4_script is None:
                raise InputError("--second-family requires --r4-script")
            load_r4(cfg.r4_script, PINS3["r4_script_sha256"])
            cfg.r4_script_sha = sha256_file(cfg.r4_script)
        cfg.env_info = check_environment(require_cuda=cfg.require_cuda)
        say("environment: " + ", ".join(f"{k}={v}" for k, v in cfg.env_info.items() if v is not None))
        if cfg.amp and not (cfg.env_info or {}).get("cuda_available"):
            say("AMP requested but CUDA is not available; mixed precision will stay off")
        write_desktop_readme(cfg.project_root)
        cache = cfg.cache_dir
        must_write_under_work(cache, cfg.project_root, "cache folder")
        prev = R2.setup_runtime_env(cache)
        say(f"library caches and temp files -> {R2.redact_path(cache, cfg.project_root)}")
        R2.import_heavy()
        with R2.Heartbeat(cfg.heartbeat_seconds):
            res = build(cfg)
    except errs("input") as e:
        print(f"STOPPED (input): {e}", file=sys.stderr)
        print("result: STOPPED (input); nothing written")
        return 2
    except errs("build") as e:
        lo = getattr(e, "leftover", None)
        tail = (f"pending folder left behind: {lo} (delete it unopened after review; never share or sync it)"
                if lo else "nothing left behind")
        print(f"BUILD FAILED: {e}", file=sys.stderr)
        print(f"result: FAILED; {tail}")
        return 3
    except Exception as e:  # noqa: BLE001
        print("BUILD FAILED (unexpected error; frames follow)", file=sys.stderr)
        lo = getattr(e, "leftover", None)
        for ln in describe_exception(e, cfg.project_root if cfg else None):
            print(ln, file=sys.stderr)
        print(f"result: FAILED; {'pending folder left behind: ' + lo if lo else 'nothing left behind'}")
        return 3
    finally:
        if cache is not None and R2 is not None:
            R2.clear_cache_temp(cache)
        if prev is not None:
            R2.restore_runtime_env(prev)
    if res["mode"] == "precheck-only":
        print("result: PRECHECK OK (Stage 06-08, Run 1, Run 2 manifests and files, Run 2 script, canonical, "
              "harmonization, split, raw ICUSTAYS / ADMISSIONS, test-use log, packages); nothing written (library "
              "cache folder and README only)")
        for k, v in sorted(res["hashes"].items()):
            print(f"{k} sha256 {v}")
        return 0
    print(res["audit_md"])
    print(f"elapsed {time.time() - t0:.0f} s")
    print("result: RUN stage15_layer_c_levels desktop (Tier 3) OK; RELEASE GATE OK; post-primary sensitivity analysis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
