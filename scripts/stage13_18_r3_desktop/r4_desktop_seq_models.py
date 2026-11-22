"""Desktop Tier 2 sequence-model sensitivity (PIC Foundations, PIC; plan v13, contract v13):
LSTM, LSTM with masked additive attention, and a small transformer encoder, each as an A+B+C Layer C
sequence model with the same inputs, Layer A+B static context at every step, Layer A baseline-risk
offset and CORAL-style cumulative-link head as the Run 2 GRU (D19).

r4_desktop_seq_models v1.0.1-desktop (Nov 21, 2026).

v1.0.1: GPU name, GPU memory, the Python executable path and other host environment strings are kept out of TEAM files (they are written only to an INTERNAL environment record). TEAM files may report a coarse cuda yes/no flag. The identifier gate is unchanged. A prior unpublished test-use attempt (status 'failed (nothing published)' or 'started') does not block a rerun; a published attempt still does. If a prior unpublished attempt exists, the audit logs that the test set was scored once before in that unpublished attempt.

This is a post-primary sensitivity analysis (a second look at the test set), not a replacement for
the laptop Run 3a primary scoring or the Run 2 GRU. Tuning and model selection use train and
validation only. Selected configurations are then scored once on test. All files this script writes
go under data/processed/ladder/DESKTOP_FULL_PLAN/stage17_seqmodels/ (published folder, pending
folder, cache, test-use log, internal files, logs and README_DESKTOP.txt). A published laptop or
desktop Run 3a/3b does not block this run.

Frozen inputs (hash-gated before anything is parsed): everything Run 2 checked (Stage 06-08 manifest
and files, Run 1 v1.0.2 manifest and files, canonical, harmonization, split, raw ICUSTAYS /
ADMISSIONS; the same pins), the Run 2 script (scripts/stage11_18_r2/r2_11p_11_12_17_models.py,
imported only after its SHA-256 matches the pin), and the Run 2 manifest (pinned by its full
SHA-256 on the command line and in this script) with every file it lists, including the INTERNAL
labels, validation predictions, tuned model and calibrators, and stage 17 models.

A test-use log (data/processed/ladder/DESKTOP_FULL_PLAN/stage17_seqmodels/test_use_log_stage17_seqmodels.json)
records each desktop attempt; a run refuses to start once a published desktop attempt exists in
that log. Device is CUDA if available, otherwise CPU, and is logged. Mixed precision is used only
when --amp is set and the device is CUDA.

Outputs (data/processed/ladder/DESKTOP_FULL_PLAN/stage17_seqmodels/published/, published atomically
from a pending folder in the same stage17_seqmodels directory): TEAM files (aggregate only) and
INTERNAL files (predictions and selected states; never opened, pasted or shared). Release gate as
in Run 2.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_THREADS = str(min(4, max(1, (os.cpu_count() or 2) // 2)))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_VERSION = "r4_desktop_seq_models v1.0.1-desktop"
SPEC_DOC = "stage17_seqmodels_desktop_spec v1.0.1-desktop"
PLAN, CONTRACT = "v13", "v13"
OUT_NAME = "stage17_seqmodels"
DESKTOP_ROOT_NAME = "DESKTOP_FULL_PLAN"
README_NAME = "README_DESKTOP.txt"
README_TEXT = (
    "These are post-primary sensitivity outputs from the desktop Tier 2 sequence-model run "
    "(r4_desktop_seq_models v1.0.1-desktop): LSTM, LSTM with attention, and a small transformer "
    "encoder, compared with the frozen Run 2 GRU and A+B on validation and scored once on test. "
    "They are a second look, not a replacement for the laptop primary Run 2 / Run 3a results. "
    "Published folders, pending folders, caches, internal files and logs for this run belong "
    "only in this folder (data/processed/ladder/DESKTOP_FULL_PLAN/stage17_seqmodels/).\n"
)
RUN2_NAME = "stage11_12_17_r2"
RUN2_MODULE = "r2_11p_11_12_17_models"
TEST_LOG_NAME = f"test_use_log_{OUT_NAME}.json"
CACHE_NAME = f"_cache_{OUT_NAME}"
PUBLISHED_NAME = "published"

PINS3 = {
    "run2_script": "r2_11p_11_12_17_models v1.0.0",
    "run2_script_sha256": "0262df9482c360baeb091f9bfcb4107e19d45712fa9f1aa8fd9db8f5c270ffbb",
    # Run 2 real run of Nov 19, 2026 (00:36 CT); tests set this to None
    "run2_manifest": "ec39434546de667b1a23b56c84be9563db747d1c95905a04a14a30f3526128ed",
}
RUN2_INTERNAL = ("stage11p_labels_landmark_INTERNAL.npz", "stage11_12_17_validation_predictions_INTERNAL.npz",
                 "stage12_tuned_model_calibrators_INTERNAL.pkl", "stage17_models_INTERNAL.pt",
                 "stage17_tabular_models_INTERNAL.pkl")
RUN2_AUDIT = "stage11_12_17_r2_audit_TEAM.json"
RUN2_TEAM_USED = ("stage11_validation_scores_TEAM.csv", "stage11_layer_c_24h_TEAM.csv", "stage11_winner_TEAM.csv")

# Expected environment (major.minor). Patch and the torch CUDA/CPU build suffix may differ.
EXPECTED_PY = (3, 14)
EXPECTED_PACKAGES = OrderedDict([
    ("numpy", "2.4"),
    ("pandas", "3.0"),
    ("scipy", "1.17"),
    ("sklearn", "1.8"),
    ("torch", "2.10"),
])

FAMILIES = ("lstm", "lstm_attn", "transformer")
FAMILY_CODE = {"lstm": 1, "lstm_attn": 2, "transformer": 3}
DEFAULT_MAX_EPOCHS = 40
DEFAULT_N_SEEDS = 1
DEFAULT_N_BOOT = 1000
DEFAULT_PATIENCE = 5
DEFAULT_BATCH = 256
EMB = 16
MAX_LEN = 64
ALPHA = 0.05
R2 = None

# Small fixed validation grids (not a full bake-off). Transformer positional encoding is a grid
# factor; n_heads is fixed at 4.
def default_grid(family: str) -> list:
    if family == "lstm":
        return [dict(hidden=h, n_layers=nl) for nl in (1, 2) for h in (32, 64)]
    if family == "lstm_attn":
        return [dict(hidden=h, n_layers=nl) for nl in (1, 2) for h in (32, 64)]
    if family == "transformer":
        return [dict(d_model=d, n_layers=nl, n_heads=4, pe=pe)
                for nl in (2, 4) for d in (64, 128) for pe in ("sinusoidal", "learned")]
    raise ValueError(family)


DECISIONS = OrderedDict([
    ("D17", "the Stage 08 base Layer C tensor is the single Layer C representation (Tier 3 Stage 15 levels are not built in this run)"),
    ("D18", "thresholds frozen before real runs: delta = 0.02 (QWK equivalence), delta_C = 0.01 (Layer C gain)"),
    ("D19", "A+B+C sequence model inputs, Layer A+B static context at every step, Layer A baseline-risk offset "
            "in a CORAL-style cumulative-link head, and early stopping on a 10% subject holdout inside train, "
            "as in the Run 2 GRU; this run swaps the recurrent encoder for LSTM, LSTM with masked additive "
            "attention, and a small transformer encoder"),
    ("D20", "validation-only tuning (no nested CV); B_eval = 1000 paired subject-cluster bootstrap replicates"),
    ("D21", "scripts written by the assistant; teammates cross-check them on the synthetic sandbox"),
])
DEVIATIONS = [
    "this is a post-primary sensitivity analysis on the desktop, not a replacement for the laptop Run 2 GRU "
    "or the laptop Run 3a test scoring; test rows are scored once under a separate refuse-rerun log",
    "Outcome 1 only, at the seven core landmarks (24, 48, 72, 120, 168, 240, 336 h); Outcome 2, ordinal8 and "
    "the non-core landmarks are not fit in this run",
    "the Stage 15 Layer C feature levels and nested cross-validation remain deferred (Tier 3)",
    "each family uses a small fixed validation grid (LSTM and LSTM-attention: hidden in {32, 64} and 1 or 2 "
    "layers; transformer: 2 or 4 layers, d_model in {64, 128}, 4 heads, sinusoidal or learned positional "
    "encoding); the selected configuration of each family is the one with the highest validation QWK at "
    "that landmark, official seed 6310 plus the landmark",
    "the transformer uses a learned CLS token, a linear map of the Layer C step (with the A+B embedding) "
    "into d_model, GELU feed-forward width 4 d_model, dropout 0.1, and learning rate 1e-3; LSTM families "
    "use learning rate 2e-3 and no dropout, matching the Run 2 GRU optimiser settings",
    "multi-seed refits (--n-seeds, default 1) are applied only to the already-selected configuration, as a "
    "stability check; they do not change the selected grid point",
    "comparisons with the frozen Run 2 GRU and A+B use the saved Run 2 models; validation is the primary "
    "comparison (delta_C = 0.01, B = 1000 paired subject-cluster bootstrap); test contrasts are descriptive "
    "sensitivity only",
    "a padding / key-padding mask is implemented, but at a fixed landmark every eligible prefix has length "
    "L/6, so the mask is all-valid in the Stage 08 base tensor",
    "no runtime cap (max_seconds default 0); mixed precision is off unless --amp is set on CUDA",
    "all writes are confined under data/processed/ladder/DESKTOP_FULL_PLAN/stage17_seqmodels/",
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


def load_run2(path: Path, expect_sha: str | None):
    """Import the Run 2 module from its file once its SHA-256 matches the pin (no bytecode written)."""
    global R2
    if not path.is_file():
        raise InputError(f"Run 2 script not found: {path.name}")
    h = sha256_file(path)
    if expect_sha and h != expect_sha:
        raise InputError("Run 2 script SHA-256 differs from the pin")
    if RUN2_MODULE in sys.modules and getattr(sys.modules[RUN2_MODULE], "__file__", None) and \
            os.path.abspath(sys.modules[RUN2_MODULE].__file__) == os.path.abspath(path):
        R2 = sys.modules[RUN2_MODULE]
        return R2, h
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(RUN2_MODULE, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[RUN2_MODULE] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    R2 = mod
    return mod, h


def errs(kind: str) -> tuple:
    if kind == "input":
        return (InputError,) + ((R2.InputError,) if R2 is not None else ())
    return (BuildError,) + ((R2.BuildError, R2.TimeLimit) if R2 is not None else ())


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
    """Refuse a write path that would land outside data/processed/ladder/DESKTOP_FULL_PLAN/stage17_seqmodels/."""
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
    if R2 is not None:
        R2.write_bytes_atomic(p, data)
    else:
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)


def pkg_major_minor(ver: str) -> str:
    parts = ver.split("+")[0].split(".")
    return ".".join(parts[:2])


def check_environment(require_cuda: bool = False) -> dict:
    """Compare major.minor versions with the laptop pins; report the device; stop on a mismatch."""
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
    """CUDA if available, otherwise CPU. Logged once; seeds stay as deterministic as the device allows."""
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
    """Unpickler for the hash-verified Run 2 model files."""
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


@dataclass
class Config:
    project_root: Path
    in_dir: Path
    run1_dir: Path
    run2_dir: Path
    run2_script: Path
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
    landmarks: tuple = field(default_factory=lambda: tuple(R2.CORE_LANDMARKS) if R2 is not None else (24, 48, 72, 120, 168, 240, 336))
    families: tuple = FAMILIES
    amp: bool = False
    require_cuda: bool = False
    max_seconds: float = 0.0
    heartbeat_seconds: float = 30.0
    repro_tol_prob: float = 1e-4
    run2_script_sha: str | None = None
    env_info: dict | None = None


def r2_config(cfg: Config):
    return R2.Config(project_root=cfg.project_root, in_dir=cfg.in_dir, run1_dir=cfg.run1_dir,
                     canonical=cfg.canonical, harmonization=cfg.harmonization, split_file=cfg.split_file,
                     raw_dir=cfg.raw_dir, out_dir=cfg.out_dir, expect_manifest_sha256=cfg.expect_manifest_sha256,
                     expect_run1_manifest_sha256=cfg.expect_run1_manifest_sha256, precheck_only=True,
                     n_boot=cfg.n_boot, gru_max_epochs=cfg.max_epochs, max_seconds=cfg.max_seconds,
                     heartbeat_seconds=cfg.heartbeat_seconds)


def read_test_log(p: Path) -> list:
    if not p.is_file():
        return []
    try:
        return list(json.loads(p.read_text(encoding="utf-8")).get("attempts", []))
    except (ValueError, AttributeError):
        raise InputError(f"test-use log is unreadable: {p.name}") from None


def write_test_log(p: Path, attempts: list) -> None:
    R2.write_bytes_atomic(p, json.dumps({"run": OUT_NAME, "attempts": attempts}, indent=1).encode("utf-8"))


def parse_landmarks(s: str | None) -> tuple:
    core = (24, 48, 72, 120, 168, 240, 336)
    if not s:
        return core
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v not in core:
            raise InputError(f"landmark {v} is not one of the core landmarks {list(core)}")
        if v not in out:
            out.append(v)
    if not out:
        raise InputError("no landmarks given")
    return tuple(out)


def parse_families(s: str | None) -> tuple:
    if not s:
        return FAMILIES
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in FAMILIES:
            raise InputError(f"unknown family {part!r}; expected one of {list(FAMILIES)}")
        if part not in out:
            out.append(part)
    if not out:
        raise InputError("no families given")
    return tuple(out)



def precheck(cfg: Config) -> dict:
    """Run 2's hash gates (same pins), then the Run 2 manifest (pinned) and every file it lists; the output
    folder must be empty, no pending folder may exist, and no published attempt may be in this run's test-use log.
    A prior 'failed (nothing published)' or 'started' attempt does not block a rerun."""
    for q in (cfg.run2_dir, cfg.out_dir, cfg.split_file, cfg.raw_dir, cfg.run2_script, cfg.work_dir, cfg.test_log):
        if R2.NEVER_READ_TOKEN in str(q):
            raise InputError(f"a path containing {R2.NEVER_READ_TOKEN} is never read or written")
    att_early = read_test_log(cfg.test_log)
    if any(a.get("status") == "published" for a in att_early):
        raise InputError("the test-use log records a published test scoring: test rows are scored once only")
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
    att = read_test_log(cfg.test_log)
    if any(a.get("status") == "published" for a in att):
        raise InputError("the test-use log records a published test scoring: test rows are scored once only")
    return {"hashes": hashes, "run2_manifest": m2, "prior_attempts": len(att)}


def load_all(cfg: Config, checks: dict) -> dict:
    """Stage 08 levels on the 24 h landmark rows of every split (test rows stay behind the TestGuard), the Run 1
    bases, the Layer C index of every landmark, the Run 2 labels, subjects and frozen artefacts."""
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
    checks["r4_labels_match_stage08_order_and_split"] = bool(
        np.array_equal(lab["encounter_id"].astype(str), eid0) and np.array_equal(lab["split"].astype(str), split0))
    if not checks["r4_labels_match_stage08_order_and_split"]:
        raise InputError("Run 2 labels: encounter order or split differs from Stage 08")
    can = R2.read_csv_str(cfg.canonical, usecols=["encounter_id", "SUBJECT_ID"])
    can["encounter_id"] = can["encounter_id"].str.strip()
    if can["encounter_id"].duplicated().any() or set(can["encounter_id"]) != set(eid0):
        raise InputError("the Stage 08 encounters differ from the canonical encounter set (count withheld)")
    subj_all = can.set_index("encounter_id").loc[eid0]["SUBJECT_ID"].map(R2.norm_id).to_numpy().astype(str)
    spl = R2.read_csv_str(cfg.split_file, usecols=["SUBJECT_ID", "split"])
    smap = dict(zip(spl["SUBJECT_ID"].map(R2.norm_id),
                    spl["split"].map(lambda v: R2.SPLIT_LABELS.get(str(v).strip().lower(), "?"))))
    checks["r4_split_matches_split_file"] = bool(np.array_equal(np.array([smap.get(s, "?") for s in subj_all]),
                                                                split0))
    if not checks["r4_split_matches_split_file"]:
        raise InputError("Stage 08 split differs from the persisted subject split (count withheld)")
    rows["subj"] = subj_all[elig24]
    rows["died"] = lab["died"].astype(bool)[elig24]
    rows["O1"] = OrderedDict((L, lab[f"outcome1_L{L}"].astype(np.int64)[elig24]) for L in R2.LANDMARKS)
    rows["O2"], rows["O8"] = lab["outcome2"].astype(np.int64)[elig24], lab["ordinal8"].astype(np.int64)[elig24]
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
    checks["r4_eligible_sets_match_labels"] = ok_el
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
    """The only path to test rows. Phase 'verify' (tuning and frozen-model checks on validation) refuses any
    test access; phase 'score' allows each named scoring exactly once. Every fit is registered with the splits of
    its rows, and a fit on any test row is refused."""

    def __init__(self, split: np.ndarray):
        self.split = split
        self.phase = "verify"
        self.scored = OrderedDict()
        self.fits = []
        self.first_test_access = None

    def te(self) -> np.ndarray:
        if self.phase != "score":
            raise BuildError("test rows requested outside the scoring phase")
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
        if self.first_test_access is None:
            self.first_test_access = time.time()
        out = fn()
        self.scored[key] = True
        return out


# ----------------------------------------------------------------------------------------- sequence models
def _coral_params(nn, torch, K, gamma0):
    g0 = np.zeros(K - 1) if gamma0 is None else np.asarray(gamma0, dtype=float)
    d = np.diff(g0) if K > 2 else np.zeros(0)
    g_first = nn.Parameter(torch.tensor([g0[0]], dtype=torch.float32))
    g_steps = nn.Parameter(torch.tensor(np.log(np.expm1(np.maximum(d, 1e-3))), dtype=torch.float32))
    return g_first, g_steps


def _gamma(nn, g_first, g_steps):
    torch = R2._torch()
    if g_steps.numel() == 0:
        return g_first
    return torch.cat([g_first, g_first + torch.cumsum(nn.functional.softplus(g_steps), 0)])


def _static_cat(emb, xs, st):
    if emb is None:
        return xs
    torch = R2._torch()
    e = emb(st).unsqueeze(1).expand(-1, xs.shape[1], -1)
    return torch.cat([xs, e], dim=2)


def _last_hidden(out, mask):
    if mask is None:
        return out[:, -1, :]
    lengths = mask.long().sum(1).clamp(min=1)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
    return out.gather(1, idx).squeeze(1)


def make_lstm_class():
    torch = R2._torch()
    nn = torch.nn

    class StaticSeqLSTM(nn.Module):
        """D19 static-plus-sequential model with an LSTM encoder and last-step (masked) pooling."""

        def __init__(self, d_seq, d_static, hidden, K, emb=EMB, gamma0=None, n_layers=1, dropout=0.0):
            super().__init__()
            self.d_static = d_static
            self.hidden = hidden
            self.n_layers = n_layers
            self.emb = nn.Linear(d_static, emb) if d_static > 0 else None
            d_in = d_seq + (emb if d_static > 0 else 0)
            self.rnn = nn.LSTM(d_in, hidden, num_layers=n_layers, batch_first=True,
                               dropout=(dropout if n_layers > 1 else 0.0))
            self.out = nn.Linear(hidden, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.g_first, self.g_steps = _coral_params(nn, torch, K, gamma0)

        def gamma(self):
            return _gamma(nn, self.g_first, self.g_steps)

        def forward(self, xs, st, off, mask=None):
            xs = _static_cat(self.emb, xs, st)
            h, _ = self.rnn(xs)
            g = self.out(_last_hidden(h, mask)).squeeze(1)
            return self.gamma().unsqueeze(0) - (off + g).unsqueeze(1)

    return StaticSeqLSTM


def make_lstm_attn_class():
    torch = R2._torch()
    nn = torch.nn

    class StaticSeqLSTMAttn(nn.Module):
        """D19 head with an LSTM encoder and masked additive attention pooling over time steps."""

        def __init__(self, d_seq, d_static, hidden, K, emb=EMB, gamma0=None, n_layers=1, dropout=0.0):
            super().__init__()
            self.d_static = d_static
            self.hidden = hidden
            self.n_layers = n_layers
            self.emb = nn.Linear(d_static, emb) if d_static > 0 else None
            d_in = d_seq + (emb if d_static > 0 else 0)
            self.rnn = nn.LSTM(d_in, hidden, num_layers=n_layers, batch_first=True,
                               dropout=(dropout if n_layers > 1 else 0.0))
            self.attn_w = nn.Linear(hidden, hidden)
            self.attn_v = nn.Linear(hidden, 1, bias=False)
            self.out = nn.Linear(hidden, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.g_first, self.g_steps = _coral_params(nn, torch, K, gamma0)

        def gamma(self):
            return _gamma(nn, self.g_first, self.g_steps)

        def forward(self, xs, st, off, mask=None):
            xs = _static_cat(self.emb, xs, st)
            h, _ = self.rnn(xs)
            e = self.attn_v(torch.tanh(self.attn_w(h))).squeeze(-1)
            if mask is not None:
                e = e.masked_fill(~mask, torch.finfo(e.dtype).min)
            alpha = torch.softmax(e, dim=1)
            pooled = (alpha.unsqueeze(-1) * h).sum(1)
            g = self.out(pooled).squeeze(1)
            return self.gamma().unsqueeze(0) - (off + g).unsqueeze(1)

    return StaticSeqLSTMAttn


def make_transformer_class():
    torch = R2._torch()
    nn = torch.nn

    class SinusoidalPE(nn.Module):
        def __init__(self, d_model, max_len=MAX_LEN):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)].to(dtype=x.dtype)

    class LearnedPE(nn.Module):
        def __init__(self, d_model, max_len=MAX_LEN):
            super().__init__()
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pe, std=0.02)

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class StaticSeqTransformer(nn.Module):
        """Small transformer encoder with a learned CLS token and the D19 CORAL head."""

        def __init__(self, d_seq, d_static, K, d_model=64, n_layers=2, n_heads=4, emb=EMB, gamma0=None,
                     pe="sinusoidal", dropout=0.1, max_len=MAX_LEN):
            super().__init__()
            self.d_static = d_static
            self.d_model = d_model
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.pe_kind = pe
            self.emb = nn.Linear(d_static, emb) if d_static > 0 else None
            d_in = d_seq + (emb if d_static > 0 else 0)
            self.proj = nn.Linear(d_in, d_model)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)
            self.pos = (SinusoidalPE if pe == "sinusoidal" else LearnedPE)(d_model, max_len=max_len + 1)
            layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
                                               batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.out = nn.Linear(d_model, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.g_first, self.g_steps = _coral_params(nn, torch, K, gamma0)

        def gamma(self):
            return _gamma(nn, self.g_first, self.g_steps)

        def forward(self, xs, st, off, mask=None):
            xs = _static_cat(self.emb, xs, st)
            h = self.proj(xs)
            cls = self.cls.expand(h.size(0), -1, -1)
            h = self.pos(torch.cat([cls, h], dim=1))
            if mask is None or bool(mask.all()):
                h = self.encoder(h)
            else:
                z = torch.zeros(mask.size(0), 1, dtype=torch.bool, device=mask.device)
                pad = torch.cat([z, ~mask], dim=1)
                h = self.encoder(h, src_key_padding_mask=pad)
            g = self.out(h[:, 0, :]).squeeze(1)
            return self.gamma().unsqueeze(0) - (off + g).unsqueeze(1)

    return StaticSeqTransformer


def _factory(family: str):
    if family == "lstm":
        return make_lstm_class()
    if family == "lstm_attn":
        return make_lstm_attn_class()
    if family == "transformer":
        return make_transformer_class()
    raise BuildError(f"unknown family {family}")


def _build_net(family: str, d_seq: int, d_static: int, K: int, hp: dict, gamma0=None):
    Net = _factory(family)
    if family in ("lstm", "lstm_attn"):
        return Net(d_seq, d_static, int(hp["hidden"]), K, emb=EMB, gamma0=gamma0,
                   n_layers=int(hp.get("n_layers", 1)), dropout=float(hp.get("dropout", 0.0)))
    return Net(d_seq, d_static, K, d_model=int(hp["d_model"]), n_layers=int(hp["n_layers"]),
               n_heads=int(hp.get("n_heads", 4)), emb=EMB, gamma0=gamma0, pe=str(hp.get("pe", "sinusoidal")),
               dropout=float(hp.get("dropout", 0.1)))


def _default_lr(family: str) -> float:
    return 1e-3 if family == "transformer" else 2e-3


def seq_fit_predict(family, hp, seq_tr, st_tr, off_tr, y_tr, g_tr, seq_va, st_va, off_va, K, seed,
                    gamma0=None, max_epochs=40, patience=5, batch=256, lr=None, wd=1e-4, amp=False,
                    mask_tr=None, mask_va=None):
    """Train with the CORAL binary cross-entropy, Adam, early stopping on a 10% subject holdout of train.
    Returns (state, validation class probabilities, epochs trained)."""
    torch = R2._torch()
    seed_torch(torch, seed)
    dev = torch_device()
    use_amp = bool(amp) and dev.type == "cuda"
    rng = np.random.default_rng(seed)
    us = np.unique(g_tr)
    hold = set(rng.choice(us, size=max(1, int(round(0.1 * len(us)))), replace=False).tolist())
    hm = np.array([g in hold for g in g_tr])
    fit_i, hold_i = np.flatnonzero(~hm), np.flatnonzero(hm)
    if len(fit_i) == 0:
        fit_i = np.arange(len(y_tr))
        hold_i = fit_i[:1]
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    B = lambda m, n: (None if m is None else torch.tensor(np.ascontiguousarray(m), dtype=torch.bool, device=dev))
    xs, xh = T(seq_tr), T(seq_va)
    st = T(st_tr) if st_tr is not None else torch.zeros((len(y_tr), 0), device=dev)
    sh = T(st_va) if st_va is not None else torch.zeros((seq_va.shape[0], 0), device=dev)
    ot, oh = T(off_tr), T(off_va)
    mt, mh = B(mask_tr, len(y_tr)), B(mask_va, seq_va.shape[0])
    Y = torch.tensor((y_tr[:, None] > np.arange(K - 1)[None, :]).astype(np.float32), device=dev)
    net = _build_net(family, seq_tr.shape[2], int(st.shape[1]), K, hp, gamma0=gamma0).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=(lr if lr is not None else _default_lr(family)), weight_decay=wd)
    bce = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_amp)

    def _fwd(xb, stb, otb, mb):
        if mb is None:
            return net(xb, stb, otb)
        return net(xb, stb, otb, mask=mb)

    def hold_loss():
        net.eval()
        with torch.no_grad():
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                z = _fwd(xs[hold_i], st[hold_i], ot[hold_i], None if mt is None else mt[hold_i])
                return float(bce(-z, Y[hold_i]))

    best, best_state, bad, ep = math.inf, None, 0, 0
    for ep in range(1, max_epochs + 1):
        net.train()
        perm = rng.permutation(fit_i)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            mb = None if mt is None else mt[b]
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = bce(-_fwd(xs[b], st[b], ot[b], mb), Y[b])
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
            F = torch.sigmoid(_fwd(xh, sh, oh, mh)).float().cpu().numpy().astype(np.float64)
    F = np.maximum.accumulate(F, axis=1)
    state = dict(state_dict=best_state, family=family, hp=dict(hp), d_seq=int(seq_tr.shape[2]),
                 d_static=int(st.shape[1]), K=K, seed=int(seed))
    return state, R2.cum_to_probs(F), ep


def seq_predict(state: dict, seq: np.ndarray, st, off: np.ndarray, mask=None, amp=False) -> np.ndarray:
    torch = R2._torch()
    dev = torch_device()
    use_amp = bool(amp) and dev.type == "cuda"
    net = _build_net(state["family"], state["d_seq"], state["d_static"], state["K"], state["hp"])
    net.load_state_dict(state["state_dict"])
    net.to(dev)
    net.eval()
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    s = T(st) if st is not None and state["d_static"] > 0 else torch.zeros((seq.shape[0], 0), device=dev)
    m = None if mask is None else torch.tensor(np.ascontiguousarray(mask), dtype=torch.bool, device=dev)
    with torch.no_grad():
        with torch.autocast(device_type=dev.type, enabled=use_amp):
            logits = net(T(seq), s, T(off), mask=m) if m is not None else net(T(seq), s, T(off))
            F = torch.sigmoid(logits).float().cpu().numpy().astype(np.float64)
    F = np.maximum.accumulate(F, axis=1)
    return R2.cum_to_probs(F)


def gru_predict(state: dict, seq: np.ndarray, st, off: np.ndarray) -> np.ndarray:
    """Class probabilities of a frozen Run 2 GRU (same forward pass and monotone repair as Run 2)."""
    torch = R2._torch()
    Net = R2.make_gru_class()
    net = Net(state["d_seq"], state["d_static"], state["hidden"], state["K"])
    net.load_state_dict(state["state_dict"])
    dev = torch_device()
    net.to(dev)
    net.eval()
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    s = T(st) if st is not None and state["d_static"] > 0 else torch.zeros((seq.shape[0], 0), device=dev)
    with torch.no_grad():
        F = torch.sigmoid(net(T(seq), s, T(off))).cpu().numpy().astype(np.float64)
    F = np.maximum.accumulate(F, axis=1)
    return R2.cum_to_probs(F)



# --------------------------------------------------------------------------------------------------- helpers
def p_gain(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    return float((np.sum(d <= 0) + 1) / (d.size + 1)) if d.size else float("nan")


def interval(v: np.ndarray) -> tuple:
    v = v[np.isfinite(v)]
    return (R2.pct(v, 2.5), R2.pct(v, 97.5)) if v.size else (float("nan"), float("nan"))


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


def boot_auc(W: np.ndarray, score: np.ndarray, lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=bool)
    o = np.argsort(score, kind="mergesort")
    s, l_ = np.asarray(score)[o], lab[o]
    Wo = W[:, o].astype(np.float64)
    _, start = np.unique(s, return_index=True)
    WP = np.add.reduceat(Wo * l_[None, :], start, axis=1)
    WN = np.add.reduceat(Wo * (~l_)[None, :], start, axis=1)
    below = np.cumsum(WN, axis=1) - WN
    num = (WP * (below + 0.5 * WN)).sum(1)
    den = WP.sum(1) * WN.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def boot_macro_auc(W, P, y, K) -> np.ndarray:
    exc = 1.0 - np.cumsum(P, axis=1)[:, :-1]
    A = np.column_stack([boot_auc(W, exc[:, j], y > j) for j in range(K - 1)])
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(A)
        return np.where(ok.any(1), np.nansum(A, 1) / np.maximum(ok.sum(1), 1), np.nan)


def score_split(P, y, K, died, W, extra: dict) -> tuple:
    row, kb = R2.score_row(P, y, K, died, W, extra)
    ka = boot_macro_auc(W, P, y, K)
    row["auroc_macro_p2_5"], row["auroc_macro_p97_5"] = interval(ka)
    return suppress_small(row, y, K, died), kb, ka


def contrast(kq1, kq0, q1, q0, ka1, ka0, a1, a0, e1, e0, delta_C) -> OrderedDict:
    d = kq1 - kq0
    lo, hi = interval(d)
    da = ka1 - ka0
    alo, ahi = interval(da)
    pt = q1 - q0
    return OrderedDict(delta_qwk=pt, delta_p2_5=lo, delta_p97_5=hi, p_one_sided_le_0=p_gain(d),
                       delta_auroc_macro=a1 - a0, delta_auroc_macro_p2_5=alo, delta_auroc_macro_p97_5=ahi,
                       delta_emd=e1 - e0, gain_at_least_delta_C=bool(pt >= delta_C and lo > 0))


def winner_X(D: dict, mats: OrderedDict) -> tuple:
    w = D["s12"]["winner"]
    kind, k = (w["rep"], None) if w["rep"] == "full" else (w["rep"].split("_")[0], int(w["rep"].split("_")[1]))
    return w, R2.rep_X(mats[w["matrix"]], kind, k)


def hp_label(hp: dict) -> str:
    if "d_model" in hp:
        return f"d{hp['d_model']}_L{hp['n_layers']}_h{hp.get('n_heads', 4)}_{hp.get('pe', 'sinusoidal')}"
    return f"hid{hp['hidden']}_L{hp.get('n_layers', 1)}"


def official_seed(family: str, L: int, seed0: int | None = None) -> int:
    return (R2.SEED if seed0 is None else int(seed0)) + int(L) + 10000 * FAMILY_CODE[family]


# --------------------------------------------------------------------------------- validation tuning
def verify_frozen(D: dict, mats: OrderedDict, Vw, Xw, cfg: Config, checks: dict, info: dict, landmarks) -> None:
    """Frozen Run 2 GRU and A+B must reproduce the Run 2 validation predictions (core landmarks, Outcome 1)."""
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
    checks["r4_frozen_gru_ab_reproduce_run2_validation"] = ok and not missing and maxd <= cfg.repro_tol_prob
    info["repro_s17_max_abs_prob_diff"] = maxd
    say("verify: frozen Run 2 GRU and A+B reproduce the Run 2 validation predictions")


def tune_landmark(D, mats, G, Vw, Xw, L, cfg, checks) -> dict:
    """Fit every grid point of every family on train (holdout early stopping); select on validation QWK."""
    R, tab = D["rows"], D["tab"]
    sp = R["split"]
    idx = D["lc"][L]["idx"]
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
    S = R2.c_inputs(D, L, "all", None)
    off_tr, off_va, th, mA = R2.crossfit_offset(XA[jtr], yy[jtr], g[jtr], XA[jva], K, ("r4", L, "outcome1"))
    W = R2.boot_weights(g[jva], cfg.n_boot, R2.rng_for("boot", "r4", L))
    # frozen comparators (no fit)
    P_ab = tab[f"L{L}_outcome1_AB"].predict_proba(Xab[jva])
    P_gru = gru_predict(D["states"][f"L{L}_outcome1_ABC"], S[jva], Xab[jva], mA.decision(XA[jva]))
    yv, dv = yy[jva], died[jva]
    row_ab, kb_ab, ka_ab = score_split(P_ab, yv, K, dv, W,
                                       dict(outcome="outcome1", landmark=L, core=True, family="AB",
                                            model="AB", config="frozen_run2", split="validation"))
    row_gru, kb_gru, ka_gru = score_split(P_gru, yv, K, dv, W,
                                          dict(outcome="outcome1", landmark=L, core=True, family="GRU",
                                               model="GRU", config="frozen_run2", split="validation"))
    frozen = dict(AB=(P_ab, row_ab, kb_ab, ka_ab), GRU=(P_gru, row_gru, kb_gru, ka_gru),
                  off_tr=off_tr, off_va=off_va, theta=th, mA=mA, S=S, Xab=Xab, XA=XA,
                  jtr=jtr, jva=jva, yy=yy, g=g, died=died, W=W, idx=idx, K=K)
    tune_rows, states, preds = [row_ab, row_gru], OrderedDict(), OrderedDict()
    selected = OrderedDict()
    for fam in cfg.families:
        best = None
        for ci, hp in enumerate(default_grid(fam)):
            seed = official_seed(fam, L)
            G.fit_rows(idx[jtr], f"{fam}_L{L}_{hp_label(hp)}")
            t0 = time.time()
            stt, P, ep = seq_fit_predict(fam, hp, S[jtr], Xab[jtr], off_tr, yy[jtr], g[jtr],
                                         S[jva], Xab[jva], off_va, K, seed, gamma0=th,
                                         max_epochs=cfg.max_epochs, patience=cfg.patience, batch=cfg.batch,
                                         amp=cfg.amp)
            elapsed = time.time() - t0
            row, kb, ka = score_split(P, yv, K, dv, W,
                                      dict(outcome="outcome1", landmark=L, core=True, family=fam, model=fam,
                                           config=hp_label(hp), hidden=hp.get("hidden", ""),
                                           d_model=hp.get("d_model", ""), n_layers=hp.get("n_layers", ""),
                                           pe=hp.get("pe", ""), epochs_trained=ep, split="validation"))
            row["fit_seconds"] = round(elapsed, 3)
            tune_rows.append(row)
            key = f"L{L}_{fam}_{hp_label(hp)}"
            states[key] = stt
            preds[f"val_{key}"] = P.astype(np.float32)
            say(f"tune L{L} {fam} {hp_label(hp)}: val QWK {row['qwk']:.4f} epochs {ep} ({elapsed:.1f} s)")
            if best is None or row["qwk"] > best["qwk"]:
                best = dict(hp=hp, state=stt, P=P, row=row, kb=kb, ka=ka, epochs=ep, key=key,
                            qwk=row["qwk"], elapsed=elapsed)
            check_time()
        selected[fam] = best
        preds[f"val_L{L}_{fam}_selected"] = best["P"].astype(np.float32)
    preds[f"val_L{L}__eid"] = R["eid"][idx][jva]
    preds[f"val_L{L}_AB"] = P_ab.astype(np.float32)
    preds[f"val_L{L}_GRU"] = P_gru.astype(np.float32)
    return dict(tune_rows=tune_rows, selected=selected, frozen=frozen, states=states, preds=preds)


def val_contrasts(L, sel, frozen, delta_C) -> list:
    rows = []
    for fam, best in sel.items():
        for ref_name, pack in (("GRU", frozen["GRU"]), ("AB", frozen["AB"])):
            P1, r1, k1, a1 = best["P"], best["row"], best["kb"], best["ka"]
            P0, r0, k0, a0 = pack
            c = contrast(k1, k0, r1["qwk"], r0["qwk"], a1, a0, r1["auroc_macro"], r0["auroc_macro"],
                         r1["emd"], r0["emd"], delta_C)
            rows.append(OrderedDict(outcome="outcome1", landmark=L, core=True, family=fam,
                                    config=hp_label(best["hp"]), contrast=f"{fam}_minus_{ref_name}",
                                    split="validation", **c))
    return rows


def multiseed_selected(D, G, L, pack, cfg) -> list:
    """Refit the selected configuration with additional seeds (stability; does not change selection)."""
    if cfg.n_seeds <= 1:
        return []
    fr = pack["frozen"]
    S, Xab, yy, g = fr["S"], fr["Xab"], fr["yy"], fr["g"]
    jtr, jva, K, th = fr["jtr"], fr["jva"], fr["K"], fr["theta"]
    off_tr, off_va, W = fr["off_tr"], fr["off_va"], fr["W"]
    yv, dv = yy[jva], fr["died"][jva]
    idx = fr["idx"]
    rows = []
    for fam, best in pack["selected"].items():
        qwks = [best["qwk"]]
        for s in range(1, cfg.n_seeds):
            seed = official_seed(fam, L) + 17 * s
            G.fit_rows(idx[jtr], f"{fam}_L{L}_seed{s}")
            t0 = time.time()
            _, P, ep = seq_fit_predict(fam, best["hp"], S[jtr], Xab[jtr], off_tr, yy[jtr], g[jtr],
                                       S[jva], Xab[jva], off_va, K, seed, gamma0=th,
                                       max_epochs=cfg.max_epochs, patience=cfg.patience, batch=cfg.batch,
                                       amp=cfg.amp)
            row, _, _ = score_split(P, yv, K, dv, W,
                                    dict(outcome="outcome1", landmark=L, core=True, family=fam, model=fam,
                                         config=hp_label(best["hp"]), seed_index=s, epochs_trained=ep,
                                         split="validation"))
            row["fit_seconds"] = round(time.time() - t0, 3)
            rows.append(row)
            qwks.append(row["qwk"])
        # summary row uses mean/sd column names that start with allowed audit keys
        rows.append(OrderedDict(outcome="outcome1", landmark=L, core=True, family=fam, model=fam,
                                config=hp_label(best["hp"]), seed_index="mean", split="validation",
                                qwk=float(np.mean(qwks)), qwk_sd=float(np.std(qwks, ddof=1) if len(qwks) > 1 else 0.0),
                                n_seeds=cfg.n_seeds))
    return rows


# --------------------------------------------------------------------------------- test scoring
def score_test_landmark(D, G, L, pack, cfg) -> dict:
    R = D["rows"]
    fr = pack["frozen"]
    sp = R["split"]
    idx, S, Xab, XA, K = fr["idx"], fr["S"], fr["Xab"], fr["XA"], fr["K"]
    # official test-row access (refused unless phase is score)
    _ = G.te()
    ite = np.flatnonzero(sp[idx] == "test")
    yy = R["O1"][L][idx]
    okr = (yy >= 0)[ite]
    jte = ite[okr]
    W = R2.boot_weights(R["subj"][idx][jte], cfg.n_boot, R2.rng_for("boot", "r4te", L))
    yv, dv = yy[jte], R["died"][idx][jte]
    off_te = fr["mA"].decision(XA[jte])
    rows, crows, internal = [], [], OrderedDict()
    Pab = G.score(f"r4_L{L}_AB", lambda: D["tab"][f"L{L}_outcome1_AB"].predict_proba(Xab[jte]))
    Pgru = G.score(f"r4_L{L}_GRU", lambda: gru_predict(D["states"][f"L{L}_outcome1_ABC"], S[jte], Xab[jte], off_te))
    scored = OrderedDict(AB=Pab, GRU=Pgru)
    packs = {}
    for name, P in (("AB", Pab), ("GRU", Pgru)):
        row, kb, ka = score_split(P, yv, K, dv, W,
                                  dict(outcome="outcome1", landmark=L, core=True, family=name, model=name,
                                       config="frozen_run2", split="test",
                                       label="post-primary sensitivity (desktop)"))
        rows.append(row)
        packs[name] = (P, row, kb, ka)
        internal[f"te_L{L}_{name}"] = P.astype(np.float32)
    dC = R2.THRESH["delta_C"]
    for fam, best in pack["selected"].items():
        def _pred(st=best["state"]):
            return seq_predict(st, S[jte], Xab[jte], off_te, amp=cfg.amp)
        P = G.score(f"r4_L{L}_{fam}", _pred)
        row, kb, ka = score_split(P, yv, K, dv, W,
                                  dict(outcome="outcome1", landmark=L, core=True, family=fam, model=fam,
                                       config=hp_label(best["hp"]), hidden=best["hp"].get("hidden", ""),
                                       d_model=best["hp"].get("d_model", ""), n_layers=best["hp"].get("n_layers", ""),
                                       pe=best["hp"].get("pe", ""), split="test",
                                       label="post-primary sensitivity (desktop)"))
        rows.append(row)
        packs[fam] = (P, row, kb, ka)
        internal[f"te_L{L}_{fam}"] = P.astype(np.float32)
        for ref in ("GRU", "AB"):
            P0, r0, k0, a0 = packs[ref]
            c = contrast(kb, k0, row["qwk"], r0["qwk"], ka, a0, row["auroc_macro"], r0["auroc_macro"],
                         row["emd"], r0["emd"], dC)
            crows.append(OrderedDict(outcome="outcome1", landmark=L, core=True, family=fam,
                                     config=hp_label(best["hp"]), contrast=f"{fam}_minus_{ref}",
                                     split="test", label="descriptive sensitivity (desktop)", **c))
    internal[f"te_L{L}__eid"] = R["eid"][idx][jte]
    return dict(rows=rows, contrasts=crows, internal=internal)


# ------------------------------------------------------------------------------------------------ figures
def figures(tune_df: pd.DataFrame, sel_df: pd.DataFrame, val_ct: pd.DataFrame) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    # selected validation QWK by landmark
    for fam, style in (("AB", "k--"), ("GRU", "k-"), ("lstm", "C0-"), ("lstm_attn", "C1-"), ("transformer", "C2-")):
        if fam in ("AB", "GRU"):
            sub = tune_df[(tune_df["family"] == fam) & (tune_df["config"] == "frozen_run2")]
        else:
            sub = sel_df[sel_df["family"] == fam] if "family" in sel_df.columns else pd.DataFrame()
        if sub is None or len(sub) == 0:
            continue
        ax.plot(sub["landmark"].astype(int), sub["qwk"].astype(float), style, marker="o", label=fam)
    ax.set_xlabel("landmark (h)")
    ax.set_ylabel("validation QWK")
    ax.set_title("Selected sequence models vs frozen GRU and A+B (validation)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    out["fig17_seq_val_qwk_TEAM.png"] = buf.getvalue()
    # delta vs GRU
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    g = val_ct[val_ct["contrast"].astype(str).str.endswith("_minus_GRU")] if len(val_ct) else val_ct
    for fam, col in (("lstm", "C0"), ("lstm_attn", "C1"), ("transformer", "C2")):
        sub = g[g["family"] == fam] if len(g) else g
        if len(sub) == 0:
            continue
        ax.errorbar(sub["landmark"].astype(int), sub["delta_qwk"].astype(float),
                    yerr=np.vstack([sub["delta_qwk"].astype(float) - sub["delta_p2_5"].astype(float),
                                    sub["delta_p97_5"].astype(float) - sub["delta_qwk"].astype(float)]),
                    fmt="o-", color=col, label=fam)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(R2.THRESH["delta_C"], color="k", ls=":", lw=0.8)
    ax.set_xlabel("landmark (h)")
    ax.set_ylabel("validation delta QWK vs frozen GRU")
    ax.set_title("Selected models minus frozen Run 2 GRU (validation)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    out["fig17_seq_val_delta_gru_TEAM.png"] = buf.getvalue()
    return out


def selected_table(packs: dict) -> pd.DataFrame:
    rows = []
    for L, pack in packs.items():
        for fam, best in pack["selected"].items():
            r = OrderedDict(best["row"])
            r["selected"] = True
            rows.append(r)
        for name in ("AB", "GRU"):
            rows.append(OrderedDict(pack["frozen"][name][1]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------- audit
_STRUCT_COL3 = re.compile(r"^(k|p|landmark|class|threshold|hidden|epochs_trained|n_layers|d_model|n_heads|n_seeds|seed_index|fit_seconds)$")


def audit_num_keys() -> tuple:
    return tuple(R2.AUDIT_NUM_KEYS) + (
        "repro", "p_holm", "alpha", "test_attempts", "epochs", "epsilon", "d_model", "n_layers",
        "n_heads", "n_seeds", "fit_seconds", "patience", "max_epochs", "seed_index", "qwk_sd",
        "cfg_", "pe",
    )


def audit_nums_ok(o, key="", inherited=False) -> bool:
    allowed = inherited or any(key.startswith(k) for k in audit_num_keys())
    if isinstance(o, bool):
        return True
    if isinstance(o, (int, float, np.integer, np.floating)):
        return allowed
    if isinstance(o, dict):
        return all(audit_nums_ok(v, str(k), allowed) for k, v in o.items())
    if isinstance(o, list):
        return all(audit_nums_ok(v, key, allowed) for v in o)
    return True


def bad_num_keys(o, key="", inherited=False, path="$") -> list:
    allowed = inherited or any(key.startswith(k) for k in audit_num_keys())
    if isinstance(o, bool):
        return []
    if isinstance(o, (int, float, np.integer, np.floating)):
        return [] if allowed else [path]
    if isinstance(o, dict):
        return [x for k, v in o.items() for x in bad_num_keys(v, str(k), allowed, f"{path}.{k}")]
    if isinstance(o, list):
        return [x for i, v in enumerate(o) for x in bad_num_keys(v, key, allowed, f"{path}[{i}]")]
    return []


def script_text_ints(audit: dict | None) -> set:
    txt = " ".join(list(DECISIONS.values()) + list(DECISIONS) + list(DEVIATIONS) + [R2.SUPP, "10 or more"])
    if audit is not None:
        try:
            txt += " " + audit_md(R2._skeleton(audit))
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
        if nm.endswith(".png"):
            continue
        text = (pend / nm).read_text(encoding="utf-8")
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
                            if c in ("item", "label", "family", "model", "config", "contrast", "split", "pe"):
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
                    for t in R2.team_text_problems(v, forbidden - allowed, ids, allowed):
                        probs.append((nm, f"json key {path}", t))
        else:
            for t in R2.team_text_problems(text, forbidden - allowed, ids, allowed):
                probs.append((nm, "md text", t))
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
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return R2.r6(v) if math.isfinite(v) else None


def environment_internal_bytes(cfg, pre) -> bytes:
    """Full host environment (GPU name, memory, executable, package versions) and prior test-use
    attempts. TEAM files must not contain these strings; this record is INTERNAL only."""
    payload = OrderedDict(
        script=SCRIPT_VERSION,
        generated_utc=utc_now(),
        environment=cfg.env_info or {},
        prior_test_use_attempts=read_test_log(cfg.test_log),
        prior_attempts=int((pre or {}).get("prior_attempts") or 0),
        note="GPU name, GPU memory, executable path and other host strings stay in this INTERNAL "
             "record only; TEAM files carry a coarse cuda yes/no flag.",
    )
    return json.dumps(payload, indent=1, default=str).encode("utf-8")


def make_audit(cfg, pre, checks, info, packs, val_ct, te_tab, te_ct, ms_tab, script_sha, n_written) -> dict:
    sel = []
    for L, pack in packs.items():
        for fam, best in pack["selected"].items():
            r = best["row"]
            rec = OrderedDict(landmark=int(L), family=fam, config=hp_label(best["hp"]),
                              epochs=int(best["epochs"]),
                              qwk=_f(r["qwk"]), qwk_ci=[_f(r.get("qwk_p2_5")), _f(r.get("qwk_p97_5"))],
                              auroc_macro=_f(r.get("auroc_macro")), hidden=best["hp"].get("hidden"),
                              d_model=best["hp"].get("d_model"), n_layers=best["hp"].get("n_layers"),
                              pe=best["hp"].get("pe"))
            sel.append(rec)
    vrows = []
    for _, r in val_ct.iterrows():
        vrows.append(OrderedDict(landmark=int(r["landmark"]), family=r["family"], contrast=r["contrast"],
                                 delta_qwk=_f(r["delta_qwk"]), delta_ci=[_f(r["delta_p2_5"]), _f(r["delta_p97_5"])],
                                 p_one_sided_le_0=_f(r["p_one_sided_le_0"]),
                                 gain_at_least_delta_C=bool(r["gain_at_least_delta_C"])))
    trows = []
    if te_tab is not None and len(te_tab):
        for _, r in te_tab[te_tab["split"] == "test"].iterrows():
            trows.append(OrderedDict(landmark=int(r["landmark"]), family=r["family"], config=str(r.get("config", "")),
                                     qwk=_f(r["qwk"]), qwk_ci=[_f(r.get("qwk_p2_5")), _f(r.get("qwk_p97_5"))],
                                     auroc_macro=_f(r.get("auroc_macro"))))
    env = cfg.env_info or {}
    return OrderedDict(
        script=SCRIPT_VERSION, script_sha256=script_sha, spec=SPEC_DOC, plan=PLAN, contract=CONTRACT,
        generated_utc=utc_now(), input_hashes=pre["hashes"], test_attempts_before=pre["prior_attempts"],
        decisions=DECISIONS, deviations=deviations_for(pre),
        configuration=OrderedDict(seed=R2.SEED, n_boot=cfg.n_boot, alpha=ALPHA,
                                  delta=R2.THRESH["delta"], delta_C=R2.THRESH["delta_C"],
                                  max_epochs=cfg.max_epochs, patience=cfg.patience, n_seeds=cfg.n_seeds,
                                  landmarks=list(cfg.landmarks), families=list(cfg.families),
                                  amp=bool(cfg.amp), max_seconds=cfg.max_seconds,
                                  torch_device=str(torch_device()) if R2 is not None else "not recorded",
                                  python=env.get("python"), torch=env.get("torch"),
                                  cuda_available=bool(env.get("cuda_available"))),
        reproduction=OrderedDict((k, _f(v)) for k, v in info.items() if k.startswith("repro")),
        selected=sel, validation_contrasts=vrows, test_scores=trows,
        acceptance_checks=OrderedDict((k, "pass" if v else "fail") for k, v in checks.items()),
        n_files=n_written, release_gate="ok",
        role="post-primary sensitivity analysis (desktop Tier 2 sequence models); a second look, not a replacement",
    )


def _ci(v):
    return "[" + ", ".join("nan" if x is None else f"{x:.4g}" for x in v) + "]"


def _n(x):
    return "nan" if x is None else f"{x:.4g}"


def audit_md(a: dict) -> str:
    L = ["# Stage 17 sequence-model sensitivity (desktop Tier 2) audit - TEAM", "",
         f"- script: {a['script']} (`{a['script_sha256']}`); spec {a['spec']}; plan {a['plan']}, contract {a['contract']}",
         f"- generated (UTC): {a['generated_utc']}",
         "- post-primary sensitivity analysis (a second look at the test set), not a replacement for the laptop "
         "Run 2 GRU or the laptop Run 3a primary scoring; test rows scored once on this desktop log; aggregate "
         "only: no patient, row, encounter or subject counts and no identifiers; the release gate re-read every "
         "TEAM file from disk",
         f"- earlier test-use attempts (failed, nothing published): {a['test_attempts_before']}", "",
         "## Input hashes", ""]
    L += [f"- {k}: `{v}`" for k, v in a["input_hashes"].items()]
    L += ["", "## Decisions applied", ""] + [f"- **{k}**: {v}" for k, v in a["decisions"].items()]
    L += ["", "## Deviations from the plan (logged)", ""] + [f"- {d}" for d in a["deviations"]]
    c = a["configuration"]
    L += ["", "## Configuration", "",
          f"- seed {c['seed']}; paired subject-cluster bootstrap B = {c['n_boot']}; alpha {c['alpha']}; "
          f"delta {c['delta']}, delta_C {c['delta_C']}; max epochs {c['max_epochs']}; patience {c['patience']}; "
          f"n_seeds {c['n_seeds']}; "
          f"{('no time limit' if not c.get('max_seconds') else 'time limit ' + str(c['max_seconds']) + ' s')}; "
          f"torch device {c.get('torch_device', 'not recorded')}; amp {c.get('amp')}",
          f"- core landmarks in this run: {c['landmarks']}; families {c['families']}",
          f"- python {c.get('python')}; torch {c.get('torch')}; "
          f"cuda: {'yes' if c.get('cuda_available') else 'no'}",
          "- this is a post-primary sensitivity analysis (a second look at the test set), not a replacement for "
          "the laptop primary results; outputs are under data/processed/ladder/DESKTOP_FULL_PLAN/stage17_seqmodels/",
          "",
          "## Frozen-model reproduction on validation (before any test row was read)", ""]
    L += [f"- {k}: {_n(v)}" for k, v in a["reproduction"].items()]
    L += ["", "## Selected configuration per family (validation QWK, official seed)", "",
          "| landmark | family | config | epochs | QWK [95% CI] | macro AUROC |",
          "|---|---|---|---|---|---|"]
    L += [f"| {r['landmark']} | {r['family']} | {r['config']} | {r['epochs']} | {_n(r['qwk'])} {_ci(r['qwk_ci'])} | "
          f"{_n(r['auroc_macro'])} |" for r in a["selected"]]
    L += ["", "## Validation contrasts (selected model minus frozen Run 2 GRU or A+B)", "",
          "| landmark | family | contrast | delta QWK [95% CI] | p (delta <= 0) | gain >= delta_C and CI > 0 |",
          "|---|---|---|---|---|---|"]
    L += [f"| {r['landmark']} | {r['family']} | {r['contrast']} | {_n(r['delta_qwk'])} {_ci(r['delta_ci'])} | "
          f"{_n(r['p_one_sided_le_0'])} | {r['gain_at_least_delta_C']} |" for r in a["validation_contrasts"]]
    L += ["", "## Test scores (post-primary sensitivity; descriptive)", "",
          "| landmark | family | config | QWK [95% CI] | macro AUROC |",
          "|---|---|---|---|---|"]
    L += [f"| {r['landmark']} | {r['family']} | {r['config']} | {_n(r['qwk'])} {_ci(r['qwk_ci'])} | "
          f"{_n(r['auroc_macro'])} |" for r in a["test_scores"]]
    L += ["", "## Acceptance checks", ""] + [f"- {k}: {v}" for k, v in a["acceptance_checks"].items()]
    L += ["", "Release gate: ok. Detail tables: stage17_seq_*_TEAM.csv; *_INTERNAL files are "
          "team-internal (never opened, pasted or shared).", ""]
    return "\n".join(L)


def publish(cfg, D, pre, T, internal, checks, info, packs, val_ct, te_tab, te_ct, ms_tab) -> dict:
    out = cfg.out_dir
    pend = out.parent / f"{out.name}.pending-{os.getpid()}"
    out.parent.mkdir(parents=True, exist_ok=True)
    pend.mkdir()
    try:
        script_sha = sha256_file(Path(__file__))
        files, team = OrderedDict(), []
        for nm, df in T.items():
            files[nm] = R2.write_bytes_atomic(pend / nm, R2.fmt_df(df))
            team.append(nm)
        sel_df = T.get("stage17_seq_selected_TEAM.csv", pd.DataFrame())
        tune_df = T.get("stage17_seq_tuning_TEAM.csv", pd.DataFrame())
        for nm, data in figures(tune_df, sel_df, val_ct).items():
            files[nm] = R2.write_bytes_atomic(pend / nm, data)
            team.append(nm)
        for nm, data in internal.items():
            if "_INTERNAL" not in nm:
                raise BuildError("an internal file name lacks the _INTERNAL marker")
            files[nm] = R2.write_bytes_atomic(pend / nm, data)
        env_nm = f"{OUT_NAME}_environment_INTERNAL.json"
        files[env_nm] = R2.write_bytes_atomic(pend / env_nm, environment_internal_bytes(cfg, pre))
        audit = make_audit(cfg, pre, checks, info, packs, val_ct, te_tab, te_ct, ms_tab, script_sha, len(files))
        if not audit_nums_ok(audit):
            raise BuildError("release gate: the shareable audit contains a number outside the structural whitelist "
                             "(keys: " + ", ".join(sorted(set(re.sub(r"\[\d+\]", "[]", k) for k in bad_num_keys(audit))))
                             + ")")
        ajn, amn = f"{OUT_NAME}_audit_TEAM.json", f"{OUT_NAME}_audit_TEAM.md"
        aj = json.dumps(audit, indent=1, ensure_ascii=False).encode("utf-8")
        files[ajn] = R2.write_bytes_atomic(pend / ajn, aj)
        files[amn] = R2.write_bytes_atomic(pend / amn, audit_md(audit).encode("utf-8"))
        team += [ajn, amn]
        back = json.loads((pend / ajn).read_text(encoding="utf-8"))
        if back != json.loads(aj.decode("utf-8")) or not audit_nums_ok(back):
            raise BuildError("release gate: disk reload of the audit differs")
        say(f"release gate: re-reading {len(team)} TEAM files from disk")
        release_gate(pend, team, sensitive_numbers(D), set(map(str, D["eid_all"])) | set(map(str, D["subj_all"])))
        say("release gate: ok")
        for nm_, h in files.items():
            if sha256_file(pend / nm_) != h:
                raise BuildError(f"release gate: {nm_} hash differs after writing")
        man = dict(script=SCRIPT_VERSION, script_sha256=script_sha, spec=SPEC_DOC, plan=PLAN, contract=CONTRACT,
                   generated_utc=utc_now(), input_hashes=pre["hashes"],
                   files={k: {"sha256": v, "bytes": (pend / k).stat().st_size} for k, v in files.items()},
                   team_files=team, internal_files=[k for k in files if k not in team],
                   acceptance_checks={k: ("pass" if v else "fail") for k, v in checks.items()}, release_gate="ok")
        R2.write_bytes_atomic(pend / f"{OUT_NAME}_manifest.json", json.dumps(man, indent=1).encode("utf-8"))
        if out.exists():
            R2.retry(lambda: out.rmdir(), "remove empty output folder")
        R2.retry(lambda: os.replace(pend, out), "rename pending folder to the output folder")
    except BaseException as e:
        try:
            if pend.exists():
                R2.retry(lambda: R2.rmtree_robust(pend), "remove pending folder")
        except errs("build"):
            e.leftover = pend.name  # type: ignore[attr-defined]
        raise
    say(f"published {len(files) + 1} files to {out.name}")
    return dict(mode="run", audit=audit, audit_md=audit_md(audit), n_files=len(files) + 1)



def build(cfg: Config) -> dict:
    if cfg.max_seconds:
        R2._STATE["deadline"] = time.time() + cfg.max_seconds
    say("precheck: hashes")
    pre = precheck(cfg)
    if cfg.precheck_only:
        return {"mode": "precheck-only", "hashes": pre["hashes"]}
    checks, info = OrderedDict(), OrderedDict()
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
    packs, tune_rows, val_crows, ms_rows = OrderedDict(), [], [], []
    all_states, all_preds = OrderedDict(), OrderedDict()
    for L in cfg.landmarks:
        say(f"tuning families {list(cfg.families)} at landmark {L} h (train/validation only)")
        pack = tune_landmark(D, mats, G, Vw, Xw, L, cfg, checks)
        packs[L] = pack
        tune_rows.extend(pack["tune_rows"])
        val_crows.extend(val_contrasts(L, pack["selected"], pack["frozen"], R2.THRESH["delta_C"]))
        ms_rows.extend(multiseed_selected(D, G, L, pack, cfg))
        all_states.update(pack["states"])
        for fam, best in pack["selected"].items():
            all_states[f"L{L}_{fam}_selected"] = best["state"]
        all_preds.update(pack["preds"])
        checks[f"r4_L{L}_prefix_bins"] = pack["frozen"]["S"].shape[1] == L // 6
    val_ct = pd.DataFrame(val_crows)
    sel_df = selected_table(packs)
    tune_df = pd.DataFrame(tune_rows)
    ms_tab = pd.DataFrame(ms_rows) if ms_rows else pd.DataFrame()
    checks["r4_no_test_rows_in_tuning"] = all("test" not in sp for _, _, sp in G.fits) and G.phase == "verify"
    checks["r4_all_requested_landmarks_tuned"] = set(packs) == set(cfg.landmarks)
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise BuildError("acceptance checks failed before any test row was read (test-use log unchanged): " +
                         "; ".join(failed))
    attempts = read_test_log(cfg.test_log)
    script_sha = sha256_file(Path(__file__))
    attempts.append(dict(script=SCRIPT_VERSION, script_sha256=script_sha, started_utc=utc_now(), status="started"))
    write_test_log(cfg.test_log, attempts)
    G.phase = "score"
    try:
        te_rows, te_crows, te_internal = [], [], OrderedDict()
        for L in cfg.landmarks:
            say(f"test scoring (post-primary sensitivity) at landmark {L} h")
            sc = score_test_landmark(D, G, L, packs[L], cfg)
            te_rows.extend(sc["rows"])
            te_crows.extend(sc["contrasts"])
            te_internal.update(sc["internal"])
        G.phase = "done"
        checks["r4_no_fit_on_test_rows"] = all("test" not in sp for _, _, sp in G.fits)
        checks["r4_fits_only_before_scoring"] = all(ph == "verify" for _, ph, _ in G.fits)
        n_expected = len(cfg.landmarks) * (2 + len(cfg.families))  # AB, GRU, each family; plus te() calls are not scored keys
        checks["r4_each_test_scoring_once"] = len(G.scored) == len(set(G.scored)) and len(G.scored) == n_expected
        te_tab = pd.DataFrame(te_rows)
        te_ct = pd.DataFrame(te_crows)
        failed = [k for k, v in checks.items() if not v]
        if failed:
            raise BuildError("acceptance checks failed: " + "; ".join(failed))
        T = OrderedDict()
        T["stage17_seq_tuning_TEAM.csv"] = tune_df
        T["stage17_seq_selected_TEAM.csv"] = sel_df
        T["stage17_seq_val_contrast_TEAM.csv"] = val_ct
        T["stage17_seq_test_TEAM.csv"] = te_tab
        T["stage17_seq_test_contrast_TEAM.csv"] = te_ct
        if len(ms_tab):
            T["stage17_seq_multiseed_TEAM.csv"] = ms_tab
        # INTERNAL: selected states + predictions (validation and test)
        torch = R2._torch()
        buf = io.BytesIO()
        torch.save({k: v for k, v in all_states.items()}, buf)
        internal = OrderedDict([
            ("stage17_seq_predictions_INTERNAL.npz", R2._npz_bytes({**all_preds, **te_internal})),
            ("stage17_seq_models_INTERNAL.pt", buf.getvalue()),
        ])
        res = publish(cfg, D, pre, T, internal, checks, info, packs, val_ct, te_tab, te_ct, ms_tab)
    except BaseException:
        attempts[-1]["status"] = "failed (nothing published)"
        attempts[-1]["ended_utc"] = utc_now()
        write_test_log(cfg.test_log, attempts)
        raise
    attempts[-1]["status"] = "published"
    attempts[-1]["ended_utc"] = utc_now()
    attempts[-1]["manifest_sha256"] = sha256_file(cfg.out_dir / f"{OUT_NAME}_manifest.json")
    write_test_log(cfg.test_log, attempts)
    return res


def describe_exception(e: BaseException, root, max_chain: int = 4) -> list:
    lines, seen, cur, depth = [], set(), e, 0
    while cur is not None and id(cur) not in seen and depth < max_chain:
        seen.add(id(cur))
        lines.append(("unexpected " if depth == 0 else "  caused by ") + type(cur).__name__)
        for frame, lineno in traceback.walk_tb(cur.__traceback__):
            co = frame.f_code
            base = os.path.basename(co.co_filename) or co.co_filename
            func = getattr(co, "co_qualname", co.co_name)
            mod = frame.f_globals.get("__name__", "?")
            ours = os.path.abspath(co.co_filename) in (os.path.abspath(__file__),
                                                       os.path.abspath(getattr(R2, "__file__", "") or "-"))
            lines.append(f"  frame {base}:{lineno}:{func}" + ("" if ours else f" [{mod}]"))
        if isinstance(cur, OSError) and (cur.filename is not None or cur.filename2 is not None) and R2 is not None:
            fn = R2.redact_path(cur.filename, root) if cur.filename is not None else "-"
            lines.append(f"  OSError errno {cur.errno} winerror {getattr(cur, 'winerror', None)}; file {fn}")
        cur = cur.__cause__ or cur.__context__
        depth += 1
    return lines


def build_config(argv=None) -> Config:
    ap = argparse.ArgumentParser(description=SCRIPT_VERSION)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--run1-dir", default=None)
    ap.add_argument("--run2-dir", default=None)
    ap.add_argument("--run2-script", default=None)
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
    ap.add_argument("--families", default=None, help="comma-separated subset of lstm,lstm_attn,transformer")
    ap.add_argument("--amp", action="store_true", help="mixed precision; ignored unless the device is CUDA")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=30.0)
    a = ap.parse_args(argv)
    if a.n_seeds < 1:
        raise InputError("--n-seeds must be at least 1")
    if a.max_epochs < 1 or a.patience < 1 or a.n_boot < 1 or a.batch < 1:
        raise InputError("a run setting is out of range")
    root = Path(a.project_root)
    lad = root / "data" / "processed" / "ladder"
    here = Path(__file__).resolve().parent
    r2s = Path(a.run2_script) if a.run2_script else here.parent / "stage11_18_r2" / f"{RUN2_MODULE}.py"
    work = work_root(root)
    cfg = Config(
        project_root=root,
        in_dir=Path(a.in_dir) if a.in_dir else lad / "stage06_08",
        run1_dir=Path(a.run1_dir) if a.run1_dir else lad / "stage09_10_r1",
        run2_dir=Path(a.run2_dir) if a.run2_dir else lad / RUN2_NAME,
        run2_script=r2s,
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
        families=parse_families(a.families), amp=bool(a.amp), require_cuda=bool(a.require_cuda),
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
    print("result: RUN stage17_seqmodels desktop (Tier 2) OK; RELEASE GATE OK; post-primary sensitivity analysis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
