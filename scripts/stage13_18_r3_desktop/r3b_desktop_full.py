"""Run 3b of stages 09-18 (PIC Foundations, PIC; plan v13, contract v13): attribution and the Layer C tensor
analysis, on train and validation rows only. Stage 13 (attribution stability of the ordinal LR and the tabular
winner, 24 h landmark rows, Outcome 1 with Outcome 2 secondary: subject-bootstrap refits of train, Frank-Hall SHAP of
E[Y | x] on a fixed seeded validation sample, Kendall tau, within-group credit split, interventional vs conditional
SHAP gap; H7), stage 18 sequence attribution (the D19 GRU A+B+C at the later core landmarks: expected-gradients
SHAP over subject-bootstrap refits, Kendall tau at three granularities, permutation cross-check, TreeSHAP of the
frozen A+B landmark models) and stage 16 (Layer C tensor SVD: HOSVD mode spectra with nulls, Tucker by HOOI, CP-ALS
with CORCONDIA and congruence, validation projection, principal angles across landmarks, organ-block alignment,
the labeled flattened baseline).

r3b_desktop_full v2.0.2-desktop (Nov 21, 2026). This is the desktop full-settings variant of Run 3b: stages 13, 16
and 18 run together as one run (no --part split, no 60-minute cap), with the plan D20 refit counts restored
(stage 13 GB B = 50, stage 18 GRU B = 20) at all seven core landmarks. All files this script writes go under
data/processed/ladder/DESKTOP_FULL_PLAN/ (published folder, pending folder, cache, internal files and
README_DESKTOP.txt).

Plan v13 section 11 and section 6.7 place attribution on a fixed validation sample with refits on train; no test row
is read, so Run 3b needs no Run 3a output. Test rows are dropped when the inputs are loaded (only their split labels
and outcome labels are kept, for the release gate's list of sizes that must never be printed).

Frozen inputs (hash-gated before anything is parsed): everything Run 2 checked (Stage 06-08 manifest and files, Run 1
v1.0.2 manifest and files, canonical, harmonization, split, raw ICUSTAYS / ADMISSIONS; the same pins), the Run 2
script (scripts/stage11_18_r2/r2_11p_11_12_17_models.py, imported only after its SHA-256 matches the pin), and the
Run 2 manifest (pinned by its full SHA-256 on the command line and in this script) with every file it lists.

Outputs (data/processed/ladder/DESKTOP_FULL_PLAN/stage13_16_18_r3b/, published atomically from a pending folder in
the same DESKTOP_FULL_PLAN directory): TEAM files (aggregate
only: importance by column name, channel, bin or organ block; no per-row SHAP and no patient loadings) and one
INTERNAL file (per-refit importance vectors, train factor matrices and the validation-sample encounter ids; never
opened, pasted or shared). Release gate as in Run 2.
Exit codes: 0 ok; 2 stopped on an input check (nothing written); 3 build or release-gate failure (nothing published).
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

SCRIPT_VERSION = "r3b_desktop_full v2.0.2-desktop"
SPEC_DOC = "stage13_16_18_r3b_desktop_spec v2.0.2-desktop"
PLAN, CONTRACT = "v13", "v13"
OUT_NAME = "stage13_16_18_r3b"
DESKTOP_ROOT_NAME = "DESKTOP_FULL_PLAN"
README_NAME = "README_DESKTOP.txt"
README_TEXT = (
    "These are post-primary sensitivity outputs from the full-settings desktop run "
    "(v2.0.2-desktop). They are a second look, not a replacement for the laptop "
    "primary Run 3a/3b results. Published folders, pending folders, caches, "
    "internal files and logs for the desktop run belong only in this folder.\n"
)
RUN2_NAME = "stage11_12_17_r2"
RUN2_MODULE = "r2_11p_11_12_17_models"
CACHE_NAME = f"_cache_{OUT_NAME}"
INTERNAL_NAME = f"{OUT_NAME}_importances_factors_INTERNAL.npz"


def stages() -> tuple:
    return (13, 16, 18)

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
RUN2_TEAM_USED = ("stage11_validation_scores_TEAM.csv", "stage11_winner_TEAM.csv")
H9_FOCUS = (72, 120, 168, 240, 336)       # Vinai's framing (Nov 19): day 3, 5, 7, 10 and 14 lead every Layer C table
R2 = None                                # the hash-verified Run 2 module (set by load_run2)

# ---- frozen Run 3b budgets and thresholds (D20; plan 11, 6.7, 12.2 H7, 12.3; D18)
S13_B = 50                               # SHAP refits, stage 13 ordinal LR (D20; plan 200)
S13_TREE_B = 50                          # SHAP refits, stage 13 gradient boosting (D20)
S18_B = 20                               # neural attribution refits, stage 18 (D20)
S13_SAMPLE = 500                         # fixed seeded validation sample for global importance (plan 11 [sample size])
S13_GAP_SAMPLE = 200                     # rows of that sample used for the interventional vs conditional gap
S13_BACKGROUND = 50                      # train background rows for interventional TreeSHAP
S13_LR_MATRICES = ("a0", "a1", "a2", "b0", "b1", "b2s", "b2d", "b2p")   # + the winner's matrix
S13_TREE_LEVELS = ("b1", "b2d")          # + the winner's matrix; b2s is read off b2d when the fits coincide
S13_O2_LEVELS = ("b1", "b2s", "b2d")     # Outcome 2 (secondary): ordinal LR only
H7_FALL, H7_PRUNED = 0.10, 0.05          # plan 12.2 H7 bracketed values
S18_LANDMARKS = (24, 48, 72, 120, 168, 240, 336)  # all seven core landmarks (D20)
S18_SAMPLE = 300                         # validation rows per landmark for expected gradients
S18_BACKGROUND = 100                     # train background rows (baselines) per landmark
S18_EG_DRAWS = 16                        # baseline-and-alpha draws per explained row
S18_TOP = 10
S16_LANDMARKS = (24, 48, 72, 120, 168, 240, 336)   # core landmarks (all 14 for the angle stability table)
S16_R1 = (5, 10, 20, 40)
S16_CP_R = tuple(range(1, 16))
S16_CP_STARTS = 10                       # random starts (+ the HOSVD start)
S16_BOOT = 50                            # subject bootstraps for CP congruence (plan 12.3)
S16_NULL_DRAWS = 20
S16_CONGRUENCE, S16_CORCONDIA = 0.85, 80.0          # D18 frozen thresholds
S16_ANGLE_R2, S16_ANGLE_R3 = 3, 10
S16_TOPLOAD = 5
S16_FLAT_K = 20


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
    install_device_gru(R2)
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


def must_write_under_desktop(path: Path, project_root: Path, role: str, *, allow_root: bool = False) -> None:
    """Refuse a write path that would land outside data/processed/ladder/DESKTOP_FULL_PLAN/."""
    base = desktop_root(project_root).resolve()
    p = path.resolve()
    if p == base:
        if allow_root:
            return
        raise InputError(f"{role} must be a subfolder of data/processed/ladder/{DESKTOP_ROOT_NAME}")
    try:
        p.relative_to(base)
    except ValueError:
        raise InputError(f"{role} must be under data/processed/ladder/{DESKTOP_ROOT_NAME}") from None


def write_desktop_readme(project_root: Path) -> None:
    d = desktop_root(project_root)
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


def torch_device():
    """CUDA if available, otherwise CPU. Logged once; seeds stay as deterministic as the device allows."""
    torch = R2._torch()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not getattr(torch_device, "_logged", False):
        say(f"torch device: {dev}")
        torch_device._logged = True
        if dev.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return dev


def verify_device():
    """CPU device for frozen-model reproduction forward passes, matching the laptop Run 2, even when CUDA is available.

    Training and expected-gradients attribution may use torch_device() (CUDA if available). Tensor dtype at the
    call site stays float32.
    """
    torch = R2._torch()
    return torch.device("cpu")


def _fmt_repro(x) -> str:
    """Aggregate float for progress and failure lines (no counts)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(v):
        return "nan"
    return f"{v:.6g}"


def repro_fail_detail(failed, info: dict, cfg) -> str:
    """Failed check names plus the measured max differences and tolerances that apply."""
    bits = []
    for k in failed:
        extra = ""
        if k == "13_frozen_winner_reproduces_run2_validation":
            extra = (f" (max abs prob diff={_fmt_repro(info.get('repro_winner_max_abs_prob_diff'))}, "
                     f"tol={_fmt_repro(cfg.repro_tol_prob)})")
        elif k == "18_frozen_abc_reproduces_run2_validation":
            extra = (f" (max abs prob diff={_fmt_repro(info.get('repro_s18_abc_max_abs_prob_diff'))}, "
                     f"tol={_fmt_repro(cfg.repro_tol_prob)})")
        bits.append(k + extra)
    return "; ".join(bits)


def seed_torch(torch, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def install_device_gru(mod) -> None:
    """Replace the hash-pinned Run 2 GRU trainer with a device-aware copy (the Run 2 file is never edited)."""
    if getattr(mod, "_desktop_device_gru", False):
        return

    def gru_fit_predict(seq_tr, st_tr, off_tr, y_tr, g_tr, seq_va, st_va, off_va, K, hidden, seed, gamma0=None,
                        max_epochs=40, patience=5, batch=256, lr=2e-3, wd=1e-4):
        torch = mod._torch()
        seed_torch(torch, seed)
        dev = torch_device()
        rng = np.random.default_rng(seed)
        Net = mod.make_gru_class()
        us = np.unique(g_tr)
        hold = set(rng.choice(us, size=max(1, int(round(0.1 * len(us)))), replace=False).tolist())
        hm = np.array([g in hold for g in g_tr])
        fit_i, hold_i = np.flatnonzero(~hm), np.flatnonzero(hm)
        T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
        xs, xh = T(seq_tr), T(seq_va)
        st = T(st_tr) if st_tr is not None else torch.zeros((len(y_tr), 0), device=dev)
        sh = T(st_va) if st_va is not None else torch.zeros((seq_va.shape[0], 0), device=dev)
        ot, oh = T(off_tr), T(off_va)
        Y = torch.tensor((y_tr[:, None] > np.arange(K - 1)[None, :]).astype(np.float32), device=dev)
        net = Net(seq_tr.shape[2], int(st.shape[1]), hidden, K, gamma0=gamma0).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        bce = torch.nn.BCEWithLogitsLoss()

        def hold_loss():
            net.eval()
            with torch.no_grad():
                z = net(xs[hold_i], st[hold_i], ot[hold_i])
                return float(bce(-z, Y[hold_i]))

        best, best_state, bad, ep = math.inf, None, 0, 0
        for ep in range(1, max_epochs + 1):
            net.train()
            perm = rng.permutation(fit_i)
            for i in range(0, len(perm), batch):
                b = perm[i:i + batch]
                opt.zero_grad()
                loss = bce(-net(xs[b], st[b], ot[b]), Y[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
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
            F = torch.sigmoid(net(xh, sh, oh)).cpu().numpy().astype(np.float64)
        F = np.maximum.accumulate(F, axis=1)
        state = dict(state_dict=best_state, d_seq=int(seq_tr.shape[2]), d_static=int(st.shape[1]), hidden=hidden, K=K)
        return state, mod.cum_to_probs(F), ep

    mod.gru_fit_predict = gru_fit_predict
    mod._desktop_device_gru = True



class R2Unpickler(pickle.Unpickler):
    """Unpickler for the hash-verified Run 2 model files: Run 2 classes (pickled as __main__ when Run 2 ran as a
    script, or under the module name in the tests) resolve to the verified Run 2 module; otherwise only numpy,
    scikit-learn, scipy and the standard containers."""
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
            # scikit-learn's Cython loss objects pickle under their extension-module name "_loss"
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


# ------------------------------------------------------------------------------------------------------- config
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
    out_dir: Path
    expect_manifest_sha256: str | None
    expect_run1_manifest_sha256: str | None
    expect_run2_manifest_sha256: str | None
    precheck_only: bool
    gru_max_epochs: int = 40
    max_seconds: float = 0.0
    heartbeat_seconds: float = 30.0
    run2_script_sha: str = ""
    s13_refits: int = S13_B
    s13_tree_refits: int = S13_TREE_B
    s18_refits: int = S18_B
    s16_boot: int = S16_BOOT
    s16_cp_starts: int = S16_CP_STARTS
    s16_landmarks: tuple = S16_LANDMARKS
    repro_tol_prob: float = 1e-4


def r2_config(cfg: Config):
    return R2.Config(project_root=cfg.project_root, in_dir=cfg.in_dir, run1_dir=cfg.run1_dir,
                     canonical=cfg.canonical, harmonization=cfg.harmonization, split_file=cfg.split_file,
                     raw_dir=cfg.raw_dir, out_dir=cfg.out_dir, expect_manifest_sha256=cfg.expect_manifest_sha256,
                     expect_run1_manifest_sha256=cfg.expect_run1_manifest_sha256, precheck_only=True,
                     n_boot=1000, gru_max_epochs=cfg.gru_max_epochs, max_seconds=cfg.max_seconds,
                     heartbeat_seconds=cfg.heartbeat_seconds)


def precheck(cfg: Config) -> dict:
    """Run 2's hash gates (same pins), then the Run 2 manifest (pinned) and every file it lists; the output folder
    must be empty and no pending folder may exist. No Run 3a file is read."""
    for p in (cfg.run2_dir, cfg.out_dir, cfg.split_file, cfg.raw_dir, cfg.run2_script):
        if R2.NEVER_READ_TOKEN in str(p):
            raise InputError(f"a path containing {R2.NEVER_READ_TOKEN} is never read or written")
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
        p = cfg.run2_dir / name
        if not p.is_file() or sha256_file(p) != rec.get("sha256"):
            raise InputError(f"Run 2 file {name} is missing or its SHA-256 differs from the Run 2 manifest")
    hashes["stage11_12_17_r2_manifest.json"] = s2
    for n in RUN2_INTERNAL:
        hashes[n] = f2[n]["sha256"]
    hashes["run2_script"] = cfg.run2_script_sha
    return {"hashes": hashes, "run2_manifest": m2}


# ------------------------------------------------------------------------------------------------------- loading
def load_all(cfg: Config, checks: dict) -> dict:
    """Stage 08 levels on the 24 h landmark rows of train and validation (test rows are dropped here), the Run 1
    bases, the Layer C index of every landmark (train and validation rows), the Run 2 labels, subjects (canonical
    file) and frozen artefacts. Split and outcome labels of every row are kept for the release gate only."""
    d = cfg.in_dir
    lv = OrderedDict()
    eid0 = split0 = elig24 = keep = None
    for k in R2.LEVELS:
        z = R2.read_npz(d / f"stage08_{k}.npz", ("encounter_id", "split", "eligible_24h", "X_z", "columns"))
        eid, split = z["encounter_id"].astype(str), z["split"].astype(str)
        if eid0 is None:
            eid0, split0, elig24 = eid, split, z["eligible_24h"].astype(bool)
            keep = split0[elig24] != "test"
        elif not (np.array_equal(eid, eid0) and np.array_equal(split, split0)):
            raise InputError(f"level {k}: encounter order or split differs from level a0")
        X = z["X_z"][elig24][keep].astype(np.float64)
        if not np.isfinite(X).all():
            raise InputError(f"level {k}: non-finite values in X_z")
        lv[k] = dict(X=X, cols=[str(c) for c in z["columns"]])
    if not set(np.unique(split0)) <= {"train", "validation", "test"}:
        raise InputError("unexpected split labels")
    rows = OrderedDict(eid=eid0[elig24][keep], split=split0[elig24][keep])
    emap = {e: i for i, e in enumerate(rows["eid"])}
    lab = R2.read_npz(cfg.run2_dir / RUN2_INTERNAL[0],
                      ["encounter_id", "split", "died", "outcome2", "ordinal8"] +
                      [f"outcome1_L{L}" for L in R2.LANDMARKS])
    checks["r3b_labels_match_stage08_order_and_split"] = bool(
        np.array_equal(lab["encounter_id"].astype(str), eid0) and np.array_equal(lab["split"].astype(str), split0))
    if not checks["r3b_labels_match_stage08_order_and_split"]:
        raise InputError("Run 2 labels: encounter order or split differs from Stage 08")
    can = R2.read_csv_str(cfg.canonical, usecols=["encounter_id", "SUBJECT_ID"])
    can["encounter_id"] = can["encounter_id"].str.strip()
    if can["encounter_id"].duplicated().any() or set(can["encounter_id"]) != set(eid0):
        raise InputError("the Stage 08 encounters differ from the canonical encounter set (count withheld)")
    subj_all = can.set_index("encounter_id").loc[eid0]["SUBJECT_ID"].map(R2.norm_id).to_numpy().astype(str)
    spl = R2.read_csv_str(cfg.split_file, usecols=["SUBJECT_ID", "split"])
    smap = dict(zip(spl["SUBJECT_ID"].map(R2.norm_id),
                    spl["split"].map(lambda v: R2.SPLIT_LABELS.get(str(v).strip().lower(), "?"))))
    checks["r3b_split_matches_split_file"] = bool(np.array_equal(np.array([smap.get(s, "?") for s in subj_all]),
                                                                 split0))
    if not checks["r3b_split_matches_split_file"]:
        raise InputError("Stage 08 split differs from the persisted subject split (count withheld)")
    sel = np.flatnonzero(elig24)[keep]
    rows["subj"] = subj_all[sel]
    rows["died"] = lab["died"].astype(bool)[sel]
    rows["O1"] = OrderedDict((L, lab[f"outcome1_L{L}"].astype(np.int64)[sel]) for L in R2.LANDMARKS)
    rows["O2"] = lab["outcome2"].astype(np.int64)[sel]
    lc = OrderedDict()
    elig_all = OrderedDict()
    ok_el = True
    for L in R2.LANDMARKS:
        z = R2.read_npz(d / f"stage08_layer_c_L{L}.npz", ("encounter_id", "split", "channels"))
        e, s = z["encounter_id"].astype(str), z["split"].astype(str)
        m = s != "test"
        idx = np.array([emap.get(x, -1) for x in e[m]])
        if (idx < 0).any() or not np.array_equal(rows["split"][idx], s[m]):
            raise InputError(f"Layer C L{L}: encounters or split do not match the 24 h tabular rows")
        lc[L] = dict(idx=idx, mask=m, channels=[str(c) for c in z["channels"]], path=d / f"stage08_layer_c_L{L}.npz")
        el = np.isin(eid0, e)
        elig_all[L] = el
        ok_el &= bool(np.array_equal(el, lab[f"outcome1_L{L}"] >= 0))
    checks["r3b_eligible_sets_match_labels"] = ok_el
    checks["r3b_no_test_rows_loaded"] = bool((rows["split"] != "test").all() and
                                             all((rows["split"][r["idx"]] != "test").all() for r in lc.values()))
    bz = np.load(cfg.run1_dir / "stage10_matched_width_bases_INTERNAL.npz", allow_pickle=False)
    with bz:
        bases = {k: np.array(bz[k]) for k in bz.files}
    s12 = load_pickle(cfg.run2_dir / RUN2_INTERNAL[2])
    tab = load_pickle(cfg.run2_dir / RUN2_INTERNAL[4])
    torch = R2._torch()
    states = torch.load(io.BytesIO((cfg.run2_dir / RUN2_INTERNAL[3]).read_bytes()), weights_only=True)
    with np.load(cfg.run2_dir / RUN2_INTERNAL[1], allow_pickle=False) as z:
        vpred = {k: np.array(z[k]) for k in z.files}
    team2 = {n: pd.read_csv(cfg.run2_dir / n, dtype=str, keep_default_na=False) for n in RUN2_TEAM_USED}
    a2 = json.loads((cfg.run2_dir / RUN2_AUDIT).read_text(encoding="utf-8"))
    r2cfg = a2.get("config") or {}
    if not isinstance(r2cfg.get("gru_max_epochs"), int) or r2cfg.get("seed") != R2.SEED:
        raise InputError("Run 2 audit lacks its GRU configuration (gru_max_epochs, seed)")
    if s12["winner"]["family"] not in ("GB", "LR"):
        raise InputError("Run 3b attribution supports a GB or LR stage 11 winner only")
    return dict(r2cfg=r2cfg, levels=lv, rows=rows, eid_all=eid0, split_all=split0, elig24_all=elig24,
                subj_all=subj_all, lab_all=lab, elig_all=elig_all, lc=lc, bases=bases, s12=s12, tab=tab,
                states=states, vpred=vpred, team2=team2)


class FitLog:
    """Registers the splits of every fit's rows and of every explained sample; any test row fails closed."""

    def __init__(self, split: np.ndarray):
        self.split, self.fits = split, []

    def rows(self, idx: np.ndarray, what: str) -> np.ndarray:
        sp = tuple(sorted(set(np.unique(self.split[idx]).tolist())))
        self.fits.append((what, sp))
        if "test" in sp:
            raise BuildError("a fit or attribution sample was given test rows")
        return idx


def boot_rows(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Subject bootstrap: draw subjects with replacement and take all rows of each drawn subject (positions)."""
    us, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    starts = np.searchsorted(inv[order], np.arange(len(us)))
    ends = np.r_[starts[1:], len(order)]
    pick = rng.integers(0, len(us), len(us))
    return np.concatenate([order[starts[u]:ends[u]] for u in pick])


def kendall_pairs(M: np.ndarray, max_pairs: int = 0, rng=None) -> np.ndarray:
    """Kendall tau-b between the importance vectors (rows of M) of every pair of refits."""
    from scipy.stats import kendalltau
    B = M.shape[0]
    pairs = [(i, j) for i in range(B) for j in range(i + 1, B)]
    if max_pairs and len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    out = []
    for i, j in pairs:
        if np.ptp(M[i]) == 0 or np.ptp(M[j]) == 0:
            out.append(np.nan)
            continue
        out.append(kendalltau(M[i], M[j]).statistic)
    return np.array(out, dtype=float)


def tau_summary(t: np.ndarray) -> OrderedDict:
    t = t[np.isfinite(t)]
    if not t.size:
        return OrderedDict(tau_median=np.nan, tau_q25=np.nan, tau_q75=np.nan)
    return OrderedDict(tau_median=float(np.median(t)), tau_q25=float(np.percentile(t, 25)),
                       tau_q75=float(np.percentile(t, 75)))
# ------------------------------------------------------------------------------------- home-made SHAP (no shap)
def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _secant(f_x, f_0, m_x, m_0, slope0):
    """Rescale factor (f(x) - f(base)) / (m(x) - m(base)) of a monotone link, the derivative at the base when the
    two margins coincide (keeps SHAP additive on the outcome scale; DeepLIFT rescale rule)."""
    d = m_x - m_0
    small = np.abs(d) < 1e-9
    return np.where(small, slope0, (f_x - f_0) / np.where(small, 1.0, d))


def linear_shap_ey(model, X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Ordinal LR (cumulative logit, P(Y > j | x) = sigmoid(x w - theta_j)): interventional SHAP of the linear
    predictor against the background mean mu (exact for a linear function), carried to the target
    E[Y | x] = sum_j P(Y > j | x) by the rescale rule. Returns n x p; rows sum to E[Y|x] - E[Y|mu]."""
    w, th = np.asarray(model.w, dtype=np.float64), np.asarray(model.theta, dtype=np.float64)
    eta = X @ w
    eta0 = float(mu @ w)
    ey = _sig(eta[:, None] - th[None, :]).sum(1)
    ey0 = float(_sig(eta0 - th).sum())
    s0 = float((_sig(eta0 - th) * (1 - _sig(eta0 - th))).sum())
    r = _secant(ey, ey0, eta, eta0, s0)
    return (X - mu[None, :]) * w[None, :] * r[:, None]


def _hist_leaves(pred) -> list:
    """Leaves of one scikit-learn HistGradientBoosting tree: (value, [(feature, lo, hi, cover ratio)]) with the
    conditions on one feature merged into the interval lo < x <= hi and the cover ratios multiplied."""
    nd = pred.nodes
    out = []
    stack = [(0, {})]
    while stack:
        i, path = stack.pop()
        if nd["is_leaf"][i]:
            out.append((float(nd["value"][i]), path))
            continue
        if nd["is_categorical"][i]:
            raise BuildError("categorical tree splits are not supported by the stage 13 TreeSHAP")
        f, t = int(nd["feature_idx"][i]), float(nd["num_threshold"][i])
        c = float(nd["count"][i])
        for ch, go_left in ((int(nd["left"][i]), True), (int(nd["right"][i]), False)):
            lo, hi, r = path.get(f, (-np.inf, np.inf, 1.0))
            lo2, hi2 = (lo, min(hi, t)) if go_left else (max(lo, t), hi)
            p2 = dict(path)
            p2[f] = (lo2, hi2, r * (float(nd["count"][ch]) / c if c > 0 else 0.0))
            stack.append((ch, p2))
    return out


class TreeTable:
    """All leaves of a Frank-Hall gradient-boosting model, grouped by the number m of distinct features on the
    path: per group feature (L x m), lo, hi, cover ratio R, leaf value v and threshold index j."""

    def __init__(self, fh, p: int):
        self.J, self.p = fh.K - 1, p
        self.base = np.zeros(self.J)
        self.const = np.zeros(self.J, dtype=bool)
        groups = {}
        for j, m in enumerate(fh.models):
            if isinstance(m, float):
                self.const[j] = True
                continue
            if not hasattr(m, "_predictors"):
                raise BuildError("stage 13 TreeSHAP supports the HistGradientBoosting Frank-Hall model only")
            self.base[j] = float(np.ravel(m._baseline_prediction)[0])
            for it in m._predictors:
                for v, path in _hist_leaves(it[0]):
                    if not path:
                        self.base[j] += v
                        continue
                    items = sorted(path.items())
                    groups.setdefault(len(items), []).append((j, v, items))
        self.groups = {}
        for m, lst in groups.items():
            self.groups[m] = dict(
                j=np.array([a for a, _, _ in lst], dtype=np.int64),
                v=np.array([b for _, b, _ in lst]),
                f=np.array([[it[0] for it in c] for _, _, c in lst], dtype=np.int64),
                lo=np.array([[it[1][0] for it in c] for _, _, c in lst]),
                hi=np.array([[it[1][1] for it in c] for _, _, c in lst]),
                R=np.array([[it[1][2] for it in c] for _, _, c in lst]))
        # path-dependent expectation of each margin (cover-weighted)
        self.mean_margin = self.base.copy()
        for g in self.groups.values():
            np.add.at(self.mean_margin, g["j"], g["v"] * g["R"].prod(1))

    def margins(self, X: np.ndarray) -> np.ndarray:
        M = np.repeat(self.base[None, :], X.shape[0], 0)
        for g in self.groups.values():
            for s in range(0, len(g["v"]), 2000):
                sl = slice(s, s + 2000)
                A = ((X[:, g["f"][sl]] > g["lo"][sl]) & (X[:, g["f"][sl]] <= g["hi"][sl])).all(2)
                M += (A * g["v"][sl][None, :]) @ _onehot(g["j"][sl], self.J)
        return M


def _onehot(idx: np.ndarray, width: int):
    from scipy import sparse
    return sparse.csr_matrix((np.ones(len(idx)), (np.arange(len(idx)), idx)), shape=(len(idx), width))


def tree_shap_margins(T: TreeTable, X: np.ndarray, chunk_cells: int = 4_000_000) -> np.ndarray:
    """Path-dependent (conditional) TreeSHAP of every threshold margin, exact: each leaf is the product game
    v * prod_d (d in S ? 1[x_d in interval] : cover ratio r_d), whose Shapley value for feature k is
    (a_k - r_k) v sum_t w_t e_t(other factors) = (a_k - r_k) v int_0^1 prod_{e != k} (r_e (1 - z) + a_e z) dz
    (the Shapley weights are Beta integrals), evaluated exactly by Gauss-Legendre quadrature with ceil(m / 2)
    nodes, summed over leaves. Returns J x n x p (margin scale)."""
    n, J, p = X.shape[0], T.J, T.p
    out = np.zeros((n, J * p))
    for m, g in T.groups.items():
        Q = max(1, (m + 1) // 2)
        zx, zw = np.polynomial.legendre.leggauss(Q)
        z, wq = (zx + 1.0) / 2.0, zw / 2.0
        step = max(1, chunk_cells // max(1, n * m * Q))
        for s in range(0, len(g["v"]), step):
            sl = slice(s, s + step)
            f, R, v, jj = g["f"][sl], g["R"][sl], g["v"][sl], g["j"][sl]
            A = ((X[:, f] > g["lo"][sl]) & (X[:, f] <= g["hi"][sl])).astype(np.float64)   # n x L x m
            F = R[None, :, :, None] * (1.0 - z) + A[..., None] * z                      # n x L x m x Q
            P = F.prod(2)                                                                # n x L x Q
            integ = ((P[:, :, None, :] / F) * wq).sum(3)                                 # n x L x m
            phi = (A - R[None]) * v[None, :, None] * integ
            for k in range(m):
                out += phi[:, :, k] @ _onehot(jj * p + f[:, k], J * p)
    return out.reshape(n, J, p).transpose(1, 0, 2)


_IC_CACHE = {}


def _int_coefs(M: int) -> tuple:
    if M not in _IC_CACHE:
        cp, cn = np.zeros((M + 1, M + 1)), np.zeros((M + 1, M + 1))
        for P in range(M + 1):
            for N in range(M + 1 - P):
                if P >= 1:
                    cp[P, N] = math.factorial(P - 1) * math.factorial(N) / math.factorial(P + N)
                if N >= 1:
                    cn[P, N] = math.factorial(P) * math.factorial(N - 1) / math.factorial(P + N)
        _IC_CACHE[M] = (cp, cn)
    return _IC_CACHE[M]


def tree_shap_interventional(T: TreeTable, X: np.ndarray, Z: np.ndarray, chunk_cells: int = 8_000_000) -> np.ndarray:
    """Interventional TreeSHAP against the background rows Z (exact Shapley value of f(x_S, z_~S), averaged over
    z): per leaf and (x, z) pair the game is a product of 0/1 factors, with the closed form
    +v (P-1)! N! / (P+N)! for the P features that only x satisfies and -v P! (N-1)! / (P+N)! for the N features
    that only z satisfies (zero when a feature fails for both). Returns J x n x p (margin scale)."""
    n, nz, J, p = X.shape[0], Z.shape[0], T.J, T.p
    out = np.zeros((n, J * p))
    for m, g in T.groups.items():
        cp, cn = _int_coefs(m)
        step = max(1, chunk_cells // max(1, n * nz))
        for s in range(0, len(g["v"]), step):
            sl = slice(s, s + step)
            f, v, jj = g["f"][sl], g["v"][sl], g["j"][sl]
            lo, hi = g["lo"][sl], g["hi"][sl]
            A = ((X[:, f] > lo) & (X[:, f] <= hi)).astype(np.float64)      # n x L x m
            B = ((Z[:, f] > lo) & (Z[:, f] <= hi)).astype(np.float64)      # nz x L x m
            P = np.einsum("nlm,zlm->nzl", A, 1 - B)
            N = np.einsum("nlm,zlm->nzl", 1 - A, B)
            dead = np.einsum("nlm,zlm->nzl", 1 - A, 1 - B) > 0
            Pi, Ni = P.astype(np.int64), N.astype(np.int64)
            WP = np.where(dead, 0.0, cp[Pi, Ni])
            WN = np.where(dead, 0.0, cn[Pi, Ni])
            SP = np.einsum("nzl,zlm->nlm", WP, 1 - B)
            SN = np.einsum("nzl,zlm->nlm", WN, B)
            phi = (A * SP - (1 - A) * SN) * (v / nz)[None, :, None]
            for k in range(m):
                out += phi[:, :, k] @ _onehot(jj * p + f[:, k], J * p)
    return out.reshape(n, J, p).transpose(1, 0, 2)


def fh_shap_ey(T: TreeTable, X: np.ndarray, Z: np.ndarray | None = None) -> tuple:
    """Frank-Hall SHAP of E[Y | x] = sum_j sigmoid(margin_j(x)) (invariant to the monotone rearrangement):
    per-threshold margin SHAP (path-dependent, or interventional against Z) carried to the probability scale by
    the rescale rule and summed over thresholds. Returns (n x p SHAP, additivity residual max)."""
    Mx = T.margins(X)
    if Z is None:
        S = tree_shap_margins(T, X)
        m0 = np.repeat(T.mean_margin[None, :], X.shape[0], 0)
    else:
        S = tree_shap_interventional(T, X, Z)
        m0 = np.repeat(T.margins(Z).mean(0)[None, :], X.shape[0], 0)
    resid = float(np.abs(S.sum(2).T - (Mx - m0))[:, ~T.const].max()) if (~T.const).any() else 0.0
    out = np.zeros((X.shape[0], T.p))
    for j in range(T.J):
        if T.const[j]:
            continue
        s0 = _sig(m0[:, j]) * (1 - _sig(m0[:, j]))
        r = _secant(_sig(Mx[:, j]), _sig(m0[:, j]), Mx[:, j], m0[:, j], s0)
        out += S[j] * r[:, None]
    return out, resid
# ---------------------------------------------------------------------------------- stage 13: attribution stability
ORGANS = ("cardiovascular", "electrolytes", "endocrine", "gi", "hematologic", "hepatic", "infectious", "neurologic",
          "renal", "respiratory")
_SUFFIXES = ("_agez_missing", "_zsd_missing", "_ratio_podium_missing", "_first_ratio_podium", "_worst_ratio_podium",
             "_input_missing", "_median_z", "_min_z", "_max_z", "_first_z", "_worst_z", "_last_z", "_z_flag", "_zsd",
             "_missing", "_measured", "_median", "_min", "_max", "_range", "_first", "_worst", "_last")
_VALUE_SUFFIXES = {"_median_z", "_min_z", "_max_z", "_first_z", "_worst_z", "_last_z", "_median", "_min", "_max",
                   "_range", "_first", "_worst", "_last", "_first_ratio_podium", "_worst_ratio_podium"}
_Z_SUFFIXES = {"_median_z", "_min_z", "_max_z", "_first_z", "_worst_z", "_last_z", "_z_flag", "_zsd"}


def strip_name(col: str) -> tuple:
    """(organ block or '', is a phi2-dup copy, base name) of a Stage 08 column (level prefix removed)."""
    s = col.split("_", 1)[1] if "_" in col else col
    organ = ""
    head, sep, rest = s.partition("__")
    if sep and head in ORGANS:
        organ, s = head, rest
    dup = s.startswith("dup__")
    if dup:
        s = s[len("dup__"):]
    return organ, dup, s


def stem_of(base: str) -> tuple:
    for suf in _SUFFIXES:
        if base.endswith(suf) and len(base) > len(suf):
            return base[:-len(suf)], suf
    return base, ""


def column_groups(cols: list) -> OrderedDict:
    """Pre-registered redundant groups (plan 11), from the column names (derivation graph): 'summary' = the value
    summaries of one channel (min / median / max / first / worst / last, raw or z); 'zscore_flag' = the z-scores,
    z-SD and abnormality flag of one channel (organ-block copies included); 'renal' = creatinine values, AKI stage and
    BUN/Cr ratio; 'dup' = a phi2-dup column with its copies. Only groups with two or more members."""
    g = OrderedDict()
    for i, c in enumerate(cols):
        _, dup, base = strip_name(c)
        stem, suf = stem_of(base)
        if suf in _VALUE_SUFFIXES:
            g.setdefault(("summary", stem), []).append(i)
        if suf in _Z_SUFFIXES:
            g.setdefault(("zscore_flag", stem), []).append(i)
        if stem in ("creatinine", "bun_cr_ratio") and suf in _VALUE_SUFFIXES | {""} or base in ("aki_stage",
                                                                                                "bun_cr_ratio"):
            g.setdefault(("renal", "creatinine_aki"), []).append(i)
        g.setdefault(("dup", base), []).append(i)
    out = OrderedDict()
    for (typ, name), idx in g.items():
        idx = sorted(set(idx))
        if len(idx) < 2:
            continue
        if typ == "zscore_flag" and not any(stem_of(strip_name(cols[i])[2])[1] == "_z_flag" for i in idx):
            continue
        if typ == "dup" and not any(strip_name(cols[i])[1] for i in idx):
            continue
        out[(typ, name)] = np.array(idx)
    return out


def credit_split(M: np.ndarray, groups: OrderedDict) -> list:
    """Per group: share of the total |SHAP| (median, IQR over refits) and the member shares within the group
    (mean over members of the IQR across refits; the largest member's median share)."""
    tot = M.sum(1, keepdims=True)
    tot = np.where(tot > 0, tot, 1.0)
    rows = []
    for (typ, name), idx in groups.items():
        G = M[:, idx]
        gs = G.sum(1)
        share = gs / tot[:, 0]
        mem = G / np.where(gs > 0, gs, 1.0)[:, None]
        iqr = np.percentile(mem, 75, axis=0) - np.percentile(mem, 25, axis=0)
        rows.append(OrderedDict(group_type=typ, group=name, members=len(idx),
                                group_share_median=float(np.median(share)),
                                group_share_q25=float(np.percentile(share, 25)),
                                group_share_q75=float(np.percentile(share, 75)),
                                group_share_cv=float(np.std(share) / np.mean(share)) if np.mean(share) > 0 else np.nan,
                                member_share_iqr_mean=float(iqr.mean()),
                                member_share_sd_mean=float(mem.std(0).mean()),
                                top_member_share_median=float(np.median(mem.max(1)))))
    return rows


def polr_fit_warm(K: int, X: np.ndarray, y: np.ndarray, x0: np.ndarray | None):
    """Run 2's proportional-odds ordinal LR (lam = 1), L-BFGS warm-started at the full-train solution (same convex
    objective and optimum; fewer iterations)."""
    from scipy.optimize import minimize
    m = R2.OrdinalLinear(K, 1.0, "polr")
    if x0 is None:
        return m.fit(X, y)
    yk = np.asarray(y, dtype=np.int64)
    x0 = np.array(x0, dtype=np.float64)
    if np.unique(yk).size < K:
        return m.fit(X, yk)
    r = minimize(m._loss, x0, args=(X, yk, np.zeros(len(yk))), jac=True, method="L-BFGS-B",
                 options={"maxiter": m.max_iter, "gtol": 1e-6})
    p = X.shape[1]
    m.w, m.theta = r.x[:p], R2._theta(r.x[p:])
    return m


def _par_of(m) -> np.ndarray:
    th = np.asarray(m.theta, dtype=np.float64)
    return np.r_[np.asarray(m.w, dtype=np.float64), th[0], np.log(np.maximum(np.diff(th), 1e-12))]


def attribute(model, kind: str, X: np.ndarray, mu: np.ndarray, Z=None) -> np.ndarray:
    if kind == "LR":
        return linear_shap_ey(model, X, mu)
    T = TreeTable(model, X.shape[1])
    S, resid = fh_shap_ey(T, X, Z)
    if resid > 1e-6:
        raise BuildError("TreeSHAP additivity check failed")
    return S


def planted_gate() -> OrderedDict:
    """Plan 13.2 planted redundant-group gate (synthetic, in-script, seeded): a planted near-duplicate pair must show
    (i) a stable group total but unstable member shares under bootstrap refits of the ordinal LR, (ii) an exact
    duplicate split equally by the ridge penalty, (iii) a larger interventional-vs-conditional TreeSHAP gap than an
    independent design, and (iv) TreeSHAP additivity on every threshold."""
    rng = np.random.default_rng(13013)
    n, K = 1500, 5
    z = rng.normal(size=(n, 4))
    near = z[:, 0] + 0.05 * rng.normal(size=n)
    X = np.column_stack([z[:, 0], near, z[:, 0], z[:, 1], z[:, 2], z[:, 3]])   # cols 0/1 near-dup, 0/2 exact dup
    eta = 1.2 * z[:, 0] + 0.8 * z[:, 1] + 0.4 * z[:, 2]
    y = np.digitize(eta + rng.logistic(size=n) * 0.6, [-1.5, -0.5, 0.5, 1.5])
    mu = X.mean(0)
    Ms = []
    for b in range(12):
        ix = rng.integers(0, n, n)
        m = R2.OrdinalLinear(K, 1.0, "polr").fit(X[ix], y[ix])
        Ms.append(np.abs(linear_shap_ey(m, X[:300], mu)).mean(0))
    M = np.array(Ms)
    trio = M[:, [0, 1, 2]]
    share_near = M[:, 1] / trio.sum(1)
    total = trio.sum(1) / M.sum(1)
    ok_i = float(np.std(total) / np.mean(total)) < float(np.std(share_near) / np.mean(share_near))
    ok_ii = bool(np.allclose(M[:, 0], M[:, 2], rtol=1e-3))
    fh = R2.FrankHall(K, "gb", dict(max_iter=40, learning_rate=0.1, max_leaf_nodes=7), R2.SEED).fit(X, y)
    Xi = np.column_stack([z, rng.normal(size=(n, 2))])
    fi = R2.FrankHall(K, "gb", dict(max_iter=40, learning_rate=0.1, max_leaf_nodes=7), R2.SEED).fit(Xi, y)
    gaps = []
    for mdl, XX in ((fh, X), (fi, Xi)):
        T = TreeTable(mdl, XX.shape[1])
        c, r1 = fh_shap_ey(T, XX[:150])
        it, r2 = fh_shap_ey(T, XX[:150], XX[600:640])
        a, b = np.abs(c).mean(0), np.abs(it).mean(0)
        gaps.append((0.5 * np.abs(a / a.sum() - b / b.sum()).sum(), max(r1, r2)))
    ok_iii = gaps[0][0] > gaps[1][0]
    ok_iv = max(g[1] for g in gaps) < 1e-8
    return OrderedDict(near_dup_group_stable_members_unstable=bool(ok_i), exact_dup_split_equally=ok_ii,
                       gap_larger_with_redundancy=bool(ok_iii), treeshap_additive=bool(ok_iv))


def s13_matrices(D: dict, mats: OrderedDict) -> tuple:
    wm = D["s12"]["winner"]["matrix"]
    lr = list(S13_LR_MATRICES) + ([wm] if wm not in S13_LR_MATRICES else [])
    tree = []
    if D["s12"]["winner"]["family"] == "GB":
        tree = list(S13_TREE_LEVELS) + ([wm] if wm not in S13_TREE_LEVELS else [])
    return lr, tree, wm


def stage13(D: dict, mats: OrderedDict, FL: FitLog, cfg: Config, checks: dict, info: dict) -> dict:
    R = D["rows"]
    sp = R["split"]
    tr, va = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "validation")
    rng = R2.rng_for("s13", "sample")
    samp = np.sort(rng.choice(va, min(S13_SAMPLE, len(va)), replace=False))
    gap_s = samp[:min(S13_GAP_SAMPLE, len(samp))]
    bg = np.sort(R2.rng_for("s13", "background").choice(tr, min(S13_BACKGROUND, len(tr)), replace=False))
    FL.rows(samp, "stage 13 validation sample")
    FL.rows(bg, "stage 13 interventional background")
    boots = [boot_rows(R["subj"][tr], R2.rng_for("s13", "boot", b)) for b in range(max(cfg.s13_refits, cfg.s13_tree_refits))]
    lr_m, tree_m, wm = s13_matrices(D, mats)
    hp = dict(D["s12"]["hp"])
    wmodel = D["s12"]["winner"]["model"]
    jobs = [("polr", m, "outcome1") for m in lr_m]
    if wmodel != "polr" and D["s12"]["winner"]["family"] == "LR":
        jobs += [(wmodel, wm, "outcome1")]
    jobs += [("gb_fh", m, "outcome1") for m in tree_m if m != "b2s"]
    jobs += [("polr", m, "outcome2") for m in S13_O2_LEVELS]
    imp, tau_rows, split_rows, top_rows, gap_rows, internal = OrderedDict(), [], [], [], [], OrderedDict()

    def summarise(outc, mod, m, M, cols, note=""):
        groups = column_groups(cols)
        gidx = np.unique(np.concatenate(list(groups.values()))) if groups else np.array([], dtype=int)
        ung = np.setdiff1d(np.arange(len(cols)), gidx)
        row = OrderedDict(outcome=outc, model=mod, matrix=m, refits=M.shape[0], **tau_summary(kendall_pairs(M)))
        for nm, sel in (("grouped", gidx), ("ungrouped", ung)):
            ts = tau_summary(kendall_pairs(M[:, sel])) if len(sel) >= 3 else tau_summary(np.array([]))
            row.update({f"{k}_{nm}": v for k, v in ts.items()})
        row["note"] = note
        tau_rows.append(row)
        for r in credit_split(M, groups):
            split_rows.append(OrderedDict(outcome=outc, model=mod, matrix=m, **r))
        tot = M.sum(1, keepdims=True)
        shares = M / np.where(tot > 0, tot, 1.0)
        ranks = ((-M).argsort(1).argsort(1) + 1) / M.shape[1]
        med = np.median(shares, 0)
        for r_, j in enumerate(np.argsort(-med)[:20]):
            top_rows.append(OrderedDict(outcome=outc, model=mod, matrix=m, position=r_ + 1, item=cols[j],
                                        share_median=float(med[j]),
                                        rank_fraction_q25=float(np.percentile(ranks[:, j], 25)),
                                        rank_fraction_q75=float(np.percentile(ranks[:, j], 75))))
        return groups

    def run_job(mod, m, outc):
        t0 = time.time()
        X = mats[m]["X"]
        cols = mats[m]["cols"]
        y = R["O1"][24] if outc == "outcome1" else R["O2"]
        K = R2.K_OUT[outc]
        ok = y >= 0
        mu = X[tr].mean(0)
        kind = "LR" if mod in ("polr", "allthr") else "GB"
        B = cfg.s13_refits if kind == "LR" else cfg.s13_tree_refits
        trk = tr[ok[tr]]
        FL.rows(trk, f"stage 13 {mod} {m} full-train fit")
        if kind == "GB" and m == wm and D["s12"]["winner"]["rep"] == "full":
            full = D["s12"]["model"]                 # the frozen stage 12 model is this full-train fit
        elif kind == "GB":
            full = R2.make_model("gb_fh", K, hp, R2.SEED).fit(X[trk], y[trk])
        elif mod == "polr":
            full = polr_fit_warm(K, X[trk], y[trk], None)
        else:
            full = R2.make_model(mod, K, hp, R2.SEED).fit(X[trk], y[trk])
        par0 = _par_of(full) if mod == "polr" else None
        Ms, t_ref = [], time.time()
        for b in range(B):
            ib = tr[boots[b]]
            ib = ib[ok[ib]]
            FL.rows(ib, f"stage 13 {mod} {m} refit")
            if kind == "GB":
                mb = R2.make_model("gb_fh", K, hp, R2.SEED).fit(X[ib], y[ib])
            elif mod == "polr":
                mb = polr_fit_warm(K, X[ib], y[ib], par0)
            else:
                mb = R2.make_model(mod, K, hp, R2.SEED).fit(X[ib], y[ib])
            Ms.append(np.abs(attribute(mb, kind, X[samp], mu)).mean(0))
            check_time()
        t_ref_end = time.time()
        M = np.array(Ms)
        imp[f"{outc}|{mod}|{m}"] = M
        internal[f"s13__{outc}__{mod}__{m.replace('+', '_')}"] = M.astype(np.float32)
        groups = summarise(outc, mod, m, M, cols)
        if kind == "GB":
            from scipy.stats import kendalltau
            Xg = X[gap_s]
            c = np.abs(attribute(full, "GB", Xg, mu)).mean(0)
            it = np.abs(attribute(full, "GB", Xg, mu, X[bg])).mean(0)
            a, b_ = c / max(c.sum(), 1e-300), it / max(it.sum(), 1e-300)
            g = OrderedDict(outcome=outc, model=mod, matrix=m, tau_conditional_vs_interventional=(
                float(kendalltau(c, it).statistic) if np.ptp(c) > 0 and np.ptp(it) > 0 else np.nan),
                tv_distance_all=float(0.5 * np.abs(a - b_).sum()))
            for typ in ("summary", "zscore_flag", "renal", "dup"):
                sel = [ix for (t_, _), ix in groups.items() if t_ == typ]
                if sel:
                    s_ = np.unique(np.concatenate(sel))
                    g[f"tv_distance_{typ}"] = float(0.5 * np.abs(a[s_] - b_[s_]).sum())
                    g[f"share_shift_{typ}"] = float(b_[s_].sum() - a[s_].sum())
            gap_rows.append(g)
        say(f"stage 13: {mod} {m} {outc} done ({time.time() - t0:.0f} s; {(t_ref_end - t_ref) / max(B, 1):.1f} s per refit)")
        return M

    for mod, m, outc in jobs:
        run_job(mod, m, outc)
    # gradient boosting on b2s: read off the b2d refits when the b2d trees never split on a phi2-dup copy and the
    # full-train b2s and b2d fits coincide (exact duplicates tie and the first column wins); otherwise refitted
    if "b2d" in tree_m:
        dm = np.array([strip_name(c)[1] for c in mats["b2d"]["cols"]])
        Md = imp["outcome1|gb_fh|b2d"]
        same_cols = ([strip_name(c)[2] for c in mats["b2s"]["cols"]] ==
                     [strip_name(c)[2] for c, d_ in zip(mats["b2d"]["cols"], dm) if not d_])
        y, trk = R["O1"][24], tr[R["O1"][24][tr] >= 0]
        fs = R2.make_model("gb_fh", 9, hp, R2.SEED).fit(mats["b2s"]["X"][trk], y[trk])
        fd = R2.make_model("gb_fh", 9, hp, R2.SEED).fit(mats["b2d"]["X"][trk], y[trk])
        same_pred = float(np.abs(fs.predict_proba(mats["b2s"]["X"][samp]) -
                                 fd.predict_proba(mats["b2d"]["X"][samp])).max()) <= 1e-12
        derived = bool(same_cols and same_pred and dm.any() and (Md[:, dm] == 0).all())
        info["s13_tree_b2s_read_off_b2d"] = derived
        if derived:
            M = Md[:, ~dm]
            imp["outcome1|gb_fh|b2s"] = M
            internal["s13__outcome1__gb_fh__b2s"] = M.astype(np.float32)
            summarise("outcome1", "gb_fh", "b2s", M, mats["b2s"]["cols"],
                      "read off the b2d refits (the trees never split on a phi2-dup copy)")
        else:
            run_job("gb_fh", "b2s", "outcome1")
    tau = pd.DataFrame(tau_rows)
    h7 = h7_verdicts(tau, pd.DataFrame(split_rows))
    checks["13_validation_sample_fixed_and_seeded"] = bool(np.array_equal(
        samp, np.sort(R2.rng_for("s13", "sample").choice(va, min(S13_SAMPLE, len(va)), replace=False))))
    internal["s13__validation_sample_eid"] = R["eid"][samp]
    return dict(tau=tau, split=pd.DataFrame(split_rows), top=pd.DataFrame(top_rows), gap=pd.DataFrame(gap_rows),
                h7=h7, internal=internal)


def h7_verdicts(tau: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """H7 per model (Outcome 1, plan 12.2): median tau falls from phi1 (b1) to phi2-dup (b2d) by at least 0.10, the
    fall is concentrated in the redundant groups (grouped-column tau falls more than ungrouped), and phi2-single
    (b2s) is within 0.05 of phi1."""
    out = []
    t = tau[tau["outcome"] == "outcome1"]
    for mod, g in t.groupby("model", sort=False):
        v = {r.matrix: r for r in g.itertuples()}
        if not {"b1", "b2d", "b2s"} <= set(v):
            continue
        fall = float(v["b1"].tau_median - v["b2d"].tau_median)
        fg = float(v["b1"].tau_median_grouped - v["b2d"].tau_median_grouped)
        fu = float(v["b1"].tau_median_ungrouped - v["b2d"].tau_median_ungrouped)
        pr = float(abs(v["b2s"].tau_median - v["b1"].tau_median))
        sup = bool(fall >= H7_FALL and fg > fu and pr <= H7_PRUNED)
        out.append(OrderedDict(model=mod, tau_b1=float(v["b1"].tau_median), tau_b2d=float(v["b2d"].tau_median),
                               tau_b2s=float(v["b2s"].tau_median), fall_b1_to_b2d=fall, fall_grouped=fg,
                               fall_ungrouped=fu, abs_diff_b2s_vs_b1=pr, threshold_fall=H7_FALL,
                               threshold_pruned=H7_PRUNED, h7_met=sup))
    return pd.DataFrame(out)
# ------------------------------------------------------------------------------- stage 18: sequence attribution
def winner_X(D: dict, mats: OrderedDict) -> tuple:
    w = D["s12"]["winner"]
    kind, k = (w["rep"], None) if w["rep"] == "full" else (w["rep"].split("_")[0], int(w["rep"].split("_")[1]))
    X = R2.rep_X(mats[w["matrix"]], kind, k)
    names = list(mats[w["matrix"]]["cols"]) if kind == "full" else (
        [mats[w["matrix"]]["cols"][j] for j in mats[w["matrix"]]["cpqr"][:k]] if kind == "cpqr" else
        [f"{w['matrix']}_svd_component_{j + 1}" for j in range(k)])
    return w, X, names


def channel_organs(mats: OrderedDict, channels: list) -> list:
    """Organ system of each Layer C channel, read off the organ-block prefixes of the b2s column names."""
    votes = {}
    for c in mats["b2s"]["cols"]:
        organ, _, base = strip_name(c)
        if organ:
            st = stem_of(base)[0]
            votes.setdefault(st, {}).setdefault(organ, 0)
            votes[st][organ] += 1
    return [max(sorted(votes[c].items()), key=lambda kv: kv[1])[0] if c in votes else "other" for c in channels]


def gru_net(state: dict, device=None):
    Net = R2.make_gru_class()
    net = Net(state["d_seq"], state["d_static"], state["hidden"], state["K"])
    net.load_state_dict(state["state_dict"])
    net.to(device if device is not None else torch_device())
    net.eval()
    return net


def gru_ey(net, seq, st, off) -> np.ndarray:
    torch = R2._torch()
    dev = torch_device()
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    s = T(st) if st is not None else torch.zeros((seq.shape[0], 0), device=dev)
    with torch.no_grad():
        return torch.sigmoid(-net(T(seq), s, T(off))).sum(1).cpu().numpy().astype(np.float64)


def expected_gradients(net, seq, st, off, bseq, bst, boff, draws: int, rng) -> tuple:
    """Expected-gradients SHAP (GradientSHAP; Erion et al. 2021) of E[Y | x] = sum_j P(Y > j | x) for the GRU:
    for each explained row, `draws` baselines from the train background and alpha ~ U(0, 1);
    phi = mean (x - x') * grad f(x' + alpha (x - x')), for the sequence inputs, the static A+B context and the
    Layer A offset. Returns (seq phi n x b x d, static phi summed n, offset phi n)."""
    torch = R2._torch()
    dev = torch_device()
    n = seq.shape[0]
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    ps, pst, po = np.zeros(seq.shape), np.zeros(n), np.zeros(n)
    for _ in range(draws):
        j = rng.integers(0, bseq.shape[0], n)
        a = rng.uniform(0, 1, n)
        dx, ds, do = seq - bseq[j], (st - bst[j]) if st is not None else None, off - boff[j]
        xs = T(bseq[j] + a[:, None, None] * dx).requires_grad_(True)
        xo = T(boff[j] + a * do).requires_grad_(True)
        if st is not None:
            xt = T(bst[j] + a[:, None] * ds).requires_grad_(True)
        else:
            xt = torch.zeros((n, 0), device=dev)
        # cuDNN's fused GRU backward only runs in training mode; the frozen net is in eval mode, so the
        # attribution forward/backward uses the native (non-cuDNN) kernels. No-op on CPU. (v2.0.2 fix)
        with torch.backends.cudnn.flags(enabled=False):
            f = torch.sigmoid(-net(xs, xt, xo)).sum()
            gr = torch.autograd.grad(f, [xs, xo] + ([xt] if st is not None else []))
        ps += dx * gr[0].detach().cpu().numpy()
        po += do * gr[1].detach().cpu().numpy()
        if st is not None:
            pst += (ds * gr[2].detach().cpu().numpy()).sum(1)
    return ps / draws, pst / draws, po / draws


def seq_importances(ps, pst, po, q: int, organs: list) -> OrderedDict:
    """Global importances (mean |phi| over the explained rows) at the stage 18 granularities."""
    n, b, _ = ps.shape
    per = ps[:, :, :q] + ps[:, :, q:2 * q] + ps[:, :, 2 * q:3 * q]            # n x b x q (value + mask + carried)
    blocks = np.arange(b) // 4                                                # 24 h blocks of 6 h bins
    ub = np.unique(blocks)
    org = sorted(set(organs))
    oi = np.array([org.index(o) for o in organs])
    out = OrderedDict()
    out["bin_channel"] = np.abs(per).mean(0).ravel()
    out["channel"] = np.abs(per.sum(1)).mean(0)
    out["day_block"] = np.array([np.abs(ps[:, blocks == u, :].sum((1, 2))).mean() for u in ub])
    out["organ"] = np.array([np.abs(per.sum(1)[:, oi == k].sum(1)).mean() for k in range(len(org))])
    ty = np.array([np.abs(ps[:, :, :q].sum((1, 2))).mean(), np.abs(ps[:, :, q:2 * q].sum((1, 2))).mean(),
                   np.abs(ps[:, :, 2 * q:3 * q].sum((1, 2))).mean(), np.abs(ps[:, :, -1].sum(1)).mean(),
                   np.abs(pst).mean(), np.abs(po).mean()])
    out["input_type"] = ty
    out["adjacent"] = np.abs(per).mean(0)                                     # b x q (for the credit split)
    return out


INPUT_TYPES = ("layer_c_values", "layer_c_missing_masks", "layer_c_carried_forward_flags", "bin_index",
               "static_ab_context", "layer_a_offset")


def permutation_channels(net, seq, st, off, q: int, rng) -> np.ndarray:
    """Permutation cross-check: mean |change in E[Y]| when one channel's value, mask and carried-forward inputs
    are permuted across the explained rows (all bins together)."""
    base = gru_ey(net, seq, st, off)
    out = np.zeros(q)
    for c in range(q):
        p = rng.permutation(seq.shape[0])
        s2 = seq.copy()
        for k in (c, q + c, 2 * q + c):
            s2[:, :, k] = seq[p, :, k]
        out[c] = np.abs(gru_ey(net, s2, st, off) - base).mean()
    return out


def stage18_attr(D: dict, mats: OrderedDict, FL: FitLog, cfg: Config, checks: dict, info: dict) -> dict:
    from scipy.stats import kendalltau
    R, tab, sts, vp = D["rows"], D["tab"], D["states"], D["vpred"]
    sp = R["split"]
    w, Xw, wnames = winner_X(D, mats)
    A_name = w["matrix"].split("+")[0]
    tau_rows, imp_rows, cc_rows, ab_rows, internal = [], [], [], [], OrderedDict()
    maxd, ok_order = 0.0, True
    for L in S18_LANDMARKS:
        t0 = time.time()
        r = D["lc"][L]
        idx = r["idx"]
        spL = sp[idx]
        y = R["O1"][L][idx]
        g = R["subj"][idx]
        itr = np.flatnonzero((spL == "train") & (y >= 0))
        iva = np.flatnonzero((spL == "validation") & (y >= 0))
        K = 9
        S = R2.c_inputs(D, L, "all", None)
        chans = [str(c) for c in D["bases"][f"C_L{L}__channels"]]
        q = len(chans)
        organs = channel_organs(mats, chans)
        org_names = sorted(set(organs))
        XA, Xab = mats[A_name]["X"][idx], Xw[idx]
        samp = np.sort(R2.rng_for("s18", "sample", L).choice(iva, min(S18_SAMPLE, len(iva)), replace=False))
        bg = np.sort(R2.rng_for("s18", "background", L).choice(itr, min(S18_BACKGROUND, len(itr)), replace=False))
        FL.rows(idx[samp], f"stage 18 L{L} validation sample")
        FL.rows(idx[bg], f"stage 18 L{L} background")
        st0 = sts[f"L{L}_outcome1_ABC"]
        mA = tab[f"L{L}_outcome1_A"]
        # frozen Run 2 model: must reproduce its Run 2 validation predictions (CPU float32)
        vdev = verify_device()
        net0 = gru_net(st0, device=vdev)
        ek = f"s17_L{L}_outcome1__eid"
        jva_all = np.flatnonzero(spL == "validation")
        jva_all = jva_all[(y >= 0)[jva_all]]
        ok_order &= ek in vp and bool(np.array_equal(vp[ek].astype(str), R["eid"][idx][jva_all]))
        ref = vp.get(f"s17_L{L}_outcome1_ABC")
        Pf = gru_predict_probs(net0, S[jva_all], Xab[jva_all], mA.decision(XA[jva_all]), device=vdev)
        net0 = gru_net(st0)  # attribution / permutation on torch_device() (CUDA if available)
        if ref is None or ref.shape != Pf.shape:
            ok_order = False
        else:
            maxd = max(maxd, float(np.abs(Pf - ref.astype(np.float64)).max()))
        rng = R2.rng_for("s18", "eg", L, 0)
        ps, pst, po = expected_gradients(net0, S[samp], Xab[samp], mA.decision(XA[samp]), S[bg], Xab[bg],
                                         mA.decision(XA[bg]), S18_EG_DRAWS, rng)
        I0 = seq_importances(ps, pst, po, q, organs)
        perm = permutation_channels(net0, S[samp], Xab[samp], mA.decision(XA[samp]), q, R2.rng_for("s18", "perm", L))
        # refits on subject-bootstrap resamples of train (frozen configuration of the Run 2 model at L)
        Is, t_ref = [], time.time()
        for b in range(cfg.s18_refits):
            jb = itr[boot_rows(g[itr], R2.rng_for("s18", "boot", L, b))]
            FL.rows(idx[jb], f"stage 18 L{L} refit")
            off_tr, off_s, th, mfull = R2.crossfit_offset(XA[jb], y[jb], g[jb], XA[samp], K, ("s18", L, b))
            stt, _, _ = R2.gru_fit_predict(S[jb], Xab[jb], off_tr, y[jb], g[jb], S[samp], Xab[samp], off_s, K,
                                           st0["hidden"], R2.SEED + L + 7919 * (b + 1), gamma0=th,
                                           max_epochs=D["r2cfg"]["gru_max_epochs"])
            net = gru_net(stt)
            ps, pst, po = expected_gradients(net, S[samp], Xab[samp], off_s, S[bg], Xab[bg], mfull.decision(XA[bg]),
                                             S18_EG_DRAWS, R2.rng_for("s18", "eg", L, b + 1))
            Is.append(seq_importances(ps, pst, po, q, organs))
            check_time()
        t_ref = (time.time() - t_ref) / max(cfg.s18_refits, 1)
        for gran in ("bin_channel", "channel", "day_block", "organ"):
            M = np.array([x[gran] for x in Is])
            internal[f"s18__L{L}__{gran}"] = M.astype(np.float32)
            tau_rows.append(OrderedDict(landmark=L, focus=L in H9_FOCUS, granularity=gran, refits=M.shape[0],
                                        **tau_summary(kendall_pairs(M))))
        names = OrderedDict(channel=chans, organ=org_names, input_type=list(INPUT_TYPES),
                            day_block=[f"day {u + 1}" for u in range(len(I0["day_block"]))])
        for gran, nm in names.items():
            M = np.array([x[gran] for x in Is])
            sh = M / np.where(M.sum(1, keepdims=True) > 0, M.sum(1, keepdims=True), 1.0)
            med = np.median(sh, 0)
            order = np.argsort(-med)[:15] if gran == "channel" else np.arange(len(nm))
            for j in order:
                imp_rows.append(OrderedDict(landmark=L, focus=L in H9_FOCUS, granularity=gran, item=nm[j],
                                            organ=(organs[j] if gran == "channel" else ""),
                                            share_median=float(med[j]), share_q25=float(np.percentile(sh[:, j], 25)),
                                            share_q75=float(np.percentile(sh[:, j], 75)),
                                            share_frozen_model=float(I0[gran][j] / max(I0[gran].sum(), 1e-300))))
        Mc = np.array([x["channel"] for x in Is])
        top = np.argsort(-np.median(Mc, 0))[:S18_TOP]
        adj = np.array([x["adjacent"] for x in Is])                                  # B x b x q
        nb = adj.shape[1]
        if nb >= 2:
            last2 = adj[:, -2:, :][:, :, top]
            sh_last = last2[:, 1, :] / np.where(last2.sum(1) > 0, last2.sum(1), 1.0)
            adj_iqr = float((np.percentile(sh_last, 75, 0) - np.percentile(sh_last, 25, 0)).mean())
            adj_med = float(np.median(sh_last))
        else:
            adj_iqr = adj_med = np.nan
        blocks = np.arange(nb) // 4
        lastday = adj[:, blocks == blocks.max(), :].sum(1)[:, top] / np.where(adj.sum(1)[:, top] > 0,
                                                                              adj.sum(1)[:, top], 1.0)
        tf = [kendalltau(I0["channel"], m).statistic for m in Mc if np.ptp(m) > 0]
        cc_rows.append(OrderedDict(
            landmark=L, focus=L in H9_FOCUS,
            tau_gradient_vs_permutation_frozen=float(kendalltau(I0["channel"], perm).statistic)
            if np.ptp(perm) > 0 and np.ptp(I0["channel"]) > 0 else np.nan,
            tau_frozen_vs_refits_median=float(np.median(tf)) if tf else np.nan,
            adjacent_bin_last_share_median=adj_med, adjacent_bin_last_share_iqr_mean=adj_iqr,
            last_day_share_top_channels_median=float(np.median(lastday))))
        # TreeSHAP (or linear SHAP) of the frozen A+B landmark model on the same rows
        mab = tab[f"L{L}_outcome1_AB"]
        kind = "GB" if isinstance(mab, R2.FrankHall) else "LR"
        if isinstance(mab, R2.FrankHall) and mab.base != "gb":
            kind = None
        if kind is not None:
            phi = attribute(mab, kind, Xab[samp], Xab[itr].mean(0))
            ia = np.abs(phi).mean(0)
            sh = ia / max(ia.sum(), 1e-300)
            isA = np.array([n_.startswith("a") for n_ in wnames])
            ab_rows.append(OrderedDict(landmark=L, focus=L in H9_FOCUS, position=0, item="(layer A share)",
                                       share=float(sh[isA].sum())))
            for pos, j in enumerate(np.argsort(-sh)[:15]):
                ab_rows.append(OrderedDict(landmark=L, focus=L in H9_FOCUS, position=pos + 1, item=wnames[j],
                                           share=float(sh[j])))
        say(f"stage 18: attribution at landmark {L} done ({time.time() - t0:.0f} s; {t_ref:.1f} s per refit)")
    checks["18_frozen_abc_reproduces_run2_validation"] = ok_order and maxd <= cfg.repro_tol_prob
    info["repro_s18_abc_max_abs_prob_diff"] = maxd
    say(f"stage 18 verify: frozen A+B+C models on CPU (float32); max abs prob diff {_fmt_repro(maxd)} "
        f"(tol {_fmt_repro(cfg.repro_tol_prob)}); "
        f"{'ok' if checks['18_frozen_abc_reproduces_run2_validation'] else 'FAIL'}")
    return dict(tau=pd.DataFrame(tau_rows), imp=pd.DataFrame(imp_rows), cross=pd.DataFrame(cc_rows),
                ab=pd.DataFrame(ab_rows), internal=internal)


def gru_predict_probs(net, seq, st, off, device=None) -> np.ndarray:
    torch = R2._torch()
    dev = device if device is not None else torch_device()
    net.to(dev)
    net.eval()
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    s = T(st) if st is not None else torch.zeros((seq.shape[0], 0), device=dev)
    with torch.no_grad():
        F = torch.sigmoid(net(T(seq), s, T(off))).cpu().numpy().astype(np.float64)
    return R2.cum_to_probs(np.maximum.accumulate(F, axis=1))
# --------------------------------------------------------------------------------- stage 16: Layer C tensor SVD
S16_MAX_PATIENTS = 2000                  # seeded train-patient subsample: reference HOOI fits and the flattened baseline (runtime)
S16_HOOI_ITERS = 5
S16_CP_COMPRESS = 60                     # patient-mode compression rank for CP (CANDELINC; exact for B, C, CORCONDIA)
S16_NULL_DRAWS_PATIENT = 5               # null draws for the patient-mode spectrum (n x n Gram; runtime)
S16_CP_ITERS, S16_CP_TOL = 60, 1e-6


def _es(*a):
    return np.einsum(*a, optimize=True)


def s16_tensors(D: dict, L: int) -> OrderedDict:
    """Train and validation landmark tensors (patients x prefix bins x Run 1 channels) at landmark L: normalized
    (X_filled_z), raw (X_raw_z, NaN where not observed) and engineered-lite (normalized plus the missing mask and
    carried-forward flag as extra feature-mode channels; stage 15 levels are deferred, D17)."""
    r = D["lc"][L]
    chans = [str(c) for c in D["bases"][f"C_L{L}__channels"]]
    z = R2.read_npz(r["path"], ("X_filled_z", "X_raw_z", "missing", "carried_forward", "channels"))
    allc = [str(c) for c in z["channels"]]
    pos = [allc.index(c) for c in chans]
    m = r["mask"]
    Xf = z["X_filled_z"][m][:, :, pos].astype(np.float64)
    Xr = z["X_raw_z"][m][:, :, pos].astype(np.float64)
    Mi = z["missing"][m][:, :, pos].astype(np.float64)
    Cf = z["carried_forward"][m][:, :, pos].astype(np.float64)
    if Xf.shape[1] != L // 6 or not np.isfinite(Xf).all():
        raise BuildError(f"Layer C L{L}: prefix bins or values not as expected")
    sp = D["rows"]["split"][r["idx"]]
    return OrderedDict(channels=chans, split=sp, normalized=Xf, raw=Xr,
                       engineered=np.concatenate([Xf, Mi, Cf], axis=2))


def mode_gram(X: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        Y = X.reshape(X.shape[0], -1)
        return Y @ Y.T if Y.shape[0] <= Y.shape[1] else None
    if mode == 1:
        return _es("nbq,ncq->bc", X, X)
    return _es("nbq,nbr->qr", X, X)


def mode_svals(X: np.ndarray, mode: int) -> np.ndarray:
    """Singular values of the mode-n unfolding (from the smaller Gram matrix)."""
    if mode == 0:
        Y = X.reshape(X.shape[0], -1)
        G = Y @ Y.T if Y.shape[0] <= Y.shape[1] else Y.T @ Y
    else:
        G = mode_gram(X, mode)
    ev = np.linalg.eigvalsh(G)[::-1]
    return np.sqrt(np.clip(ev, 0, None))


def mode_basis(X: np.ndarray, mode: int, r: int) -> np.ndarray:
    G = mode_gram(X, mode)
    ev, V = np.linalg.eigh(G)
    return V[:, ::-1][:, :r]


def spec_stats(s: np.ndarray) -> OrderedDict:
    e = s ** 2
    tot = e.sum()
    if tot <= 0:
        return OrderedDict(k95=np.nan, erank_sigma2=np.nan)
    k95 = int(np.searchsorted(np.cumsum(e) / tot, 0.95) + 1)
    return OrderedDict(k95=k95, erank_sigma2=float(tot ** 2 / (e ** 2).sum()))


def null_tensor(X: np.ndarray, kind: str, rng) -> np.ndarray:
    """'channel': each channel's patient order permuted independently (bins kept together); 'bin': each bin's
    patient order permuted independently (the per-bin patient shuffle)."""
    Y = np.empty_like(X)
    if kind == "channel":
        for c in range(X.shape[2]):
            Y[:, :, c] = X[rng.permutation(X.shape[0]), :, c]
    else:
        for b in range(X.shape[1]):
            Y[:, b, :] = X[rng.permutation(X.shape[0]), b, :]
    return Y


def hooi(X: np.ndarray, ranks: tuple, iters: int, mask: np.ndarray | None = None, init=None) -> tuple:
    """Tucker by HOOI from the HOSVD start; with a mask, EM imputation of the unobserved cells inside each sweep
    (masked weighted fit; unobserved values never enter the error). Returns (factors, core, rel. error)."""
    r1, r2, r3 = ranks
    Xw = X if mask is None else np.where(mask, X, 0.0)
    if mask is not None and init is None:
        obs = np.where(mask, X, np.nan)
        cm = np.nanmean(obs.reshape(-1, X.shape[2]), axis=0)
        Xw = np.where(mask, X, np.nan_to_num(cm)[None, None, :])
    U2 = mode_basis(Xw, 1, r2) if init is None else init[1]
    U3 = mode_basis(Xw, 2, r3) if init is None else init[2]
    U1 = None
    for _ in range(max(1, iters)):
        Y = _es("nbq,br,qs->nrs", Xw, U2, U3).reshape(Xw.shape[0], -1)
        U1 = np.linalg.svd(Y, full_matrices=False)[0][:, :r1]
        Y = _es("nbq,nr,qs->brs", Xw, U1, U3).reshape(Xw.shape[1], -1)
        U2 = np.linalg.svd(Y, full_matrices=False)[0][:, :r2]
        Y = _es("nbq,nr,bs->qrs", Xw, U1, U2).reshape(Xw.shape[2], -1)
        U3 = np.linalg.svd(Y, full_matrices=False)[0][:, :r3]
        if mask is not None:
            G = _es("nbq,nr,bs,qt->rst", Xw, U1, U2, U3)
            Xh = _es("rst,nr,bs,qt->nbq", G, U1, U2, U3)
            Xw = np.where(mask, X, Xh)
    G = _es("nbq,nr,bs,qt->rst", Xw, U1, U2, U3)
    if mask is None:
        err = math.sqrt(max(0.0, float((Xw ** 2).sum() - (G ** 2).sum())) / max(float((Xw ** 2).sum()), 1e-300))
    else:
        Xh = _es("rst,nr,bs,qt->nbq", G, U1, U2, U3)
        err = math.sqrt(float(((X - Xh)[mask] ** 2).sum()) / max(float((X[mask] ** 2).sum()), 1e-300))
    return (U1, U2, U3), G, err


def proj_error(Xv: np.ndarray, U2: np.ndarray, U3: np.ndarray, mask=None) -> float:
    """Relative error of validation projected on the fixed train bin and channel factors (patients free)."""
    Xw = Xv if mask is None else np.where(mask, Xv, 0.0)
    P = _es("nbq,bs,qt->nst", Xw, U2, U3)
    Xh = _es("nst,bs,qt->nbq", P, U2, U3)
    if mask is None:
        return math.sqrt(float(((Xw - Xh) ** 2).sum()) / max(float((Xw ** 2).sum()), 1e-300))
    return math.sqrt(float(((Xv - Xh)[mask] ** 2).sum()) / max(float((Xv[mask] ** 2).sum()), 1e-300))


def _khatri(B: np.ndarray, C: np.ndarray) -> np.ndarray:
    return (B[:, None, :] * C[None, :, :]).reshape(-1, B.shape[1])


def cp_als(X: np.ndarray, R: int, init: tuple, iters: int, tol: float) -> tuple:
    """CP-ALS (rank R) from the given factors; returns (A, B, C, rel. error). MTTKRPs use the three unfoldings."""
    A, B, C = (np.array(f, dtype=np.float64) for f in init)
    n, b, q = X.shape
    X0 = X.reshape(n, b * q)
    X1 = np.ascontiguousarray(X.transpose(1, 0, 2)).reshape(b, n * q)
    X2 = np.ascontiguousarray(X.transpose(2, 0, 1)).reshape(q, n * b)
    nx = float((X0 ** 2).sum())
    prev = np.inf
    err = np.nan
    for _ in range(iters):
        A = (X0 @ _khatri(B, C)) @ np.linalg.pinv((B.T @ B) * (C.T @ C))
        B = (X1 @ _khatri(A, C)) @ np.linalg.pinv((A.T @ A) * (C.T @ C))
        M = X2 @ _khatri(A, B)
        C = M @ np.linalg.pinv((A.T @ A) * (B.T @ B))
        inner = float((M * C).sum())
        nrm = float(((A.T @ A) * (B.T @ B) * (C.T @ C)).sum())
        err = math.sqrt(max(0.0, nx - 2 * inner + nrm) / max(nx, 1e-300))
        if abs(prev - err) < tol:
            break
        prev = err
    return A, B, C, err


def compress_patients(X: np.ndarray, k: int, rng) -> tuple:
    """Orthonormal patient-mode basis U (n x k) from a randomized range finder (3 power iterations) and the
    compressed tensor U'X (k x b x q). CP-ALS on U'X with A = U A~ is the CANDELINC fit; Tucker congruence and the
    CORCONDIA core are unchanged by the orthonormal U."""
    n, b, q = X.shape
    Y = X.reshape(n, b * q)
    k = int(min(k, n, b * q))
    Q = np.linalg.qr(Y @ rng.normal(size=(b * q, min(k + 10, b * q))))[0]
    for _ in range(3):
        Q = np.linalg.qr(Y @ (Y.T @ Q))[0]
    U = np.linalg.svd(Q.T @ Y, full_matrices=False)[0]
    U = (Q @ U)[:, :k]
    return U, (U.T @ Y).reshape(k, b, q)


def cp_path(X: np.ndarray, R: int, starts: int, rng, comp=None) -> list:
    """CP-ALS at rank R from the HOSVD start and `starts` random starts on the patient-compressed tensor; returns
    [(rel. error on X, start name, (A~, B, C))] sorted by error, plus the compressed tensor."""
    U, Xc = comp if comp is not None else compress_patients(X, S16_CP_COMPRESS, rng)
    nx = float((X ** 2).sum())
    resid = max(0.0, nx - float((Xc ** 2).sum()))
    nc = float((Xc ** 2).sum())
    out = []
    for nm, init in cp_inits(Xc, R, starts, rng):
        A, B, C, e = cp_als(Xc, R, init, S16_CP_ITERS, S16_CP_TOL)
        out.append((math.sqrt((resid + e * e * nc) / max(nx, 1e-300)), nm, (A, B, C)))
    out.sort(key=lambda t: t[0])
    return out, Xc


def corcondia(X: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """Core consistency (Bro and Kiers 2003): the least-squares Tucker core of X on the CP factors against the
    superdiagonal; 100 is a perfect trilinear model."""
    R = A.shape[1]
    s = [np.linalg.norm(F, axis=0) for F in (A, B, C)]
    A1, B1, C1 = A / np.where(s[0] > 0, s[0], 1), B / np.where(s[1] > 0, s[1], 1), C / np.where(s[2] > 0, s[2], 1)
    lam = s[0] * s[1] * s[2]
    G = _es("nbq,rn,sb,tq->rst", X, np.linalg.pinv(A1 * lam[None, :]), np.linalg.pinv(B1), np.linalg.pinv(C1))
    T = np.zeros((R, R, R))
    T[np.arange(R), np.arange(R), np.arange(R)] = 1.0
    return float(100.0 * (1.0 - ((G - T) ** 2).sum() / R))


def congruence(F1: tuple, F2: tuple) -> float:
    """Tucker congruence of two CP solutions over the given modes, components matched (Hungarian) on the product of
    the per-mode absolute cosines; returns the smallest matched product."""
    from scipy.optimize import linear_sum_assignment
    R = F1[0].shape[1]
    P = np.ones((R, R))
    for a, b in zip(F1, F2):
        an = a / np.where(np.linalg.norm(a, axis=0) > 0, np.linalg.norm(a, axis=0), 1)
        bn = b / np.where(np.linalg.norm(b, axis=0) > 0, np.linalg.norm(b, axis=0), 1)
        P *= np.abs(an.T @ bn)
    i, j = linear_sum_assignment(-P)
    return float(P[i, j].min())


def cp_inits(X: np.ndarray, R: int, starts: int, rng) -> list:
    n, b, q = X.shape
    out = []
    hs = []
    for mode, dim in ((0, n), (1, b), (2, q)):
        if mode == 0:
            Y = X.reshape(n, -1)
            U = np.linalg.svd(Y, full_matrices=False)[0]
        else:
            U = mode_basis(X, mode, dim)
        k = min(R, U.shape[1])
        F = np.zeros((dim, R))
        F[:, :k] = U[:, :k]
        if k < R:
            F[:, k:] = rng.normal(size=(dim, R - k)) / math.sqrt(dim)
        hs.append(F)
    out.append(("hosvd", tuple(hs)))
    for s in range(starts):
        out.append((f"random_{s + 1}", tuple(rng.normal(size=(d, R)) for d in (n, b, q))))
    return out


def principal_angles(Q1: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    a = np.linalg.qr(Q1)[0]
    b = np.linalg.qr(Q2)[0]
    s = np.clip(np.linalg.svd(a.T @ b, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(s))


def organ_alignment(U3: np.ndarray, organs: list, top: int, rng, draws: int = 1000) -> tuple:
    """Mean share of each factor's top-loading channels that fall in that factor's modal organ block, and its
    permutation p-value (organ labels permuted across channels)."""
    org = np.array(organs)

    def stat(lbl):
        v = []
        for j in range(U3.shape[1]):
            t = np.argsort(-np.abs(U3[:, j]))[:top]
            _, cnt = np.unique(lbl[t], return_counts=True)
            v.append(cnt.max() / len(t))
        return float(np.mean(v))

    s0 = stat(org)
    null = np.array([stat(rng.permutation(org)) for _ in range(draws)])
    return s0, float((np.sum(null >= s0) + 1) / (draws + 1)), float(np.mean(null))


def cpqr_pivots(Y: np.ndarray, k: int) -> list:
    """First k column pivots of column-pivoted QR (Businger-Golub), by greedy modified Gram-Schmidt."""
    R = np.array(Y, dtype=np.float64, copy=True)
    nrm = (R ** 2).sum(axis=0)
    piv = []
    for _ in range(k):
        j = int(np.argmax(nrm))
        if nrm[j] <= 0:
            break
        piv.append(j)
        v = R[:, j] / math.sqrt(nrm[j])
        R -= np.outer(v, v @ R)
        nrm = (R ** 2).sum(axis=0)
        nrm[piv] = -1.0
    return piv


def deim(V: np.ndarray) -> list:
    """DEIM row selection on the leading right singular vectors (columns of V)."""
    sel = [int(np.argmax(np.abs(V[:, 0])))]
    for j in range(1, V.shape[1]):
        Vj = V[:, :j]
        c = np.linalg.solve(Vj[sel, :], V[sel, j])
        r = V[:, j] - Vj @ c
        sel.append(int(np.argmax(np.abs(r))))
    return sel


def stage16(D: dict, mats: OrderedDict, FL: FitLog, cfg: Config, checks: dict, info: dict) -> dict:
    spec_rows, tuck_rows, cp_rows, ang_rows, load_rows, org_rows, flat_rows = [], [], [], [], [], [], []
    internal = OrderedDict()
    bases2, bases3 = OrderedDict(), OrderedDict()
    for L in R2.LANDMARKS:
        t0 = time.time()
        T = s16_tensors(D, L)
        chans, sp = T["channels"], T["split"]
        itr, iva = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "validation")
        FL.rows(D["lc"][L]["idx"][itr], f"stage 16 L{L} train tensor")
        Xn = T["normalized"][itr]
        b, q = Xn.shape[1], Xn.shape[2]
        bases2[L] = mode_basis(Xn, 1, min(S16_ANGLE_R2, b))
        bases3[L] = (mode_basis(Xn, 2, min(S16_ANGLE_R3, q)), chans)
        if L not in cfg.s16_landmarks:
            continue
        organs = channel_organs(mats, chans)
        rng = R2.rng_for("s16", L)
        sub = np.sort(rng.choice(len(itr), min(S16_MAX_PATIENTS, len(itr)), replace=False))
        for lvl in ("normalized", "engineered"):
            X = T[lvl][itr]
            for mode, nm in ((0, "patients"), (1, "bins"), (2, "channels")):
                s = mode_svals(X, mode)
                kind = "bin" if mode == 1 else "channel"
                nulls = np.array([mode_svals(null_tensor(X, kind, R2.rng_for("s16null", L, lvl, mode, d)), mode)
                                  for d in range(S16_NULL_DRAWS if mode else S16_NULL_DRAWS_PATIENT)])
                thr = np.percentile(nulls, 95, axis=0)
                kk = min(len(s), thr.shape[0])
                row = OrderedDict(landmark=L, level=lvl, mode=nm, null=("per-bin patient shuffle" if kind == "bin"
                                                                        else "channel-wise patient permutation"),
                                  **spec_stats(s), above_null=int(np.sum(s[:kk] > thr[:kk])))
                for j in range(min(10, len(s))):
                    row[f"sv_share_{j + 1}"] = float(s[j] ** 2 / max((s ** 2).sum(), 1e-300))
                spec_rows.append(row)
            check_time()
        # Tucker error surface. Grid: HOSVD bin and channel factors (train), patient factor optimal for each r1
        # (one SVD gives every r1). HOOI (5 sweeps) at the reference ranks; raw level by masked EM-HOOI at the
        # reference ranks only (train subsample). Validation projected on the train bin / channel factors.
        ref = (20, min(8, b), None)
        for lvl in ("normalized", "engineered"):
            Xs, Xv = T[lvl][itr], T[lvl][iva]
            qq = Xs.shape[2]
            nx = float((Xs ** 2).sum())
            r2g = sorted({r for r in (1, 2, 4, 8, 16, b) if r <= b})
            r3g = sorted({r for r in (5, 10, 20, 40, qq) if r <= qq})
            V2, V3 = mode_basis(Xs, 1, b), mode_basis(Xs, 2, qq)
            for r2 in r2g:
                for r3 in r3g:
                    Y = _es("nbq,bs,qt->nst", Xs, V2[:, :r2], V3[:, :r3]).reshape(Xs.shape[0], -1)
                    sv = np.linalg.svd(Y, compute_uv=False) ** 2
                    ev = proj_error(Xv, V2[:, :r2], V3[:, :r3])
                    for r1 in S16_R1:
                        if r1 > Xs.shape[0] or r1 > r2 * r3:
                            continue
                        tuck_rows.append(OrderedDict(landmark=L, level=lvl, method="HOSVD bins/channels, optimal patients",
                                                     r1=r1, r2=r2, r3=r3,
                                                     rel_error_train=math.sqrt(max(0.0, nx - sv[:r1].sum()) / nx),
                                                     rel_error_validation_projected=ev))
            check_time()
            r3r = min(20, qq)
            F, G, err = hooi(Xs[sub], (ref[0], ref[1], r3r), S16_HOOI_ITERS)
            tuck_rows.append(OrderedDict(landmark=L, level=lvl, method="HOOI (train subsample)", r1=ref[0], r2=ref[1],
                                         r3=r3r, rel_error_train=err,
                                         rel_error_validation_projected=proj_error(Xv, F[1], F[2])))
        Xs, Xv = T["raw"][itr][sub], T["raw"][iva]
        mask, mv = np.isfinite(Xs), np.isfinite(Xv)
        r3r = min(20, Xs.shape[2])
        F, G, err = hooi(np.nan_to_num(Xs), (ref[0], ref[1], r3r), 3 * S16_HOOI_ITERS, mask)
        tuck_rows.append(OrderedDict(landmark=L, level="raw", method="masked EM-HOOI (train subsample)", r1=ref[0],
                                     r2=ref[1], r3=r3r, rel_error_train=err,
                                     rel_error_validation_projected=proj_error(np.nan_to_num(Xv), F[1], F[2], mv)))
        check_time()
        # CP-ALS (normalized level, all train patients, patient mode compressed): rank 1..15, HOSVD + random starts,
        # CORCONDIA, congruence across starts
        X = T["normalized"][itr]
        comp = compress_patients(X, S16_CP_COMPRESS, R2.rng_for("s16comp", L))
        best = OrderedDict()
        for R in S16_CP_R:
            if R > min(X.shape[1] * X.shape[2], comp[1].shape[0]):
                continue
            sols, Xc = cp_path(X, R, cfg.s16_cp_starts, R2.rng_for("s16cp", L, R), comp)
            check_time()
            e0, n0, F0 = sols[0]
            cc = corcondia(Xc, *F0)
            cong = [congruence(F0, F) for _, _, F in sols[1:]]
            best[R] = (cc, F0, float(np.median(cong)) if cong else np.nan)
            cp_rows.append(OrderedDict(landmark=L, level="normalized", rank=R, rel_error_best=e0, best_start=n0,
                                       core_consistency=cc / 100.0, congruence_starts_median=float(np.median(cong)) if cong else np.nan,
                                       congruence_starts_share_ge_threshold=float(np.mean(np.array(cong) >= S16_CONGRUENCE))
                                       if cong else np.nan))
        Rsel = select_cp_rank({R: (v[0], v[2]) for R, v in best.items()})
        # subject bootstraps at the selected rank: congruence of the bin and channel factors
        g = D["rows"]["subj"][D["lc"][L]["idx"]][itr]
        F0 = best[Rsel][1]
        bc = []
        for bb in range(cfg.s16_boot):
            jb = boot_rows(g, R2.rng_for("s16boot", L, bb))
            sols, _ = cp_path(X[jb], Rsel, 0, R2.rng_for("s16bi", L, bb))
            bc.append(congruence(F0[1:], sols[0][2][1:]))
            check_time()
        bc = np.array(bc) if bc else np.array([np.nan])
        cp_rows.append(OrderedDict(landmark=L, level="normalized", rank=Rsel, selected=True,
                                   core_consistency=best[Rsel][0] / 100.0, bootstrap_congruence_median=float(np.nanmedian(bc)),
                                   bootstrap_congruence_q25=float(np.nanpercentile(bc, 25)),
                                   bootstrap_congruence_share_ge_threshold=float(np.nanmean(bc >= S16_CONGRUENCE)),
                                   bootstraps=int(cfg.s16_boot)))
        # factor loadings over channels and bins (train factors only; never patient loadings)
        U3, _ = bases3[L]
        U2 = mode_basis(Xn, 1, min(4, b))
        internal[f"s16__L{L}__U2"] = U2.astype(np.float32)
        internal[f"s16__L{L}__U3"] = U3.astype(np.float32)
        internal[f"s16__L{L}__cp_B"] = F0[1].astype(np.float32)
        internal[f"s16__L{L}__cp_C"] = F0[2].astype(np.float32)
        for j in range(min(3, U3.shape[1])):
            sgn = np.sign(U3[np.argmax(np.abs(U3[:, j])), j]) or 1.0
            for pos, c in enumerate(np.argsort(-np.abs(U3[:, j]))[:8]):
                load_rows.append(OrderedDict(landmark=L, level="normalized", mode="channels", factor=j + 1,
                                             position=pos + 1, item=chans[c], organ=organs[c],
                                             loading=float(sgn * U3[c, j])))
        for j in range(U2.shape[1]):
            sgn = np.sign(U2[np.argmax(np.abs(U2[:, j])), j]) or 1.0
            for bi in range(b):
                load_rows.append(OrderedDict(landmark=L, level="normalized", mode="bins", factor=j + 1,
                                             position=bi + 1, item=f"bin {bi + 1}", organ="",
                                             loading=float(sgn * U2[bi, j])))
        s0, pv, nm0 = organ_alignment(U3, organs, S16_TOPLOAD, R2.rng_for("s16org", L))
        org_rows.append(OrderedDict(landmark=L, level="normalized", factors=U3.shape[1], top_channels=S16_TOPLOAD,
                                    modal_block_share=s0, null_mean=nm0, p_permutation=pv))
        # flattened baseline (labeled): effective rank as bins are added, and CPQR / DEIM of (bin, channel) pairs
        Xf = Xn[sub]
        steps = sorted({s for s in (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56) if s <= b} | {b})
        for bb in steps:
            Y = Xf[:, :bb, :].reshape(Xf.shape[0], -1)
            s = mode_svals(Xf[:, :bb, :], 0)
            sn = mode_svals(null_tensor(Xf[:, :bb, :], "bin", R2.rng_for("s16flat", L, bb)), 0)
            flat_rows.append(OrderedDict(label="flattened baseline (never chooses a rank, level or model)",
                                         landmark=L, bins_used=bb, **spec_stats(s),
                                         erank_sigma2_null=spec_stats(sn)["erank_sigma2"],
                                         erank_ratio_to_columns=spec_stats(s)["erank_sigma2"] / (bb * q)))
        Y = Xf.reshape(Xf.shape[0], -1)
        kf = min(S16_FLAT_K, min(Y.shape))
        piv = cpqr_pivots(Y, kf)
        Uc, Yc = compress_patients(Xf, kf + 20, R2.rng_for("s16deim", L))
        Vt = np.linalg.svd(Yc.reshape(Yc.shape[0], -1), full_matrices=False)[2]
        dsel = deim(Vt[:kf].T)
        topc = {chans[c] for j in range(U3.shape[1]) for c in np.argsort(-np.abs(U3[:, j]))[:S16_TOPLOAD]}
        for meth, sel in (("CPQR", piv), ("DEIM", dsel)):
            for pos, cidx in enumerate(sel):
                bi, ci = divmod(int(cidx), q)
                flat_rows.append(OrderedDict(label="flattened baseline (never chooses a rank, level or model)",
                                             landmark=L, method=meth, position=pos + 1,
                                             item=f"bin {bi + 1} / {chans[ci]}",
                                             channel_in_tucker_top_loadings=chans[ci] in topc))
        say(f"stage 16: landmark {L} done ({time.time() - t0:.0f} s)")
    # principal angles between consecutive landmarks (all 14; normalized level)
    Ls = list(R2.LANDMARKS)
    for L1, L2 in zip(Ls[:-1], Ls[1:]):
        b1 = bases2[L1].shape[0]
        r = min(bases2[L1].shape[1], bases2[L2].shape[1])
        a2 = principal_angles(bases2[L1][:, :r], bases2[L2][:b1, :r])
        U31, c1 = bases3[L1]
        U32, c2 = bases3[L2]
        sh = [c for c in c1 if c in c2]
        r3 = min(U31.shape[1], U32.shape[1], len(sh))
        a3 = principal_angles(U31[[c1.index(c) for c in sh], :r3], U32[[c2.index(c) for c in sh], :r3])
        ang_rows.append(OrderedDict(landmark_from=L1, landmark_to=L2, mode="bins (shared prefix)", rank=r,
                                    angle_max_deg=float(a2.max()), angle_mean_deg=float(a2.mean())))
        ang_rows.append(OrderedDict(landmark_from=L1, landmark_to=L2, mode="channels (shared)", rank=r3,
                                    angle_max_deg=float(a3.max()), angle_mean_deg=float(a3.mean())))
    checks["16_no_patient_loadings_published"] = True
    return dict(spectra=pd.DataFrame(spec_rows), tucker=pd.DataFrame(tuck_rows), cp=pd.DataFrame(cp_rows),
                angles=pd.DataFrame(ang_rows), loadings=pd.DataFrame(load_rows), organ=pd.DataFrame(org_rows),
                flat=pd.DataFrame(flat_rows), internal=internal)


def select_cp_rank(rows: dict) -> int:
    """Largest R whose best fit has CORCONDIA >= 80 % and whose random starts agree with it (median congruence
    >= 0.85; D18 thresholds). On large-n data with unstructured noise CORCONDIA alone stays high when the model is
    over-factored (box experiments); the across-start congruence drops to near zero there."""
    ok = [R for R, (cc, cg) in rows.items() if cc >= S16_CORCONDIA and (not np.isfinite(cg) or cg >= S16_CONGRUENCE)]
    return max(ok) if ok else 1


def planted_tensor_gate() -> OrderedDict:
    """Plan 12.3 synthetic gate (seeded, in-script): a planted CP rank-3 tensor plus noise is recovered (the stage 16
    rank rule selects 3; the HOOI error at multilinear rank (3, 3, 3) is at the noise floor), the planted bin factors
    are recovered (principal angles under 10 degrees), and extreme values in unobserved cells change nothing in the
    masked fit."""
    rng = np.random.default_rng(16016)
    n, b, q, R = 400, 8, 10, 3
    A, B, C = rng.normal(size=(n, R)), rng.normal(size=(b, R)), rng.normal(size=(q, R))
    S = _es("nr,br,qr->nbq", A, B, C)
    S = S / np.sqrt((S ** 2).mean())
    noise = 0.2 * rng.normal(size=S.shape)
    X = S + noise
    rows = {}
    for r in range(1, 6):
        sols, Xc = cp_path(X, r, 5, np.random.default_rng(r))
        cg = float(np.median([congruence(sols[0][2], s_[2]) for s_ in sols[1:]]))
        rows[r] = (corcondia(Xc, *sols[0][2]), cg)
    sel = select_cp_rank(rows)
    _, _, e3 = hooi(X, (3, 3, 3), 5)
    floor = math.sqrt(float((noise ** 2).sum()) / float((X ** 2).sum()))
    ang = principal_angles(mode_basis(X, 1, 3), B)
    mask = rng.uniform(size=X.shape) > 0.2
    Xa, Xb = np.where(mask, X, 0.0), np.where(mask, X, 1e6)
    _, _, ea = hooi(Xa, (3, 3, 3), 4, mask)
    _, _, eb = hooi(Xb, (3, 3, 3), 4, mask)
    return OrderedDict(cp_rank_recovered=bool(sel == 3), tucker_rank3_at_noise_floor=bool(e3 <= 1.05 * floor),
                       planted_bin_factors_recovered=bool(ang.max() < 10.0),
                       unobserved_extreme_values_change_nothing=bool(abs(ea - eb) < 1e-9))

# ------------------------------------------------------------------------------------------- decisions, gate
DECISIONS = OrderedDict([
    ("D16-D21", "applied as in Run 2 (dedup by Run 1 column names; Stage 08 base Layer C, D17; D18 thresholds incl. "
                "congruence 0.85 and CORCONDIA 80 %; the D19 GRU; D20 budgets restored on the desktop: "
                "stage 13 GB B = 50, stage 18 GRU B = 20 at all seven core landmarks, no 60-minute cap)"),
    ("no test rows", "plan 11 and 6.7: attribution uses subject-bootstrap refits of train and a fixed seeded "
                     "validation sample; stage 16 fits on train and projects validation; test rows are dropped at "
                     "load, so Run 3b needs no Run 3a output; stages 13, 16 and 18 run together as one run"),
    ("SHAP target", "E[Y | x] = sum_j P(Y > j | x) (plan 11); ordinal LR: exact interventional linear SHAP of the "
                    "linear predictor (train-mean background) carried to E[Y] by the rescale rule; Frank-Hall "
                    "gradient boosting: exact path-dependent TreeSHAP of each threshold margin carried to "
                    "P(Y > j | x) by the rescale rule and summed (E[Y] is unchanged by the monotone rearrangement); "
                    "home-made (no shap package on the laptop), checked against brute-force Shapley values"),
    ("H7", "median Kendall tau falls from phi1 (b1) to phi2-dup (b2d) by at least 0.10, more in the grouped than in "
           "the ungrouped columns, and phi2-single (b2s) within 0.05 of phi1 (plan 12.2 bracketed values)"),
    ("stage 18", "expected-gradients SHAP (GradientSHAP) of E[Y | x] for the D19 GRU A+B+C, Outcome 1, at all "
                 "seven core landmarks 24, 48, 72, 120, 168, 240 and 336 h (tables may still lead with the later "
                 "core landmarks in Vinai's H9 framing); refits on subject "
                 "bootstraps of train with the frozen Run 2 configuration; permutation importance of the frozen "
                 "Run 2 model as the cross-check; TreeSHAP of the frozen A+B landmark models"),
    ("stage 16", "train tensors at the seven core landmarks (all 14 for the angle table); normalized, raw (masked "
                 "EM fit) and engineered-lite (normalized + masks and carried-forward flags as channels) levels; "
                 "validation projected on fixed train factors; nothing refitted on validation"),
])
DEVIATIONS = [
    "desktop full-settings run (v2.0.2-desktop): stages 13, 16 and 18 run together as one run (no --part split); "
    "no 60-minute time cap; stage 13 gradient boosting B = 50 and stage 18 GRU B = 20 at all seven core landmarks "
    "24, 48, 72, 120, 168, 240 and 336 h (D20 restored); stage 13 ordinal LR keeps B = 50; training and "
    "expected-gradients attribution use CUDA if available, otherwise CPU (logged; seeds set on CPU and CUDA); "
    "frozen A+B+C reproduction forward passes run on CPU in float32, matching the laptop Run 2, even when CUDA "
    "is available; every file this script writes is under "
    "data/processed/ladder/DESKTOP_FULL_PLAN/ (published folder, pending folder, cache, internal files and "
    "README_DESKTOP.txt)",
    "frozen A+B+C reproduction uses a 1e-4 probability tolerance on a CPU float32 forward pass; the measured max "
    "abs probability differences are written in the audit and in the progress and failure lines. The frozen "
    "stage 12 winner check is a sklearn predict_proba of a saved model (not a torch forward pass) and keeps the "
    "same 1e-4 tolerance",
    "the preceding v2.0.0-desktop Run 3a attempt on this desktop failed during validation-only verification, "
    "before any test row was read; the desktop test-use log is unchanged. This Run 3b does not score test rows",
    "stage 13 ordinal LR at every level (a0-b2p) and on the winner's matrix; the tabular winner (gradient "
    "boosting, tuned stage 12 hyperparameters) on b1, b2d and the winner's matrix, all at full width; b2s gradient "
    "boosting is read off the b2d refits when the b2d trees never split on a phi2-dup copy and the full-train b2s "
    "and b2d fits coincide (otherwise refitted)",
    "stage 13 Outcome 2 (secondary): ordinal LR on b1, b2s and b2d only",
    "stage 13 interventional vs conditional gap on the full-train fit only (not per refit), on a sub-sample of the "
    "fixed validation sample against a fixed train background",
    "stage 13 ordinal LR refits are warm-started at the full-train solution (same objective and optimum)",
    "stage 18 credit split across temporally adjacent bins summarised as the share of the last bin within the last "
    "two bins for the ten most important channels (spread across refits), plus the last-day share",
    "stage 18 attributions to masks, carried-forward flags, the bin index, the static A+B context and the Layer A "
    "offset are reported as input-type shares; the bin x channel, channel and organ granularities sum each "
    "channel's value, mask and carried-forward inputs",
    "stage 16: HOOI grid r1 in {5, 10, 20, 40}, r2 in {1, 2, 4, 8, 16, b_L}, r3 in {5, 10, 20, 40, q} (capped), 5 "
    "sweeps from the HOSVD start (15 EM sweeps for the raw level); HOOI and CP fits on a seeded sub-sample of at "
    "most 2000 train patients (spectra, nulls and angles use all train patients); CP-ALS on the normalized level "
    "only, at most 60 sweeps",
    "stage 16 CP rank: the largest R whose best fit has CORCONDIA >= 80 % and median congruence across starts >= "
    "0.85 (both D18 thresholds); on large-n data with unstructured noise CORCONDIA alone stays high when the model is "
    "over-factored (synthetic experiments), the congruence across starts does not",
    "stage 16 nulls use 20 draws; the H17 padding probe is deferred (it needs the full 56-bin Stage 3 grid; the "
    "Stage 08 prefix tensors contain no padding); the optional loading-based model is omitted",
]

_STRUCT_COL3 = re.compile(r"^(k|p|landmark|landmark_from|landmark_to|class|threshold|hidden|epochs_trained|refits|"
                          r"members|position|rank|r1|r2|r3|factor|factors|bins_used|bootstraps|top_channels|k95|"
                          r"above_null)$")
NAME_INTS: set = set()                   # digits inside Stage 08 column and channel names (schema, not counts)


def audit_num_keys() -> tuple:
    return tuple(R2.AUDIT_NUM_KEYS) + ("repro", "tau", "fall", "threshold", "abs_diff", "refits", "boot", "share",
                                       "seed", "budget", "sample", "background", "draws", "landmark", "rank",
                                       "corcondia", "core_consistency", "congruence", "angle", "max_seconds", "h7", "r1", "cp", "null",
                                       "s13_refits", "s13_tree_refits", "s13_sample", "s13_gap_sample", "s13_background",
                                       "s16_boot", "s16_cp_starts", "s16_landmarks", "s16_null_draws", "s18_background",
                                       "s18_draws", "s18_landmarks", "s18_refits", "s18_sample", "tv_distance",
                                       "modal_block_share", "p_permutation", "adjacent_bin")


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
    """Key paths (never values) of numbers outside the structural whitelist."""
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
        except Exception:  # noqa: BLE001  (fail closed: no template exemption)
            pass
    return {int(t) for t in R2.id_tokens(txt) if t.isdigit()}


def gate_problems(pend: Path, team_files: list, forbidden: set, ids: set) -> list:
    """Run 2 release-gate rules (v1.0.2 / Run 2) on every TEAM file: collect-all, file and location only."""
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
                    if not size_hit and R2._size_hit(cell, forbidden - NAME_INTS if c in ('item', 'label', 'group') else forbidden):
                        size_hit = True
                    if not id_hit:
                        if R2._NUMERIC_CELL.match(cell):
                            id_hit = bool(re.fullmatch(r"[-+]?\d+", cell)) and cell.lstrip("+") in ids
                        else:
                            toks = R2.id_tokens(cell)
                            if c in ("item", "label", "family", "group"):
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
    """Every size that must never appear in a TEAM file (fail closed): cohort, subjects, splits, each landmark's
    eligible set by split (and train + validation, and stacked totals), assignable sets, class counts and the
    per-threshold class splits and deaths of every scored set."""
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


def figures(T: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {}
    if "stage13_tau_TEAM.csv" in T:
        out.update(_fig13(T))
    if "stage18_seq_importance_TEAM.csv" in T:
        out.update(_fig18(T))
    if "stage16_angles_TEAM.csv" in T:
        out.update(_fig16(T))
    return out


def _fig13(T: dict) -> dict:
    import matplotlib.pyplot as plt
    out = {}
    t = T["stage13_tau_TEAM.csv"]
    t = t[t["outcome"] == "outcome1"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, (mod, g) in enumerate(t.groupby("model", sort=False)):
        x = np.arange(len(g)) + 0.18 * i
        md = g["tau_median"].astype(float).to_numpy()
        lo, hi = g["tau_q25"].astype(float).to_numpy(), g["tau_q75"].astype(float).to_numpy()
        ax.errorbar(x, md, yerr=[md - lo, hi - md], fmt="o", capsize=3, ms=4, label=mod)
        for xx, m in zip(x, g["matrix"]):
            ax.annotate(str(m), (xx, 0.0), xycoords=("data", "axes fraction"), fontsize=6, rotation=90,
                        ha="center", va="bottom")
    ax.set_xticks([])
    ax.set_ylabel("Kendall tau between refits (median, IQR)")
    ax.set_title("Stage 13: attribution stability by level (Outcome 1, 24 h rows)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, metadata={"Software": None})
    plt.close(fig)
    out["fig13_1_tau_by_level_TEAM.png"] = buf.getvalue()
    return out


def _fig18(T: dict) -> dict:
    import matplotlib.pyplot as plt
    out = {}
    s = T["stage18_seq_importance_TEAM.csv"]
    s = s[s["granularity"] == "input_type"]
    if len(s):
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        Ls = sorted(s["landmark"].astype(int).unique())
        bottom = np.zeros(len(Ls))
        for it in INPUT_TYPES:
            v = np.array([float(s[(s["landmark"].astype(int) == L) & (s["item"] == it)]["share_median"].iloc[0])
                          for L in Ls])
            ax.bar([str(L) for L in Ls], v, bottom=bottom, label=it)
            bottom += v
        ax.set_xlabel("landmark (h)")
        ax.set_ylabel("share of mean |SHAP| (median over refits)")
        ax.set_title("Stage 18: GRU A+B+C attribution by input type (validation sample)", fontsize=9)
        ax.legend(fontsize=6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, metadata={"Software": None})
        plt.close(fig)
        out["fig18_2_attribution_by_input_type_TEAM.png"] = buf.getvalue()
    return out


def _fig16(T: dict) -> dict:
    import matplotlib.pyplot as plt
    out = {}
    a = T["stage16_angles_TEAM.csv"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for mode, g in a.groupby("mode", sort=False):
        ax.plot(g["landmark_to"].astype(int), g["angle_max_deg"].astype(float), "o-", ms=3, label=f"{mode} (max)")
    ax.set_xlabel("landmark (h), compared with the previous landmark")
    ax.set_ylabel("largest principal angle (degrees)")
    ax.set_title("Stage 16: stability of the Layer C bin and channel subspaces across landmarks", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, metadata={"Software": None})
    plt.close(fig)
    out["fig16_1_principal_angles_TEAM.png"] = buf.getvalue()
    return out


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return R2.r6(v) if math.isfinite(v) else None



def _rows(df: pd.DataFrame, cols: list) -> list:
    out = []
    for _, r in df.iterrows():
        out.append(OrderedDict((c, (_f(r[c]) if isinstance(r[c], (int, float, np.integer, np.floating))
                                    and not isinstance(r[c], (bool, np.bool_)) else
                                    (bool(r[c]) if isinstance(r[c], (bool, np.bool_)) else str(r[c]))))
                               for c in cols if c in r.index))
    return out


def make_audit(cfg, pre, checks, info, S13, S18, S16, gates, script_sha, n_written) -> dict:
    E = pd.DataFrame()
    S13 = S13 if S13 is not None else dict(tau=E, h7=E, gap=E)
    S18 = S18 if S18 is not None else dict(tau=E, cross=E)
    S16 = S16 if S16 is not None else dict(cp=E, organ=E)
    tau = S13["tau"]
    cp = S16["cp"]
    sel = cp[cp.get("selected", pd.Series(False, index=cp.index)).fillna(False).astype(bool)] if "selected" in cp else cp.iloc[0:0]
    return OrderedDict(
        script=SCRIPT_VERSION, script_sha256=script_sha, spec=SPEC_DOC, plan=PLAN, contract=CONTRACT,
        part="all", part_stages="13, 16, 18",
        generated_utc=utc_now(), input_hashes=pre["hashes"], decisions=DECISIONS, deviations=DEVIATIONS,
        configuration=OrderedDict(seed=R2.SEED, s13_refits=cfg.s13_refits, s13_tree_refits=cfg.s13_tree_refits,
                                  s13_sample=S13_SAMPLE, s13_gap_sample=S13_GAP_SAMPLE,
                                  s13_background=S13_BACKGROUND, s18_refits=cfg.s18_refits,
                                  s18_landmarks=list(S18_LANDMARKS), s18_sample=S18_SAMPLE,
                                  s18_background=S18_BACKGROUND, s18_draws=S18_EG_DRAWS,
                                  s16_landmarks=list(cfg.s16_landmarks), s16_cp_starts=cfg.s16_cp_starts,
                                  s16_boot=cfg.s16_boot, s16_null_draws=S16_NULL_DRAWS,
                                  threshold_congruence=S16_CONGRUENCE, threshold_core_consistency=S16_CORCONDIA / 100.0,
                                  h7_fall=H7_FALL, h7_pruned=H7_PRUNED, max_seconds=cfg.max_seconds,
                                  torch_device=str(torch_device()) if R2 is not None else "not recorded"),
        reproduction=OrderedDict((k, _f(v)) for k, v in info.items() if k.startswith("repro")),
        s13_tree_b2s_read_off_b2d=bool(info.get("s13_tree_b2s_read_off_b2d", False)),
        s13_tau=_rows(tau, ["outcome", "model", "matrix", "refits", "tau_median", "tau_q25", "tau_q75",
                            "tau_median_grouped", "tau_median_ungrouped"]),
        h7=_rows(S13["h7"], ["model", "tau_b1", "tau_b2d", "tau_b2s", "fall_b1_to_b2d", "fall_grouped",
                             "fall_ungrouped", "abs_diff_b2s_vs_b1", "h7_met"]),
        s13_gap=_rows(S13["gap"], ["outcome", "model", "matrix", "tau_conditional_vs_interventional",
                                   "tv_distance_all"]),
        s18_tau=_rows(S18["tau"], ["landmark", "granularity", "refits", "tau_median", "tau_q25", "tau_q75"]),
        s18_cross=_rows(S18["cross"], ["landmark", "tau_gradient_vs_permutation_frozen",
                                       "tau_frozen_vs_refits_median", "adjacent_bin_last_share_iqr_mean"]),
        s16_cp_selected=_rows(sel, ["landmark", "rank", "core_consistency", "bootstrap_congruence_median",
                                    "bootstrap_congruence_share_ge_threshold"]),
        s16_organ=_rows(S16["organ"], ["landmark", "modal_block_share", "null_mean", "p_permutation"]),
        gates=OrderedDict((k, "pass" if v else "fail") for k, v in gates.items()),
        acceptance_checks=OrderedDict((k, "pass" if v else "fail") for k, v in checks.items()),
        n_files=n_written, release_gate="ok")


def _n(x):
    return "nan" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))


def audit_md(a: dict) -> str:
    st = set(stages())
    L = ["# Stages 13, 16 and 18 attribution (Run 3b desktop full settings) audit - TEAM", "",
         f"- script: {a['script']} (`{a['script_sha256']}`); spec {a['spec']}; plan {a['plan']}, contract {a['contract']}",
         "- one run of stages 13, 16 and 18 together (no part split, no 60-minute cap); outputs under "
         "data/processed/ladder/DESKTOP_FULL_PLAN/",
         f"- generated (UTC): {a['generated_utc']}",
         "- train and validation rows only (no test row is read); aggregate only: importance by column name, "
         "channel, bin or organ block, no per-row SHAP, no patient loadings, no patient, row, encounter or subject "
         "counts and no identifiers; the release gate re-read every TEAM file from disk", "",
         "## Input hashes", ""]
    L += [f"- {k}: `{v}`" for k, v in a["input_hashes"].items()]
    L += ["", "## Decisions applied", ""] + [f"- **{k}**: {v}" for k, v in a["decisions"].items()]
    L += ["", "## Deviations from the plan (logged)", ""] + [f"- {d}" for d in a["deviations"]]
    c = a["configuration"]
    L += ["", "## Configuration", ""] + [f"- {k}: {v}" for k, v in c.items()]
    L += ["- no time limit" if not c.get("max_seconds") else f"- time limit {c['max_seconds']} s"]
    L += ["", "## Frozen-model reproduction (validation)", ""] + [f"- {k}: {_n(v)}" for k, v in a["reproduction"].items()]
    if 13 in st:
        L += _md13(a)
    if 18 in st:
        L += _md18(a)
    if 16 in st:
        L += _md16(a)
    L += ["", "## Synthetic gates (in-script, seeded)", ""] + [f"- {k}: {v}" for k, v in a["gates"].items()]
    L += ["", "## Acceptance checks", ""] + [f"- {k}: {v}" for k, v in a["acceptance_checks"].items()]
    det = ", ".join(f"stage{k}_*_TEAM.csv" for k in sorted(st))
    L += ["", f"Release gate: ok. Detail tables: {det}; "
          "*_INTERNAL files are team-internal (never opened, pasted or shared)."]
    return "\n".join(L) + "\n"


def _md13(a: dict) -> list:
    L = ["", "## Stage 13: Kendall tau between refits (24 h rows, validation sample)", "",
          "| outcome | model | matrix | refits | tau median [IQR] | grouped | ungrouped |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['outcome']} | {r['model']} | {r['matrix']} | {r['refits']} | {_n(r['tau_median'])} "
          f"[{_n(r['tau_q25'])}, {_n(r['tau_q75'])}] | {_n(r.get('tau_median_grouped'))} | "
          f"{_n(r.get('tau_median_ungrouped'))} |" for r in a["s13_tau"]]
    L += ["", f"- gradient boosting b2s read off the b2d refits: {a['s13_tree_b2s_read_off_b2d']}", "",
          "## H7 (attribution stability; Outcome 1)", "",
          "| model | tau b1 | tau b2d | tau b2s | fall b1->b2d | fall grouped | fall ungrouped | abs(b2s - b1) | supported |",
          "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['model']} | {_n(r['tau_b1'])} | {_n(r['tau_b2d'])} | {_n(r['tau_b2s'])} | {_n(r['fall_b1_to_b2d'])} | "
          f"{_n(r['fall_grouped'])} | {_n(r['fall_ungrouped'])} | {_n(r['abs_diff_b2s_vs_b1'])} | {r['h7_met']} |"
          for r in a["h7"]]
    L += ["", "## Stage 13: interventional vs conditional TreeSHAP (full-train fit)", "",
          "| outcome | model | matrix | tau(cond, interv) | TV distance |", "|---|---|---|---|---|"]
    L += [f"| {r['outcome']} | {r['model']} | {r['matrix']} | {_n(r['tau_conditional_vs_interventional'])} | "
          f"{_n(r['tv_distance_all'])} |" for r in a["s13_gap"]]
    return L


def _md18(a: dict) -> list:
    L = ["", "## Stage 18: GRU A+B+C attribution stability (Outcome 1; all seven core landmarks)", "",
          "| landmark | granularity | refits | tau median [IQR] |", "|---|---|---|---|"]
    L += [f"| {r['landmark']} | {r['granularity']} | {r['refits']} | {_n(r['tau_median'])} [{_n(r['tau_q25'])}, "
          f"{_n(r['tau_q75'])}] |" for r in a["s18_tau"]]
    L += ["", "| landmark | tau(gradient, permutation), frozen model | tau(frozen, refits) median | adjacent-bin split IQR |",
          "|---|---|---|---|"]
    L += [f"| {r['landmark']} | {_n(r['tau_gradient_vs_permutation_frozen'])} | {_n(r['tau_frozen_vs_refits_median'])} | "
          f"{_n(r['adjacent_bin_last_share_iqr_mean'])} |" for r in a["s18_cross"]]
    return L


def _md16(a: dict) -> list:
    L = ["", "## Stage 16: CP rank selected by CORCONDIA (normalized level, core landmarks; core consistency as a fraction, 1 = perfect)", "",
          "| landmark | rank | core consistency | bootstrap congruence median | share >= 0.85 |", "|---|---|---|---|---|"]
    L += [f"| {r['landmark']} | {r['rank']} | {_n(r['core_consistency'])} | {_n(r['bootstrap_congruence_median'])} | "
          f"{_n(r['bootstrap_congruence_share_ge_threshold'])} |" for r in a["s16_cp_selected"]]
    L += ["", "| landmark | organ-block share of top channels | null mean | permutation p |", "|---|---|---|---|"]
    L += [f"| {r['landmark']} | {_n(r['modal_block_share'])} | {_n(r['null_mean'])} | {_n(r['p_permutation'])} |"
          for r in a["s16_organ"]]
    return L


def publish(cfg, D, pre, T, internal, checks, info, S13, S18, S16, gates) -> dict:
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
        for nm, data in figures(T).items():
            files[nm] = R2.write_bytes_atomic(pend / nm, data)
            team.append(nm)
        for nm, data in internal.items():
            if "_INTERNAL" not in nm:
                raise BuildError("an internal file name lacks the _INTERNAL marker")
            files[nm] = R2.write_bytes_atomic(pend / nm, data)
        audit = make_audit(cfg, pre, checks, info, S13, S18, S16, gates, script_sha, len(files))
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
            e.leftover = pend.name   # type: ignore[attr-defined]
        raise
    say(f"published {len(files) + 1} files to {out.name}")
    return dict(mode="run", audit=audit, audit_md=audit_md(audit), n_files=len(files) + 1)


# ---------------------------------------------------------------------------------------------------- build
def build(cfg: Config) -> dict:
    if cfg.max_seconds:
        R2._STATE["deadline"] = time.time() + cfg.max_seconds
    say("precheck: hashes")
    pre = precheck(cfg)
    if cfg.precheck_only:
        return {"mode": "precheck-only", "hashes": pre["hashes"]}
    checks, info = OrderedDict(), OrderedDict()
    gates = OrderedDict()
    if 13 in stages():
        say("synthetic gate: planted redundant group (stage 13)")
        gates.update({f"13_{k}": v for k, v in planted_gate().items()})
    if 16 in stages():
        say("synthetic gate: planted tensor (stage 16)")
        gates.update({f"16_{k}": v for k, v in planted_tensor_gate().items()})
    if not all(gates.values()):
        raise BuildError("synthetic gate failed: " + "; ".join(k for k, v in gates.items() if not v))
    say("loading Stage 08 levels (24 h train and validation rows), Layer C index, Run 1 bases, Run 2 labels and "
        "frozen models")
    D = load_all(cfg, checks)
    global NAME_INTS
    NAME_INTS = {int(t) for k in D["levels"] for c in D["levels"][k]["cols"] for t in re.findall(r"\d+", c)}
    NAME_INTS |= {int(t) for r in D["lc"].values() for c in r["channels"] for t in re.findall(r"\d+", c)}
    mats = R2.build_matrices(D, checks)
    FL = FitLog(D["rows"]["split"])
    w = D["s12"]["winner"]
    if w["rep"] == "full":
        _, Xw, _ = winner_X(D, mats)
        va = np.flatnonzero(D["rows"]["split"] == "validation")
        vp = D["vpred"]
        ok_order = bool(np.array_equal(vp["s12_eid"].astype(str), D["rows"]["eid"][va]))
        dw = float(np.abs(D["s12"]["model"].predict_proba(Xw[va]) - vp["s12_tuned_P"].astype(np.float64)).max())
        checks["13_frozen_winner_reproduces_run2_validation"] = ok_order and dw <= cfg.repro_tol_prob
        info["repro_winner_max_abs_prob_diff"] = dw
        say(f"stage 13 verify: frozen winner max abs prob diff {_fmt_repro(dw)} "
            f"(tol {_fmt_repro(cfg.repro_tol_prob)}); "
            f"{'ok' if checks['13_frozen_winner_reproduces_run2_validation'] else 'FAIL'}")
    S13 = S18 = S16 = None
    if 13 in stages():
        say("stage 13: attribution stability (24 h rows)")
        S13 = stage13(D, mats, FL, cfg, checks, info)
    if 18 in stages():
        say("stage 18: sequence attribution (all seven core landmarks)")
        S18 = stage18_attr(D, mats, FL, cfg, checks, info)
    if 16 in stages():
        say("stage 16: Layer C tensor SVD")
        S16 = stage16(D, mats, FL, cfg, checks, info)
    checks["r3b_no_test_rows_in_any_fit_or_sample"] = all("test" not in sp for _, sp in FL.fits)
    if S13 is not None:
        checks["13_all_importances_finite"] = all(np.isfinite(v).all() for k, v in S13["internal"].items()
                                                  if not k.endswith("_eid"))
    if S18 is not None:
        checks["18_all_importances_finite"] = all(np.isfinite(v).all() for v in S18["internal"].values())
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise BuildError("acceptance checks failed: " + repro_fail_detail(failed, info, cfg))
    T = OrderedDict()
    if S13 is not None:
        T["stage13_tau_TEAM.csv"] = S13["tau"]
        T["stage13_h7_TEAM.csv"] = S13["h7"]
        T["stage13_credit_split_TEAM.csv"] = S13["split"]
        T["stage13_shap_gap_TEAM.csv"] = S13["gap"]
        T["stage13_importance_top_TEAM.csv"] = S13["top"]
    if S18 is not None:
        T["stage18_seq_tau_TEAM.csv"] = S18["tau"]
        T["stage18_seq_importance_TEAM.csv"] = S18["imp"]
        T["stage18_seq_crosscheck_TEAM.csv"] = S18["cross"]
        T["stage18_ab_treeshap_TEAM.csv"] = S18["ab"]
    if S16 is not None:
        T["stage16_mode_spectra_TEAM.csv"] = S16["spectra"]
        T["stage16_tucker_error_TEAM.csv"] = S16["tucker"]
        T["stage16_cp_TEAM.csv"] = S16["cp"]
        T["stage16_angles_TEAM.csv"] = S16["angles"]
        T["stage16_loadings_TEAM.csv"] = S16["loadings"]
        T["stage16_organ_alignment_TEAM.csv"] = S16["organ"]
        T["stage16_flattened_baseline_TEAM.csv"] = S16["flat"]
    pr = OrderedDict()
    for S in (S13, S18, S16):
        if S is not None:
            pr.update(S["internal"])
    internal = OrderedDict([(INTERNAL_NAME, R2._npz_bytes(dict(pr)))])
    return publish(cfg, D, pre, T, internal, checks, info, S13, S18, S16, gates)


# ----------------------------------------------------------------------------------------------- runtime, main
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
    ap.add_argument("--s13-refits", type=int, default=S13_B)
    ap.add_argument("--s13-tree-refits", type=int, default=S13_TREE_B)
    ap.add_argument("--s18-refits", type=int, default=S18_B)
    ap.add_argument("--s16-boot", type=int, default=S16_BOOT)
    ap.add_argument("--s16-cp-starts", type=int, default=S16_CP_STARTS)
    ap.add_argument("--gru-max-epochs", type=int, default=40)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=30.0)
    a = ap.parse_args(argv)
    if min(a.s13_refits, a.s13_tree_refits, a.s18_refits) < 2 or a.s16_boot < 1 or a.s16_cp_starts < 0:
        raise InputError("refit counts must be at least 2 (bootstraps at least 1)")
    root = Path(a.project_root)
    lad = root / "data" / "processed" / "ladder"
    here = Path(__file__).resolve().parent
    r2s = Path(a.run2_script) if a.run2_script else here.parent / "stage11_18_r2" / f"{RUN2_MODULE}.py"
    cfg = Config(project_root=root, in_dir=Path(a.in_dir) if a.in_dir else lad / "stage06_08",
                 run1_dir=Path(a.run1_dir) if a.run1_dir else lad / "stage09_10_r1",
                 run2_dir=Path(a.run2_dir) if a.run2_dir else lad / RUN2_NAME, run2_script=r2s,
                 canonical=Path(a.canonical) if a.canonical else root / "data" / "interim" / "canonical_icu_encounters.csv",
                 harmonization=Path(a.harmonization) if a.harmonization else root / "data" / "interim" /
                 "encounter_harmonization_fields.csv",
                 split_file=Path(a.split_file) if a.split_file else root / "data" / "interim" / "r2_inputs" /
                 "subject_splits.csv",
                 raw_dir=Path(a.raw_dir) if a.raw_dir else root / "data" / "interim" / "r2_inputs" / "raw",
                 out_dir=Path(a.out_dir) if a.out_dir else lad / DESKTOP_ROOT_NAME / OUT_NAME,
                 expect_manifest_sha256=(a.expect_stage06_08_manifest_sha256 or "").strip().lower() or None,
                 expect_run1_manifest_sha256=(a.expect_run1_manifest_sha256 or "").strip().lower() or None,
                 expect_run2_manifest_sha256=(a.expect_run2_manifest_sha256 or "").strip().lower() or None,
                 precheck_only=a.precheck_only, gru_max_epochs=a.gru_max_epochs, max_seconds=a.max_seconds,
                 heartbeat_seconds=a.heartbeat_seconds, s13_refits=a.s13_refits, s13_tree_refits=a.s13_tree_refits,
                 s18_refits=a.s18_refits, s16_boot=a.s16_boot, s16_cp_starts=a.s16_cp_starts)
    must_write_under_desktop(cfg.out_dir, root, "--out-dir")
    must_write_under_desktop(cfg.out_dir.parent / f"{cfg.out_dir.name}.pending-0", root, "pending folder")
    must_write_under_desktop(desktop_root(root) / CACHE_NAME, root, "cache folder")
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
        write_desktop_readme(cfg.project_root)
        cache = desktop_root(cfg.project_root) / CACHE_NAME
        must_write_under_desktop(cache, cfg.project_root, "cache folder")
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
    except Exception as e:  # noqa: BLE001  (fail closed; types, frames and redacted file names only)
        if R2 is not None and isinstance(e, R2.ImportStageError):
            print(f"BUILD FAILED (import of {e.module} raised an unexpected error)", file=sys.stderr)
            e = e.__cause__ or e
        else:
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
              "harmonization, split, raw ICUSTAYS / ADMISSIONS, packages); nothing written (library cache folder only)")
        for k, v in sorted(res["hashes"].items()):
            print(f"{k} sha256 {v}")
        return 0
    print(res["audit_md"])
    print(f"elapsed {time.time() - t0:.0f} s")
    print("result: RUN 3b desktop (13, 16, 18) OK; RELEASE GATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
