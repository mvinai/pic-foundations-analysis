"""Run 3a of stages 09-18 (PIC Foundations, PIC; plan v13, contract v13): the single test scoring of the
frozen Run 2 choices. Stage 14 (24 h landmark rows, Layers A and B: the frozen stage 12 winner with its calibrators,
the family (a) H6 adjacent-level equivalence tests with Holm, Layer C at 24 h) and stage 18 scoring (landmark
models A, A+B, A+B+C and C only at all 14 landmarks: the family (b) H9 contrast at the seven core landmarks with
Holm, the 14-landmark added-value curve labeled descriptive, Outcome 2 and ordinal8), plus the stacked
(encounter, landmark) Layer C sensitivity model deferred from Run 2.

r3a_desktop_full v2.0.1-desktop (Nov 21, 2026).

This is the desktop full-settings variant of Run 3a. It is a post-primary sensitivity analysis
(a second look at the test set), not a replacement for the laptop Run 3a primary scoring. All
files this script writes go under data/processed/ladder/DESKTOP_FULL_PLAN/ (published folder,
pending folder, cache, test-use log, internal files and README_DESKTOP.txt). A published laptop
Run 3a does not block this run.

Frozen inputs (hash-gated before anything is parsed): everything Run 2 checked (Stage 06-08 manifest and files, Run 1
v1.0.2 manifest and files, canonical, harmonization, split, raw ICUSTAYS / ADMISSIONS; the same pins), the Run 2
script (scripts/stage11_18_r2/r2_11p_11_12_17_models.py, imported only after its SHA-256 matches the pin), and the
Run 2 manifest (pinned by its full SHA-256 on the command line and in this script) with every file it lists,
including the INTERNAL labels, validation predictions, tuned model and calibrators, and stage 17 models.

Test rows are scored exactly once, with frozen models only. Models that Run 2 did not save (the stage 11 ordinal LR
at every width and the Layer C 24 h GRU) are refitted on train with the frozen Run 2 configuration, and the refit
must reproduce the Run 2 validation result before any test row is read (ordinal LR QWK within 0.005; Layer C 24 h GRU QWK within 0.005; frozen landmark-model probabilities within 1e-4 on a CPU float32 forward pass). No model is fitted or selected on test rows.
A test-use log (data/processed/ladder/DESKTOP_FULL_PLAN/test_use_log_stage14_18_r3a.json) records each desktop
attempt; a run refuses to start once a published desktop attempt exists in that log.

Outputs (data/processed/ladder/DESKTOP_FULL_PLAN/stage14_18_r3a/, published atomically from a pending folder in
the same DESKTOP_FULL_PLAN directory): TEAM files (aggregate
only) and INTERNAL files (test predictions; never opened, pasted or shared). Release gate as in Run 2.
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

SCRIPT_VERSION = "r3a_desktop_full v2.0.1-desktop"
SPEC_DOC = "stage14_18_r3a_desktop_spec v2.0.1-desktop"
PLAN, CONTRACT = "v13", "v13"
OUT_NAME = "stage14_18_r3a"
DESKTOP_ROOT_NAME = "DESKTOP_FULL_PLAN"
README_NAME = "README_DESKTOP.txt"
README_TEXT = (
    "These are post-primary sensitivity outputs from the full-settings desktop run "
    "(v2.0.1-desktop). They are a second look, not a replacement for the laptop "
    "primary Run 3a/3b results. Published folders, pending folders, caches, "
    "internal files and logs for the desktop run belong only in this folder.\n"
)
RUN2_NAME = "stage11_12_17_r2"
RUN2_MODULE = "r2_11p_11_12_17_models"
TEST_LOG_NAME = f"test_use_log_{OUT_NAME}.json"
CACHE_NAME = f"_cache_{OUT_NAME}"

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
# Vinai's H9 framing (Nov 19): the later core landmarks day 3, 5, 7, 10 and 14 lead every H9 table. The Holm family
# (b) stays the contract's seven core landmarks (contract v13 "Multiplicity"; plan 12.2 H9).
H9_FOCUS = (72, 120, 168, 240, 336)
FAMILY_A_MODEL = "polr"                  # the Run 2 H6 headline model (fitted at every width)
DESCRIPTIVE_MODELS = ("allthr",)         # test-scored at every width, no test of hypotheses
ALPHA = 0.05
STACK_HIDDEN = 32
STACK_MAX_EPOCHS = 200                   # stacked sensitivity: D19 early stopping, generous max (no 20-epoch cap)
R2 = None                                # the hash-verified Run 2 module (set by load_run2)


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

    Training and the stacked sensitivity fit may use torch_device() (CUDA if available). Tensor dtype at the
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
        if k == "14_frozen_winner_reproduces_run2_validation":
            extra = (f" (max abs prob diff={_fmt_repro(info.get('repro_winner_max_abs_prob_diff'))}, "
                     f"tol={_fmt_repro(cfg.repro_tol_prob)})")
        elif k == "14_ordinal_lr_refits_reproduce_run2_validation_qwk":
            extra = (f" (max abs QWK diff={_fmt_repro(info.get('repro_lr_max_abs_qwk_diff'))}, "
                     f"tol={_fmt_repro(cfg.repro_tol_lr_qwk)})")
        elif k == "14_layer_c_24h_refit_reproduces_run2_validation_qwk":
            extra = (f" (max abs QWK diff={_fmt_repro(info.get('repro_c24_abs_qwk_diff'))}, "
                     f"tol={_fmt_repro(cfg.repro_tol_gru_qwk)})")
        elif k == "18_frozen_landmark_models_reproduce_run2_validation":
            extra = (f" (max abs prob diff={_fmt_repro(info.get('repro_s17_max_abs_prob_diff'))}, "
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
    test_log: Path
    expect_manifest_sha256: str | None
    expect_run1_manifest_sha256: str | None
    expect_run2_manifest_sha256: str | None
    precheck_only: bool
    n_boot: int = 1000
    n_boot_family_a: int = 20000
    gru_max_epochs: int = 40
    stacked_max_epochs: int = STACK_MAX_EPOCHS
    max_seconds: float = 0.0
    heartbeat_seconds: float = 30.0
    stacked: bool = True
    repro_tol_tab: float = 2e-6
    repro_tol_prob: float = 1e-4
    repro_tol_gru_qwk: float = 0.005
    repro_tol_lr_qwk: float = 0.005


def r2_config(cfg: Config):
    return R2.Config(project_root=cfg.project_root, in_dir=cfg.in_dir, run1_dir=cfg.run1_dir,
                     canonical=cfg.canonical, harmonization=cfg.harmonization, split_file=cfg.split_file,
                     raw_dir=cfg.raw_dir, out_dir=cfg.out_dir, expect_manifest_sha256=cfg.expect_manifest_sha256,
                     expect_run1_manifest_sha256=cfg.expect_run1_manifest_sha256, precheck_only=True,
                     n_boot=cfg.n_boot, gru_max_epochs=cfg.gru_max_epochs, max_seconds=cfg.max_seconds,
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


def precheck(cfg: Config) -> dict:
    """Run 2's hash gates (same pins), then the Run 2 manifest (pinned) and every file it lists; the output folder
    must be empty, no pending folder may exist, and no published attempt may be in the test-use log."""
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
    att = read_test_log(cfg.test_log)
    if any(a.get("status") == "published" for a in att):
        raise InputError("the test-use log records a published test scoring: test rows are scored once only")
    return {"hashes": hashes, "run2_manifest": m2, "prior_attempts": len(att)}


# ------------------------------------------------------------------------------------------------------- loading
def load_all(cfg: Config, checks: dict) -> dict:
    """Stage 08 levels on the 24 h landmark rows of every split (test rows stay behind the TestGuard), the Run 1
    bases, the Layer C index of every landmark, the Run 2 labels, subjects (canonical file) and frozen artefacts."""
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
    # labels (Run 2 INTERNAL; same encounter order and split as Stage 08)
    lab = R2.read_npz(cfg.run2_dir / RUN2_INTERNAL[0],
                      ["encounter_id", "split", "died", "outcome2", "ordinal8"] +
                      [f"outcome1_L{L}" for L in R2.LANDMARKS])
    checks["r3_labels_match_stage08_order_and_split"] = bool(
        np.array_equal(lab["encounter_id"].astype(str), eid0) and np.array_equal(lab["split"].astype(str), split0))
    if not checks["r3_labels_match_stage08_order_and_split"]:
        raise InputError("Run 2 labels: encounter order or split differs from Stage 08")
    # subjects from the canonical file (hash-verified)
    can = R2.read_csv_str(cfg.canonical, usecols=["encounter_id", "SUBJECT_ID"])
    can["encounter_id"] = can["encounter_id"].str.strip()
    if can["encounter_id"].duplicated().any() or set(can["encounter_id"]) != set(eid0):
        raise InputError("the Stage 08 encounters differ from the canonical encounter set (count withheld)")
    subj_all = can.set_index("encounter_id").loc[eid0]["SUBJECT_ID"].map(R2.norm_id).to_numpy().astype(str)
    # the persisted subject split (same reading rule as Run 2) must equal the Stage 08 split
    spl = R2.read_csv_str(cfg.split_file, usecols=["SUBJECT_ID", "split"])
    smap = dict(zip(spl["SUBJECT_ID"].map(R2.norm_id),
                    spl["split"].map(lambda v: R2.SPLIT_LABELS.get(str(v).strip().lower(), "?"))))
    checks["r3_split_matches_split_file"] = bool(np.array_equal(np.array([smap.get(s, "?") for s in subj_all]),
                                                                split0))
    if not checks["r3_split_matches_split_file"]:
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
    checks["r3_eligible_sets_match_labels"] = ok_el
    bz = np.load(cfg.run1_dir / "stage10_matched_width_bases_INTERNAL.npz", allow_pickle=False)
    with bz:
        bases = {k: np.array(bz[k]) for k in bz.files}
    # frozen Run 2 artefacts
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
    return dict(r2cfg=r2cfg, levels=lv, rows=rows, eid_all=eid0, split_all=split0, elig24_all=elig24, subj_all=subj_all,
                lab_all=lab, elig_all=elig_all, lc=lc, bases=bases, s12=s12, tab=tab, states=states,
                vpred=vpred, team2=team2)


# ------------------------------------------------------------------------------------------------ test guard
class TestGuard:
    """The only path to test rows. Phase 'verify' (frozen-model and refit reproduction on validation) refuses any
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


# --------------------------------------------------------------------------------------------------- helpers
def gru_predict(state: dict, seq: np.ndarray, st, off: np.ndarray, device=None) -> np.ndarray:
    """Class probabilities of a frozen Run 2 GRU (same forward pass and monotone repair as Run 2).

    Pass device=verify_device() for frozen-model reproduction and for test scoring of the frozen
    Run 2 landmark models (CPU float32, matching the laptop). Leave device None to use torch_device()
    (CUDA if available) for models trained in this run (Layer C 24 h refit, stacked sensitivity).
    """
    torch = R2._torch()
    Net = R2.make_gru_class()
    net = Net(state["d_seq"], state["d_static"], state["hidden"], state["K"])
    net.load_state_dict(state["state_dict"])
    dev = device if device is not None else torch_device()
    net.to(dev)
    net.eval()
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    s = T(st) if st is not None and state["d_static"] > 0 else torch.zeros((seq.shape[0], 0), device=dev)
    with torch.no_grad():
        F = torch.sigmoid(net(T(seq), s, T(off))).cpu().numpy().astype(np.float64)
    F = np.maximum.accumulate(F, axis=1)
    return R2.cum_to_probs(F)


def boot_auc(W: np.ndarray, score: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Weighted Mann-Whitney AUROC for each bootstrap weight row (ties count one half); nan when undefined."""
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


def p_tost(d: np.ndarray, delta: float) -> float:
    """Equivalence (TOST) bootstrap p-value: the larger of the two one-sided p-values against -delta and +delta,
    with the (count + 1) / (B + 1) rule."""
    d = d[np.isfinite(d)]
    if not d.size:
        return float("nan")
    lo = (np.sum(d <= -delta) + 1) / (d.size + 1)
    hi = (np.sum(d >= delta) + 1) / (d.size + 1)
    return float(max(lo, hi))


def p_gain(d: np.ndarray) -> float:
    """One-sided bootstrap p-value for delta <= 0, (count + 1) / (B + 1)."""
    d = d[np.isfinite(d)]
    return float((np.sum(d <= 0) + 1) / (d.size + 1)) if d.size else float("nan")


def holm(p) -> tuple:
    """Holm step-down: adjusted p-values (monotone) and rejections at ALPHA, in the input order."""
    p = np.asarray(p, dtype=float)
    m = int(np.isfinite(p).sum())
    adj = np.full(p.shape, np.nan)
    o = [i for i in np.argsort(np.where(np.isfinite(p), p, np.inf), kind="mergesort") if np.isfinite(p[i])]
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj, np.where(np.isfinite(adj), adj <= ALPHA, False)


def interval(v: np.ndarray) -> tuple:
    v = v[np.isfinite(v)]
    return (R2.pct(v, 2.5), R2.pct(v, 97.5)) if v.size else (float("nan"), float("nan"))


def suppress_small(row: dict, y: np.ndarray, K: int, died: np.ndarray | None) -> dict:
    """Counts gate: a per-threshold metric whose positive or negative class has fewer than 10 rows, and the
    mortality AUROC when deaths or survivors are fewer than 10, are written as '<10' (never the count)."""
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


def score_test(P, y, K, died, W, extra: dict) -> tuple:
    """Run 2's score_row on test rows, plus the bootstrap macro AUROC, with the counts gate applied."""
    row, kb = R2.score_row(P, y, K, died, W, extra)
    ka = boot_macro_auc(W, P, y, K)
    row["auroc_macro_p2_5"], row["auroc_macro_p97_5"] = interval(ka)
    return suppress_small(row, y, K, died), kb, ka


# ------------------------------------------------------------------------------------- stage 14: verify (no test)
def winner_X(D: dict, mats: OrderedDict) -> tuple:
    w = D["s12"]["winner"]
    kind, k = (w["rep"], None) if w["rep"] == "full" else (w["rep"].split("_")[0], int(w["rep"].split("_")[1]))
    return w, R2.rep_X(mats[w["matrix"]], kind, k)


def verify14(D: dict, mats: OrderedDict, G: TestGuard, cfg: Config, checks: dict, info: dict) -> dict:
    """Validation-only reproduction of the frozen stage 12 winner, the stage 11 ordinal LR refits (family (a)
    model and the descriptive all-threshold model) and the Layer C 24 h GRU refit. No test row is read."""
    R = D["rows"]
    sp = R["split"]
    tr, va = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "validation")
    y, K = R["O1"][24], 9
    s12, vp = D["s12"], D["vpred"]
    w, Xw = winner_X(D, mats)
    ok_order = bool(np.array_equal(vp["s12_eid"].astype(str), R["eid"][va]))
    Pv = s12["model"].predict_proba(Xw[va])
    dw = float(np.abs(Pv - vp["s12_tuned_P"].astype(np.float64)).max())
    checks["14_frozen_winner_reproduces_run2_validation"] = ok_order and dw <= cfg.repro_tol_prob
    Pc = R2.apply_calibrators(s12["calibrators"], Pv)
    checks["14_frozen_calibrators_monotone"] = bool((np.diff(np.cumsum(Pc, axis=1), axis=1) >= -1e-12).all())
    info["repro_winner_max_abs_prob_diff"] = dw
    say(f"stage 14 verify: frozen winner max abs prob diff {_fmt_repro(dw)} "
        f"(tol {_fmt_repro(cfg.repro_tol_prob)}); "
        f"{'ok' if checks['14_frozen_winner_reproduces_run2_validation'] else 'FAIL'}")
    team = D["team2"]["stage11_validation_scores_TEAM.csv"]
    tq = {(r.model, r.matrix, r.rep): float(r.qwk) for r in team.itertuples() if r.outcome == "outcome1"}
    fitted, maxd, missing = OrderedDict(), 0.0, False
    G.fit_rows(tr, "stage 11 ordinal LR refits")
    for mod in (FAMILY_A_MODEL,) + DESCRIPTIVE_MODELS:
        t0 = time.time()
        for m, M in mats.items():
            for kind, k in R2.reps_of(M):
                X = R2.rep_X(M, kind, k)
                mdl, P = R2.fit_predict(mod, K, X[tr], y[tr], X[va], seed=R2.SEED)
                rn = R2.rep_name(kind, k)
                ref = tq.get((mod, m, rn))
                if ref is None:
                    missing = True
                else:
                    maxd = max(maxd, abs(R2.qwk(y[va], R2.median_class(P), K) - ref))
                fitted[(mod, m, rn)] = (mdl, kind, k)
                check_time()
        say(f"stage 14 verify: {mod} refits at every width done ({time.time() - t0:.0f} s)")
    checks["14_ordinal_lr_refits_reproduce_run2_validation_qwk"] = (not missing) and maxd <= cfg.repro_tol_lr_qwk
    info["repro_lr_max_abs_qwk_diff"] = maxd
    say(f"stage 14 verify: ordinal LR refits max abs QWK diff {_fmt_repro(maxd)} "
        f"(tol {_fmt_repro(cfg.repro_tol_lr_qwk)}); "
        f"{'ok' if checks['14_ordinal_lr_refits_reproduce_run2_validation_qwk'] else 'FAIL'}")
    # Layer C at 24 h: the Run 2 validation-selected representation, refitted with the frozen configuration
    c24 = D["team2"]["stage11_layer_c_24h_TEAM.csv"]
    b = c24.loc[c24["qwk"].astype(float).idxmax()]
    sel, kc = b["c_rep"], (int(float(b["k"])) if b["k"] not in ("", "nan") else None)
    r = D["lc"][24]
    idx = r["idx"]
    spL = sp[idx]
    itr, iva = np.flatnonzero(spL == "train"), np.flatnonzero(spL == "validation")
    yL, gL = y[idx], R["subj"][idx]
    A_name = w["matrix"].split("+")[0]
    XA, Xs = mats[A_name]["X"][idx], Xw[idx]
    G.fit_rows(idx[itr], "Layer C 24 h GRU refit")
    off_tr, off_va, th, mA = R2.crossfit_offset(XA[itr], yL[itr], gL[itr], XA[iva], K, ("c24",))
    S = R2.c_inputs(D, 24, sel, kc)
    state, Pg, ep = R2.gru_fit_predict(S[itr], Xs[itr], off_tr, yL[itr], gL[itr], S[iva], Xs[iva], off_va, K, 32,
                                       R2.SEED, gamma0=th, max_epochs=D["r2cfg"]["gru_max_epochs"])
    dq = abs(R2.qwk(yL[iva], R2.median_class(Pg), K) - float(b["qwk"]))
    checks["14_layer_c_24h_refit_reproduces_run2_validation_qwk"] = dq <= cfg.repro_tol_gru_qwk
    info["repro_c24_abs_qwk_diff"] = dq
    info["c24_rep"] = sel if kc is None else f"{sel}_{kc}"
    say(f"stage 14 verify: Layer C 24 h refit max abs QWK diff {_fmt_repro(dq)} "
        f"(tol {_fmt_repro(cfg.repro_tol_gru_qwk)}); "
        f"{'ok' if checks['14_layer_c_24h_refit_reproduces_run2_validation_qwk'] else 'FAIL'}")
    return dict(w=w, Xw=Xw, fitted=fitted, c24_rep=info["c24_rep"], c24=dict(state=state, mA=mA, S=S, idx=idx, XA=XA, Xs=Xs, sel=sel, k=kc,
                                                    epochs=ep))


# ------------------------------------------------------------------------------------------ stage 14: scoring
def h6_rows(KB: dict, Q: dict, mods, family_model: str, delta: float, delta_a: float) -> list:
    out = []
    for mod in mods:
        for a, b in R2.ADJACENT:
            reps = [k_[2] for k_ in KB if len(k_) == 3 and k_[0] == mod and k_[1] == a]
            for rn in reps:
                if (mod, b, rn) not in KB:
                    continue
                (kqa, kaa, kma), (kqb, kab, kmb) = KB[(mod, a, rn)], KB[(mod, b, rn)]
                qa, qb = Q[(mod, a, rn)], Q[(mod, b, rn)]
                ds = R2.diff_summary(kqb, kqa, qb["qwk"] - qa["qwk"], delta)
                row = OrderedDict(model=mod, rep=rn, from_matrix=a, to_matrix=b, matched_width=rn != "full",
                                  family_a=mod == family_model, delta_qwk=ds["delta_qwk"],
                                  delta_p2_5=ds["delta_p2_5"], delta_p97_5=ds["delta_p97_5"], verdict=ds["verdict"],
                                  p_tost=p_tost(kqb - kqa, delta))
                for nm, ka_, kb_, key in (("auroc_macro", kaa, kab, "auroc_macro"),
                                          ("mortality_auroc", kma, kmb, "mortality_auroc")):
                    dd = kb_ - ka_
                    lo, hi = interval(dd)
                    pt = qb[key] - qa[key] if np.isfinite(qb[key]) and np.isfinite(qa[key]) else float("nan")
                    row[f"delta_{nm}"] = pt
                    row[f"delta_{nm}_p2_5"], row[f"delta_{nm}_p97_5"] = lo, hi
                    row[f"{nm}_equivalent"] = bool(lo > -delta_a and hi < delta_a)
                out.append(row)
    return out


def family_a_pvalues(subj, y, K, YH, pairs, c24, B, delta, chunk=500) -> list:
    """TOST p-values (larger one-sided p, (count + 1) / (B + 1)) for the family (a) contrasts from a dedicated
    paired subject-cluster bootstrap of B draws, generated in chunks (memory) from a fixed seed. pairs: (to, from)
    keys of YH (median-class test predictions); c24: (Layer C yhat, winner yhat, positions within the test rows)."""
    rng = R2.rng_for("boot", "s14_family_a")
    us, inv = np.unique(subj, return_inverse=True)
    lo = np.zeros(len(pairs) + 1)
    hi = np.zeros(len(pairs) + 1)
    yc_hat, yw_hat, pos = c24
    done = 0
    while done < B:
        b = min(chunk, B - done)
        W = np.zeros((b, len(subj)), dtype=np.float32)
        for i in range(b):
            W[i] = np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us)).astype(np.float32)[inv]
        for j, (kb, ka) in enumerate(pairs):
            d = R2.boot_qwk(W, y, YH[kb], K) - R2.boot_qwk(W, y, YH[ka], K)
            lo[j] += np.sum(d <= -delta)
            hi[j] += np.sum(d >= delta)
        Wc = W[:, pos]
        d = R2.boot_qwk(Wc, y[pos], yc_hat, K) - R2.boot_qwk(Wc, y[pos], yw_hat, K)
        lo[-1] += np.sum(d <= -delta)
        hi[-1] += np.sum(d >= delta)
        done += b
        check_time()
    return [float(max(a + 1, c + 1) / (B + 1)) for a, c in zip(lo, hi)]


def score14(D: dict, mats: OrderedDict, G: TestGuard, V: dict, cfg: Config, checks: dict) -> dict:
    R = D["rows"]
    te = G.te()
    y, K = R["O1"][24][te], 9
    died = R["died"][te]
    s12 = D["s12"]
    W = R2.boot_weights(R["subj"][te], cfg.n_boot, R2.rng_for("boot", "s14"))
    rows, KB, Q, PK = [], {}, {}, {}

    def add(key, P, extra, keep=True):
        row, kq, ka = score_test(P, y, K, died, W, extra)
        rows.append(row)
        if keep:
            if key[0] == FAMILY_A_MODEL:
                PK[key] = P
            km = boot_auc(W, P[:, K - 1], died)
            KB[key] = (kq, ka, km)
            mt = R2.metrics(P, y, K, died)
            Q[key] = dict(qwk=mt["qwk"], auroc_macro=mt["auroc_macro"], mortality_auroc=mt["mortality_auroc"])
        return kq

    w, Xw = V["w"], V["Xw"]
    Pw = G.score("s14_winner", lambda: s12["model"].predict_proba(Xw[te]))
    Pwc = R2.apply_calibrators(s12["calibrators"], Pw)
    base = dict(matrix=w["matrix"], rep=w["rep"], model=w["model"], family=w["family"])
    add(("winner",), Pw, dict(item="stage 12 winner (frozen)", calibrated=False, **base))
    add(("winner_cal",), Pwc, dict(item="stage 12 winner (frozen)", calibrated=True, **base))
    for (mod, m, rn), (mdl, kind, k) in V["fitted"].items():
        X = R2.rep_X(mats[m], kind, k)
        P = G.score(f"s14_{mod}_{m}_{rn}", lambda: mdl.predict_proba(X[te]))
        add((mod, m, rn), P, dict(item="stage 11 refit (frozen configuration)", calibrated=False, matrix=m,
                                  rep=rn, model=mod, family=R2.FAMILY[mod]))
        check_time()
    h6 = h6_rows(KB, Q, (FAMILY_A_MODEL,) + DESCRIPTIVE_MODELS, FAMILY_A_MODEL, R2.THRESH["delta"],
                 R2.THRESH["delta"])
    # Layer C at 24 h vs the frozen A+B winner (uncalibrated, as in Run 2), paired on the same test rows
    c = V["c24"]
    idx, spL = c["idx"], R["split"][c["idx"]]
    ite = np.flatnonzero(spL == "test")
    pos = np.searchsorted(te, idx[ite])
    if not np.array_equal(te[pos], idx[ite]):
        raise BuildError("Layer C 24 h test rows are not a subset of the 24 h test rows")
    Pc = G.score("s14_c24", lambda: gru_predict(c["state"], c["S"][ite], c["Xs"][ite], c["mA"].decision(c["XA"][ite])))
    Wc = W[:, pos]
    yc = y[pos]
    rowc, kqc, kac = score_test(Pc, yc, K, died[pos], Wc,
                                dict(item="Layer C 24 h GRU on A+B (refit, frozen configuration)", calibrated=False,
                                     matrix=w["matrix"] + "+C24", rep=V["c24_rep"], model="gru", family="GRU"))
    rows.append(rowc)
    kq_abc = R2.boot_qwk(Wc, yc, R2.median_class(Pw[pos]), K)
    dqc = rowc["qwk"] - R2.qwk(yc, R2.median_class(Pw[pos]), K)
    ds = R2.diff_summary(kqc, kq_abc, dqc, R2.THRESH["delta"])
    c24row = OrderedDict(model="gru", rep=V["c24_rep"], from_matrix=w["matrix"], to_matrix=w["matrix"] + "+C24",
                         matched_width=False, family_a=True, delta_qwk=ds["delta_qwk"], delta_p2_5=ds["delta_p2_5"],
                         delta_p97_5=ds["delta_p97_5"], verdict=ds["verdict"],
                         p_tost=p_tost(kqc - kq_abc, R2.THRESH["delta"]))
    h6.append(c24row)
    h6 = pd.DataFrame(h6)
    fam = h6["family_a"].astype(bool).to_numpy()
    # family (a) p-values: a separate subject-cluster bootstrap with B_family_a draws (131 tests: Holm's first
    # threshold 0.05 / 131 is below 1 / (1000 + 1), the smallest p-value B = 1000 can give)
    pairs = [((r_["model"], r_["to_matrix"], r_["rep"]), (r_["model"], r_["from_matrix"], r_["rep"]))
             for _, r_ in h6[fam].iterrows() if r_["model"] != "gru"]
    if h6[fam].iloc[-1]["model"] != "gru" or len(pairs) != int(fam.sum()) - 1:
        raise BuildError("family (a) table order is unexpected")
    YH = {k_: R2.median_class(P_) for k_, P_ in PK.items()}
    pf = family_a_pvalues(R["subj"][te], y, K, YH, pairs, (R2.median_class(Pc), R2.median_class(Pw[pos]), pos),
                          cfg.n_boot_family_a, R2.THRESH["delta"])
    h6.loc[fam, "p_tost"] = pf
    adj, rej = holm(h6.loc[fam, "p_tost"].to_numpy())
    h6["p_tost_holm"] = np.nan
    h6["equivalent_holm"] = False
    h6.loc[fam, "p_tost_holm"] = adj
    h6.loc[fam, "equivalent_holm"] = rej
    fam_tab = h6[fam][["model", "rep", "from_matrix", "to_matrix", "delta_qwk", "delta_p2_5", "delta_p97_5",
                       "p_tost", "p_tost_holm", "equivalent_holm", "verdict"]].copy()
    fam_tab = fam_tab.rename(columns={"verdict": "ci_verdict_unadjusted"})   # the Holm columns carry the test
    fam_tab["c24_gain_p_one_sided_le_0"] = np.nan
    fam_tab.loc[fam_tab["model"] == "gru", "c24_gain_p_one_sided_le_0"] = p_gain(kqc - kq_abc)
    internal = OrderedDict(s14_test_eid=R["eid"][te], s14_winner_P=Pw.astype(np.float32),
                           s14_winner_cal_P=Pwc.astype(np.float32), s14_c24_P=Pc.astype(np.float32),
                           s14_c24_eid=R["eid"][idx[ite]])
    say("stage 14: test scoring done")
    return dict(test=pd.DataFrame(rows), h6=h6, holm=fam_tab, internal=internal)


# --------------------------------------------------------------------------------- stage 18: landmark scoring
def todo18(R: dict, idx: np.ndarray, L: int) -> list:
    out = [("outcome1", R["O1"][L][idx])]
    if L in R2.CORE_LANDMARKS:
        out.append(("outcome2", R["O2"][idx]))
    if L in R2.ORD8_LANDMARKS:
        out.append(("ordinal8", R["O8"][idx]))
    return out


def verify18(D: dict, mats: OrderedDict, G: TestGuard, V: dict, cfg: Config, checks: dict, info: dict) -> None:
    """Validation-only: every frozen stage 17 model (A, A+B, A+B+C, C only) must reproduce its Run 2 validation
    predictions before any test row is read."""
    R, tab, st, vp = D["rows"], D["tab"], D["states"], D["vpred"]
    sp = R["split"]
    A_name = V["w"]["matrix"].split("+")[0]
    maxd, ok, missing = 0.0, True, False
    vdev = verify_device()
    for L in R2.LANDMARKS:
        idx = D["lc"][L]["idx"]
        iva = np.flatnonzero(sp[idx] == "validation")
        S = R2.c_inputs(D, L, "all", None)
        XA, Xab = mats[A_name]["X"][idx], V["Xw"][idx]
        for outc, yy in todo18(R, idx, L):
            jva = iva[(yy >= 0)[iva]]
            ek = f"s17_L{L}_{outc}__eid"
            ok &= ek in vp and bool(np.array_equal(vp[ek].astype(str), R["eid"][idx][jva]))
            mA = tab[f"L{L}_{outc}_A"]
            P = OrderedDict(A=mA.predict_proba(XA[jva]), AB=tab[f"L{L}_{outc}_AB"].predict_proba(Xab[jva]),
                            ABC=gru_predict(st[f"L{L}_{outc}_ABC"], S[jva], Xab[jva], mA.decision(XA[jva]),
                                            device=vdev))
            if outc == "outcome1":
                P["C_only"] = gru_predict(st[f"L{L}_{outc}_C_only"], S[jva], None, np.zeros(len(jva)),
                                          device=vdev)
            for m, Pm in P.items():
                ref = vp.get(f"s17_L{L}_{outc}_{m}")
                if ref is None or ref.shape != Pm.shape:
                    missing = True
                    continue
                maxd = max(maxd, float(np.abs(Pm - ref.astype(np.float64)).max()))
        check_time()
    checks["18_frozen_landmark_models_reproduce_run2_validation"] = ok and not missing and maxd <= cfg.repro_tol_prob
    info["repro_s17_max_abs_prob_diff"] = maxd
    say(f"stage 18 verify: frozen landmark models on CPU (float32); max abs prob diff {_fmt_repro(maxd)} "
        f"(tol {_fmt_repro(cfg.repro_tol_prob)}); "
        f"{'ok' if checks['18_frozen_landmark_models_reproduce_run2_validation'] else 'FAIL'}")


def contrast(kq1, kq0, q1, q0, ka1, ka0, a1, a0, e1, e0, delta_C) -> OrderedDict:
    d = kq1 - kq0
    lo, hi = interval(d)
    da = ka1 - ka0
    alo, ahi = interval(da)
    pt = q1 - q0
    return OrderedDict(delta_qwk=pt, delta_p2_5=lo, delta_p97_5=hi, p_one_sided_le_0=p_gain(d),
                       delta_auroc_macro=a1 - a0, delta_auroc_macro_p2_5=alo, delta_auroc_macro_p97_5=ahi,
                       delta_emd=e1 - e0, gain_at_least_delta_C=bool(pt >= delta_C and lo > 0))


def score18(D: dict, mats: OrderedDict, G: TestGuard, V: dict, cfg: Config, checks: dict) -> dict:
    R, tab, st = D["rows"], D["tab"], D["states"]
    sp = R["split"]
    G.te()
    A_name = V["w"]["matrix"].split("+")[0]
    rows, crows, internal, keep_ab = [], [], OrderedDict(), OrderedDict()
    dC = R2.THRESH["delta_C"]
    for L in R2.LANDMARKS:
        t0 = time.time()
        idx = D["lc"][L]["idx"]
        ite = np.flatnonzero(sp[idx] == "test")
        S = R2.c_inputs(D, L, "all", None)
        XA, Xab = mats[A_name]["X"][idx], V["Xw"][idx]
        W = R2.boot_weights(R["subj"][idx][ite], cfg.n_boot, R2.rng_for("boot", "s18", L))
        for outc, yy in todo18(R, idx, L):
            K = R2.K_OUT[outc]
            okr = (yy >= 0)[ite]
            jte, Wv = ite[okr], W[:, okr]
            yv, dv = yy[jte], R["died"][idx][jte]
            mA = tab[f"L{L}_{outc}_A"]
            P = OrderedDict()
            P["A"] = G.score(f"s18_L{L}_{outc}_A", lambda: mA.predict_proba(XA[jte]))
            P["AB"] = G.score(f"s18_L{L}_{outc}_AB", lambda: tab[f"L{L}_{outc}_AB"].predict_proba(Xab[jte]))
            P["ABC"] = G.score(f"s18_L{L}_{outc}_ABC",
                               lambda: gru_predict(st[f"L{L}_{outc}_ABC"], S[jte], Xab[jte],
                                                   mA.decision(XA[jte]), device=verify_device()))
            if outc == "outcome1":
                P["C_only"] = G.score(f"s18_L{L}_{outc}_C_only",
                                      lambda: gru_predict(st[f"L{L}_{outc}_C_only"], S[jte], None, np.zeros(len(jte)),
                                                          device=verify_device()))
                for m in ("AB", "ABC"):
                    P[f"{m}_cal"] = R2.apply_calibrators(tab[f"L{L}_{outc}_{m}__calibrators"], P[m])
                keep_ab[L] = dict(pos=idx[jte], P=P["AB"], y=yv)
            res = OrderedDict()
            for m, Pm in P.items():
                row, kq, ka = score_test(Pm, yv, K, dv, Wv, dict(outcome=outc, landmark=L, core=L in R2.CORE_LANDMARKS,
                                                                 h9_focus=L in H9_FOCUS, model=m))
                rows.append(row)
                res[m] = (kq, ka, R2.metrics(Pm, yv, K, dv))
                internal[f"s18_L{L}_{outc}_{m}"] = Pm.astype(np.float32)
            internal[f"s18_L{L}_{outc}__eid"] = R["eid"][idx][jte]
            pairs = [("ABC_minus_AB", "ABC", "AB")]
            if outc == "outcome1":
                pairs += [("ABC_cal_minus_AB_cal", "ABC_cal", "AB_cal"), ("C_only_minus_AB", "C_only", "AB")]
            for nm, m1, m0 in pairs:
                (k1, a1, t1), (k0, a0, t0_) = res[m1], res[m0]
                c = contrast(k1, k0, t1["qwk"], t0_["qwk"], a1, a0, t1["auroc_macro"], t0_["auroc_macro"],
                             t1["emd"], t0_["emd"], dC)
                crows.append(OrderedDict(outcome=outc, landmark=L, core=L in R2.CORE_LANDMARKS, h9_focus=L in H9_FOCUS,
                                         contrast=nm, **c))
            check_time()
        say(f"stage 18: landmark {L} test scoring done ({time.time() - t0:.0f} s)")
    ct = pd.DataFrame(crows)
    # family (b): Holm over the seven core-landmark Outcome 1 contrasts (contract v13 "Multiplicity"; plan 12.2 H9)
    fam = []
    for outc in ("outcome1", "outcome2"):
        g = ct[(ct["outcome"] == outc) & (ct["contrast"] == "ABC_minus_AB") & ct["core"].astype(bool)].copy()
        adj, rej = holm(g["p_one_sided_le_0"].to_numpy())
        g["p_holm"], g["rejected_holm"] = adj, rej
        g["h9_claim_met"] = g["rejected_holm"].astype(bool) & (g["delta_qwk"].astype(float) >= dC)
        g["family"] = "b (H9, confirmatory)" if outc == "outcome1" else "Outcome 2 (key secondary; Holm within outcome)"
        g["order"] = [H9_FOCUS.index(L) if L in H9_FOCUS else len(H9_FOCUS) + R2.CORE_LANDMARKS.index(L)
                      for L in g["landmark"]]
        fam.append(g.sort_values("order").drop(columns="order"))
    holm_tab = pd.concat(fam, ignore_index=True)
    cur = ct[(ct["outcome"] == "outcome1") & (ct["contrast"] == "ABC_minus_AB")][
        ["landmark", "core", "h9_focus", "delta_qwk", "delta_p2_5", "delta_p97_5", "delta_auroc_macro",
         "delta_auroc_macro_p2_5", "delta_auroc_macro_p97_5", "delta_emd"]].copy()
    cur["label"] = "descriptive (no test; intervals are descriptive bootstrap 95% intervals)"
    checks["18_all_landmarks_scored"] = set(cur["landmark"]) == set(R2.LANDMARKS)
    checks["18_probabilities_valid"] = all(bool(np.allclose(v.sum(1), 1, atol=1e-5) and (v >= -1e-7).all())
                                           for k, v in internal.items() if not k.endswith("__eid"))
    ct_out = ct.drop(columns=["p_one_sided_le_0"]).copy()
    ct_out["p_one_sided_le_0"] = np.where(ct["core"].astype(bool) & (ct["contrast"] == "ABC_minus_AB") &
                                          ct["outcome"].isin(["outcome1", "outcome2"]),
                                          ct["p_one_sided_le_0"], np.nan)
    return dict(test=pd.DataFrame(rows), contrast=ct_out, holm=holm_tab, curve=cur, internal=internal, keep_ab=keep_ab)


# ------------------------------------------------------------------- stacked (encounter, landmark) sensitivity
def stacked_fit(blocks: list, K: int, hidden: int, seed: int, mode: str, max_epochs: int, gamma0=None,
                patience: int = 5, batch: int = 256, lr: float = 2e-3, wd: float = 1e-4) -> tuple:
    """One GRU (Run 2 D19 architecture) on every (encounter, landmark) train row; batches never mix landmarks, so
    no padding enters. mode 'per_row' (each row weight 1) or 'per_patient' (each subject's rows share weight 1).
    Early stopping on a 10% subject holdout of train (Run 2 rule)."""
    torch = R2._torch()
    seed_torch(torch, seed)
    dev = torch_device()
    rng = np.random.default_rng(seed)
    g_all = np.concatenate([b["g"] for b in blocks])
    us, inv, cnt = np.unique(g_all, return_inverse=True, return_counts=True)
    hold = set(rng.choice(us, size=max(1, int(round(0.1 * len(us)))), replace=False).tolist())
    w_all = np.ones(len(g_all)) if mode == "per_row" else 1.0 / cnt[inv]
    w_all = w_all / w_all.mean()
    o = 0
    for b in blocks:
        n = len(b["g"])
        b["w"] = w_all[o:o + n]
        o += n
        hm = np.array([g in hold for g in b["g"]])
        b["fit_i"], b["hold_i"] = np.flatnonzero(~hm), np.flatnonzero(hm)
    Net = R2.make_gru_class()
    d_seq, d_st = blocks[0]["seq"].shape[2], blocks[0]["st"].shape[1]
    net = Net(d_seq, d_st, hidden, K, gamma0=gamma0).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)  # noqa: E731

    def wloss(b, ii):
        z = net(T(b["seq"][ii].astype(np.float32)), T(b["st"][ii]), T(b["off"][ii]))
        Y = T((b["y"][ii][:, None] > np.arange(K - 1)[None, :]).astype(np.float32))
        l_ = torch.nn.functional.binary_cross_entropy_with_logits(-z, Y, reduction="none").mean(1)
        wt = T(b["w"][ii])
        return (wt * l_).sum(), float(wt.sum())

    def hold_loss():
        net.eval()
        num = den = 0.0
        with torch.no_grad():
            for b in blocks:
                for i in range(0, len(b["hold_i"]), 2048):
                    s, w_ = wloss(b, b["hold_i"][i:i + 2048])
                    num, den = num + float(s), den + w_
        return num / max(den, 1e-12)

    best, best_state, bad, ep = math.inf, None, 0, 0
    for ep in range(1, max_epochs + 1):
        net.train()
        plan = []
        for bi, b in enumerate(blocks):
            perm = rng.permutation(b["fit_i"])
            plan += [(bi, perm[i:i + batch]) for i in range(0, len(perm), batch)]
        for j in rng.permutation(len(plan)):
            bi, ii = plan[j]
            opt.zero_grad()
            s, w_ = wloss(blocks[bi], ii)
            (s / max(w_, 1e-12)).backward()
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
    state = dict(state_dict=best_state, d_seq=int(d_seq), d_static=int(d_st), hidden=hidden, K=K)
    return state, ep


def stacked_channels(D: dict) -> list:
    """Channels shared by every landmark's Run 1 deduplicated channel list (order of the 24 h list): the stacked
    model needs one input width across landmarks."""
    B = D["bases"]
    common = set.intersection(*[set(map(str, B[f"C_L{L}__channels"])) for L in R2.LANDMARKS])
    return [c for c in map(str, B["C_L24__channels"]) if c in common]


def stacked_train(D: dict, mats: OrderedDict, G: TestGuard, V: dict, cfg: Config) -> dict:
    """Train phase (no test row): per-landmark train blocks with the cross-fitted Layer A offset (Run 2 seed keys),
    then the per-row and per-patient weighted stacked GRUs; validation predictions for the record."""
    R = D["rows"]
    sp = R["split"]
    A_name = V["w"]["matrix"].split("+")[0]
    K = 9
    blocks = []
    chan = stacked_channels(D)
    for L in R2.LANDMARKS:
        idx = D["lc"][L]["idx"]
        itr, iva = np.flatnonzero(sp[idx] == "train"), np.flatnonzero(sp[idx] == "validation")
        G.fit_rows(idx[itr], "stacked sensitivity GRU")
        y = R["O1"][L][idx]
        XA, Xab = mats[A_name]["X"][idx], V["Xw"][idx]
        g = R["subj"][idx]
        off_tr, _, _, _ = R2.crossfit_offset(XA[itr], y[itr], g[itr], XA[iva], K, ("s17", L, "outcome1"))
        S = R2.load_layer_c(D["lc"][L], chan)
        blocks.append(dict(L=L, seq=S[itr].astype(np.float16), st=Xab[itr].astype(np.float32), off=off_tr,
                           y=y[itr], g=g[itr]))
        check_time()
    y_all = np.concatenate([b["y"] for b in blocks])
    out = OrderedDict()
    for mode in ("per_row", "per_patient"):
        t0 = time.time()
        stt, ep = stacked_fit(blocks, K, STACK_HIDDEN, R2.SEED + 5000, mode, cfg.stacked_max_epochs,
                              gamma0=R2.class_logit_init(y_all, K))
        out[mode] = dict(state=stt, epochs=ep)
        say(f"stacked sensitivity: {mode} GRU trained ({time.time() - t0:.0f} s, {ep} epochs)")
    out["channels"] = chan
    return out


def stacked_score(D: dict, mats: OrderedDict, G: TestGuard, V: dict, ST: dict, keep_ab: dict, cfg: Config) -> dict:
    R, tab = D["rows"], D["tab"]
    sp = R["split"]
    A_name = V["w"]["matrix"].split("+")[0]
    K = 9
    G.te()
    chan = ST["channels"]
    ST = OrderedDict((k, v) for k, v in ST.items() if k != "channels")
    preds = {m: [] for m in ST}
    ys, ab, subj, lms = [], [], [], []
    for L in R2.LANDMARKS:
        idx = D["lc"][L]["idx"]
        ite = np.flatnonzero(sp[idx] == "test")
        S = R2.load_layer_c(D["lc"][L], chan)
        XA, Xab = mats[A_name]["X"][idx], V["Xw"][idx]
        mA = tab[f"L{L}_outcome1_A"]
        if not np.array_equal(keep_ab[L]["pos"], idx[ite]):
            raise BuildError("stacked sensitivity: test rows differ from the stage 18 rows")
        for m, rec in ST.items():
            preds[m].append(G.score(f"stacked_{m}_L{L}", lambda: gru_predict(rec["state"], S[ite], Xab[ite],
                                                                              mA.decision(XA[ite]))))
        ys.append(keep_ab[L]["y"])
        ab.append(keep_ab[L]["P"])
        subj.append(R["subj"][idx][ite])
        lms.append(np.full(len(ite), L))
        check_time()
    y, Pab, g, lm = np.concatenate(ys), np.concatenate(ab), np.concatenate(subj), np.concatenate(lms)
    W = R2.boot_weights(g, cfg.n_boot, R2.rng_for("boot", "stacked"))
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    pw = (1.0 / cnt[inv]).astype(np.float32)
    rows = []
    for m in ST:
        P = np.concatenate(preds[m])
        for wname, wt in (("per_row", np.ones(len(y), dtype=np.float32)), ("per_patient", pw)):
            Wt = W * wt[None, :]
            k1 = R2.boot_qwk(Wt, y, R2.median_class(P), K)
            k0 = R2.boot_qwk(Wt, y, R2.median_class(Pab), K)
            q1 = float(R2.boot_qwk(wt[None, :], y, R2.median_class(P), K)[0])
            q0 = float(R2.boot_qwk(wt[None, :], y, R2.median_class(Pab), K)[0])
            lo, hi = interval(k1 - k0)
            rows.append(OrderedDict(training_weights=m, scoring_weights=wname, scope="pooled", landmark="",
                                    qwk_stacked=q1, qwk_ab_per_landmark=q0, delta_qwk=q1 - q0, delta_p2_5=lo,
                                    delta_p97_5=hi, label="sensitivity (descriptive)"))
        for L in R2.LANDMARKS:
            mm = lm == L
            Wl = W[:, mm]
            k1 = R2.boot_qwk(Wl, y[mm], R2.median_class(P[mm]), K)
            k0 = R2.boot_qwk(Wl, y[mm], R2.median_class(Pab[mm]), K)
            q1, q0 = R2.qwk(y[mm], R2.median_class(P[mm]), K), R2.qwk(y[mm], R2.median_class(Pab[mm]), K)
            lo, hi = interval(k1 - k0)
            rows.append(OrderedDict(training_weights=m, scoring_weights="per_row", scope="landmark", landmark=L,
                                    qwk_stacked=q1, qwk_ab_per_landmark=q0, delta_qwk=q1 - q0, delta_p2_5=lo,
                                    delta_p97_5=hi, label="sensitivity (descriptive)"))
    say("stacked sensitivity: test scoring done")
    return dict(table=pd.DataFrame(rows), internal={f"stacked_{m}_P": np.concatenate(preds[m]).astype(np.float32)
                                                    for m in ST})


# ----------------------------------------------------------------------------------------- decisions, deviations
DECISIONS = OrderedDict([
    ("D16-D21", "applied as in Run 2 (dedup by Run 1 column names; Stage 08 base Layer C; D18 thresholds delta = "
                "delta_a = 0.02, delta_C = 0.01, epsilon = 0.01; the D19 GRU; B_eval = 1000; scripts by the assistant)"),
    ("test once", "plan I4 and contract: test rows are scored a single time with frozen choices; this desktop "
                  "run is a post-primary sensitivity analysis (a second look at the test set), not a replacement "
                  "for the laptop primary scoring, and uses its own test-use log under DESKTOP_FULL_PLAN; that "
                  "log refuses a second published desktop scoring"),
    ("family (a)", "H6 on test: ordinal LR (proportional odds, the Run 2 H6 headline model) at every width (full, "
                   "SVD and CPQR k grid) for the ten adjacent steps, plus Layer C at 24 h added to the frozen A+B "
                   "winner; equivalence p-value = the larger one-sided bootstrap p against -delta and +delta; Holm "
                   "at alpha 0.05"),
    ("family (b)", "H9 on test: A+B+C minus A+B QWK at the seven core landmarks (24, 48, 72, 120, 168, 240, 336 h; "
                   "contract v13); one-sided bootstrap p for delta <= 0; Holm at alpha 0.05; supported where Holm "
                   "rejects and the gain is at least delta_C; tables lead with the later core landmarks (72, 120, "
                   "168, 240, 336 h), the 14-landmark curve is descriptive only"),
])
DEVIATIONS = [
    "the Run 2 module is imported (after its SHA-256 matches the pin) for the models, metrics, bootstrap and GRU; "
    "frozen model files are read with a restricted unpickler",
    "Run 2 saved no stage 11 bake-off models and no Layer C 24 h GRU: the ordinal LR (proportional odds and "
    "all-threshold) at every width and the validation-selected Layer C 24 h GRU are refitted on train with the frozen "
    "Run 2 configuration and seeds, and must reproduce the Run 2 validation QWK before any test row is read "
    "(ordinal LR: within 0.005, an environment-reproducibility tolerance; Layer C 24 h GRU: within 0.005)",
    "family (a) uses the ordinal LR only (the model fitted at every width); random forest, gradient boosting and the "
    "multinomial reference are not refitted at every width for test (runtime); the frozen stage 12 winner is scored",
    "family (a) includes one Layer C 24 h representation (the Run 2 validation choice), compared with the frozen "
    "uncalibrated A+B winner as in Run 2",
    "bootstrap p-values on test use the (count + 1) / (B + 1) rule; Run 2 validation p-values used count / B",
    "family (a) TOST p-values come from a dedicated paired subject-cluster bootstrap with B = 20000 (intervals stay "
    "at B = 1000): with 131 tests Holm's first threshold (0.05 / 131) is below 1 / 1001, the smallest p-value "
    "B = 1000 can give, so no equivalence could ever be declared",
    "H6 AUROC parts (macro AUROC and death-threshold AUROC within delta_a) are reported with bootstrap intervals "
    "but are not in the Holm family (QWK only, as in the Run 2 verdicts)",
    "the H9 frozen-rule check (landmark features under the frozen 3.0 whole-stay rule, epsilon) is not evaluated: "
    "no such feature build exists",
    "stacked (encounter, landmark) sensitivity: one GRU over all 14 landmarks (hidden 32, batches never mix "
    "landmarks; the Layer C channels shared by every landmark's Run 1 channel list; at most 200 epochs with the D19 "
    "early stopping), per-row and per-patient weighted training; compared with the frozen "
    "per-landmark A+B on the same "
    "test rows; descriptive only",
    "desktop full-settings run (v2.0.1-desktop): no 60-minute time cap; training and the stacked sensitivity fit "
    "use CUDA if available, otherwise CPU (logged; seeds set on CPU and CUDA); frozen-model reproduction forward "
    "passes, and test scoring of the frozen Run 2 landmark GRUs, run on CPU in float32, matching the laptop Run 2, "
    "even when CUDA is available; every file this script writes is under "
    "data/processed/ladder/DESKTOP_FULL_PLAN/ (published folder, pending folder, cache, test-use log, internal "
    "files and README_DESKTOP.txt), so a published laptop Run 3a does not block this sensitivity run",
    "environment-reproducibility tolerance: desktop ordinal LR refits are not required to be bit-identical to the "
    "laptop Run 2; their validation QWK must fall within 0.005 of the Run 2 TEAM QWK (repro_tol_lr_qwk). The "
    "measured max abs differences (QWK and frozen-model probabilities) are written in the audit and in the "
    "progress and failure lines. Frozen landmark-model probabilities still use a 1e-4 tolerance, checked on a "
    "CPU float32 forward pass",
    "the v2.0.0-desktop attempt on this desktop failed during validation-only verification (the 2e-6 ordinal LR "
    "QWK tolerance and frozen GRU forward passes on CUDA), before any test row was read; the desktop test-use "
    "log is unchanged, so this v2.0.1 run is not a second test scoring",
    "counts gate: per-threshold metrics with fewer than 10 rows in either class, and the mortality AUROC with fewer "
    "than 10 deaths or survivors, are written as <10",
]

_STRUCT_COL3 = re.compile(r"^(k|p|landmark|class|threshold|hidden|epochs_trained)$")
AUDIT_NUM_KEYS3 = None


def audit_num_keys() -> tuple:
    return tuple(R2.AUDIT_NUM_KEYS) + ("repro", "p_holm", "p_tost", "alpha", "test_attempts", "epochs", "focus", "epsilon",
                                       "c24_epochs", "stacked_max_epochs")


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
                    if not size_hit and R2._size_hit(cell, forbidden):
                        size_hit = True
                    if not id_hit:
                        if R2._NUMERIC_CELL.match(cell):
                            id_hit = bool(re.fullmatch(r"[-+]?\d+", cell)) and cell.lstrip("+") in ids
                        else:
                            toks = R2.id_tokens(cell)
                            if c in ("item", "label", "family"):
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
    cur = T["stage18_curve_TEAM.csv"].sort_values("landmark")
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    x = cur["landmark"].astype(int).to_numpy()
    d = cur["delta_qwk"].astype(float).to_numpy()
    lo, hi = cur["delta_p2_5"].astype(float).to_numpy(), cur["delta_p97_5"].astype(float).to_numpy()
    ax.errorbar(x, d, yerr=[d - lo, hi - d], fmt="o-", capsize=3, lw=1.0, ms=3, color="0.45", label="all 14 (descriptive)")
    f = np.isin(x, H9_FOCUS)
    ax.errorbar(x[f], d[f], yerr=[(d - lo)[f], (hi - d)[f]], fmt="s", capsize=4, ms=6, color="C3",
                label="H9 later core landmarks (day 3, 5, 7, 10, 14)")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(R2.THRESH["delta_C"], color="grey", lw=0.6, ls="--")
    ax.set_xlabel("landmark (h)")
    ax.set_ylabel("test QWK(A+B+C) - QWK(A+B)")
    ax.set_title("Stage 18: test gain of Layer C (95% subject-bootstrap interval)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, metadata={"Software": None})
    plt.close(fig)
    out["fig18_1_test_delta_qwk_by_landmark_TEAM.png"] = buf.getvalue()
    h6 = T["stage14_h6_test_TEAM.csv"]
    g = h6[(h6["model"] == FAMILY_A_MODEL) & h6["rep"].str.startswith("svd_")].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for (a, b), gg in g.groupby(["from_matrix", "to_matrix"], sort=False):
        gg = gg.assign(kk=gg["rep"].str.split("_").str[1].astype(int)).sort_values("kk")
        ax.plot(gg["kk"], gg["delta_qwk"].astype(float), "o-", lw=1.0, ms=3, label=f"{a} -> {b}")
    ax.axhspan(-R2.THRESH["delta"], R2.THRESH["delta"], color="0.9")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("k (SVD width)")
    ax.set_ylabel("test delta QWK (ordinal LR)")
    ax.set_title("Stage 14: adjacent-level test delta QWK at matched width (shaded: +/- delta)", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, metadata={"Software": None})
    plt.close(fig)
    out["fig14_1_test_h6_matched_width_TEAM.png"] = buf.getvalue()
    return out


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return R2.r6(v) if math.isfinite(v) else None


# ---------------------------------------------------------------------------------------------------- audit
def make_audit(cfg, pre, checks, info, S14, S18, SS, script_sha, n_written) -> dict:
    hb = S18["holm"]
    h9 = [OrderedDict(landmark=int(r["landmark"]), focus=bool(r["h9_focus"]), delta_qwk=_f(r["delta_qwk"]),
                      delta_ci=[_f(r["delta_p2_5"]), _f(r["delta_p97_5"])], p_one_sided_le_0=_f(r["p_one_sided_le_0"]),
                      p_holm=_f(r["p_holm"]), rejected=bool(r["rejected_holm"]), supported=bool(r["h9_claim_met"]))
          for _, r in hb[hb["outcome"] == "outcome1"].iterrows()]
    o2 = [OrderedDict(landmark=int(r["landmark"]), focus=bool(r["h9_focus"]), delta_qwk=_f(r["delta_qwk"]),
                      delta_ci=[_f(r["delta_p2_5"]), _f(r["delta_p97_5"])], p_holm=_f(r["p_holm"]),
                      rejected=bool(r["rejected_holm"]))
          for _, r in hb[hb["outcome"] == "outcome2"].iterrows()]
    cur = [OrderedDict(landmark=int(r["landmark"]), delta_qwk=_f(r["delta_qwk"]),
                       delta_ci=[_f(r["delta_p2_5"]), _f(r["delta_p97_5"])],
                       delta_auroc_macro=_f(r["delta_auroc_macro"])) for _, r in S18["curve"].iterrows()]
    n_sup = sum(x["supported"] for x in h9)
    verdict = ("supported at every core landmark" if n_sup == len(h9) else
               ("supported at some core landmarks only" if n_sup else "not supported at any core landmark"))
    t = S14["test"]
    win = []
    for cal in (False, True):
        r = t[(t["item"] == "stage 12 winner (frozen)") & (t["calibrated"] == cal)].iloc[0]
        win.append(OrderedDict(calibrated=cal, qwk=_f(r["qwk"]), qwk_ci=[_f(r["qwk_p2_5"]), _f(r["qwk_p97_5"])],
                               emd=_f(r["emd"]), auroc_macro=_f(r["auroc_macro"]), ece_mean=_f(r["ece_mean"]),
                               mortality_auroc=(_f(r["mortality_auroc"]) if r["mortality_auroc"] != R2.SUPP else None)))
    fa = S14["holm"]
    eq = OrderedDict()
    for (a, b), g in fa[fa["model"] == FAMILY_A_MODEL].groupby(["from_matrix", "to_matrix"], sort=False):
        eq[f"{a} -> {b}"] = [str(x) for x in g.loc[g["equivalent_holm"].astype(bool), "rep"]]
    c24 = fa[fa["model"] == "gru"].iloc[0]
    st = SS["table"] if SS else None
    stk = []
    if st is not None:
        for _, r in st[st["scope"] == "pooled"].iterrows():
            stk.append(OrderedDict(training=r["training_weights"], scoring=r["scoring_weights"],
                                   delta_qwk=_f(r["delta_qwk"]), delta_ci=[_f(r["delta_p2_5"]), _f(r["delta_p97_5"])]))
    return OrderedDict(
        script=SCRIPT_VERSION, script_sha256=script_sha, spec=SPEC_DOC, plan=PLAN, contract=CONTRACT,
        generated_utc=utc_now(), input_hashes=pre["hashes"], test_attempts_before=pre["prior_attempts"],
        decisions=DECISIONS, deviations=DEVIATIONS,
        configuration=OrderedDict(seed=R2.SEED, n_boot=cfg.n_boot, n_boot_family_a=cfg.n_boot_family_a, alpha=ALPHA, delta=R2.THRESH["delta"],
                                  delta_C=R2.THRESH["delta_C"], epsilon=R2.THRESH["epsilon"],
                                  core_landmarks=list(R2.CORE_LANDMARKS), focus_landmarks=list(H9_FOCUS),
                                  max_seconds=cfg.max_seconds, stacked=cfg.stacked,
                                  stacked_max_epochs=cfg.stacked_max_epochs,
                                  torch_device=str(torch_device()) if R2 is not None else "not recorded"),
        reproduction=OrderedDict((k, _f(v)) for k, v in info.items() if k.startswith("repro")),
        c24_rep=info.get("c24_rep"),
        h9_family_b=h9, h9_verdict=verdict, outcome2_core=o2, curve_descriptive=cur,
        s14_winner=OrderedDict(model=S14["winner"]["model"], matrix=S14["winner"]["matrix"],
                               rep=S14["winner"]["rep"], test=win),
        s14_family_a_equivalent_reps=eq,
        s14_c24=OrderedDict(delta_qwk=_f(c24["delta_qwk"]), delta_ci=[_f(c24["delta_p2_5"]), _f(c24["delta_p97_5"])],
                            p_tost=_f(c24["p_tost"]), p_holm=_f(c24["p_tost_holm"]),
                            equivalent=bool(c24["equivalent_holm"]),
                            p_one_sided_le_0=_f(c24["c24_gain_p_one_sided_le_0"])),
        stacked=stk, acceptance_checks=OrderedDict((k, "pass" if v else "fail") for k, v in checks.items()),
        n_files=n_written, release_gate="ok")


def _ci(v):
    return "[" + ", ".join("nan" if x is None else f"{x:.4g}" for x in v) + "]"


def _n(x):
    return "nan" if x is None else f"{x:.4g}"


def audit_md(a: dict) -> str:
    L = ["# Stages 14 and 18 test scoring (Run 3a desktop full settings) audit - TEAM", "",
         f"- script: {a['script']} (`{a['script_sha256']}`); spec {a['spec']}; plan {a['plan']}, contract {a['contract']}",
         f"- generated (UTC): {a['generated_utc']}",
         "- post-primary sensitivity analysis (a second look at the test set), not a replacement for the laptop "
         "Run 3a primary scoring; test rows scored once on this desktop log with frozen Run 2 choices; aggregate "
         "only: no patient, row, encounter or subject "
         "counts and no identifiers; the release gate re-read every TEAM file from disk",
         f"- earlier test-use attempts (failed, nothing published): {a['test_attempts_before']}", "",
         "## Input hashes", ""]
    L += [f"- {k}: `{v}`" for k, v in a["input_hashes"].items()]
    L += ["", "## Decisions applied", ""] + [f"- **{k}**: {v}" for k, v in a["decisions"].items()]
    L += ["", "## Deviations from the plan (logged)", ""] + [f"- {d}" for d in a["deviations"]]
    c = a["configuration"]
    L += ["", "## Configuration", "",
          f"- seed {c['seed']}; paired subject-cluster bootstrap B = {c['n_boot']} (family (a) TOST p-values: B = "
          f"{c['n_boot_family_a']}); alpha {c['alpha']}; delta "
          f"{c['delta']}, delta_C {c['delta_C']}, epsilon {c['epsilon']}; "
          f"{('no time limit' if not c.get('max_seconds') else 'time limit ' + str(c['max_seconds']) + ' s')}; "
          f"torch device {c.get('torch_device', 'not recorded')}",
          f"- core landmarks {c['core_landmarks']}; H9 focus (later core landmarks) {c['focus_landmarks']}; stacked "
          f"sensitivity {c['stacked']}; stacked max epochs {c.get('stacked_max_epochs', STACK_MAX_EPOCHS)}",
          "- this is a post-primary sensitivity analysis (a second look at the test set), not a replacement for "
          "the laptop Run 3a primary scoring; outputs are under data/processed/ladder/DESKTOP_FULL_PLAN/", "",
          "## Frozen-model reproduction on validation (before any test row was read)", ""]
    L += [f"- {k}: {_n(v)}" for k, v in a["reproduction"].items()]
    L += [f"- Layer C 24 h representation (Run 2 validation choice): {a['c24_rep']}", "",
          "## H9 family (b): test QWK(A+B+C) - QWK(A+B), Outcome 1, core landmarks (Holm, alpha 0.05)", "",
          "Later core landmarks first (day 3, 5, 7, 10, 14), then 24 and 48 h.", "",
          "| landmark | focus | delta QWK [95% CI] | p (delta <= 0) | Holm p | rejected | gain >= delta_C and rejected |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {r['landmark']} | {r['focus']} | {_n(r['delta_qwk'])} {_ci(r['delta_ci'])} | {_n(r['p_one_sided_le_0'])} | "
          f"{_n(r['p_holm'])} | {r['rejected']} | {r['supported']} |" for r in a["h9_family_b"]]
    L += ["", f"- **H9 on test: {a['h9_verdict']}**", "",
          "## Outcome 2 (key secondary): A+B+C - A+B at the core landmarks (Holm within Outcome 2)", "",
          "| landmark | focus | delta QWK [95% CI] | Holm p | rejected |", "|---|---|---|---|---|"]
    L += [f"| {r['landmark']} | {r['focus']} | {_n(r['delta_qwk'])} {_ci(r['delta_ci'])} | {_n(r['p_holm'])} | "
          f"{r['rejected']} |" for r in a["outcome2_core"]]
    L += ["", "## Added-value curve, all 14 landmarks (descriptive; no test)", "",
          "| landmark | delta QWK [95% CI] | delta macro AUROC |", "|---|---|---|"]
    L += [f"| {r['landmark']} | {_n(r['delta_qwk'])} {_ci(r['delta_ci'])} | {_n(r['delta_auroc_macro'])} |"
          for r in a["curve_descriptive"]]
    w = a["s14_winner"]
    L += ["", f"## Stage 14: frozen winner on test ({w['model']} on {w['matrix']}, {w['rep']})", "",
          "| calibrated | QWK [95% CI] | EMD | macro AUROC | mean ECE | mortality AUROC |", "|---|---|---|---|---|---|"]
    L += [f"| {r['calibrated']} | {_n(r['qwk'])} {_ci(r['qwk_ci'])} | {_n(r['emd'])} | {_n(r['auroc_macro'])} | "
          f"{_n(r['ece_mean'])} | {_n(r['mortality_auroc'])} |" for r in w["test"]]
    L += ["", "## Stage 14 family (a): H6 equivalence on test (ordinal LR; Holm, alpha 0.05)", "",
          "Widths at which the adjacent step is equivalent within +/- delta after Holm:", ""]
    L += [f"- {k}: {', '.join(v) if v else 'none'}" for k, v in a["s14_family_a_equivalent_reps"].items()]
    c = a["s14_c24"]
    L += [f"- Layer C at 24 h added to the A+B winner: delta QWK {_n(c['delta_qwk'])} {_ci(c['delta_ci'])}; "
          f"equivalence p {_n(c['p_tost'])}, Holm p {_n(c['p_holm'])}, equivalent {c['equivalent']}; one-sided p "
          f"(gain <= 0) {_n(c['p_one_sided_le_0'])}", "",
          "## Stacked (encounter, landmark) sensitivity (descriptive)", "",
          "| training weights | scoring weights | pooled delta QWK vs per-landmark A+B [95% CI] |", "|---|---|---|"]
    L += [f"| {r['training']} | {r['scoring']} | {_n(r['delta_qwk'])} {_ci(r['delta_ci'])} |" for r in a["stacked"]]
    L += ["", "## Acceptance checks", ""] + [f"- {k}: {v}" for k, v in a["acceptance_checks"].items()]
    L += ["", "Release gate: ok. Detail tables: stage14_*_TEAM.csv, stage18_*_TEAM.csv; *_INTERNAL files are "
          "team-internal (never opened, pasted or shared).", ""]
    return "\n".join(L)


# -------------------------------------------------------------------------------------------------- publish
def publish(cfg, D, pre, T, internal, checks, info, S14, S18, SS) -> dict:
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
        audit = make_audit(cfg, pre, checks, info, S14, S18, SS, script_sha, len(files))
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
    say("loading Stage 08 levels (24 h rows, all splits; test rows behind the guard), Layer C index, Run 1 bases, "
        "Run 2 labels and frozen models")
    D = load_all(cfg, checks)
    mats = R2.build_matrices(D, checks)
    G = TestGuard(D["rows"]["split"])
    say("verification (validation only): frozen winner, ordinal LR refits, Layer C 24 h refit")
    V = verify14(D, mats, G, cfg, checks, info)
    verify18(D, mats, G, V, cfg, checks, info)
    ST = None
    if cfg.stacked:
        say("stacked sensitivity: training on train rows")
        ST = stacked_train(D, mats, G, V, cfg)
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise BuildError("verification failed before any test row was read (test-use log unchanged): " +
                         repro_fail_detail(failed, info, cfg))
    attempts = read_test_log(cfg.test_log)
    script_sha = sha256_file(Path(__file__))
    attempts.append(dict(script=SCRIPT_VERSION, script_sha256=script_sha, started_utc=utc_now(), status="started"))
    write_test_log(cfg.test_log, attempts)
    G.phase = "score"
    try:
        say("stage 14: test scoring (frozen winner, ordinal LR refits, Layer C 24 h)")
        S14 = score14(D, mats, G, V, cfg, checks)
        S14["winner"] = V["w"]
        say("stage 18: test scoring of the frozen landmark models")
        S18 = score18(D, mats, G, V, cfg, checks)
        SS = stacked_score(D, mats, G, V, ST, S18["keep_ab"], cfg) if ST else None
        G.phase = "done"
        checks["r3_no_fit_on_test_rows"] = all("test" not in sp for _, _, sp in G.fits)
        checks["r3_fits_only_before_scoring"] = all(ph == "verify" for _, ph, _ in G.fits)
        checks["r3_each_test_scoring_once"] = len(G.scored) == len(set(G.scored))
        failed = [k for k, v in checks.items() if not v]
        if failed:
            raise BuildError("acceptance checks failed: " + "; ".join(failed))
        T = OrderedDict()
        T["stage14_test_TEAM.csv"] = S14["test"]
        T["stage14_h6_test_TEAM.csv"] = S14["h6"]
        T["stage14_holm_family_a_TEAM.csv"] = S14["holm"]
        T["stage18_test_landmarks_TEAM.csv"] = S18["test"]
        T["stage18_contrast_TEAM.csv"] = S18["contrast"]
        T["stage18_holm_core_TEAM.csv"] = S18["holm"]
        T["stage18_curve_TEAM.csv"] = S18["curve"]
        if SS:
            T["stage18_stacked_sensitivity_TEAM.csv"] = SS["table"]
        pr = OrderedDict(S14["internal"])
        pr.update(S18["internal"])
        if SS:
            pr.update(SS["internal"])
        internal = OrderedDict([("stage14_18_test_predictions_INTERNAL.npz", R2._npz_bytes(dict(pr)))])
        res = publish(cfg, D, pre, T, internal, checks, info, S14, S18, SS)
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


def build_config(argv=None) -> tuple:
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
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-family-a", type=int, default=20000)
    ap.add_argument("--gru-max-epochs", type=int, default=40)
    ap.add_argument("--stacked-max-epochs", type=int, default=STACK_MAX_EPOCHS)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=30.0)
    ap.add_argument("--no-stacked", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.project_root)
    lad = root / "data" / "processed" / "ladder"
    here = Path(__file__).resolve().parent
    r2s = Path(a.run2_script) if a.run2_script else here.parent / "stage11_18_r2" / f"{RUN2_MODULE}.py"
    desk = lad / DESKTOP_ROOT_NAME
    cfg = Config(project_root=root, in_dir=Path(a.in_dir) if a.in_dir else lad / "stage06_08",
                 run1_dir=Path(a.run1_dir) if a.run1_dir else lad / "stage09_10_r1",
                 run2_dir=Path(a.run2_dir) if a.run2_dir else lad / RUN2_NAME, run2_script=r2s,
                 canonical=Path(a.canonical) if a.canonical else root / "data" / "interim" / "canonical_icu_encounters.csv",
                 harmonization=Path(a.harmonization) if a.harmonization else root / "data" / "interim" /
                 "encounter_harmonization_fields.csv",
                 split_file=Path(a.split_file) if a.split_file else root / "data" / "interim" / "r2_inputs" /
                 "subject_splits.csv",
                 raw_dir=Path(a.raw_dir) if a.raw_dir else root / "data" / "interim" / "r2_inputs" / "raw",
                 out_dir=Path(a.out_dir) if a.out_dir else desk / OUT_NAME, test_log=desk / TEST_LOG_NAME,
                 expect_manifest_sha256=(a.expect_stage06_08_manifest_sha256 or "").strip().lower() or None,
                 expect_run1_manifest_sha256=(a.expect_run1_manifest_sha256 or "").strip().lower() or None,
                 expect_run2_manifest_sha256=(a.expect_run2_manifest_sha256 or "").strip().lower() or None,
                 precheck_only=a.precheck_only, n_boot=a.n_boot, n_boot_family_a=a.n_boot_family_a, gru_max_epochs=a.gru_max_epochs,
                 stacked_max_epochs=a.stacked_max_epochs,
                 max_seconds=a.max_seconds, heartbeat_seconds=a.heartbeat_seconds, stacked=not a.no_stacked)
    must_write_under_desktop(cfg.out_dir, root, "--out-dir")
    must_write_under_desktop(cfg.test_log, root, "test-use log")
    must_write_under_desktop(cfg.out_dir.parent / f"{cfg.out_dir.name}.pending-0", root, "pending folder")
    must_write_under_desktop(desktop_root(root) / CACHE_NAME, root, "cache folder", allow_root=False)
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
              "harmonization, split, raw ICUSTAYS / ADMISSIONS, test-use log, packages); nothing written (library "
              "cache folder only)")
        for k, v in sorted(res["hashes"].items()):
            print(f"{k} sha256 {v}")
        return 0
    print(res["audit_md"])
    print(f"elapsed {time.time() - t0:.0f} s")
    print("result: RUN 3a desktop (14, 18) OK; RELEASE GATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
