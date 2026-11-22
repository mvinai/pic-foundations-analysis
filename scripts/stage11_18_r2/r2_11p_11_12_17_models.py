"""Run 2 of stages 09-18 (PIC Foundations, PIC; plan v13, contract v13): the outcome-label build "11p"
(preflight of stage 11, plan gap G1), stage 11 (model bake-off at matched width on the 24 h landmark rows), stage 12
(tuning and calibration of the winner, validation only) and stage 17 (landmark models, A+B vs A+B+C, all 14
landmarks), in one pass. Validation only: no test row is scored or used in any fit.

r2_11p_11_12_17_models v1.0.0 (Nov 18, 2026).

Inputs (hash-gated before anything is parsed):
  data/processed/ladder/stage06_08/stage06_08_manifest.json pinned by its full SHA-256 on the command line, and every
  file it lists; data/processed/ladder/stage09_10_r1/stage09_10_r1_manifest.json (Run 1 v1.0.2) pinned by its full
  SHA-256 on the command line, and every file it lists; the canonical encounter file, the harmonization fields, the
  persisted subject split, and two raw PIC tables (ICUSTAYS and ADMISSIONS, read with usecols), each against its full
  plan v13 Table 3.1 pin.

Outputs (data/processed/ladder/stage11_12_17_r2/, published atomically from a stage11_12_17_r2.pending-<pid>
folder): TEAM files (aggregate only: metrics, intervals, ranks of models; class shares with cells under 10 and their
complements suppressed; no cohort, split or landmark sizes; no identifiers; no per-patient labels or scores) and
INTERNAL files (labels per landmark, validation predictions, fitted models and calibrators; never opened, pasted or
shared). A release gate re-reads every TEAM file from disk (Run 1 v1.0.2 rules).

Exit codes: 0 ok; 2 stopped on an input check (nothing written); 3 build or release-gate failure or a Windows file
lock (nothing published; a pending folder is removed, or named if Windows refuses to remove it).
Assistants never run this script on patient data; it runs on the real data only through Copilot, unedited.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# BLAS / OpenMP thread cap, set before numpy loads (an existing environment setting wins).
_THREADS = str(min(4, max(1, (os.cpu_count() or 2) // 2)))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402

SCRIPT_VERSION = "r2_11p_11_12_17_models v1.0.0"
SPEC_DOC = "stage11_12_17_r2_spec v1.0.0"
PLAN, CONTRACT = "v13", "v13"
OUT_NAME = "stage11_12_17_r2"
RUN1_NAME = "stage09_10_r1"

# ----------------------------------------------------------------------------------------------- pins and settings
PINS = {
    "stage06_08_script": "06_08_build_levels_preprocess v1.2.1",
    "stage06_08_script_sha256": "f55cdaf5d7dd51470a9bd9494b1ed37730fcc29c7948de29eb8461d33daec8e7",
    "stage06_08_files": 42,
    "run1_script": "r1_09p_09_10_spectral_lowrank v1.0.2",
    # Full pins (plan v13 Table 3.1; Run 1 v1.0.2 real run of Nov 18, 2026). Tests set these to None.
    "canonical": "e5feeb22b5457ab89c4c73235e6a4722d538aec5f88595b6f049004171286418",
    "harmonization": "ffd1b8ae6a13e491be2f7a6631eb4efc1f323656795b80b91e105a0644209d6d",
    "split": "60196792457af718155a878f25b3d502ae12dbc71ccabb4c2709518c62e40e31",
    "ICUSTAYS": "449119e698acaaf948921c34375db535b9fd3f6eb6e996a86069da95e90eea1b",
    "ADMISSIONS": "1451ae8ea0ff463269b348522047a4bc99efd2665c770e9c5cef0ffe2647e1ae",
    "run1_script_sha256": "707dafc80827f6cd378d8ed18ec2db6c0e3ea68c9532d376b8ae6186e08fea2f",
    "run1_manifest": "703d7b466d3f56e0c7b68427a325a2455a4ba70f713835852a6b61bcab367f61",
    "run1_bases": "6b52a538da8d79dba84d5c66aa4f67a4d4395c87c533606968497425ddeb6933",
    # raw table rows (Table 3.1) and Stage 1 valid stays; checked internally, never printed
    "raw_rows": None,  # public copy: table-size pins omitted
    "valid_stays": None,
}
RAW_COLS = {"ICUSTAYS": ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"],
            "ADMISSIONS": ["SUBJECT_ID", "HADM_ID", "DISCHTIME"]}
NEVER_READ_TOKEN = "_raw_LOCAL"
LEVELS_A = ("a0", "a1", "a2")
LEVELS_B = ("b0", "b1", "b2s", "b2d", "b2p")
LEVELS = LEVELS_A + LEVELS_B
AB_PAIRS = (("a0", "b0"), ("a1", "b1"), ("a2", "b2s"), ("a2", "b2d"), ("a2", "b2p"))
MATRICES = LEVELS + tuple(f"{a}+{b}" for a, b in AB_PAIRS)
ADJACENT = (("a0", "a1"), ("a1", "a2"), ("b0", "b1"), ("b1", "b2s"), ("b1", "b2d"), ("b1", "b2p"),
            ("a0+b0", "a1+b1"), ("a1+b1", "a2+b2s"), ("a1+b1", "a2+b2d"), ("a1+b1", "a2+b2p"))
LANDMARKS = (24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336)
CORE_LANDMARKS = (24, 48, 72, 120, 168, 240, 336)
K_GRID = (5, 10, 20, 30, 40, 59)
TREE_K = (20, 59)
SEED = 6310
# Outcome 1 (contract v13 2.3 item 1): K = 9; level 8 died in hospital; survivors by remaining ICU time r = OUT - (t0+L)
O1_EDGES = (24.0, 48.0, 72.0, 120.0, 168.0, 240.0, 336.0)
K_OUT = {"outcome1": 9, "outcome2": 5, "ordinal8": 8}
ORD8_LANDMARKS = (24, 48, 72)
SUPPRESS_BELOW = 10
SUPP = "<10"
# D18 thresholds (frozen before any real stage 09-18 run)
THRESH = dict(delta=0.02, delta_C=0.01, epsilon=0.01)
FAMILY = {"polr": "LR", "allthr": "LR", "multinom": "reference", "rf_fh": "RF", "gb_fh": "GB"}
SIMPLICITY = ("LR", "RF", "GB")
CLASS_WEIGHTING = "none (unweighted likelihoods and losses for every model)"
DECISIONS = OrderedDict([
    ("D16", "D15 exact-duplicate collapse on every level and A+B concatenation (Run 1 dedup, reused by name), except "
            "the by-design __dup__ copies of b2d"),
    ("D17", "the Stage 08 base Layer C tensor is the single Layer C representation (C at 24 h in stage 11; stage 17)"),
    ("D18", "thresholds frozen before real runs: delta = delta_a = 0.02 (QWK equivalence), delta_C = 0.01 (Layer C "
            "gain), epsilon = 0.01"),
    ("D19", "A+B+C sequence model: one small GRU, Layer A+B static context at every step, Layer A baseline-risk "
            "offset (ordinal LR on A, cross-fitted on train) in a CORAL-style cumulative-link head, fixed small grid; "
            "TCN and Transformer deferred (reduced bake-off)"),
    ("D20", "validation-only tuning (no nested CV); B_eval = 1000 paired subject-cluster bootstrap replicates"),
    ("D21", "scripts written by the assistant; teammates cross-check them on the synthetic sandbox"),
])
DEVIATIONS = [
    "no lightgbm / xgboost / shap on the laptop: gradient boosting is scikit-learn HistGradientBoostingClassifier and "
    "random forest is RandomForestClassifier, both as Frank-Hall threshold models with monotone repair by "
    "rearrangement (sorting P(Y > j) in j); no SHAP is computed in Run 2 (attribution is stages 13 and 18, Run 3)",
    "proportional-odds ridge LR and the all-threshold variant are implemented directly (L-BFGS with analytic "
    "gradients, ridge penalty on the slopes only); the multinomial reference is scikit-learn LogisticRegression",
    "stage 11 tree models (RF and GB) are fitted at k in {20, 59} (top-k SVD scores and k CPQR columns) and at full "
    "width; the three linear models are fitted at every k of the grid; runtime",
    "stage 11 hyperparameters are fixed defaults (light grid deferred to stage 12, which tunes the winner only)",
    "the winner is chosen among LR (proportional odds or all-threshold), RF and GB on the A+B levels; the multinomial "
    "model is a reference and cannot win",
    "ordinal8 is computed from the canonical icu_los_hours and hospital_los_hours with the 3.0 contract 5.5 rule (the "
    "inherited label file is not a v13 pinned input)",
    "GRU inputs per bin: Stage 08 X_filled_z, missing and carried_forward of the Run 1 deduplicated channels plus the "
    "bin index; time since last measurement and the imputed flags are not used",
    "GRU early stopping on a 10% subject holdout inside train; the grid (hidden 32 or 64) is chosen on validation QWK",
    "stage 17 sequence-only comparator: GRU with hidden 32 and no static input or offset (Outcome 1 only); Outcome 2 "
    "at the core landmarks and ordinal8 at 24, 48 and 72 h use the A, A+B and A+B+C models, the GRU with the hidden "
    "size chosen for Outcome 1 at the same landmark",
    "the stage 11 winner is written as stage11_winner_TEAM.csv (plan: stage11_winner.json) so the release gate scans "
    "every cell",
    "the stacked (encounter, landmark) sensitivity model is deferred to Run 3",
    "calibration: isotonic regression per cumulative threshold fitted on validation, monotone by rearrangement; the "
    "reported post-calibration metrics are 5-fold subject-grouped cross-fits on validation",
    "class tables give within-row percentages (train and validation), never counts or row totals; cells under 10 and "
    "complementary cells are shown as <10",
]


class InputError(Exception):
    pass


class BuildError(Exception):
    pass


class TimeLimit(Exception):
    pass
# ----------------------------------------------------------------------------------------------------- utilities
_T0 = time.time()
_LOCK = threading.Lock()
_STATE = {"last": time.time(), "label": "starting", "deadline": None}


def say(msg: str) -> None:
    with _LOCK:
        print(f"[{time.time() - _T0:9.1f}s] {msg}", file=sys.stderr, flush=True)
        _STATE["last"] = time.time()
        _STATE["label"] = msg[:90]


class Heartbeat:
    def __init__(self, every: float):
        self.every, self.stop = every, threading.Event()

    def _run(self):
        while not self.stop.wait(1.0):
            with _LOCK:
                if time.time() - _STATE["last"] >= self.every:
                    left = (f"; time left {int(_STATE['deadline'] - time.time())} s" if _STATE["deadline"] else "")
                    print(f"[{time.time() - _T0:9.1f}s] heartbeat: still working - {_STATE['label']}{left}",
                          file=sys.stderr, flush=True)
                    _STATE["last"] = time.time()

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()


def check_time() -> None:
    if _STATE["deadline"] and time.time() > _STATE["deadline"]:
        raise TimeLimit("time limit reached")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def retry(op, what: str, attempts: int = 10):
    """Windows file-lock retry (cloud sync, Explorer, antivirus): 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30 s."""
    waits = (0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30)
    for i in range(attempts):
        try:
            return op()
        except PermissionError as e:
            if i == attempts - 1:
                raise BuildError(f"file operation failed after {attempts} attempts: {what} (PermissionError, "
                                 f"WinError {getattr(e, 'winerror', None)}); is cloud sync paused?") from None
            say(f"file operation retry: {what} (PermissionError, WinError {getattr(e, 'winerror', None)}); "
                f"attempt {i + 1}, waiting {waits[i]} s")
            time.sleep(waits[i])


def rmtree_robust(path: Path) -> None:
    """shutil.rmtree that clears the read-only attribute (Windows, cloud sync) and retries the entry once."""
    def _fix(func, p, _exc):
        try:
            os.chmod(p, 0o700 if os.path.isdir(p) else 0o600)
            func(p)
        except FileNotFoundError:
            pass
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_fix)
    else:  # pragma: no cover
        shutil.rmtree(path, onerror=_fix)


def write_bytes_atomic(path: Path, data: bytes) -> str:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:              # handle closed before the rename (Windows refuses to rename open files)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    retry(lambda: os.replace(tmp, path), f"replace {path.name}")
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rng_for(*key) -> np.random.Generator:
    """Deterministic generator per task: seed 6310 plus a stable key."""
    h = int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16)
    return np.random.default_rng(np.random.SeedSequence([SEED, h]))


def r6(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
        return float(f"{float(x):.6g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def pct(a, q):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")



def grouped_folds(groups: np.ndarray, k: int, rng: np.random.Generator) -> list:
    ug = np.unique(groups)
    perm = rng.permutation(ug)
    fold_of = {g: i % k for i, g in enumerate(perm)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]



# ------------------------------------------------------------------------------------------ ordinal model helpers
def cum_to_probs(F: np.ndarray) -> np.ndarray:
    """Cumulative P(Y <= j), j = 0..K-2 (n x K-1, non-decreasing in j) -> class probabilities (n x K)."""
    n = F.shape[0]
    Fz = np.c_[np.zeros(n), F, np.ones(n)]
    P = np.diff(Fz, axis=1)
    P = np.clip(P, 0.0, None)
    s = P.sum(1, keepdims=True)
    return P / np.where(s > 0, s, 1.0)


def exceed_to_probs(G: np.ndarray) -> np.ndarray:
    """Frank-Hall P(Y > j) (n x K-1) -> class probabilities, after monotone repair by rearrangement (sorting each
    row in decreasing order, Chernozhukov et al. 2009)."""
    G = -np.sort(-np.clip(G, 0.0, 1.0), axis=1)
    return cum_to_probs(1.0 - G)


def _theta(par: np.ndarray) -> np.ndarray:
    """Ordered thresholds: theta_0 = par[0], theta_j = theta_{j-1} + exp(par[j])."""
    return np.r_[par[0], par[0] + np.cumsum(np.exp(par[1:]))] if par.size > 1 else par.copy()


def _theta_jac(par: np.ndarray) -> np.ndarray:
    """d theta_j / d par_i (K-1 x K-1)."""
    m = par.size
    J = np.zeros((m, m))
    J[:, 0] = 1.0
    e = np.exp(par[1:])
    for j in range(1, m):
        J[j, 1:j + 1] = e[:j]
    return J


def _init_par(y: np.ndarray, K: int) -> np.ndarray:
    cum = np.array([(y <= j).mean() for j in range(K - 1)])
    cum = np.clip(cum, 1e-3, 1 - 1e-3)
    cum = np.maximum.accumulate(cum + 1e-6 * np.arange(K - 1))
    th = np.log(cum / (1 - cum))
    d = np.diff(th)
    return np.r_[th[0], np.log(np.maximum(d, 1e-3))]


class OrdinalLinear:
    """Cumulative-link ordinal regression with a ridge penalty lam/2 * ||w||^2 on the slopes (never the
    thresholds). kind = 'polr': proportional-odds likelihood, P(Y <= j | x) = sigmoid(theta_j - x w).
    kind = 'allthr': all-threshold logistic loss sum_j log(1 + exp(-s_j (theta_j - x w))), s_j = +1 if y <= j
    else -1 (Rennie and Srebro 2005); predictive distribution from the same cumulative form."""

    def __init__(self, K: int, lam: float = 1.0, kind: str = "polr", max_iter: int = 500):
        self.K, self.lam, self.kind, self.max_iter = K, float(lam), kind, max_iter

    def _loss(self, par, X, y, off):
        K, p = self.K, X.shape[1]
        w, tp = par[:p], par[p:]
        th = _theta(tp)
        eta = X @ w + off
        if self.kind == "polr":
            thz = np.r_[-np.inf, th, np.inf]
            a, b = thz[y + 1] - eta, thz[y] - eta
            Fa, Fb = expit(a), expit(b)
            pr = np.clip(Fa - Fb, 1e-300, None)
            fa, fb = Fa * (1 - Fa), Fb * (1 - Fb)
            loss = -np.log(pr).sum()
            ga, gb = -fa / pr, fb / pr                       # d loss / d a, d loss / d b
            d_eta = -(ga + gb)
            g_th = np.zeros(K - 1)
            m_a, m_b = y <= K - 2, y >= 1
            np.add.at(g_th, y[m_a], ga[m_a])
            np.add.at(g_th, y[m_b] - 1, gb[m_b])
        else:
            S = np.where(y[:, None] <= np.arange(K - 1)[None, :], 1.0, -1.0)
            Z = S * (th[None, :] - eta[:, None])
            loss = np.logaddexp(0.0, -Z).sum()
            dZ = -expit(-Z)                                  # d loss / d Z
            g_th = (dZ * S).sum(0)
            d_eta = -(dZ * S).sum(1)
        loss += 0.5 * self.lam * float(w @ w)
        gw = X.T @ d_eta + self.lam * w
        gt = _theta_jac(tp).T @ g_th
        return loss, np.r_[gw, gt]

    def fit(self, X, y, offset=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=np.float64)
        p = X.shape[1]
        x0 = np.r_[np.zeros(p), _init_par(y, self.K)]
        r = minimize(self._loss, x0, args=(X, y, off), jac=True, method="L-BFGS-B",
                     options={"maxiter": self.max_iter, "gtol": 1e-6})
        self.w, self.theta = r.x[:p], _theta(r.x[p:])
        return self

    def decision(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w

    def predict_proba(self, X, offset=None):
        eta = self.decision(X) + (0.0 if offset is None else offset)
        return cum_to_probs(expit(self.theta[None, :] - eta[:, None]))


class Multinom:
    """Multinomial logistic reference (scikit-learn), probabilities mapped onto all K classes."""

    def __init__(self, K: int, C: float = 1.0):
        self.K, self.C = K, C

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=self.C, max_iter=1000)
        cls = np.unique(y)
        if cls.size < 2:
            self.const = cls[0]
            return self
        self.const = None
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        P = np.zeros((X.shape[0], self.K))
        if self.const is not None:
            P[:, self.const] = 1.0
            return P
        P[:, self.m.classes_] = self.m.predict_proba(X)
        return P


class FrankHall:
    """Frank and Hall (2001): one binary classifier per threshold for P(Y > j), j = 0..K-2, repaired to a
    monotone sequence by rearrangement. base = 'gb' (HistGradientBoostingClassifier) or 'rf'
    (RandomForestClassifier)."""

    def __init__(self, K: int, base: str, params: dict, seed: int):
        self.K, self.base, self.params, self.seed = K, base, dict(params), seed

    def _make(self, j):
        if self.base == "gb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(early_stopping=False, random_state=self.seed + j, **self.params)
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_jobs=int(_THREADS), random_state=self.seed + j, **self.params)

    def fit(self, X, y):
        self.models = []
        for j in range(self.K - 1):
            t = (y > j).astype(int)
            if t.min() == t.max():
                self.models.append(float(t[0]))
            else:
                self.models.append(self._make(j).fit(X, t))
            check_time()
        return self

    def predict_proba(self, X):
        G = np.column_stack([np.full(X.shape[0], m) if isinstance(m, float) else m.predict_proba(X)[:, 1]
                             for m in self.models])
        return exceed_to_probs(G)


def make_model(name: str, K: int, hp: dict | None = None, seed: int = SEED):
    hp = dict(hp or {})
    if name in ("polr", "allthr"):
        return OrdinalLinear(K, lam=hp.get("lam", 1.0), kind=name)
    if name == "multinom":
        return Multinom(K, C=hp.get("C", 1.0))
    if name == "gb_fh":
        return FrankHall(K, "gb", {k: hp.get(k, d) for k, d in (("max_iter", 100), ("learning_rate", 0.1),
                                                                  ("max_leaf_nodes", 15))}, seed)
    if name == "rf_fh":
        return FrankHall(K, "rf", {k: hp.get(k, d) for k, d in (("n_estimators", 100), ("min_samples_leaf", 5),
                                                                  ("max_features", "sqrt"), ("max_samples", 0.5))},
                         seed)
    raise ValueError(name)


def fit_predict(name, K, Xtr, ytr, Xva, hp=None, seed=SEED):
    m = make_model(name, K, hp, seed).fit(Xtr, ytr)
    return m, m.predict_proba(Xva)


# ------------------------------------------------------------------------------------------------------ metrics
def median_class(P: np.ndarray) -> np.ndarray:
    """Decision rule (plan 10): the median of the predictive distribution."""
    return (np.cumsum(P, axis=1) >= 0.5 - 1e-12).argmax(axis=1)


def _qwk_from_conf(C: np.ndarray, K: int) -> np.ndarray:
    """Quadratic weighted kappa from confusion counts C (..., K, K); nan when undefined."""
    W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / max((K - 1) ** 2, 1)
    tot = C.sum(axis=(-1, -2), keepdims=True)
    O = C / np.where(tot > 0, tot, 1)
    E = O.sum(-1, keepdims=True) * O.sum(-2, keepdims=True)
    num, den = (W * O).sum(axis=(-1, -2)), (W * E).sum(axis=(-1, -2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 1.0 - num / den, np.nan)


def qwk(y, yhat, K) -> float:
    C = np.zeros((K, K))
    np.add.at(C, (y, yhat), 1.0)
    return float(_qwk_from_conf(C, K))


def auc_binary(score, lab) -> float:
    lab = np.asarray(lab, dtype=bool)
    n1, n0 = int(lab.sum()), int((~lab).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank(method="average").to_numpy()
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def avg_precision(score, lab) -> float:
    lab = np.asarray(lab, dtype=bool)
    if lab.sum() == 0:
        return float("nan")
    o = np.argsort(-score, kind="mergesort")
    l_ = lab[o]
    tp = np.cumsum(l_)
    prec = tp / np.arange(1, len(l_) + 1)
    return float((prec * l_).sum() / l_.sum())


def somers_d(score, y) -> float:
    """Somers' D(score | y) over pairs with different y: 2 * concordance - 1, ties in score count one half."""
    cl = np.unique(y)
    num = den = 0.0
    for i, a in enumerate(cl):
        for b in cl[i + 1:]:
            m = (y == a) | (y == b)
            au = auc_binary(score[m], y[m] == b)
            w = float((y == a).sum() * (y == b).sum())
            num += w * au
            den += w
    return float(2 * num / den - 1) if den > 0 else float("nan")


def ece(prob, lab, bins: int = 10) -> float:
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(prob[m].mean() - lab[m].mean())
    return float(tot / len(prob)) if len(prob) else float("nan")


def metrics(P: np.ndarray, y: np.ndarray, K: int, died: np.ndarray | None = None) -> dict:
    """All plan 10 validation metrics for one model (point estimates)."""
    yhat = median_class(P)
    F = np.cumsum(P, axis=1)[:, :-1]
    O = (y[:, None] <= np.arange(K - 1)[None, :]).astype(float)
    exc = 1.0 - F
    lab = y[:, None] > np.arange(K - 1)[None, :]
    au = [auc_binary(exc[:, j], lab[:, j]) for j in range(K - 1)]
    ap = [avg_precision(exc[:, j], lab[:, j]) for j in range(K - 1)]
    ec = [ece(exc[:, j], lab[:, j].astype(float)) for j in range(K - 1)]
    ev = P @ np.arange(K)
    out = dict(qwk=qwk(y, yhat, K), emd=float((P * np.abs(np.arange(K)[None, :] - y[:, None])).sum(1).mean()),
               rps=float(((F - O) ** 2).sum(1).mean() / (K - 1)), mae=float(np.abs(yhat - y).mean()),
               auroc_macro=float(np.nanmean(au)) if np.isfinite(au).any() else float("nan"),
               auprc_macro=float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan"),
               somers_d=somers_d(ev, y), ece_mean=float(np.nanmean(ec)), ece_max=float(np.nanmax(ec)))
    out["mortality_auroc"] = auc_binary(P[:, K - 1], died) if died is not None else float("nan")
    out["by_threshold"] = dict(auroc=au, auprc=ap, ece=ec)
    return out


def ece_by_threshold(P, y, K) -> list:
    exc = 1.0 - np.cumsum(P, axis=1)[:, :-1]
    return [ece(exc[:, j], (y > j).astype(float)) for j in range(K - 1)]


# ------------------------------------------------------------------------------------- paired cluster bootstrap
def boot_weights(subj: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    """Subject-cluster bootstrap: B x n row weights (each subject drawn with replacement; its rows get the draw
    count). Shared by every model scored on the same rows, so differences are paired."""
    us, inv = np.unique(subj, return_inverse=True)
    W = np.zeros((B, len(subj)), dtype=np.float32)
    for b in range(B):
        cnt = np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us)).astype(np.float32)
        W[b] = cnt[inv]
    return W


def boot_qwk(W: np.ndarray, y: np.ndarray, yhat: np.ndarray, K: int) -> np.ndarray:
    M = np.zeros((len(y), K * K), dtype=np.float32)
    M[np.arange(len(y)), y * K + yhat] = 1.0
    C = (W @ M).reshape(-1, K, K)
    return _qwk_from_conf(C.astype(np.float64), K)


def boot_mean(W: np.ndarray, v: np.ndarray) -> np.ndarray:
    s = W.sum(1)
    return (W @ v.astype(np.float32)) / np.where(s > 0, s, 1)


def emd_rows(P, y, K):
    return (P * np.abs(np.arange(K)[None, :] - y[:, None])).sum(1)


# --------------------------------------------------------------------------------------------- GRU (D19, torch)
def _torch():
    import torch
    torch.set_num_threads(int(_THREADS))
    return torch


def make_gru_class():
    torch = _torch()
    nn = torch.nn

    class StaticSeqGRU(nn.Module):
        """Plan 6.7 static-plus-sequential model. Per step: the bin's Layer C inputs, concatenated with a linear
        embedding of the static A+B vector (the same at every step; equivalent to concatenating the static columns
        to every step's input of the GRU's input map, but cheaper). Head: CORAL-style cumulative link
        P(Y <= j) = sigmoid(gamma_j - [s_static + g(x_<=t)]), gamma ordered, s_static the cross-fitted Layer A
        baseline score (offset, fixed coefficient 1), g a linear map of the last hidden state."""

        def __init__(self, d_seq, d_static, hidden, K, emb=16, gamma0=None):
            super().__init__()
            self.d_static = d_static
            self.emb = nn.Linear(d_static, emb) if d_static > 0 else None
            self.gru = nn.GRU(d_seq + (emb if d_static > 0 else 0), hidden, batch_first=True)
            self.out = nn.Linear(hidden, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            g0 = np.zeros(K - 1) if gamma0 is None else np.asarray(gamma0, dtype=float)
            d = np.diff(g0) if K > 2 else np.zeros(0)
            self.g_first = nn.Parameter(torch.tensor([g0[0]], dtype=torch.float32))
            self.g_steps = nn.Parameter(torch.tensor(np.log(np.expm1(np.maximum(d, 1e-3))), dtype=torch.float32))

        def gamma(self):
            return torch.cat([self.g_first, self.g_first + torch.cumsum(nn.functional.softplus(self.g_steps), 0)])

        def forward(self, xs, st, off):
            if self.emb is not None:
                e = self.emb(st).unsqueeze(1).expand(-1, xs.shape[1], -1)
                xs = torch.cat([xs, e], dim=2)
            h, _ = self.gru(xs)
            g = self.out(h[:, -1, :]).squeeze(1)
            return self.gamma().unsqueeze(0) - (off + g).unsqueeze(1)      # logits of P(Y <= j)

    return StaticSeqGRU


def gru_fit_predict(seq_tr, st_tr, off_tr, y_tr, g_tr, seq_va, st_va, off_va, K, hidden, seed, gamma0=None,
                    max_epochs=40, patience=5, batch=256, lr=2e-3, wd=1e-4):
    """Train with the all-threshold (CORAL) binary cross-entropy, Adam, early stopping on a 10% subject holdout of
    train; returns (state, validation class probabilities, epochs trained)."""
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Net = make_gru_class()
    us = np.unique(g_tr)
    hold = set(rng.choice(us, size=max(1, int(round(0.1 * len(us)))), replace=False).tolist())
    hm = np.array([g in hold for g in g_tr])
    fit_i, hold_i = np.flatnonzero(~hm), np.flatnonzero(hm)
    T = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)  # noqa: E731
    xs, xh = T(seq_tr), T(seq_va)
    st = T(st_tr) if st_tr is not None else torch.zeros((len(y_tr), 0))
    sh = T(st_va) if st_va is not None else torch.zeros((seq_va.shape[0], 0))
    ot, oh = T(off_tr), T(off_va)
    Y = torch.tensor((y_tr[:, None] > np.arange(K - 1)[None, :]).astype(np.float32))
    net = Net(seq_tr.shape[2], st.shape[1], hidden, K, gamma0=gamma0)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    bce = torch.nn.BCEWithLogitsLoss()

    def hold_loss():
        net.eval()
        with torch.no_grad():
            z = net(xs[hold_i], st[hold_i], ot[hold_i])
            return float(bce(-z, Y[hold_i]))          # P(Y > j) = sigmoid(-logit(Y <= j))

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
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        check_time()
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        F = torch.sigmoid(net(xh, sh, oh)).numpy().astype(np.float64)
    F = np.maximum.accumulate(F, axis=1)
    state = dict(state_dict=best_state, d_seq=int(seq_tr.shape[2]), d_static=int(st.shape[1]), hidden=hidden, K=K)
    return state, cum_to_probs(F), ep


# ------------------------------------------------------------------------------------------------------- config
@dataclass
class Config:
    project_root: Path
    in_dir: Path
    run1_dir: Path
    canonical: Path
    harmonization: Path
    split_file: Path
    raw_dir: Path
    out_dir: Path
    expect_manifest_sha256: str | None
    expect_run1_manifest_sha256: str | None
    precheck_only: bool
    n_boot: int = 1000
    gru_max_epochs: int = 40
    gru_hidden_grid: tuple = (32, 64)
    tree_k: tuple = TREE_K
    max_seconds: float = 3600.0
    heartbeat_seconds: float = 30.0
    harm_t0_col: str | None = None
    harm_out_col: str | None = None


def _hash_pin(path: Path, role: str, pin_key: str | None, hashes: dict) -> str:
    if NEVER_READ_TOKEN in str(path):
        raise InputError(f"{role}: a path containing {NEVER_READ_TOKEN} is never read")
    if not path.is_file():
        raise InputError(f"{role} not found: {path.name}")
    h = sha256_file(path)
    if pin_key and PINS.get(pin_key) and h != PINS[pin_key]:
        raise InputError(f"{role} SHA-256 differs from the plan v13 pin")
    hashes[role] = h
    return h


def precheck(cfg: Config) -> dict:
    """Hash gates before any data file is parsed: Stage 06-08 (as in Run 1), Run 1 v1.0.2, canonical, harmonization,
    split, raw ICUSTAYS / ADMISSIONS; the output folder must be empty and no pending folder may exist."""
    for p in (cfg.in_dir, cfg.run1_dir, cfg.out_dir, cfg.raw_dir):
        if NEVER_READ_TOKEN in str(p):
            raise InputError(f"a folder containing {NEVER_READ_TOKEN} is never read or written")
    d = cfg.in_dir
    man_p = d / "stage06_08_manifest.json"
    if not man_p.is_file():
        raise InputError(f"Stage 06-08 manifest not found: {man_p.name}")
    if cfg.expect_manifest_sha256 is None:
        raise InputError("--expect-stage06-08-manifest-sha256 is required")
    if cfg.expect_run1_manifest_sha256 is None:
        raise InputError("--expect-run1-manifest-sha256 is required")
    man_sha = sha256_file(man_p)
    if man_sha != cfg.expect_manifest_sha256:
        raise InputError("stage06_08_manifest.json SHA-256 differs from --expect-stage06-08-manifest-sha256")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    if man.get("script") != PINS["stage06_08_script"] or man.get("script_sha256") != PINS["stage06_08_script_sha256"]:
        raise InputError("Stage 06-08 manifest does not come from the pinned v1.2.1 builder")
    if man.get("release_gate") != "ok":
        raise InputError("Stage 06-08 release gate is not ok")
    ac = man.get("acceptance_checks") or {}
    if not ac or any(v != "pass" for v in ac.values()):
        raise InputError("Stage 06-08 acceptance checks not all pass")
    files = man.get("files", {})
    if len(files) != PINS["stage06_08_files"]:
        raise InputError(f"Stage 06-08 manifest lists {len(files)} files, expected {PINS['stage06_08_files']}")
    ignorable = {"desktop.ini", "thumbs.db"}
    on_disk = sorted(p.name for p in d.iterdir() if p.name.lower() not in ignorable)
    extra = sorted(set(on_disk) - set(files) - {"stage06_08_manifest.json"})
    missing = sorted(set(files) - set(on_disk))
    if extra or missing:
        raise InputError(f"Stage 06-08 folder does not match its manifest (extra {extra[:5]}, missing {missing[:5]})")
    need = [f"stage08_{k}.npz" for k in LEVELS] + [f"stage08_layer_c_L{L}.npz" for L in LANDMARKS]
    if any(n not in files for n in need):
        raise InputError("Stage 06-08 manifest lacks required Stage 08 files")
    hashes = {"stage06_08_manifest.json": man_sha}
    for i, (name, rec) in enumerate(sorted(files.items())):
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            say(f"hashing Stage 06-08 files: {i + 1}/{len(files)}")
        if sha256_file(d / name) != rec.get("sha256"):
            raise InputError(f"Stage 06-08 file {name} SHA-256 differs from its manifest entry")
    # Run 1 (v1.0.2): manifest pinned on the command line (and by the plan pin), every listed file verified
    r1 = cfg.run1_dir
    r1m = r1 / f"{RUN1_NAME}_manifest.json"
    if not r1m.is_file():
        raise InputError(f"Run 1 manifest not found: {r1m.name}")
    r1sha = sha256_file(r1m)
    if r1sha != cfg.expect_run1_manifest_sha256:
        raise InputError("stage09_10_r1_manifest.json SHA-256 differs from --expect-run1-manifest-sha256")
    if PINS.get("run1_manifest") and r1sha != PINS["run1_manifest"]:
        raise InputError("stage09_10_r1_manifest.json SHA-256 differs from the Run 1 pin")
    m1 = json.loads(r1m.read_text(encoding="utf-8"))
    if m1.get("script") != PINS["run1_script"]:
        raise InputError(f"Run 1 manifest script is {m1.get('script')!r}, expected {PINS['run1_script']}")
    if PINS.get("run1_script_sha256") and m1.get("script_sha256") != PINS["run1_script_sha256"]:
        raise InputError("Run 1 manifest script SHA-256 differs from the pinned v1.0.2 script")
    if m1.get("release_gate") != "ok":
        raise InputError("Run 1 release gate is not ok")
    a1 = m1.get("acceptance_checks") or {}
    if not a1 or any(v != "pass" for v in a1.values()):
        raise InputError("Run 1 acceptance checks not all pass")
    if (m1.get("input_hashes") or {}).get("stage06_08_manifest.json") != man_sha:
        raise InputError("Run 1 was built from a different Stage 06-08 manifest")
    f1 = m1.get("files", {})
    bases_nm = "stage10_matched_width_bases_INTERNAL.npz"
    if bases_nm not in f1:
        raise InputError("Run 1 manifest lacks the matched-width bases file")
    say(f"hashing Run 1 files: {len(f1)}")
    for name, rec in sorted(f1.items()):
        p = r1 / name
        if not p.is_file() or sha256_file(p) != rec.get("sha256"):
            raise InputError(f"Run 1 file {name} is missing or its SHA-256 differs from the Run 1 manifest")
    if PINS.get("run1_bases") and f1[bases_nm]["sha256"] != PINS["run1_bases"]:
        raise InputError("Run 1 matched-width bases SHA-256 differs from the Run 1 pin")
    hashes["stage09_10_r1_manifest.json"] = r1sha
    hashes[bases_nm] = f1[bases_nm]["sha256"]
    hc = _hash_pin(cfg.canonical, "canonical", "canonical", hashes)
    if hc != (man.get("input_hashes") or {}).get("canonical"):
        raise InputError("canonical file SHA-256 differs from the hash recorded in the Stage 06-08 manifest")
    _hash_pin(cfg.harmonization, "harmonization", "harmonization", hashes)
    _hash_pin(cfg.split_file, "split", "split", hashes)
    for t in ("ICUSTAYS", "ADMISSIONS"):
        _hash_pin(cfg.raw_dir / f"{t}.csv", t, t, hashes)
    if cfg.out_dir.exists() and any(cfg.out_dir.iterdir()):
        raise InputError(f"output folder exists and is not empty: {cfg.out_dir.name}")
    par = cfg.out_dir.parent
    pend = [p.name for p in par.glob(cfg.out_dir.name + ".pending-*")] if par.exists() else []
    if pend:
        raise InputError(f"a pending folder from an earlier failed run exists: {pend[0]} (remove it unopened first)")
    return {"hashes": hashes, "manifest": man, "run1_manifest": m1}


# ------------------------------------------------------------------------------------------------------- loading
_NULLS = {"", "nan", "none", "null", "nat", "na", "n/a"}
HARM_T0_CANDIDATES = ("t0", "T0", "t0_harmonized", "harmonized_t0", "first_intime", "FIRST_INTIME", "t0_intime")
HARM_OUT_CANDIDATES = ("OUT", "out", "last_outtime", "LAST_OUTTIME", "harmonized_out", "OUT_harmonized", "t_out",
                       "last_OUTTIME")
HARM_OUT_FALLBACK = ("OUTTIME",)
SPLIT_LABELS = {"train": "train", "training": "train", "val": "validation", "valid": "validation",
                "validation": "validation", "dev": "validation", "test": "test", "testing": "test"}


def norm_id(x) -> str:
    """Same rule as the v1.7 module / Stage 3: '123', 123, 123.0, '123.0', ' 123 ' -> '123'; missing -> ''."""
    if x is None:
        return ""
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return ""
        return str(int(x)) if float(x).is_integer() else repr(float(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    s = str(x).strip().strip("'\"").strip()
    if s.lower() in _NULLS:
        return ""
    m = re.fullmatch(r"([+-]?\d+)\.0*", s)
    return str(int(m.group(1))) if m else s


def parse_hadm_list(v) -> list:
    """Same rule as the v1.7 module: Python lists, JSON '[1, 2]', repr lists, '1;2' / '1,2' / '1|2'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return []
    s = str(v).strip()
    if s.lower() in {"", "nan", "none", "null", "[]", "()"}:
        return []
    items = None
    if s[0] in "[(":
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(s)
            except Exception:  # noqa: BLE001
                continue
            items = list(obj) if isinstance(obj, (list, tuple, set)) else [obj]
            break
    if items is None:
        items = [t for t in re.split(r"[,;|\s]+", s.strip("[]()")) if t]
    out = []
    for it in items:
        nid = norm_id(it)
        if nid and nid not in out:
            out.append(nid)
    return out


def read_csv_str(path: Path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, usecols=usecols, encoding="utf-8-sig")


def read_header(path: Path) -> list:
    return list(pd.read_csv(path, dtype=str, nrows=0, encoding="utf-8-sig").columns)


def fast_time(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    t = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    rest = t.isna() & ~s.str.lower().isin(_NULLS)
    if rest.any():
        try:
            t2 = pd.to_datetime(s[rest], errors="coerce", format="mixed")
        except (TypeError, ValueError):
            t2 = pd.to_datetime(s[rest], errors="coerce")
        t.loc[rest] = t2
    return t.astype("datetime64[ns]")


def hours(a, b) -> np.ndarray:
    """(a - b) in hours; NaN where either is missing."""
    x = np.asarray(pd.Series(a).to_numpy(dtype="datetime64[ns]")).astype("int64")
    y = np.asarray(pd.Series(b).to_numpy(dtype="datetime64[ns]")).astype("int64")
    out = ((x - y) / 3.6e12).astype(float)
    out[(x == np.iinfo(np.int64).min) | (y == np.iinfo(np.int64).min)] = np.nan
    return out


def resolve_column(columns, explicit, candidates, label, fallback=()) -> str:
    if explicit:
        if explicit not in columns:
            raise InputError(f"harmonization column for {label} not found: {explicit!r}")
        return explicit
    found = [c for c in candidates if c in columns] or [c for c in fallback if c in columns]
    if len(found) != 1:
        raise InputError(f"cannot resolve the harmonization {label} column ({len(found)} candidate matches)")
    return found[0]


def read_npz(path: Path, keys) -> dict:
    """Read the named arrays fully into memory and close the archive at once (no handle stays open on Windows)."""
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            raise InputError(f"{path.name} lacks arrays {missing}")
        return {k: np.array(z[k]) for k in keys}


def load_inputs(cfg: Config) -> dict:
    """Stage 08 levels (24 h landmark rows of train + validation only), Layer C (train + validation), Run 1 bases."""
    d = cfg.in_dir
    lv = OrderedDict()
    eid0 = split0 = elig24 = None
    for k in LEVELS:
        z = read_npz(d / f"stage08_{k}.npz", ("encounter_id", "split", "eligible_24h", "X_z", "columns"))
        eid, split = z["encounter_id"].astype(str), z["split"].astype(str)
        if eid0 is None:
            eid0, split0, elig24 = eid, split, z["eligible_24h"].astype(bool)
            keep = elig24 & np.isin(split0, ("train", "validation"))     # test rows are dropped here, never used
        elif not (np.array_equal(eid, eid0) and np.array_equal(split, split0)):
            raise InputError(f"level {k}: encounter order or split differs from level a0")
        X = z["X_z"][keep].astype(np.float64)
        if not np.isfinite(X).all():
            raise InputError(f"level {k}: non-finite values in X_z")
        lv[k] = dict(X=X, cols=[str(c) for c in z["columns"]])
        if len(lv[k]["cols"]) != X.shape[1]:
            raise InputError(f"level {k}: column list does not match X_z")
    if not set(np.unique(split0)) <= {"train", "validation", "test"}:
        raise InputError("unexpected split labels")
    rows = OrderedDict(eid=eid0[keep], split=split0[keep])
    emap = {e: i for i, e in enumerate(rows["eid"])}
    lc = OrderedDict()
    for L in LANDMARKS:
        z = read_npz(d / f"stage08_layer_c_L{L}.npz", ("encounter_id", "split", "channels"))
        e, s = z["encounter_id"].astype(str), z["split"].astype(str)
        m = np.isin(s, ("train", "validation"))
        idx = np.array([emap.get(x, -1) for x in e[m]])
        if (idx < 0).any() or not np.array_equal(rows["split"][idx], s[m]):
            raise InputError(f"Layer C L{L}: encounters or split do not match the 24 h tabular rows")
        lc[L] = dict(idx=idx, mask=m, channels=[str(c) for c in z["channels"]], all_eid=set(e.tolist()),
                     path=d / f"stage08_layer_c_L{L}.npz")
    bz = np.load(cfg.run1_dir / "stage10_matched_width_bases_INTERNAL.npz", allow_pickle=False)
    with bz:
        bases = {k: np.array(bz[k]) for k in bz.files}
    return dict(levels=lv, rows=rows, eid_all=eid0, split_all=split0, elig24_all=elig24, lc=lc, bases=bases)


def load_layer_c(r: dict, channels: list) -> np.ndarray:
    """Layer C prefix tensor (train + validation rows of the landmark) for the named channels: per bin the
    standardized filled value, the missing mask and the carried-forward flag of each channel, plus the bin index
    scaled to [0, 1] (n x bins x (3q + 1), float32). Read and closed at once; never flattened."""
    z = read_npz(r["path"], ("X_filled_z", "missing", "carried_forward", "channels"))
    ch = [str(c) for c in z["channels"]]
    pos = [ch.index(c) for c in channels]
    m = r["mask"]
    X = z["X_filled_z"][m][:, :, pos]
    if X.ndim != 3 or not np.isfinite(X).all():
        raise InputError("Layer C: non-finite values in X_filled_z")
    n, b, _ = X.shape
    t = np.broadcast_to((np.arange(b, dtype=np.float32) / max(55, b - 1))[None, :, None], (n, b, 1))
    return np.concatenate([X.astype(np.float32), z["missing"][m][:, :, pos].astype(np.float32),
                           z["carried_forward"][m][:, :, pos].astype(np.float32), t], axis=2)


# ----------------------------------------------------------------------------------------------- 11p: labels (G1)
def outcome1_level(r: np.ndarray, died: np.ndarray) -> np.ndarray:
    """Outcome 1 (contract v13 2.3.1): died -> 8; survivors by r = OUT - (t0 + L) > 0 with right-closed bands
    (0, 24], (24, 48], (48, 72], (72, 120], (120, 168], (168, 240], (240, 336], (336, inf) -> 0..7."""
    lev = np.searchsorted(np.asarray(O1_EDGES), r, side="left").astype(np.int64)
    return np.where(died, 8, lev)


def outcome2_stay_level(p_days: np.ndarray) -> np.ndarray:
    """Post-ICU stay p in days (no readmission): < 7 -> 0; 7 to < 14 -> 1; >= 14 -> 2."""
    return np.where(p_days < 7, 0, np.where(p_days < 14, 1, 2)).astype(np.int64)


def ordinal8_level(icu_h, hosp_h, died) -> np.ndarray:
    """Inherited ordinal8 (3.0 contract 5.5; plan v13 2.3): 7 died; survivors: ICU < 3 d with hospital < 7 d -> 0,
    7 to < 14 d -> 1, >= 14 d -> 2; ICU 3 to < 7 d with hospital < 14 d -> 3, >= 14 d -> 4; ICU 7 to 14 d -> 5
    (exactly 14.0 d is 5); ICU > 14 d -> 6. Unknown LOS -> -1 (unassignable)."""
    icu, hosp = np.asarray(icu_h, float), np.asarray(hosp_h, float)
    lev = np.where(icu < 72, np.where(hosp < 168, 0, np.where(hosp < 336, 1, 2)),
                   np.where(icu < 168, np.where(hosp < 336, 3, 4), np.where(icu <= 336, 5, 6)))
    lev = np.where(np.isnan(icu) | ((icu < 168) & np.isnan(hosp)), -1, lev)
    return np.where(died, 7, lev).astype(np.int64)


def label_selftest() -> bool:
    """Boundary unit checks of the band-support spec (check 12) on synthetic values inside the script."""
    r = np.array([24.0, 24.0001, 336.0, 336.0001, 0.5])
    ok = outcome1_level(r, np.zeros(5, bool)).tolist() == [0, 1, 6, 7, 0]
    ok &= outcome1_level(np.array([500.0]), np.array([True])).tolist() == [8]
    ok &= outcome2_stay_level(np.array([6.999, 7.0, 13.999, 14.0, 30.0001])).tolist() == [0, 1, 1, 2, 2]
    ok &= ordinal8_level([71.9, 71.9, 71.9, 100, 100, 336, 336.01, 10],
                         [100, 200, 400, 200, 400, 999, 999, 10],
                         np.array([0, 0, 0, 0, 0, 0, 0, 1], bool)).tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    return bool(ok)


def build_labels(cfg: Config, D: dict, checks: dict) -> dict:
    """Outcome 1 per landmark, Outcome 2, ordinal8 and mortality for every cohort encounter (contract v13 2.3; the
    band-support audit spec v2.3 rules). Never printed per encounter; INTERNAL file only."""
    eid_all, split_all = D["eid_all"], D["split_all"]
    ch = read_header(cfg.canonical)
    need = ["encounter_id", "SUBJECT_ID", "HADM_ID", "hadm_id_list", "Died", "icu_los_hours", "hospital_los_hours"]
    miss = [c for c in need if c not in ch]
    if miss:
        raise InputError(f"canonical file lacks columns {miss}")
    can = read_csv_str(cfg.canonical, usecols=need)
    can["encounter_id"] = can["encounter_id"].str.strip()
    if can["encounter_id"].duplicated().any():
        raise InputError("canonical encounter_id not unique")
    if set(can["encounter_id"]) != set(eid_all):
        raise InputError("the Stage 08 encounters differ from the canonical encounter set (count withheld)")
    can = can.set_index("encounter_id").loc[eid_all].reset_index()
    subj = can["SUBJECT_ID"].map(norm_id).to_numpy()
    dv = can["Died"].map(lambda v: str(v).strip().lower())
    if (~dv.isin({"0", "1", "0.0", "1.0", "true", "false"})).any():
        raise InputError("canonical Died has values that are not 0/1")
    died = dv.isin({"1", "1.0", "true"}).to_numpy()
    hl = can["hadm_id_list"].map(parse_hadm_list)
    hl = [x if x else ([norm_id(h)] if norm_id(h) else []) for x, h in zip(hl, can["HADM_ID"])]
    hh = read_header(cfg.harmonization)
    if "encounter_id" not in hh:
        raise InputError("harmonization file has no encounter_id column")
    t0c = resolve_column(hh, cfg.harm_t0_col, HARM_T0_CANDIDATES, "T0")
    outc = resolve_column(hh, cfg.harm_out_col, HARM_OUT_CANDIDATES, "OUT", HARM_OUT_FALLBACK)
    harm = read_csv_str(cfg.harmonization, usecols=["encounter_id", t0c, outc])
    harm["encounter_id"] = harm["encounter_id"].str.strip()
    if harm["encounter_id"].duplicated().any() or set(harm["encounter_id"]) != set(eid_all):
        raise InputError("harmonization encounter_id set is not unique or differs from the canonical set")
    harm = harm.set_index("encounter_id").loc[eid_all]
    t0, OUT = fast_time(harm[t0c]).to_numpy(), fast_time(harm[outc]).to_numpy()
    if pd.isna(t0).any() or pd.isna(OUT).any():
        raise InputError("harmonized t0 or OUT missing or unparseable for some encounters")
    out_h = hours(OUT, t0)
    if (out_h < 0).any():
        raise InputError("harmonized OUT earlier than t0 for some encounters")
    # split: persisted subject split, never regenerated; must equal the Stage 08 split for every encounter
    sh = read_header(cfg.split_file)
    if "SUBJECT_ID" not in sh or "split" not in sh:
        raise InputError("split file needs SUBJECT_ID and split columns")
    spl = read_csv_str(cfg.split_file, usecols=["SUBJECT_ID", "split"])
    spl = pd.DataFrame({"s": spl["SUBJECT_ID"].map(norm_id),
                        "sp": spl["split"].map(lambda v: SPLIT_LABELS.get(str(v).strip().lower(), "?"))})
    if (spl["sp"] == "?").any():
        raise InputError("split file has unknown split labels")
    spl = spl.drop_duplicates()
    if spl["s"].duplicated().any():
        raise InputError("a subject appears in more than one split")
    smap = dict(zip(spl["s"], spl["sp"]))
    sp_file = np.array([smap.get(s, "?") for s in subj])
    checks["11p_split_matches_split_file"] = bool(np.array_equal(sp_file, split_all))
    if not checks["11p_split_matches_split_file"]:
        raise InputError("Stage 08 split differs from the persisted subject split (count withheld)")
    # eligibility at every landmark must reproduce the Stage 08 landmark sets exactly
    ok = bool(np.array_equal(out_h > 24, D["elig24_all"]))
    for L in LANDMARKS:
        ok &= set(eid_all[out_h > L].tolist()) == D["lc"][L]["all_eid"]
    checks["11p_eligible_sets_match_stage08"] = ok
    if not ok:
        raise InputError("label eligibility (OUT > t0 + L) does not reproduce the Stage 08 landmark sets")
    # raw ICUSTAYS / ADMISSIONS, usecols only (band-support spec check 14)
    raws = {}
    for t, cols in RAW_COLS.items():
        hdr = read_header(cfg.raw_dir / f"{t}.csv")
        if any(c not in hdr for c in cols):
            raise InputError(f"raw {t} lacks a required column")
        raws[t] = read_csv_str(cfg.raw_dir / f"{t}.csv", usecols=cols)
        if PINS.get("raw_rows") and len(raws[t]) != PINS["raw_rows"][t]:
            raise InputError(f"raw {t} row count differs from contract Table P (count withheld)")
    st = raws["ICUSTAYS"]
    st = pd.DataFrame({"s": st["SUBJECT_ID"].map(norm_id), "h": st["HADM_ID"].map(norm_id),
                       "in": fast_time(st["INTIME"]), "out": fast_time(st["OUTTIME"])})
    st = st[st["in"].notna() & st["out"].notna() & (st["out"] > st["in"])]       # Stage 1 valid-stay rule
    if PINS.get("valid_stays") and len(st) != PINS["valid_stays"]:
        raise InputError("valid raw ICU stays differ from the Stage 1 count (count withheld)")
    ad = raws["ADMISSIONS"]
    disch = dict(zip(zip(ad["SUBJECT_ID"].map(norm_id), ad["HADM_ID"].map(norm_id)), fast_time(ad["DISCHTIME"])))
    by_s = {s: g for s, g in st.groupby("s")}
    n = len(eid_all)
    o2 = np.full(n, -1, dtype=np.int64)
    reason = np.full(n, "", dtype=object)
    ret24 = np.zeros(n, bool)
    readm_eid = np.full(n, -1, dtype=np.int64)
    ambiguous = 0
    e_index = {}
    for i, s in enumerate(subj):
        e_index.setdefault(s, []).append(i)
    readm_overlap_pairs_split_ok = True
    for i in range(n):
        if died[i]:
            o2[i] = 4
            continue
        g = by_s.get(subj[i])
        if g is None:
            reason[i] = "index_hadm_unmatched"
            continue
        cand = g[(g["out"].to_numpy() == OUT[i]) & g["h"].isin(hl[i]).to_numpy()]
        hs = sorted(set(cand["h"]) - {""})
        if not hs:
            reason[i] = "index_hadm_unmatched"
            continue
        if len(hs) > 1:
            ambiguous += 1
            dd = [(disch.get((subj[i], h), pd.NaT), h) for h in hs]
            dd = [(x if pd.notna(x) else pd.Timestamp.min, h) for x, h in dd]
            hidx = max(dd)[1]
        else:
            hidx = hs[0]
        later = g[(g["h"] == hidx) & (g["in"] > OUT[i])]
        if len(later):
            rin = later["in"].min().to_datetime64()
            gap = float((rin - OUT[i]) / np.timedelta64(1, "h"))
            if gap <= 24:
                ret24[i] = True
                continue
            o2[i] = 3
            for j in e_index[subj[i]]:
                if j != i and t0[j] <= rin <= OUT[j]:
                    readm_eid[i] = j
                    readm_overlap_pairs_split_ok &= split_all[j] == split_all[i]
            continue
        dt = disch.get((subj[i], hidx), pd.NaT)
        if pd.isna(dt):
            reason[i] = "dischtime_missing"
            continue
        pdays = float((dt.to_datetime64() - OUT[i]) / np.timedelta64(1, "h")) / 24.0
        if pdays < 0:
            reason[i] = "dischtime_before_out"
            continue
        o2[i] = int(outcome2_stay_level(np.array([pdays]))[0])
    elig_any = out_h > 24
    checks["11p_return_within_24h_zero"] = not bool((ret24 & elig_any).any())
    if not checks["11p_return_within_24h_zero"]:
        raise BuildError("label build: an ICU return within 24 h of OUT exists among eligible encounters (count "
                         "withheld); harmonization should have merged it (band-support spec rule 3)")
    checks["11p_readmit_overlap_pairs_same_split"] = bool(readm_overlap_pairs_split_ok)
    icu = pd.to_numeric(can["icu_los_hours"], errors="coerce").to_numpy()
    hosp = pd.to_numeric(can["hospital_los_hours"], errors="coerce").to_numpy()
    o8 = ordinal8_level(icu, hosp, died)
    O1 = OrderedDict()
    ok_r = ok_dp = ok_id = True
    for L in LANDMARKS:
        el = out_h > L
        r = out_h - L
        lev = np.where(el, outcome1_level(r, died), -1)
        O1[L] = lev
        ok_r &= bool((r[el & ~died] > 0).all())
        ok_dp &= bool(np.array_equal(lev[el] == 8, died[el])) and bool(np.array_equal(o2[el] == 4, died[el]))
        for s_ in ("train", "validation", "test"):
            m = el & (split_all == s_)
            ok_id &= int(((o2 >= 0) & m).sum()) + int(((o2 < 0) & m).sum()) == int(m.sum())
    ok_dp &= bool(np.array_equal(o8 == 7, died))
    checks["11p_survivor_remaining_time_positive"] = ok_r
    checks["11p_death_precedence"] = ok_dp
    checks["11p_outcome2_assignable_plus_unassignable_equals_eligible"] = ok_id
    checks["11p_label_boundary_selftest"] = label_selftest()
    unas = (o2 < 0) & elig_any
    reasons = OrderedDict((k, int((unas & (reason == k)).sum())) for k in
                          ("index_hadm_unmatched", "dischtime_missing", "dischtime_before_out"))
    status = OrderedDict((k, "none" if v == 0 else (SUPP if v < SUPPRESS_BELOW else "10 or more"))
                         for k, v in reasons.items())
    o8_unas = int(((o8 < 0) & elig_any).sum())
    info = OrderedDict(outcome2_unassignable=status,
                       ordinal8_unassignable="none" if o8_unas == 0 else (SUPP if o8_unas < SUPPRESS_BELOW
                                                                           else "10 or more"),
                       index_hadm_ambiguous="none" if ambiguous == 0 else (SUPP if ambiguous < SUPPRESS_BELOW
                                                                            else "10 or more"),
                       readmit_index_overlap_policy="keep_both",
                       readmission_overlap_present=bool((readm_eid >= 0).any()))
    return dict(subj=subj, died=died, out_h=out_h, O1=O1, O2=o2, O8=o8, info=info)


def label_class_table(Lb: dict, split_all: np.ndarray, checks: dict) -> pd.DataFrame:
    """Within-row class percentages for train and validation (never counts, never row totals). A row is an
    (outcome, landmark, split). Cells with a count under 10 are shown as <10; complementary suppression makes every
    row's suppressed set empty or at least two cells whose combined count is at least 10, and links the death cells
    (Outcome 1 level 8 = Outcome 2 level 4 = ordinal8 level 7 = deaths at the same landmark and split), so no
    suppressed cell is recoverable from a row total known elsewhere or from the death identity."""
    rows = []
    for L in LANDMARKS:
        el = Lb["out_h"] > L
        for s_ in ("train", "validation"):
            m = el & (split_all == s_)
            for outc, lab, K in (("outcome1", Lb["O1"][L], 9), ("outcome2", Lb["O2"], 5),
                                 ("ordinal8", Lb["O8"], 8)):
                if outc == "ordinal8" and L not in ORD8_LANDMARKS:
                    continue
                v = lab[m]
                v = v[v >= 0]
                cnt = np.bincount(v, minlength=K)[:K]
                rows.append(dict(outcome=outc, landmark=L, split=s_, cnt=cnt, tot=int(cnt.sum()), K=K))
    sup = [np.array(r["cnt"] < SUPPRESS_BELOW) for r in rows]
    death = {"outcome1": 8, "outcome2": 4, "ordinal8": 7}
    for _ in range(50):
        changed = False
        for r, s in zip(rows, sup):
            if s.any():
                while s.sum() == 1 or (r["cnt"][s].sum() < SUPPRESS_BELOW and (~s).any()):
                    free = np.flatnonzero(~s)
                    if free.size == 0:
                        break
                    s[free[np.argmin(r["cnt"][free])]] = True
                    changed = True
        groups = {}
        for i, r in enumerate(rows):
            groups.setdefault((r["landmark"], r["split"]), []).append(i)
        for idxs in groups.values():
            if any(sup[i][death[rows[i]["outcome"]]] for i in idxs):
                for i in idxs:
                    if not sup[i][death[rows[i]["outcome"]]]:
                        sup[i][death[rows[i]["outcome"]]] = True
                        changed = True
        if not changed:
            break
    ok = True
    for r, s in zip(rows, sup):
        if s.any():
            ok &= s.sum() >= 2 and (r["cnt"][s].sum() >= SUPPRESS_BELOW or s.all())
        ok &= not ((r["cnt"] < SUPPRESS_BELOW) & ~s).any()
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["landmark"], r["split"]), []).append(sup[i][death[r["outcome"]]])
    ok &= all(len(set(v)) == 1 for v in groups.values())
    checks["11p_class_table_no_suppressed_cell_recoverable"] = bool(ok)
    if not ok:
        raise BuildError("counts gate: a suppressed class cell would be recoverable")
    out = []
    for r, s in zip(rows, sup):
        for c in range(r["K"]):
            pc = 100.0 * r["cnt"][c] / r["tot"] if r["tot"] else float("nan")
            out.append(dict(outcome=r["outcome"], landmark=r["landmark"], split=r["split"], **{"class": c},
                            percent=SUPP if s[c] else f"{pc:.1f}"))
    return pd.DataFrame(out, columns=["outcome", "landmark", "split", "class", "percent"])


# ------------------------------------------------------------------------------------ matrices and representations
def build_matrices(D: dict, checks: dict) -> OrderedDict:
    """Deduplicated A, B and A+B matrices on the 24 h rows, rebuilt by column name from the Run 1 bases (D16), with
    the Run 1 train SVD right vectors V (up to 59) and the CPQR column order."""
    lv, B = D["levels"], D["bases"]
    src = {}
    for k in LEVELS:
        for j, c in enumerate(lv[k]["cols"]):
            src[c] = (k, j)
    mats, ok = OrderedDict(), True
    for m in MATRICES:
        key = m.replace("+", "_")
        if f"{key}__columns" not in B:
            raise InputError(f"Run 1 bases lack matrix {m}")
        cols = [str(c) for c in B[f"{key}__columns"]]
        if any(c not in src for c in cols):
            ok = False
            raise InputError(f"Run 1 deduplicated columns of {m} are not all in the Stage 08 levels")
        X = np.column_stack([lv[src[c][0]]["X"][:, src[c][1]] for c in cols])
        V = B[f"{key}__V"].astype(np.float64)
        piv = B[f"{key}__cpqr"].astype(np.int64)
        if V.shape[0] != len(cols) or (piv >= len(cols)).any():
            raise InputError(f"Run 1 bases of {m} do not match its column list")
        mats[m] = dict(X=X, cols=cols, V=V, cpqr=piv, layer="AB" if "+" in m else m[0].upper())
    checks["11_run1_dedup_columns_resolved"] = ok
    return mats


def reps_of(M: dict, tree_k=None) -> list:
    p = M["X"].shape[1]
    out = [("full", None)]
    for k in K_GRID:
        if tree_k is not None and k not in tree_k:
            continue
        if k < p and k <= M["V"].shape[1]:
            out.append(("svd", k))
        if k < p and k <= len(M["cpqr"]):
            out.append(("cpqr", k))
    return out


def rep_X(M: dict, kind: str, k) -> np.ndarray:
    if kind == "full":
        return M["X"]
    if kind == "svd":
        return M["X"] @ M["V"][:, :k]            # top-k SVD scores (Run 1 train SVD)
    return M["X"][:, M["cpqr"][:k]]              # k CPQR-selected columns (Run 1 train CPQR)


def rep_name(kind, k):
    return "full" if kind == "full" else f"{kind}_{k}"


def score_row(P, y, K, died, W, extra: dict) -> tuple:
    met = metrics(P, y, K, died)
    yhat = median_class(P)
    kb = boot_qwk(W, y, yhat, K)
    eb = boot_mean(W, emd_rows(P, y, K))
    row = OrderedDict(extra)
    row.update(qwk=met["qwk"], qwk_p2_5=pct(kb, 2.5), qwk_p97_5=pct(kb, 97.5), emd=met["emd"],
               emd_p2_5=pct(eb, 2.5), emd_p97_5=pct(eb, 97.5))
    row.update({k: met[k] for k in ("rps", "mae", "auroc_macro", "auprc_macro", "somers_d", "ece_mean", "ece_max",
                                    "mortality_auroc")})
    for nm, vals in met["by_threshold"].items():
        for j, v in enumerate(vals):
            row[f"{nm}_t{j}"] = v
    return row, kb


def diff_summary(a: np.ndarray, b: np.ndarray, point: float, delta: float) -> dict:
    d = a - b
    d = d[np.isfinite(d)]
    lo, hi = pct(d, 2.5), pct(d, 97.5)
    p2 = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if d.size else float("nan")
    if lo > -delta and hi < delta:
        v = "equivalent"
    elif lo > 0:
        v = "higher"
    elif hi < 0:
        v = "lower"
    else:
        v = "inconclusive"
    return dict(delta_qwk=point, delta_p2_5=lo, delta_p97_5=hi, p_two_sided=p2, verdict=v)


def crossfit_offset(XA_tr, y_tr, g_tr, XA_va, K, seed_key):
    """Layer A baseline risk (ordinal LR on A): out-of-fold linear score on train (5 subject-grouped folds) and the
    full-train fit on validation; also the full-train thresholds (initial GRU thresholds)."""
    off_tr = np.zeros(len(y_tr))
    for fi, (a, b) in enumerate(grouped_folds(g_tr, 5, rng_for("offset", *seed_key))):
        m = OrdinalLinear(K, 1.0, "polr").fit(XA_tr[a], y_tr[a])
        off_tr[b] = m.decision(XA_tr[b])
    full = OrdinalLinear(K, 1.0, "polr").fit(XA_tr, y_tr)
    return off_tr, full.decision(XA_va), full.theta, full


# --------------------------------------------------------------------------------------- stage 11: bake-off at 24 h
def stage11(D: dict, Lb: dict, mats: OrderedDict, cfg: Config, checks: dict) -> dict:
    R = D["rows"]
    tr, va = R["split"] == "train", R["split"] == "validation"
    checks["11_no_test_rows_used"] = not bool((R["split"] == "test").any())
    y = R["O1"][24]
    K = 9
    W = boot_weights(R["subj"][va], cfg.n_boot, rng_for("boot", "s11"))
    rows, KB, PRED = [], {}, {}
    checks["11_same_rows_every_matrix"] = len({M["X"].shape[0] for M in mats.values()}) == 1 and \
        all(M["X"].shape[0] == len(y) for M in mats.values())
    lin, trees = ("polr", "allthr", "multinom"), ("rf_fh", "gb_fh")
    for m, M in mats.items():
        t0 = time.time()
        for kind, k in reps_of(M):
            X = rep_X(M, kind, k)
            for mod in lin + (trees if (kind == "full" or k in cfg.tree_k) else ()):
                _, P = fit_predict(mod, K, X[tr], y[tr], X[va], seed=SEED)
                row, kb = score_row(P, y[va], K, R["died"][va], W,
                                    dict(outcome="outcome1", matrix=m, layer=M["layer"], rep=rep_name(kind, k),
                                         k=k if k else "", p=X.shape[1], model=mod, family=FAMILY[mod]))
                rows.append(row)
                KB[(mod, m, rep_name(kind, k))] = kb
                if M["layer"] == "AB":
                    PRED[(mod, m, rep_name(kind, k))] = P.astype(np.float32)
                check_time()
        say(f"stage 11: {m} done ({time.time() - t0:.0f} s)")
    sc = pd.DataFrame(rows)
    # winner among LR / RF / GB on the A+B levels (best validation QWK); the simplest family whose paired-bootstrap
    # interval of (best - family best) contains 0 wins (plan 10: LR < RF < GB)
    ab = sc[(sc["layer"] == "AB") & (sc["family"].isin(SIMPLICITY))]
    best_by_fam = OrderedDict()
    for f in SIMPLICITY:
        g = ab[ab["family"] == f]
        if len(g):
            r = g.loc[g["qwk"].idxmax()]
            best_by_fam[f] = r
    top_f = max(best_by_fam, key=lambda f: best_by_fam[f]["qwk"])
    top = best_by_fam[top_f]
    kb_top = KB[(top["model"], top["matrix"], top["rep"])]
    win_rows, chosen = [], None
    for f in SIMPLICITY:
        if f not in best_by_fam:
            continue
        r = best_by_fam[f]
        kb = KB[(r["model"], r["matrix"], r["rep"])]
        ds = diff_summary(kb_top, kb, float(top["qwk"] - r["qwk"]), THRESH["delta"])
        contains0 = ds["delta_p2_5"] <= 0 <= ds["delta_p97_5"] or f == top_f
        if chosen is None and contains0:
            chosen = r
        win_rows.append(OrderedDict(family=f, model=r["model"], matrix=r["matrix"], rep=r["rep"], qwk=r["qwk"],
                                    best_minus_this=ds["delta_qwk"], diff_p2_5=ds["delta_p2_5"],
                                    diff_p97_5=ds["delta_p97_5"], interval_contains_0=bool(contains0)))
    for w in win_rows:
        w["selected"] = bool(w["family"] == chosen["family"])
    win = dict(family=chosen["family"], model=chosen["model"], matrix=chosen["matrix"], rep=chosen["rep"],
               qwk=float(chosen["qwk"]))
    # H6 (validation, descriptive; Holm on test is stage 14): adjacent levels at matched width
    h6 = []
    for mod in lin + trees:
        for a, b in ADJACENT:
            for rp in sorted({r_ for (mo, mm, r_) in KB if mo == mod and mm == a} &
                             {r_ for (mo, mm, r_) in KB if mo == mod and mm == b}):
                qa = float(sc[(sc.model == mod) & (sc.matrix == a) & (sc.rep == rp)]["qwk"].iloc[0])
                qb = float(sc[(sc.model == mod) & (sc.matrix == b) & (sc.rep == rp)]["qwk"].iloc[0])
                ds = diff_summary(KB[(mod, b, rp)], KB[(mod, a, rp)], qb - qa, THRESH["delta"])
                h6.append(OrderedDict(model=mod, rep=rp, from_matrix=a, to_matrix=b, matched_width=rp != "full",
                                      **ds))
    return dict(scores=sc, winner=win, winner_rows=pd.DataFrame(win_rows), h6=pd.DataFrame(h6), W=W, PRED=PRED)


# --------------------------------------------------------------------------------- stage 12: tuning, calibration
TUNE_GRID = {
    "LR": [dict(lam=v) for v in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)],
    "GB": [dict(learning_rate=lr, max_leaf_nodes=ml, max_iter=mi) for lr in (0.05, 0.1) for ml in (7, 15, 31)
           for mi in (100, 200)],
    "RF": [dict(n_estimators=100, min_samples_leaf=ms, max_features=mf, max_samples=0.5) for ms in (1, 5, 20)
           for mf in ("sqrt", 0.2)],
}
HP_COLS = ("lam", "learning_rate", "max_leaf_nodes", "max_iter", "n_estimators", "min_samples_leaf", "max_features",
           "max_samples")


def fit_calibrators(P: np.ndarray, y: np.ndarray, K: int) -> list:
    from sklearn.isotonic import IsotonicRegression
    exc = 1.0 - np.cumsum(P, axis=1)[:, :-1]
    return [IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(exc[:, j], (y > j).astype(float))
            for j in range(K - 1)]


def apply_calibrators(cals: list, P: np.ndarray) -> np.ndarray:
    exc = 1.0 - np.cumsum(P, axis=1)[:, :-1]
    return exceed_to_probs(np.column_stack([c.predict(exc[:, j]) for j, c in enumerate(cals)]))


def crossfit_calibration(P, y, g, K, key) -> np.ndarray:
    """Honest post-calibration predictions on validation: isotonic fitted on 4 subject-grouped folds, applied to
    the fifth."""
    out = np.zeros_like(P)
    for a, b in grouped_folds(g, 5, rng_for("cal", *key)):
        out[b] = apply_calibrators(fit_calibrators(P[a], y[a], K), P[b])
    return out


def stage12(D: dict, mats: OrderedDict, S11: dict, cfg: Config, checks: dict) -> dict:
    R = D["rows"]
    tr, va = R["split"] == "train", R["split"] == "validation"
    y, K = R["O1"][24], 9
    w = S11["winner"]
    kind, k = (w["rep"], None) if w["rep"] == "full" else (w["rep"].split("_")[0], int(w["rep"].split("_")[1]))
    X = rep_X(mats[w["matrix"]], kind, k)
    grid = TUNE_GRID[w["family"]]
    model = w["model"]
    rows, best = [], None
    for ci, hp in enumerate(grid):
        _, P = fit_predict(model, K, X[tr], y[tr], X[va], hp=hp, seed=SEED)
        met = metrics(P, y[va], K, R["died"][va])
        r = OrderedDict(config_id=ci, model=model, **{c: hp.get(c, "") for c in HP_COLS})
        r.update(qwk=met["qwk"], emd=met["emd"], rps=met["rps"], ece_mean=met["ece_mean"])
        rows.append(r)
        if best is None or met["qwk"] > best[0]:
            best = (met["qwk"], ci, hp, P)
        check_time()
    for r in rows:
        r["selected"] = bool(r["config_id"] == best[1])
    hp, P = best[2], best[3]
    say(f"stage 12: tuned {model} on {w['matrix']} {w['rep']} ({len(grid)} configurations)")
    m_final = make_model(model, K, hp, SEED).fit(X[tr], y[tr])
    cals = fit_calibrators(P, y[va], K)
    Pc = crossfit_calibration(P, y[va], R["subj"][va], K, ("s12",))
    before, after = metrics(P, y[va], K, R["died"][va]), metrics(Pc, y[va], K, R["died"][va])
    eb, ea = ece_by_threshold(P, y[va], K), ece_by_threshold(Pc, y[va], K)
    cal = [OrderedDict(item="threshold", threshold=j, ece_before=eb[j], ece_after_crossfit=ea[j]) for j in range(K - 1)]
    for mk in ("qwk", "emd", "rps", "ece_mean", "auroc_macro", "mortality_auroc"):
        cal.append(OrderedDict(item=mk, threshold="", ece_before=before[mk], ece_after_crossfit=after[mk]))
    cdf = pd.DataFrame(cal).rename(columns={"ece_before": "before", "ece_after_crossfit": "after_crossfit"})
    mono = bool((np.diff(np.cumsum(apply_calibrators(cals, P), axis=1), axis=1) >= -1e-12).all())
    checks["12_calibrated_cumulative_probabilities_monotone"] = mono
    checks["12_no_test_access"] = checks.get("11_no_test_rows_used", False)
    return dict(tuning=pd.DataFrame(rows), calibration=cdf, hp=hp, model_name=model, model=m_final, cals=cals, P=P.astype(np.float32),
                kind=kind, k=k, X=X)


def stage11_secondary(D: dict, mats: OrderedDict, S11: dict, S12: dict, cfg: Config) -> pd.DataFrame:
    """Secondary outcomes at 24 h (Outcome 2, ordinal8; plan 10 / 3.0 contract 5.5): the bake-off models on the
    chosen A+B matrix and representation (stage 11 hyperparameters), rows with an assignable label only."""
    R = D["rows"]
    X = S12["X"]
    out = []
    for outc, yy in (("outcome2", R["O2"]), ("ordinal8", R["O8"])):
        K = K_OUT[outc]
        tr = (R["split"] == "train") & (yy >= 0)
        va = (R["split"] == "validation") & (yy >= 0)
        W = boot_weights(R["subj"][va], cfg.n_boot, rng_for("boot", "s11sec", outc))
        for mod in ("polr", "allthr", "rf_fh", "gb_fh"):
            _, P = fit_predict(mod, K, X[tr], yy[tr], X[va], seed=SEED)
            row, _ = score_row(P, yy[va], K, R["died"][va], W,
                               dict(outcome=outc, landmark=24, matrix=S11["winner"]["matrix"],
                                    rep=S11["winner"]["rep"], model=mod, family=FAMILY[mod]))
            out.append(row)
            check_time()
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------- Layer C inputs and GRU wrapper
def c_inputs(D: dict, L: int, sel: str, k) -> np.ndarray:
    """Layer C prefix inputs at landmark L for the train + validation rows of L (rows order of D['lc'][L]['idx']).
    sel = 'all' (every Run 1 deduplicated channel: value, missing, carried-forward, + bin index), 'cpqr' (the first k
    channels of the Run 1 CPQR order of the stacked train matrix, same three inputs each), or 'svd' (top-k
    channel-mode SVD scores of the filled values per bin, + bin index)."""
    B = D["bases"]
    chan = [str(c) for c in B[f"C_L{L}__channels"]]
    if sel == "cpqr":
        chan = [chan[j] for j in B[f"C_L{L}__cpqr"][:k]]
    S = load_layer_c(D["lc"][L], chan)
    if sel == "svd":
        q = len(chan)
        V = B[f"C_L{L}__V"][:, :k].astype(np.float32)
        S = np.concatenate([S[:, :, :q] @ V, S[:, :, -1:]], axis=2)
    return S


def run_gru(S, Xs, off, gamma0, y, idx_tr, idx_va, groups, K, hidden, seed, cfg):
    st_tr = Xs[idx_tr] if Xs is not None else None
    st_va = Xs[idx_va] if Xs is not None else None
    o_tr = off[0] if off is not None else np.zeros(len(idx_tr))
    o_va = off[1] if off is not None else np.zeros(len(idx_va))
    return gru_fit_predict(S[idx_tr], st_tr, o_tr, y[idx_tr], groups[idx_tr], S[idx_va], st_va, o_va, K, hidden,
                           seed, gamma0=gamma0, max_epochs=cfg.gru_max_epochs)


def class_logit_init(y, K):
    cum = np.array([(y <= j).mean() for j in range(K - 1)])
    cum = np.maximum.accumulate(np.clip(cum, 1e-3, 1 - 1e-3) + 1e-6 * np.arange(K - 1))
    return np.log(cum / (1 - cum))


# ------------------------------------------------------------------------------ stage 11: Layer C at 24 h (GRU)
def stage11_c24(D, mats, S11, S12, cfg, checks) -> dict:
    """Layer C on the 24 h prefix (bins 0-3) added to A+B, matched width k: k CPQR channels of the stacked train
    matrix (primary) or top-k channel-mode SVD scores (alternative), k <= q; the D19 GRU with the A+B static context
    of the chosen representation and the cross-fitted Layer A offset; compared with the tuned A+B model."""
    R, w = D["rows"], S11["winner"]
    r = D["lc"][24]
    idx = r["idx"]
    sp = R["split"][idx]
    itr, iva = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "validation")
    y, K, g = R["O1"][24][idx], 9, R["subj"][idx]
    Xs = S12["X"][idx]
    XA = mats[w["matrix"].split("+")[0]]["X"][idx]
    off_tr, off_va, th, _ = crossfit_offset(XA[itr], y[itr], g[itr], XA[iva], K, ("c24",))
    W = boot_weights(g[iva], cfg.n_boot, rng_for("boot", "c24"))
    pos = np.searchsorted(np.flatnonzero(R["split"] == "validation"), idx[iva])
    P_ab = S12["P"][pos].astype(np.float64)
    kb_ab = boot_qwk(W, y[iva], median_class(P_ab), K)
    q_ab = qwk(y[iva], median_class(P_ab), K)
    q = len(D["bases"]["C_L24__channels"])
    out = []
    plan = [("all", None)] + [(s, k) for s in ("cpqr", "svd") for k in K_GRID if k <= q and not (s == "cpqr" and k == q)]
    for sel, k in plan:
        S = c_inputs(D, 24, sel, k)
        _, P, ep = run_gru(S, Xs, (off_tr, off_va), th, y, itr, iva, g, K, 32, SEED, cfg)
        row, kb = score_row(P, y[iva], K, R["died"][idx][iva], W,
                            dict(c_rep=sel, k=k if k else "", hidden=32, epochs_trained=ep))
        row.update(diff_summary(kb, kb_ab, row["qwk"] - q_ab, THRESH["delta_C"]))
        out.append(row)
        check_time()
    say("stage 11: Layer C at 24 h done")
    return dict(table=pd.DataFrame(out))


# --------------------------------------------------------------------------- stage 17: landmark models (A+B vs C)
def stage17(D, mats, S11, S12, cfg, checks) -> dict:
    R, w = D["rows"], S11["winner"]
    Xab_all = S12["X"]
    A_name = w["matrix"].split("+")[0]
    XA_all = mats[A_name]["X"]
    vrows, crows, calrows = [], [], []
    states, tab_models, preds = OrderedDict(), OrderedDict(), OrderedDict()
    ok_prefix = True
    for L in LANDMARKS:
        t0 = time.time()
        r = D["lc"][L]
        idx = r["idx"]
        sp = R["split"][idx]
        itr, iva = np.flatnonzero(sp == "train"), np.flatnonzero(sp == "validation")
        g, died = R["subj"][idx], R["died"][idx]
        S_all = c_inputs(D, L, "all", None)
        ok_prefix &= S_all.shape[1] == L // 6
        W = boot_weights(g[iva], cfg.n_boot, rng_for("boot", "s17", L))
        todo = [("outcome1", R["O1"][L][idx])]
        if L in CORE_LANDMARKS:
            todo.append(("outcome2", R["O2"][idx]))
        if L in ORD8_LANDMARKS:
            todo.append(("ordinal8", R["O8"][idx]))
        sel_hidden = None
        for outc, yy in todo:
            K = K_OUT[outc]
            ok_rows = yy >= 0
            jtr, jva = itr[ok_rows[itr]], iva[ok_rows[iva]]
            Wv = W[:, ok_rows[iva]]
            yv = yy[jva]
            XA, Xab = XA_all[idx], Xab_all[idx]
            off_tr, off_va, th, mA = crossfit_offset(XA[jtr], yy[jtr], g[jtr], XA[jva], K, ("s17", L, outc))
            res = OrderedDict()
            res["A"] = (mA.predict_proba(XA[jva]), mA, None)
            mAB = make_model(S12["model_name"], K, S12["hp"], SEED).fit(Xab[jtr], yy[jtr])
            res["AB"] = (mAB.predict_proba(Xab[jva]), mAB, None)
            grid = cfg.gru_hidden_grid if outc == "outcome1" else (sel_hidden,)
            bestc = None
            for h in grid:
                stt, P, ep = run_gru(S_all, Xab, (off_tr, off_va), th, yy, jtr, jva, g, K, h, SEED + L, cfg)
                qq = qwk(yv, median_class(P), K)
                if bestc is None or qq > bestc[0]:
                    bestc = (qq, P, stt, h, ep)
            if outc == "outcome1":
                sel_hidden = bestc[3]
            res["ABC"] = (bestc[1], None, bestc[2])
            if outc == "outcome1":
                stt, P, ep_s = run_gru(S_all, None, None, class_logit_init(yy[jtr], K), yy, jtr, jva, g, K, 32,
                                       SEED + 1000 + L, cfg)
                res["C_only"] = (P, None, stt)
            kbs = {}
            for mname, (P, mobj, stt) in res.items():
                row, kb = score_row(P, yv, K, died[jva], Wv,
                                    dict(outcome=outc, landmark=L, core=L in CORE_LANDMARKS, model=mname,
                                         hidden=(bestc[3] if mname == "ABC" else (32 if mname == "C_only" else "")),
                                         epochs_trained=(bestc[4] if mname == "ABC" else "")))
                vrows.append(row)
                kbs[mname] = (kb, row["qwk"], P)
                preds[f"s17_L{L}_{outc}_{mname}"] = P.astype(np.float32)
                if stt is not None:
                    states[f"L{L}_{outc}_{mname}"] = stt
                if mobj is not None:
                    tab_models[f"L{L}_{outc}_{mname}"] = mobj
            ds = diff_summary(kbs["ABC"][0], kbs["AB"][0], kbs["ABC"][1] - kbs["AB"][1], THRESH["delta_C"])
            d_emd = float(emd_rows(kbs["ABC"][2], yv, K).mean() - emd_rows(kbs["AB"][2], yv, K).mean())
            kd = kbs["ABC"][0] - kbs["AB"][0]
            p1 = float((kd[np.isfinite(kd)] <= 0).mean())
            crows.append(OrderedDict(outcome=outc, landmark=L, core=L in CORE_LANDMARKS, contrast="ABC_minus_AB",
                                     delta_qwk=ds["delta_qwk"], delta_p2_5=ds["delta_p2_5"],
                                     delta_p97_5=ds["delta_p97_5"], p_one_sided_le_0=p1, delta_emd=d_emd,
                                     gain_at_least_delta_C=bool(ds["delta_qwk"] >= THRESH["delta_C"] and
                                                                ds["delta_p2_5"] > 0)))
            preds[f"s17_L{L}_{outc}__eid"] = R["eid"][idx][jva]
            if outc == "outcome1":
                for mname in ("AB", "ABC"):
                    P = kbs[mname][2]
                    cals = fit_calibrators(P, yv, K)
                    Pc = crossfit_calibration(P, yv, g[jva], K, ("s17", L, mname))
                    tab_models[f"L{L}_{outc}_{mname}__calibrators"] = cals
                    calrows.append(OrderedDict(landmark=L, model=mname,
                                               ece_mean_before=float(np.mean(ece_by_threshold(P, yv, K))),
                                               ece_mean_after_crossfit=float(np.mean(ece_by_threshold(Pc, yv, K))),
                                               qwk_before=qwk(yv, median_class(P), K),
                                               qwk_after_crossfit=qwk(yv, median_class(Pc), K)))
            check_time()
        say(f"stage 17: landmark {L} done ({time.time() - t0:.0f} s)")
    checks["17_prefix_bins_only"] = bool(ok_prefix)
    return dict(validation=pd.DataFrame(vrows), contrast=pd.DataFrame(crows), calibration=pd.DataFrame(calrows),
                states=states, tab_models=tab_models, preds=preds)


# ------------------------------------------------------------------------------------------------- release gate
AUDIT_NUM_KEYS = ("qwk", "emd", "rps", "mae", "auroc", "auprc", "somers", "ece", "mortality", "delta", "diff",
                  "best_minus", "before", "after", "p_one", "p_two", "seed", "n_boot", "gru", "tree_k", "k_grid",
                  "landmark", "core_landmarks", "threshold", "hidden", "lam", "learning_rate", "max_leaf_nodes",
                  "max_iter", "n_estimators", "min_samples_leaf", "max_features", "max_samples", "max_seconds",
                  "epochs", "n_files", "k", "configs")
_FORBID_COL = re.compile(r"(rows|subject|encounter|patient|^n$|n_train|n_obs|support|eligible|cohort|count)", re.I)


def audit_nums_ok(o, key="", inherited=False) -> bool:
    """Shareable audit: every number sits under a structural key (metrics, intervals, hyperparameters,
    configuration); no patient count anywhere."""
    allowed = inherited or any(key.startswith(k) for k in AUDIT_NUM_KEYS)
    if isinstance(o, bool):
        return True
    if isinstance(o, (int, float, np.integer, np.floating)):
        return allowed
    if isinstance(o, dict):
        return all(audit_nums_ok(v, str(k), allowed) for k, v in o.items())
    if isinstance(o, list):
        return all(audit_nums_ok(v, key, allowed) for v in o)
    return True


_SIZE_MSG = "a cohort, split, subject or landmark size"
_ID_MSG = "an identifier"


_SIZE_RE_CACHE: dict = {}


def _size_regex(forbidden_nums) -> re.Pattern:
    """One compiled alternation of every forbidden form (plain and comma-grouped), longest first; the same
    boundaries as a per-number search (Run 2: one regex pass per cell instead of one per number)."""
    key = frozenset(forbidden_nums)
    rx = _SIZE_RE_CACHE.get(key)
    if rx is None:
        forms = sorted({f for n in key for f in (str(n), f"{n:,}")}, key=lambda f: (-len(f), f))
        rx = re.compile(r"(?<![\w.])(?<![eE][+-])(?:" + "|".join(map(re.escape, forms)) + r")(?!\w|\.0*[1-9])")
        if len(_SIZE_RE_CACHE) > 8:
            _SIZE_RE_CACHE.clear()
        _SIZE_RE_CACHE[key] = rx
    return rx


def _size_hit(text: str, forbidden_nums: set) -> bool:
    """A forbidden size as a whole number: also as a CSV cell next to commas, comma-grouped ("4,646") or written as a
    float ("1234.0"); never the digits of a fraction or exponent, nor inside a hex digest or a name (fail closed: a
    match inside a thousands-grouped number also counts)."""
    if not forbidden_nums:
        return False
    return _size_regex(forbidden_nums).search(text) is not None


def id_tokens(text: str) -> set:
    """Identifier candidates in free text: whole word tokens. A purely numeric token counts only when written as an
    integer: never the digits after a decimal point or in an exponent, and never a number written with a decimal
    point ("151.0"), because this script never writes an identifier as a float (v1.0.2)."""
    return set(re.findall(r"(?<![\w.])(?<![eE][+-])[A-Za-z0-9_]+(?!\w|\.\d)", text))


def team_text_problems(text: str, forbidden_nums: set, ids: set, allowed_ints: set = frozenset()) -> list:
    """Problem types found in a piece of TEAM text (never the offending value). allowed_ints: integer tokens that
    are structural in this context (see release_gate) and are not treated as identifiers."""
    bad = []
    if forbidden_nums and _size_hit(text, forbidden_nums):
        bad.append(_SIZE_MSG)
    if ids:
        toks = id_tokens(text)
        toks -= {t for t in toks if t.isdigit() and int(t) in allowed_ints}
        if toks & ids:
            bad.append(_ID_MSG)
    return sorted(set(bad))


# TEAM CSV columns whose integers are structural by construction and therefore cannot carry a patient count or an
# identifier: both the size scan and the identifier scan skip them (Run 1 v1.0.2 rule, Run 2 column set).
#   k, p                                               representation width (fixed k grid) and matrix width
#   landmark                                           landmark hours (24..336)
#   class, threshold                                   outcome class (0..8) and cumulative threshold (0..7) indices
#   hidden, epochs_trained                             GRU hidden size (config grid) and epochs (<= gru_max_epochs)
#   config_id                                          row number of the fixed stage 12 tuning grid
#   lam, learning_rate, max_leaf_nodes, max_iter,
#   n_estimators, min_samples_leaf, max_features,
#   max_samples                                        fixed hyperparameter grids (TUNE_GRID, make_model defaults)
_STRUCT_COL = re.compile(
    r"^(k|p|landmark|class|threshold|hidden|epochs_trained|config_id|lam|learning_rate|max_leaf_nodes|max_iter|"
    r"n_estimators|min_samples_leaf|max_features|max_samples)$")
# Text columns written only from the script's fixed sentences: numeric tokens there are the script's own
# constants; word tokens are still scanned.
_SCRIPT_TEXT_COL = re.compile(r"^(item)$")
# The suppression marker written in place of a class percentage below 10 counts: its "10" is the script's constant
# (SUPPRESS_BELOW), never an identifier or a size, so an exact marker cell is skipped by both scans (Run 2).
_SUPP_CELL = SUPP
_NUMERIC_CELL = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$|^[-+]?(inf|nan)$", re.I)


def _audit_ints(o, out: set) -> set:
    if isinstance(o, bool):
        return out
    if isinstance(o, int) or (isinstance(o, float) and math.isfinite(o) and o == int(o)):
        out.add(int(o))
    elif isinstance(o, dict):
        for v in o.values():
            _audit_ints(v, out)
    elif isinstance(o, list):
        for v in o:
            _audit_ints(v, out)
    return out


def _skeleton(o):
    """The audit with every leaf replaced by a neutral value, to render the md template's own fixed text."""
    if isinstance(o, bool):
        return o
    if isinstance(o, (int, float)):
        return 0.5
    if isinstance(o, str):
        return "x"
    if isinstance(o, dict):
        return OrderedDict((k, _skeleton(v)) for k, v in o.items())
    if isinstance(o, list):
        return [_skeleton(v) for v in o]
    return o


def script_text_ints(audit: dict | None) -> set:
    """Integers written by the script's own fixed text: the decisions / deviations sentences and the audit md
    template (rendered on a skeleton audit whose leaves carry no integers). None of them derive from patient rows."""
    txt = " ".join(list(DECISIONS.values()) + list(DEVIATIONS) + [SUPP, "10 or more"])
    if audit is not None:
        try:
            txt += " " + audit_md(_skeleton(audit))
        except Exception:  # noqa: BLE001  (fail closed: no template exemption)
            pass
    return {int(t) for t in id_tokens(txt) if t.isdigit()}


def _json_leaves(o, path="$"):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _json_leaves(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _json_leaves(v, f"{path}[{i}]")
    else:
        yield path, o


def gate_problems(pend: Path, team_files: list, forbidden: set, ids: set) -> list:
    """Dry run of every release-gate check over every TEAM text file; returns all problems as
    (file, location, check) without ever including the offending value (v1.0.2).

    CSV: column names (forbidden names, identifiers); then, per NON-structural column (_STRUCT_COL), the size scan on
    every cell and the identifier scan: a numeric cell is an identifier candidate only if written as an integer with
    no decimal point; a text cell is token-scanned (word tokens always; numeric tokens except in script-sentence
    columns). Audit md / json: sizes and identifiers everywhere except integers that the audit json holds under its
    structural whitelist (audit_nums_ok) or that the script's own fixed text writes (script_text_ints); json
    problems are located by key path. Figures (PNG) carry no text."""
    probs = []
    aj = pend / f"{OUT_NAME}_audit_TEAM.json"
    audit = json.loads(aj.read_text(encoding="utf-8")) if aj.exists() else None
    audit_struct = _audit_ints(audit, set()) if audit is not None else set()
    allowed = audit_struct | script_text_ints(audit)
    if audit is not None:
        # clock digits of the generation timestamp
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
            if any(_FORBID_COL.search(c) for c in cols):
                probs.append((nm, "column names", "forbidden column name (count or identifier)"))
            if team_text_problems(" ".join(cols), set(), ids):
                probs.append((nm, "column names", _ID_MSG))
            for c in cols:
                if _STRUCT_COL.match(c):
                    continue
                size_hit = id_hit = False
                for cell in df[c].tolist():
                    cell = cell.strip()
                    if not cell or cell == _SUPP_CELL:
                        continue
                    if not size_hit and _size_hit(cell, forbidden):
                        size_hit = True
                    if not id_hit:
                        if _NUMERIC_CELL.match(cell):
                            id_hit = bool(re.fullmatch(r"[-+]?\d+", cell)) and cell.lstrip("+") in ids
                        else:
                            toks = id_tokens(cell)
                            if _SCRIPT_TEXT_COL.match(c):
                                toks = {t for t in toks if not t.isdigit()}
                            id_hit = bool(toks & ids)
                    if size_hit and id_hit:
                        break
                if size_hit:
                    probs.append((nm, f"column {c}", _SIZE_MSG))
                if id_hit:
                    probs.append((nm, f"column {c}", _ID_MSG))
        elif nm.endswith(".json"):
            try:
                obj = json.loads(text)
            except ValueError:
                probs.append((nm, "json", "unparseable"))
                continue
            for path, v in _json_leaves(obj):
                if isinstance(v, str):
                    for t in team_text_problems(v, forbidden - allowed, ids, allowed):
                        probs.append((nm, f"json key {path}", t))
                # numbers in the audit json are structural by construction: audit_nums_ok refuses any number whose key
                # is outside the structural whitelist before this gate runs
        else:
            for t in team_text_problems(text, forbidden - allowed, ids, allowed):
                probs.append((nm, "md text", t))
    return probs


def release_gate(pend: Path, team_files: list, forbidden: set, ids: set) -> None:
    """Collect every problem across every TEAM file (dry run), then refuse the release once, listing file, location
    and check type for each problem; values are never printed."""
    probs = gate_problems(pend, team_files, forbidden, ids)
    if probs:
        raise BuildError("release gate: " + str(len(probs)) + " problem(s): " +
                         "; ".join(f"{f} [{loc}]: {chk}" for f, loc, chk in probs))



def sensitive_numbers(D: dict, Lb: dict) -> set:
    """Sizes that must never appear in a TEAM file: cohort, subjects, every (landmark, split) eligible set and its
    subjects, the assignable Outcome 2 / ordinal8 sets, the train + validation landmark rows, and every class count
    (fail closed: no count is ever written, so any of them appearing is a leak)."""
    out = set()
    sp, subj, out_h = D["split_all"], Lb["subj"], Lb["out_h"]
    out.update({len(sp), len(np.unique(subj))})
    for L in LANDMARKS:
        el = out_h > L
        out.update({int(el.sum()), len(np.unique(subj[el])), int((el & (sp != "test")).sum())})
        for s in ("train", "validation", "test"):
            m = el & (sp == s)
            out.update({int(m.sum()), len(np.unique(subj[m])), int((m & (Lb["O2"] >= 0)).sum()),
                        int((m & (Lb["O8"] >= 0)).sum())})
            for lab, K in ((Lb["O1"][L], 9), (Lb["O2"], 5), (Lb["O8"], 8)):
                v = lab[m]
                out.update(int(c) for c in np.bincount(v[v >= 0], minlength=K))
    for s in ("train", "validation", "test"):
        m = sp == s
        out.update({int(m.sum()), len(np.unique(subj[m]))})
    return {x for x in out if x >= 100}


def figures(T: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {}
    sc = T["stage11_validation_scores_TEAM.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, kind in ((axes[0], "svd"), (axes[1], "cpqr")):
        for m in [x for x in MATRICES if "+" in x]:
            g = sc[(sc["model"] == "polr") & (sc["matrix"] == m) & sc["rep"].str.startswith(kind + "_")]
            if len(g):
                g = g.assign(kk=g["k"].astype(int)).sort_values("kk")
                ax.plot(g["kk"], g["qwk"].astype(float), "o-", lw=1.1, ms=3, label=m)
        ax.set_title(f"Stage 11: ordinal LR validation QWK vs matched width k ({kind.upper()})", fontsize=9)
        ax.set_xlabel("k")
        ax.set_ylabel("QWK")
        ax.legend(fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, metadata={"Software": None})
    plt.close(fig)
    out["fig11_1_matched_width_qwk_TEAM.png"] = buf.getvalue()
    ct = T["stage17_contrast_TEAM.csv"]
    g = ct[ct["outcome"] == "outcome1"].sort_values("landmark")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = g["landmark"].astype(int).to_numpy()
    d = g["delta_qwk"].astype(float).to_numpy()
    lo, hi = g["delta_p2_5"].astype(float).to_numpy(), g["delta_p97_5"].astype(float).to_numpy()
    ax.errorbar(x, d, yerr=[d - lo, hi - d], fmt="o-", capsize=3, lw=1.1, ms=4)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(THRESH["delta_C"], color="grey", lw=0.6, ls="--")
    ax.set_xlabel("landmark (h)")
    ax.set_ylabel("QWK(A+B+C) - QWK(A+B)")
    ax.set_title("Stage 17: validation gain of Layer C (95% subject-bootstrap interval)", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, metadata={"Software": None})
    plt.close(fig)
    out["fig17_1_delta_qwk_by_landmark_TEAM.png"] = buf.getvalue()
    return out


def fmt_df(df: pd.DataFrame) -> bytes:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(r6)
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _npz_bytes(arrs: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    return buf.getvalue()


def publish(cfg, D, Lb, pre, T, internal: dict, checks, summ) -> dict:
    out = cfg.out_dir
    pend = out.parent / f"{out.name}.pending-{os.getpid()}"
    out.parent.mkdir(parents=True, exist_ok=True)
    pend.mkdir()
    try:
        script_sha = sha256_file(Path(__file__))
        files, team = OrderedDict(), []
        for nm, df in T.items():
            files[nm] = write_bytes_atomic(pend / nm, fmt_df(df))
            team.append(nm)
        for nm, data in figures(T).items():
            files[nm] = write_bytes_atomic(pend / nm, data)
            team.append(nm)
        for nm, data in internal.items():
            if "_INTERNAL" not in nm:
                raise BuildError("an internal file name lacks the _INTERNAL marker")
            files[nm] = write_bytes_atomic(pend / nm, data)
        audit = make_audit(cfg, pre, checks, summ, script_sha, len(files))
        if not audit_nums_ok(audit):
            raise BuildError("release gate: the shareable audit contains a number outside the structural whitelist")
        ajn, amn = f"{OUT_NAME}_audit_TEAM.json", f"{OUT_NAME}_audit_TEAM.md"
        aj = json.dumps(audit, indent=1, ensure_ascii=False).encode("utf-8")
        files[ajn] = write_bytes_atomic(pend / ajn, aj)
        files[amn] = write_bytes_atomic(pend / amn, audit_md(audit).encode("utf-8"))
        team += [ajn, amn]
        back = json.loads((pend / ajn).read_text(encoding="utf-8"))
        if back != json.loads(aj.decode("utf-8")) or not audit_nums_ok(back):
            raise BuildError("release gate: disk reload of the audit differs")
        say(f"release gate: re-reading {len(team)} TEAM files from disk")
        release_gate(pend, team, sensitive_numbers(D, Lb), set(map(str, D["eid_all"])) | set(map(str, Lb["subj"])))
        say("release gate: ok")
        for nm_, h in files.items():
            if sha256_file(pend / nm_) != h:
                raise BuildError(f"release gate: {nm_} hash differs after writing")
        man = dict(script=SCRIPT_VERSION, script_sha256=script_sha, spec=SPEC_DOC, plan=PLAN, contract=CONTRACT,
                   generated_utc=utc_now(), input_hashes=pre["hashes"],
                   files={k: {"sha256": v, "bytes": (pend / k).stat().st_size} for k, v in files.items()},
                   team_files=team, internal_files=[k for k in files if k not in team],
                   acceptance_checks={k: ("pass" if v else "fail") for k, v in checks.items()}, release_gate="ok")
        write_bytes_atomic(pend / f"{OUT_NAME}_manifest.json", json.dumps(man, indent=1).encode("utf-8"))
        if out.exists():
            retry(lambda: out.rmdir(), "remove empty output folder")
        retry(lambda: os.replace(pend, out), "rename pending folder to the output folder")
    except BaseException as e:
        try:
            if pend.exists():
                retry(lambda: rmtree_robust(pend), "remove pending folder")
        except BuildError:
            e.leftover = pend.name   # type: ignore[attr-defined]
        raise
    say(f"published {len(files) + 1} files to {out.name}")
    return dict(mode="run", audit=audit, audit_md=audit_md(audit), n_files=len(files) + 1)


def _f(x):
    return r6(float(x)) if x is not None and x != "" else None


def make_audit(cfg, pre, checks, summ, script_sha, n_written) -> dict:
    w = summ["winner"]
    return OrderedDict(
        script=SCRIPT_VERSION, script_sha256=script_sha, spec=SPEC_DOC, plan=PLAN, contract=CONTRACT,
        generated_utc=utc_now(), input_hashes=OrderedDict(sorted(pre["hashes"].items())),
        split_mask_sha256=summ["split_mask_sha256"],
        decisions=DECISIONS, deviations=DEVIATIONS,
        config=OrderedDict(seed=SEED, n_boot=cfg.n_boot, gru_max_epochs=cfg.gru_max_epochs,
                           gru_hidden_grid=list(cfg.gru_hidden_grid), tree_k=list(cfg.tree_k), k_grid=list(K_GRID),
                           landmarks=list(LANDMARKS), core_landmarks=list(CORE_LANDMARKS),
                           max_seconds=cfg.max_seconds, class_weighting=CLASS_WEIGHTING,
                           thresholds=OrderedDict((f"threshold_{k}", v) for k, v in THRESH.items())),
        labels=summ["label_info"],
        stage11_winner=OrderedDict(family=w["family"], model=w["model"], matrix=w["matrix"], rep=w["rep"],
                                   qwk=_f(w["qwk"])),
        stage11_family_comparison=[OrderedDict(family=r["family"], model=r["model"], matrix=r["matrix"],
                                               rep=r["rep"], qwk=_f(r["qwk"]), best_minus_this=_f(r["best_minus_this"]),
                                               diff_ci=[_f(r["diff_p2_5"]), _f(r["diff_p97_5"])],
                                               selected=bool(r["selected"]))
                                   for r in summ["winner_rows"].to_dict("records")],
        stage11_best_per_matrix=[OrderedDict(matrix=r["matrix"], model=r["model"], rep=r["rep"], qwk=_f(r["qwk"]),
                                             qwk_ci=[_f(r["qwk_p2_5"]), _f(r["qwk_p97_5"])])
                                 for r in summ["best_per_matrix"]],
        stage11_h6_polr_k20=[OrderedDict(from_matrix=r["from_matrix"], to_matrix=r["to_matrix"], rep=r["rep"],
                                         delta_qwk=_f(r["delta_qwk"]),
                                         delta_ci=[_f(r["delta_p2_5"]), _f(r["delta_p97_5"])], verdict=r["verdict"])
                             for r in summ["h6_headline"]],
        stage11_layer_c_24h_best=summ["c24_best"],
        stage12=OrderedDict(model=summ["s12_model"], hyperparameters=summ["s12_hp"],
                            configs_in_grid=summ["s12_n_configs"], budget="validation-only tuning (D20), no nested CV",
                            qwk_before=_f(summ["s12_cal"]["qwk"][0]), qwk_after_crossfit=_f(summ["s12_cal"]["qwk"][1]),
                            ece_mean_before=_f(summ["s12_cal"]["ece_mean"][0]),
                            ece_mean_after_crossfit=_f(summ["s12_cal"]["ece_mean"][1])),
        stage17_outcome1=[OrderedDict(landmark=int(r["landmark"]), delta_qwk=_f(r["delta_qwk"]),
                                      delta_ci=[_f(r["delta_p2_5"]), _f(r["delta_p97_5"])],
                                      p_one_sided_le_0=_f(r["p_one_sided_le_0"]),
                                      gain_at_least_delta_C=bool(r["gain_at_least_delta_C"]))
                          for r in summ["s17_o1"]],
        stage17_H9_descriptive=summ["h9"],
        acceptance_checks=OrderedDict((k, "pass" if v else "fail") for k, v in checks.items()),
        release_gate="ok", n_files=n_written + 3)


def audit_md(a: dict) -> str:
    L = ["# Label build 11p and stages 11, 12 and 17 (Run 2) audit - TEAM", "",
         f"- script: {a['script']} (`{a['script_sha256']}`); spec {a['spec']}; plan {a['plan']}, contract {a['contract']}",
         f"- generated (UTC): {a['generated_utc']}; split mask `{a['split_mask_sha256']}`",
         "- validation split only (test rows dropped at load, never scored); aggregate only: no patient, row, "
         "encounter or subject counts and no identifiers; the release gate re-read every TEAM file from disk", "",
         "## Input hashes", ""]
    L += [f"- {k}: `{v}`" for k, v in a["input_hashes"].items()]
    L += ["", "## Decisions applied", ""] + [f"- **{k}**: {v}" for k, v in a["decisions"].items()]
    L += ["", "## Deviations from the plan (logged)", ""] + [f"- {d}" for d in a["deviations"]]
    c = a["config"]
    L += ["", "## Configuration", "",
          f"- seed {c['seed']}; paired subject-cluster bootstrap B = {c['n_boot']}; GRU max epochs "
          f"{c['gru_max_epochs']}, hidden grid {c['gru_hidden_grid']}; tree k {c['tree_k']}; k grid {c['k_grid']}",
          f"- landmarks {c['landmarks']} (core {c['core_landmarks']}); time limit {c['max_seconds']} s",
          "- thresholds: " + ", ".join(f"{k.replace('threshold_', '')} {v:.4g}" for k, v in c["thresholds"].items()),
          f"- class weighting: {c['class_weighting']}"]
    lb = a["labels"]
    L += ["", "## Label build 11p", "",
          "- Outcome 2 unassignable: " + ", ".join(f"{k} {v}" for k, v in lb["outcome2_unassignable"].items()),
          f"- ordinal8 unassignable: {lb['ordinal8_unassignable']}; index HADM ambiguous (tie broken by latest "
          f"discharge): {lb['index_hadm_ambiguous']}",
          f"- readmission / index overlap policy: {lb['readmit_index_overlap_policy']} (overlap present: "
          f"{lb['readmission_overlap_present']})",
          "- class percentages per (outcome, landmark, split): stage11p_label_classes_TEAM.csv (cells under 10 "
          "shown as <10 with complementary suppression)"]
    w = a["stage11_winner"]
    L += ["", "## Stage 11 winner (validation, A+B levels; simplicity rule LR < RF < GB)", "",
          f"- **{w['family']}**: {w['model']} on {w['matrix']} ({w['rep']}), QWK {w['qwk']}", "",
          "| family | model | matrix | rep | QWK | best minus this [95% CI] | selected |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['family']} | {r['model']} | {r['matrix']} | {r['rep']} | {r['qwk']} | {r['best_minus_this']} "
          f"[{r['diff_ci'][0]}, {r['diff_ci'][1]}] | {r['selected']} |" for r in a["stage11_family_comparison"]]
    L += ["", "## Stage 11 best per matrix (validation QWK [95% CI])", "", "| matrix | model | rep | QWK [CI] |",
          "|---|---|---|---|"]
    L += [f"| {r['matrix']} | {r['model']} | {r['rep']} | {r['qwk']} [{r['qwk_ci'][0]}, {r['qwk_ci'][1]}] |"
          for r in a["stage11_best_per_matrix"]]
    L += ["", "## H6 (validation, descriptive): ordinal LR at matched width (SVD k = 20)", "",
          "| from | to | rep | delta QWK [CI] | verdict |", "|---|---|---|---|---|"]
    L += [f"| {r['from_matrix']} | {r['to_matrix']} | {r['rep']} | {r['delta_qwk']} [{r['delta_ci'][0]}, "
          f"{r['delta_ci'][1]}] | {r['verdict']} |" for r in a["stage11_h6_polr_k20"]]
    cb = a["stage11_layer_c_24h_best"]
    L += ["", "## Stage 11 Layer C at 24 h (GRU on A+B; vs tuned A+B)", "",
          f"- best: {cb['c_rep']} k {cb['k']}: QWK {cb['qwk']}, delta {cb['delta_qwk']} [{cb['delta_ci'][0]}, "
          f"{cb['delta_ci'][1]}] ({cb['verdict']})"]
    s = a["stage12"]
    L += ["", "## Stage 12 tuning and calibration (validation)", "",
          f"- {s['model']}: " + ", ".join(f"{k} {v}" for k, v in s["hyperparameters"].items()) +
          f" (selected from {s['configs_in_grid']} configurations; {s['budget']})",
          f"- QWK {s['qwk_before']} -> {s['qwk_after_crossfit']} after isotonic calibration (5-fold cross-fit); "
          f"mean ECE {s['ece_mean_before']} -> {s['ece_mean_after_crossfit']}"]
    L += ["", "## Stage 17 Outcome 1: QWK(A+B+C) - QWK(A+B) by landmark (validation)", "",
          "| landmark | delta QWK [95% CI] | p (delta <= 0) | gain >= delta_C |", "|---|---|---|---|"]
    L += [f"| {r['landmark']} | {r['delta_qwk']} [{r['delta_ci'][0]}, {r['delta_ci'][1]}] | {r['p_one_sided_le_0']} | "
          f"{r['gain_at_least_delta_C']} |" for r in a["stage17_outcome1"]]
    h = a["stage17_H9_descriptive"]
    L += ["", f"- H9 (descriptive, validation; inference is stage 14 on test): {h['verdict']}"]
    L += ["", "## Acceptance checks", ""] + [f"- {k}: {v}" for k, v in a["acceptance_checks"].items()]
    L += ["", f"Release gate: {a['release_gate']}. Detail tables: stage11*_TEAM.csv, stage12_*_TEAM.csv, "
          "stage17_*_TEAM.csv; *_INTERNAL files are team-internal (never opened, pasted or shared).", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------- build
def attach_labels(D: dict, Lb: dict) -> None:
    """Row-aligned labels for the train + validation 24 h rows (D['rows'])."""
    pos = np.flatnonzero(D["elig24_all"] & np.isin(D["split_all"], ("train", "validation")))
    R = D["rows"]
    if not np.array_equal(D["eid_all"][pos], R["eid"]):
        raise BuildError("row alignment of labels failed")
    R["subj"], R["died"] = Lb["subj"][pos], Lb["died"][pos]
    R["O1"] = OrderedDict((L, Lb["O1"][L][pos]) for L in LANDMARKS)
    R["O2"], R["O8"] = Lb["O2"][pos], Lb["O8"][pos]


def build(cfg: Config) -> dict:
    if cfg.max_seconds:
        _STATE["deadline"] = time.time() + cfg.max_seconds
    say("precheck: hashes")
    pre = precheck(cfg)
    if cfg.precheck_only:
        return {"mode": "precheck-only", "hashes": pre["hashes"]}
    checks = OrderedDict()
    say("loading Stage 08 levels (24 h train + validation rows), Layer C index, Run 1 bases")
    D = load_inputs(cfg)
    say("label build 11p")
    Lb = build_labels(cfg, D, checks)
    classes = label_class_table(Lb, D["split_all"], checks)
    attach_labels(D, Lb)
    checks["11p_layer_c_rows_have_labels"] = all(bool((D["rows"]["O1"][L][D["lc"][L]["idx"]] >= 0).all())
                                                for L in LANDMARKS)
    mats = build_matrices(D, checks)
    say("stage 11: bake-off")
    S11 = stage11(D, Lb, mats, cfg, checks)
    say("stage 12: tuning and calibration")
    S12 = stage12(D, mats, S11, cfg, checks)
    sec = stage11_secondary(D, mats, S11, S12, cfg)
    say("stage 11: Layer C at 24 h")
    C24 = stage11_c24(D, mats, S11, S12, cfg, checks)
    say("stage 17: landmark models")
    S17 = stage17(D, mats, S11, S12, cfg, checks)
    sc = S11["scores"]
    checks["11_multinom_never_wins"] = S11["winner"]["model"] != "multinom"
    checks["11_all_metrics_finite"] = bool(np.isfinite(sc[["qwk", "emd", "rps", "mae"]].astype(float).to_numpy()).all())
    checks["17_all_landmarks_scored"] = set(S17["contrast"].query("outcome == 'outcome1'")["landmark"]) == set(LANDMARKS)
    checks["17_probabilities_valid"] = all(bool(np.allclose(P.sum(1), 1, atol=1e-5) and (P >= -1e-7).all())
                                           for k, P in S17["preds"].items() if not k.endswith("__eid"))
    T = OrderedDict()
    T["stage11p_label_classes_TEAM.csv"] = classes
    T["stage11_validation_scores_TEAM.csv"] = sc
    T["stage11_h6_adjacent_TEAM.csv"] = S11["h6"]
    T["stage11_winner_TEAM.csv"] = S11["winner_rows"]
    T["stage11_layer_c_24h_TEAM.csv"] = C24["table"]
    T["stage11_secondary_TEAM.csv"] = sec
    T["stage12_tuning_TEAM.csv"] = S12["tuning"]
    T["stage12_calibration_TEAM.csv"] = S12["calibration"]
    T["stage17_validation_TEAM.csv"] = S17["validation"]
    T["stage17_contrast_TEAM.csv"] = S17["contrast"]
    T["stage17_calibration_TEAM.csv"] = S17["calibration"]
    # internal artefacts (never opened, pasted or shared)
    lab = dict(encounter_id=D["eid_all"], split=D["split_all"], died=Lb["died"], outcome2=Lb["O2"], ordinal8=Lb["O8"])
    lab.update({f"outcome1_L{L}": Lb["O1"][L] for L in LANDMARKS})
    va = D["rows"]["split"] == "validation"
    pr = OrderedDict(s12_tuned_P=S12["P"], s12_eid=D["rows"]["eid"][va])
    pr.update(S17["preds"])
    for (mod, m, rp), P in S11["PRED"].items():
        if (mod, m, rp) == (S11["winner"]["model"], S11["winner"]["matrix"], S11["winner"]["rep"]):
            pr["s11_winner_P"] = P

    import torch
    tb = io.BytesIO()
    torch.save(S17["states"], tb)
    internal = OrderedDict()
    internal["stage11p_labels_landmark_INTERNAL.npz"] = _npz_bytes(lab)
    internal["stage11_12_17_validation_predictions_INTERNAL.npz"] = _npz_bytes(dict(pr))
    internal["stage12_tuned_model_calibrators_INTERNAL.pkl"] = pickle.dumps(
        dict(winner=S11["winner"], hp=S12["hp"], model=S12["model"], calibrators=S12["cals"]))
    internal["stage17_models_INTERNAL.pt"] = tb.getvalue()
    internal["stage17_tabular_models_INTERNAL.pkl"] = pickle.dumps(S17["tab_models"])
    best = []
    for m in MATRICES:
        g = sc[(sc["matrix"] == m) & (sc["model"] != "multinom")]
        best.append(g.loc[g["qwk"].astype(float).idxmax()].to_dict())
    h6 = S11["h6"]
    h6h = h6[(h6["model"] == "polr") & (h6["rep"] == "svd_20")].to_dict("records")
    c24 = C24["table"]
    cb = c24.loc[c24["qwk"].astype(float).idxmax()]
    cal = S12["calibration"].set_index("item")
    o1 = S17["contrast"][S17["contrast"]["outcome"] == "outcome1"].to_dict("records")
    core = [r for r in o1 if r["landmark"] in CORE_LANDMARKS]
    n_gain = sum(bool(r["gain_at_least_delta_C"]) for r in core)
    h9 = OrderedDict(delta_C=THRESH["delta_C"],
                     verdict=("Layer C gain of at least delta_C with interval above 0 at every core landmark"
                              if n_gain == len(core) else
                              ("at some core landmarks only" if n_gain else "at no core landmark")))
    summ = dict(winner=S11["winner"], winner_rows=S11["winner_rows"], best_per_matrix=best, h6_headline=h6h,
                c24_best=OrderedDict(c_rep=cb["c_rep"], k=(int(cb["k"]) if cb["k"] != "" else None),
                                     qwk=_f(cb["qwk"]), delta_qwk=_f(cb["delta_qwk"]),
                                     delta_ci=[_f(cb["delta_p2_5"]), _f(cb["delta_p97_5"])], verdict=cb["verdict"]),
                s12_model=S12["model_name"], s12_n_configs=len(S12["tuning"]), s12_hp=OrderedDict((k, v) for k, v in S12["hp"].items()),
                s12_cal={k: (cal.loc[k, "before"], cal.loc[k, "after_crossfit"]) for k in ("qwk", "ece_mean")},
                s17_o1=o1, h9=h9, label_info=Lb["info"],
                split_mask_sha256=hashlib.sha256("\n".join(D["split_all"].tolist()).encode()).hexdigest())
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise BuildError("acceptance checks failed: " + "; ".join(failed))
    return publish(cfg, D, Lb, pre, T, internal, checks, summ)


# ------------------------------------------------------------------------------ runtime environment (v1.0.1)
CACHE_NAME = "_cache_stage11_12_17_r2"     # next to the output folder; never a leftover, never scanned by the gate
_ENV_DIRS = OrderedDict([
    ("MPLCONFIGDIR", "matplotlib"),         # matplotlib config + font cache (default ~/.matplotlib)
    ("JOBLIB_TEMP_FOLDER", "joblib"),       # joblib / loky memmapping folder
    ("SCIKIT_LEARN_DATA", "sklearn_data"),  # sklearn data home (default ~/scikit_learn_data)
    ("XDG_CACHE_HOME", "xdg_cache"),        # generic caches (fontconfig etc.)
    ("XDG_CONFIG_HOME", "xdg_config"),
    ("TMP", "tmp"), ("TEMP", "tmp"), ("TMPDIR", "tmp"),   # temp files of this process and any subprocess (loky)
])
HEAVY_IMPORTS = ("sklearn", "sklearn.ensemble", "sklearn.linear_model", "sklearn.isotonic", "matplotlib",
                 "matplotlib.pyplot", "torch")
_PIP = {"sklearn": "scikit-learn", "matplotlib": "matplotlib", "torch": "torch"}


def cache_dir_for(cfg: Config) -> Path:
    return cfg.out_dir.parent / CACHE_NAME


def setup_runtime_env(cache: Path) -> dict:
    """Point every library cache / temp location at folders inside `cache` (inside the project for the real run,
    inside pytest's tmp for the tests) before scikit-learn or matplotlib is imported. Returns what it replaced so a
    caller (tests, in-process runs) can restore it with restore_runtime_env."""
    prev = {"env": {k: os.environ.get(k) for k in _ENV_DIRS}, "tempdir": tempfile.tempdir,
            "dont_write_bytecode": sys.dont_write_bytecode}
    for k, sub in _ENV_DIRS.items():
        d = cache / sub
        d.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(d.resolve())
    tempfile.tempdir = str((cache / "tmp").resolve())
    sys.dont_write_bytecode = True               # no __pycache__ next to the script (synced folder)
    return prev


def restore_runtime_env(prev: dict) -> None:
    for k, v in prev["env"].items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    tempfile.tempdir = prev["tempdir"]
    sys.dont_write_bytecode = prev["dont_write_bytecode"]


def clear_cache_temp(cache: Path) -> None:
    """Best effort: empty the temp and joblib subfolders (they could hold memmapped arrays); keep the font cache."""
    for sub in ("tmp", "joblib"):
        d = cache / sub
        if d.is_dir():
            for p in d.iterdir():
                try:
                    rmtree_robust(p) if p.is_dir() else p.unlink()
                except OSError:
                    pass


def import_heavy() -> None:
    """Guarded imports (after setup_runtime_env). ImportError -> input stop naming the package; any other error ->
    re-raised as ImportStageError carrying the module name, reported with every frame by main."""
    for mod in HEAVY_IMPORTS:
        try:
            m = __import__(mod)
            if mod == "matplotlib":
                m.use("Agg")
        except ImportError as e:
            pip = _PIP[mod.split(".")[0]]
            raise InputError(f"import of {mod} failed ({type(e).__name__}); python package {pip} is required "
                             f"(pip install {pip})") from None
        except Exception as e:  # noqa: BLE001
            raise ImportStageError(mod) from e


class ImportStageError(Exception):
    def __init__(self, mod):
        super().__init__(mod)
        self.module = mod


def redact_path(p, root: Path | None) -> str:
    """A path reduced for the console: relative to the project root when inside it; otherwise '~\\<first folder>
    \\...\\<name>' under the user profile, or '<drive>\\<first folder>\\...\\<name>' elsewhere."""
    try:
        pp = Path(os.fsdecode(p))
    except Exception:  # noqa: BLE001
        return "<unprintable>"
    sep = "\\" if os.name == "nt" else "/"
    try:
        if root is not None:
            return sep.join(pp.resolve().relative_to(Path(root).resolve()).parts) or "."
    except (ValueError, OSError):
        pass
    try:
        rel = pp.resolve().relative_to(Path.home().resolve()).parts
        if rel:
            return sep.join(["~", rel[0]] + (["...", rel[-1]] if len(rel) > 1 else []))
        return "~"
    except (ValueError, OSError, RuntimeError):
        pass
    parts = pp.parts
    if len(parts) <= 2:
        return sep.join(parts)
    return sep.join([parts[0].rstrip("\\/"), parts[1], "...", parts[-1]]) if len(parts) > 3 else \
        sep.join([parts[0].rstrip("\\/"), parts[1], parts[-1]])


def describe_exception(e: BaseException, root: Path | None, max_chain: int = 4) -> list:
    """Console lines for an unexpected error: exception type, then every traceback frame as basename:line:function
    (library frames also give their module name), then the same for the chained cause / context. No messages and
    no locals are printed; an OSError with a filename adds errno / winerror and the redacted filename(s)."""
    lines, seen, cur, depth = [], set(), e, 0
    while cur is not None and id(cur) not in seen and depth < max_chain:
        seen.add(id(cur))
        lines.append(("unexpected " if depth == 0 else "  caused by ") + type(cur).__name__)
        for frame, lineno in traceback.walk_tb(cur.__traceback__):
            co = frame.f_code
            base = os.path.basename(co.co_filename) or co.co_filename
            func = getattr(co, "co_qualname", co.co_name)
            mod = frame.f_globals.get("__name__", "?")
            ours = os.path.abspath(co.co_filename) == os.path.abspath(__file__)
            lines.append(f"  frame {base}:{lineno}:{func}" + ("" if ours else f" [{mod}]"))
        if isinstance(cur, OSError) and (cur.filename is not None or cur.filename2 is not None):
            fn = redact_path(cur.filename, root) if cur.filename is not None else "-"
            extra = f"; file2 {redact_path(cur.filename2, root)}" if cur.filename2 is not None else ""
            lines.append(f"  OSError errno {cur.errno} winerror {getattr(cur, 'winerror', None)}; file {fn}{extra}")
        cur = cur.__cause__ or cur.__context__
        depth += 1
    return lines


# ---------------------------------------------------------------------------------------------------------- main
def build_config(argv=None) -> Config:
    ap = argparse.ArgumentParser(description=SCRIPT_VERSION)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--run1-dir", default=None)
    ap.add_argument("--canonical", default=None)
    ap.add_argument("--harmonization", default=None)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--expect-stage06-08-manifest-sha256", default=None)
    ap.add_argument("--expect-run1-manifest-sha256", default=None)
    ap.add_argument("--precheck-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--gru-max-epochs", type=int, default=40)
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=30.0)
    ap.add_argument("--harm-t0-col", default=None)
    ap.add_argument("--harm-out-col", default=None)
    a = ap.parse_args(argv)
    root = Path(a.project_root)
    lad = root / "data" / "processed" / "ladder"
    return Config(project_root=root, in_dir=Path(a.in_dir) if a.in_dir else lad / "stage06_08",
                  run1_dir=Path(a.run1_dir) if a.run1_dir else lad / RUN1_NAME,
                  canonical=Path(a.canonical) if a.canonical else root / "data" / "interim" /
                  "canonical_icu_encounters.csv",
                  harmonization=Path(a.harmonization) if a.harmonization else root / "data" / "interim" /
                  "encounter_harmonization_fields.csv",
                  split_file=Path(a.split_file) if a.split_file else root.parent / "PIC_Analysis_3_0" / "data" /
                  "processed" / "subject_splits.csv",
                  raw_dir=Path(a.raw_dir) if a.raw_dir else root.parent / "PICDataset",
                  out_dir=Path(a.out_dir) if a.out_dir else lad / OUT_NAME,
                  expect_manifest_sha256=(a.expect_stage06_08_manifest_sha256 or "").strip().lower() or None,
                  expect_run1_manifest_sha256=(a.expect_run1_manifest_sha256 or "").strip().lower() or None,
                  precheck_only=a.precheck_only, n_boot=a.n_boot, gru_max_epochs=a.gru_max_epochs,
                  max_seconds=a.max_seconds, heartbeat_seconds=a.heartbeat_seconds, harm_t0_col=a.harm_t0_col,
                  harm_out_col=a.harm_out_col)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    cfg = prev = cache = None
    try:
        cfg = build_config(argv)
        cache = cache_dir_for(cfg)
        prev = setup_runtime_env(cache)
        say(f"library caches and temp files -> {redact_path(cache, cfg.project_root)}")
        import_heavy()
        with Heartbeat(cfg.heartbeat_seconds):
            res = build(cfg)
    except InputError as e:
        print(f"STOPPED (input): {e}", file=sys.stderr)
        print("result: STOPPED (input); nothing written")
        return 2
    except (BuildError, TimeLimit) as e:
        lo = getattr(e, "leftover", None)
        tail = (f"pending folder left behind: {lo} (delete it unopened after review; never share or sync it)"
                if lo else "nothing left behind")
        print(f"BUILD FAILED: {e}", file=sys.stderr)
        print(f"result: FAILED; {tail}")
        return 3
    except ImportStageError as e:
        print(f"BUILD FAILED (import of {e.module} raised an unexpected error)", file=sys.stderr)
        for ln in describe_exception(e.__cause__ or e, cfg.project_root if cfg else None):
            print(ln, file=sys.stderr)
        print("result: FAILED; nothing left behind")
        return 3
    except Exception as e:  # noqa: BLE001  (fail closed; types, frames and redacted file names only)
        lo = getattr(e, "leftover", None)
        print("BUILD FAILED (unexpected error; frames follow)", file=sys.stderr)
        for ln in describe_exception(e, cfg.project_root if cfg else None):
            print(ln, file=sys.stderr)
        print(f"result: FAILED; {'pending folder left behind: ' + lo if lo else 'nothing left behind'}")
        return 3
    finally:
        if cache is not None:
            clear_cache_temp(cache)
        if prev is not None:
            restore_runtime_env(prev)
    if res["mode"] == "precheck-only":
        h = res["hashes"]
        print("result: PRECHECK OK (Stage 06-08 manifest and files, Run 1 manifest and files, canonical, "
              "harmonization, split, raw ICUSTAYS / ADMISSIONS, packages); nothing written (library cache folder only)")
        for k, v in sorted(h.items()):
            print(f"{k} sha256 {v}")
        return 0
    print(res["audit_md"])
    print(f"elapsed {time.time() - t0:.0f} s")
    print("result: RUN 2 (11p, 11, 12, 17) OK; RELEASE GATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
