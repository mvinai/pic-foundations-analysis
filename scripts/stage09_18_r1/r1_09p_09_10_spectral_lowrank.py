"""Run 1 of stages 09-18 (PIC Foundations, PIC; plan v13, contract v13): stage 09p (D15/D16 exact-duplicate
collapse), stage 09 (spectral diagnostics and uncertainty, plan section 8) and stage 10 (low-rank approximation,
CUR, principal angles, linear vs nonlinear recovery, plan section 9), in one pass on the Stage 06-08 v1.2.1 release.

r1_09p_09_10_spectral_lowrank v1.0.1 (Nov 18, 2026). v1.0.1: library caches and temp files are redirected into
data/processed/ladder/_cache_stage09_10_r1/ before scikit-learn and matplotlib are imported (guarded imports at the
start of main); unexpected errors print every traceback frame as basename:line:function (no messages, no paths);
every .npz handle is closed right after reading; read-only files no longer block removing a pending folder.
v1.0.2: release gate only (numerical work unchanged): structural integer columns are exempt from the identifier
scan as well as the size scan; numeric cells count as identifier candidates only when written as integers; all
problems are collected (dry run) and reported as file / column (or md text / json key) / check, never the value.

Inputs (hash-gated before anything is parsed): data/processed/ladder/stage06_08/stage06_08_manifest.json pinned by
its full SHA-256 on the command line, every file that manifest lists, and the canonical encounter file
(data/interim/canonical_icu_encounters.csv; only encounter_id and SUBJECT_ID are read, for subject-level resampling
and grouped folds) checked against the hash recorded in the Stage 06-08 manifest and the plan pin.

Outputs (data/processed/ladder/stage09_10_r1/, published atomically from a stage09_10_r1.pending-<pid> folder):
TEAM files (aggregate only; no patient, row, encounter or subject counts; no identifiers; numbers are relative
singular values, ranks, ratios, angles, R^2 and percentile intervals of those) and one INTERNAL file (matched-width
bases for stages 11 and 17). A release gate re-reads every TEAM file from disk and refuses any cohort / split /
landmark size, any identifier, or any audit number outside a structural whitelist.

Exit codes: 0 ok; 2 stopped on an input check (nothing written); 3 build or release-gate failure or a Windows file
lock (nothing published; a pending folder is removed, or named if Windows refuses to remove it).
Assistants never run this script on patient data; it runs on the real data only through Copilot, unedited.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
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

# BLAS / OpenMP thread cap, set before numpy loads (an existing environment setting wins). The work is many
# tall-skinny factorizations of a few hundred columns, where more than ~4 threads only spin and, on a busy machine,
# made the run several times slower in testing.
_THREADS = str(min(4, max(1, (os.cpu_count() or 2) // 2)))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as sla  # noqa: E402

SCRIPT_VERSION = "r1_09p_09_10_spectral_lowrank v1.0.2"
SPEC_DOC = "stage09_10_r1_spec v1.0.2"
PLAN, CONTRACT = "v13", "v13"
OUT_NAME = "stage09_10_r1"

# ----------------------------------------------------------------------------------------------- pins and settings
PINS = {
    "stage06_08_script": "06_08_build_levels_preprocess v1.2.1",
    "stage06_08_script_sha256": "f55cdaf5d7dd51470a9bd9494b1ed37730fcc29c7948de29eb8461d33daec8e7",
    "stage06_08_files": 42,          # files listed in stage06_08_manifest.json (43 in the folder with the manifest)
    "canonical": "e5feeb22b5457ab89c4c73235e6a4722d538aec5f88595b6f049004171286418",   # plan pin; None in tests
}
LEVELS_A = ("a0", "a1", "a2")
LEVELS_B = ("b0", "b1", "b2s", "b2d", "b2p")
LEVELS = LEVELS_A + LEVELS_B
AB_PAIRS = (("a0", "b0"), ("a1", "b1"), ("a2", "b2s"), ("a2", "b2d"), ("a2", "b2p"))   # contract: AB_k = A_k + B_k
LANDMARKS = (24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336)
CORE_LANDMARKS = (24, 48, 72, 120, 168, 240, 336)
K_GRID = (5, 10, 20, 30, 40, 59)
ANGLE_PAIRS = (("a0", "a1"), ("a1", "a2"), ("a0", "a2"),
               ("b0", "b1"), ("b1", "b2s"), ("b1", "b2d"), ("b1", "b2p"), ("b2s", "b2d"), ("b2s", "b2p"),
               ("b0", "b2s"), ("a0", "b0"), ("a1", "b1"), ("a2", "b2s"))
RECOVERY_PREV = {"a1": "a0", "a2": "a1", "b1": "b0", "b2s": "b1", "b2d": "b1", "b2p": "b1"}
SEED = 6310
DUP_TOKEN = "__dup__"

# D18 thresholds (frozen Nov 18, 2026, before any real stage 09-18 run; departure from plan 13.3 logged)
THRESH = dict(H1_kappa=100.0, H2_frac_p=1.0 / 3.0, H2_rise=2, H3_freq=0.80, belsley_eta=30.0, belsley_prop=0.5,
              H4_gap_z=0.05, H4_gap_flag=0.30)
DECISIONS = OrderedDict([
    ("D15", "collapse exactly identical columns before modeling; record dropped copies; keep near-constant columns"),
    ("D16", "D15 everywhere (levels, AB concatenations, Layer C channels) except the by-design __dup__ copies of b2d; "
            "stage 09 reported before and after dedup"),
    ("D17", "stage 15 skipped; the Stage 08 base Layer C tensor is the single Layer C representation"),
    ("D18", "thresholds frozen before real runs (bracketed plan values; delta 0.02, delta_C 0.01, epsilon 0.01); "
            "departure from plan 13.3 logged"),
    ("D20", "spectral subject bootstrap B = 200 (other budgets apply to later runs)"),
])
DEVIATIONS = [
    "permutation nulls use 100 draws per matrix (plan 8: 200) to keep the laptop run under about 25 min",
    "random CUR rules: 50 repeats in Frobenius norm, spectral norm on the first 10 repeats",
    "CUR rows: r = c = k; row rules mirror the column rules (uniform, leverage from U_k, top leverage, CPQR on U_k^T "
    "(Q-DEIM), DEIM on U_k); columns by CPQR on A itself",
    "DMM 2008 bound reported in order form c = k ln k / eps^2 (explicit constant not applied)",
    "recovery: targets are the columns added at a level relative to the previous level (not identical on train to a "
    "column of it), capped per (level, family); fitted on a subject-level train subsample; 5-fold CV grouped by "
    "subject; kNN mutual information deferred",
    "Layer C: the per-bin patient-shuffle null leaves the stacked channel spectrum unchanged (a row permutation), so "
    "it moves to stage 16; stage 09 uses the within-bin column permutation null",
    "Layer C bootstrap and permutation null at the seven core landmarks; the other landmarks get point estimates",
    "spectral and angle bootstraps share one subject draw per replicate across the eight levels; spectra and "
    "Davis-Kahan use all B replicates (QR then SVD of R), the between-level angle intervals use the first "
    "n_boot_angles replicates (default 50), which need the left singular vectors",
    "largest singular values used only as norms (Davis-Kahan perturbation norm, CUR spectral errors) are read from "
    "the top eigenvalue of the small Gram matrix (well conditioned); every reported spectrum comes from an SVD",
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


# --------------------------------------------------------------------------------------------- spectral helpers
def svdvals(A: np.ndarray) -> np.ndarray:
    try:
        return sla.svdvals(A, check_finite=False)
    except np.linalg.LinAlgError:
        return sla.svd(A, compute_uv=False, lapack_driver="gesvd", check_finite=False)


def r_svd(A: np.ndarray):
    """Singular values and right vectors of a tall matrix via Householder QR then the SVD of the small R factor
    (backward stable, same spectrum as the SVD of A itself; never A^T A). About half the cost of a thin SVD."""
    if A.shape[0] <= A.shape[1]:
        _, s, Vt = thin_svd(A)
        return s, Vt
    R = sla.qr(A, mode="r", check_finite=False)[0][: A.shape[1]]
    _, s, Vt = thin_svd(R)
    return s, Vt


def thin_svd(A: np.ndarray):
    try:
        return sla.svd(A, full_matrices=False, check_finite=False)
    except np.linalg.LinAlgError:
        return sla.svd(A, full_matrices=False, lapack_driver="gesvd", check_finite=False)


def spec_norm(M: np.ndarray) -> float:
    """Largest singular value (exact SVD for small matrices; top eigenvalue of the small Gram matrix otherwise)."""
    if M.size == 0:
        return 0.0
    if min(M.shape) <= 60:
        return float(svdvals(M)[0])
    # Only the LARGEST singular value is needed here (perturbation and CUR error norms); it is well conditioned
    # in the small Gram matrix (relative error ~ machine eps), unlike the trailing spectrum, which is always
    # taken from an SVD of the matrix itself (plan 8). Much faster and deterministic compared with Lanczos.
    M64 = M.astype(np.float64, copy=False)
    G = M64.T @ M64 if M.shape[0] >= M.shape[1] else M64 @ M64.T
    try:
        ev = np.linalg.eigvalsh(G)
        return float(math.sqrt(max(float(ev[-1]), 0.0)))
    except Exception:  # noqa: BLE001
        return float(svdvals(M)[0])


def rank_tol(s: np.ndarray, shape) -> float:
    return max(shape) * np.finfo(np.float64).eps * (s[0] if s.size else 0.0)


def numerical_rank(s: np.ndarray, shape) -> int:
    return int((s > rank_tol(s, shape)).sum()) if s.size else 0


def erank_sigma2(s):
    w = s.astype(float) ** 2
    if w.sum() <= 0:
        return 0.0
    w = w[w > 0] / w.sum()
    return float(np.exp(-(w * np.log(w)).sum()))


def erank_sigma(s):
    w = s.astype(float)
    if w.sum() <= 0:
        return 0.0
    w = w[w > 0] / w.sum()
    return float(np.exp(-(w * np.log(w)).sum()))


def k_energy(s, power: int, frac: float = 0.95) -> int:
    e = s.astype(float) ** power
    if e.sum() <= 0:
        return 0
    c = np.cumsum(e) / e.sum()
    return int(np.searchsorted(c, frac - 1e-12) + 1)


def stable_rank(s) -> float:
    return float((s.astype(float) ** 2).sum() / float(s[0]) ** 2) if s.size and s[0] > 0 else 0.0


def spectral_summary(s: np.ndarray, shape) -> dict:
    p = shape[1]
    r = numerical_rank(s, shape)
    tol3 = int((s > 1e-3 * s[0]).sum()) if s.size else 0
    tol2 = int((s > 1e-2 * s[0]).sum()) if s.size else 0
    kap = float(s[0] / s[-1]) if s.size and s[-1] > rank_tol(s, shape) and len(s) == p else float("inf")
    kap_r = float(s[0] / s[r - 1]) if r > 0 else float("inf")
    return dict(p=p, rank_numerical=r, rank_1e3=tol3, rank_1e2=tol2, rank_deficiency=p - r, kappa2=kap,
                kappa2_over_rank=kap_r, stable_rank=stable_rank(s), erank_sigma2=erank_sigma2(s),
                erank_sigma=erank_sigma(s), k95_sigma2=k_energy(s, 2), k95_sigma=k_energy(s, 1))


def decay_fit(s: np.ndarray, shape, max_index: int = 50) -> dict:
    """Power law, exponential and rank-plus-noise fits of log sigma_i over i = 1..m (m = min(50, numerical rank)),
    compared by BIC (plan 8)."""
    r = numerical_rank(s, shape)
    m = int(min(max_index, r))
    out = dict(decay_m=m, decay_family="na", bic_power=float("nan"), bic_exponential=float("nan"),
               bic_rank_noise=float("nan"), alpha=float("nan"), beta=float("nan"), rank_noise_r=0)
    if m < 6:
        return out
    y = np.log(s[:m].astype(float))
    i = np.arange(1, m + 1, dtype=float)

    def bic(rss, k):
        return m * math.log(max(rss, 1e-300) / m) + k * math.log(m)

    Xp = np.c_[np.ones(m), np.log(i)]
    bp, *_ = np.linalg.lstsq(Xp, y, rcond=None)
    rss_p = float(((y - Xp @ bp) ** 2).sum())
    Xe = np.c_[np.ones(m), i]
    be, *_ = np.linalg.lstsq(Xe, y, rcond=None)
    rss_e = float(((y - Xe @ be) ** 2).sum())
    best_rn = (float("inf"), 0)
    for rr in range(1, m - 2):
        tail = y[rr:]
        rss = float(((tail - tail.mean()) ** 2).sum())
        b = bic(rss, rr + 2)        # r free leading values, the floor, and the choice of r
        if b < best_rn[0]:
            best_rn = (b, rr)
    out.update(bic_power=bic(rss_p, 2), bic_exponential=bic(rss_e, 2), bic_rank_noise=best_rn[0],
               alpha=float(-bp[1]), beta=float(-be[1]), rank_noise_r=int(best_rn[1]))
    fams = {"power": out["bic_power"], "exponential": out["bic_exponential"], "rank_plus_noise": out["bic_rank_noise"]}
    out["decay_family"] = min(fams, key=fams.get)
    return out


def perm_null(A: np.ndarray, n_perm: int, rng: np.random.Generator, blocks: list | None = None) -> np.ndarray:
    """Each column permuted independently (within each row block if blocks are given); returns n_perm x min(n,p)."""
    n, p = A.shape
    out = np.zeros((n_perm, min(n, p)))
    A32 = A.astype(np.float32)
    for b in range(n_perm):
        if blocks is None:
            idx = np.argsort(rng.random((n, p), dtype=np.float32), axis=0)
            P = np.take_along_axis(A32, idx, axis=0)
        else:
            P = np.empty_like(A32)
            for lo, hi in blocks:
                idx = np.argsort(rng.random((hi - lo, p), dtype=np.float32), axis=0)
                P[lo:hi] = np.take_along_axis(A32[lo:hi], idx, axis=0)
        out[b] = svdvals(P)
        check_time()
    return out


def mp_edge(n: int, p: int) -> float:
    return math.sqrt(n) * (1.0 + math.sqrt(p / n))


def belsley(s: np.ndarray, Vt: np.ndarray, cols: list, shape, eta_cut: float, prop_cut: float, cap: int = 25) -> list:
    """Belsley condition indices with variance-decomposition proportions (standardized train matrix). Null
    directions (sigma below tolerance) are listed with the columns loading on them (|v| > 0.1); near-dependencies:
    eta >= eta_cut with at least two proportions > prop_cut."""
    tol = rank_tol(s, shape)
    rows = []
    for j in np.where(s <= tol)[0]:
        v = Vt[j]
        idx = np.argsort(-np.abs(v))
        named = [(cols[t], float(abs(v[t]))) for t in idx if abs(v[t]) > 0.1][:cap]
        rows.append(dict(direction=int(j + 1), eta="inf", kind="exact_null", columns=";".join(c for c, _ in named),
                         weights=";".join(f"{w:.3g}" for _, w in named)))
    keep = np.where(s > tol)[0]
    if keep.size:
        phi = (Vt[keep].T ** 2) / (s[keep] ** 2)
        tot = phi.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        prop = phi / tot
        eta = s[0] / s[keep]
        for jj, j in enumerate(keep):
            if eta[jj] < eta_cut:
                continue
            big = np.where(prop[:, jj] > prop_cut)[0]
            if big.size >= 2:
                order = big[np.argsort(-prop[big, jj])][:cap]
                rows.append(dict(direction=int(j + 1), eta=r6(float(eta[jj])), kind="near_dependency",
                                 columns=";".join(cols[t] for t in order),
                                 weights=";".join(f"{prop[t, jj]:.3g}" for t in order)))
    return rows


def principal_angles(QU: np.ndarray, QV: np.ndarray) -> np.ndarray:
    """Angles (radians, increasing) between the spans of two orthonormal bases (plan 9: SVD of QU^T QV)."""
    if QU.shape[1] == 0 or QV.shape[1] == 0:
        return np.zeros(0)
    c = svdvals(QU.T @ QV)
    return np.arccos(np.clip(c, -1.0, 1.0))


def orth_basis(A: np.ndarray) -> np.ndarray:
    U, s, _ = thin_svd(A)
    return U[:, :numerical_rank(s, A.shape)]


# --------------------------------------------------------------------------------------------------- dedup (09p)
def dedup_columns(X_train: np.ndarray, cols: list, exempt=lambda c: False):
    """Exact-duplicate collapse on train rows (D15/D16): columns bit-identical (float32) on train are grouped; the
    first in canonical order is kept. Columns with exempt(name) true are never dropped and never keepers.
    Returns (keep_indices, groups) with groups = [(kept_name, [dropped names])]."""
    p = X_train.shape[1]
    Xc = np.ascontiguousarray(X_train.astype(np.float32).T)
    buckets = OrderedDict()
    for j in range(p):
        if exempt(cols[j]):
            continue
        h = hashlib.sha256(Xc[j].tobytes()).hexdigest()
        buckets.setdefault(h, []).append(j)
    drop, groups = set(), []
    for js in buckets.values():
        rem = list(js)
        while len(rem) > 1:
            k0 = rem[0]
            same = [j for j in rem[1:] if np.array_equal(Xc[k0], Xc[j])]
            if same:
                groups.append((cols[k0], [cols[j] for j in same]))
                drop.update(same)
            rem = [j for j in rem[1:] if j not in same]
    keep = [j for j in range(p) if j not in drop]
    return keep, groups


# ------------------------------------------------------------------------------------------------------ CUR tools
def leverage(Vk: np.ndarray) -> np.ndarray:
    lv = (Vk ** 2).sum(axis=1)
    return lv / lv.sum()


def deim(Vk: np.ndarray) -> list:
    """DEIM index selection on the rows of the basis Vk (m x k)."""
    k = Vk.shape[1]
    idx = [int(np.argmax(np.abs(Vk[:, 0])))]
    for j in range(1, k):
        P = np.array(idx)
        c = np.linalg.lstsq(Vk[P, :j], Vk[P, j], rcond=None)[0]
        r = Vk[:, j] - Vk[:, :j] @ c
        r[P] = 0.0
        idx.append(int(np.argmax(np.abs(r))))
    return idx


def cpqr_order(M: np.ndarray) -> np.ndarray:
    out = sla.qr(M, mode="r", pivoting=True, check_finite=False)
    return np.asarray(out[-1])


def _top_eig(G: np.ndarray) -> float:
    return float(max(np.linalg.eigvalsh((G + G.T) * 0.5)[-1], 0.0))


def cur_errors(A, J, I, normA_k_f, normA_k_2, do_spec=True, G=None, nA2=None):
    """Errors of CX (C C^+ A), CUR with U = C^+ A R^+ and CUR with U = A(I,J)^+, relative to the rank-k optimum.

    Every approximation has the form C M (M = k x p), so the residual norms follow from small matrices without
    forming the n x p residual: ||A - CM||_F^2 = ||A||_F^2 - 2<C^T A, M> + <M, C^T C M> and
    (A - CM)^T (A - CM) = G - (C^T A)^T M - M^T (C^T A) + M^T (C^T C) M with G = A^T A (p x p, formed once per
    matrix; only its top eigenvalue is read, which is well conditioned). Falls back to the explicit residual
    when the residual is tiny relative to ||A|| (cancellation guard)."""
    C = A[:, J]
    R = A[I, :]
    if G is None:
        G = A.T @ A
    if nA2 is None:
        nA2 = float(np.trace(G))
    CtA = C.T @ A
    CtC = C.T @ C
    Qc, _ = np.linalg.qr(C)
    B = Qc.T @ A                                             # CX = Qc B
    Cp = np.linalg.pinv(C)
    M1 = (Cp @ A) @ np.linalg.pinv(R) @ R                    # C^+ A R^+ R
    M2 = np.linalg.pinv(A[np.ix_(I, J)]) @ R                 # A(I,J)^+ R
    out = {}
    for name, M_ in (("CX", None), ("CUR_CpARp", M1), ("CUR_AIJp", M2)):
        if M_ is None:
            f2 = nA2 - float((B * B).sum())
        else:
            f2 = nA2 - 2.0 * float((CtA * M_).sum()) + float((M_ * (CtC @ M_)).sum())
        if f2 <= 1e-8 * nA2:                                 # cancellation guard: explicit residual
            E = A - (Qc @ B if M_ is None else C @ M_)
            f2 = float((E * E).sum())
            e2 = spec_norm(E) ** 2 if do_spec else None
        elif do_spec:
            if M_ is None:
                GE = G - B.T @ B
            else:
                X1 = CtA.T @ M_
                GE = G - X1 - X1.T + M_.T @ (CtC @ M_)
            e2 = _top_eig(GE)
        out[(name, "fro")] = float(math.sqrt(max(f2, 0.0)) / normA_k_f) if normA_k_f > 0 else float("nan")
        if do_spec:
            out[(name, "spec")] = float(math.sqrt(e2) / normA_k_2) if normA_k_2 > 0 else float("nan")
    return out


# ------------------------------------------------------------------------------------------------- recovery tools
def family_of(name: str, kind: str) -> str:
    s = name.lower()
    if kind == "mask" or s.endswith("_missing") or "_missing_" in s:
        return "mask"
    if "psofa" in s or "prism" in s:
        return "organ_score"
    if "aki" in s or "kdigo" in s:
        return "aki_stage"
    if "drug" in s:
        return "drug_group"
    if kind == "onehot":
        return "pooled_category"
    if "agez" in s or re.search(r"(^|_)z($|_)", s) or "zsd" in s:
        return "zscore"
    if kind == "bin" and re.search(r"(high|low|flag|tachy|brady|hypo|hyper|early|abnormal)", s):
        return "flag"
    if "ratio" in s or "podium" in s:
        return "ratio"
    if kind == "ordinal" or "band" in s:
        return "band_ordinal"
    if kind == "bin":
        return "other_binary"
    return "other_continuous"


def grouped_folds(groups: np.ndarray, k: int, rng: np.random.Generator) -> list:
    ug = np.unique(groups)
    perm = rng.permutation(ug)
    fold_of = {g: i % k for i, g in enumerate(perm)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def balanced_accuracy(y, yhat):
    accs = [np.mean(yhat[y == c] == c) for c in (0, 1) if np.any(y == c)]
    return float(np.mean(accs)) if accs else float("nan")


def r2(y, yhat):
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - yhat) ** 2).sum() / ss) if ss > 0 else float("nan")


# ------------------------------------------------------------------------------------------------------- config
@dataclass
class Config:
    project_root: Path
    in_dir: Path
    canonical: Path
    out_dir: Path
    expect_manifest_sha256: str | None
    precheck_only: bool
    n_boot: int = 200
    n_boot_angles: int = 50
    n_perm: int = 100
    n_cur_rep: int = 50
    n_cur_spec_rep: int = 10
    n_splithalf: int = 20
    n_rsvd_rep: int = 5
    recovery_max_per_family: int = 6
    recovery_max_rows: int = 3000
    max_seconds: float = 3600.0
    heartbeat_seconds: float = 30.0


def precheck(cfg: Config) -> dict:
    """Hash gates before any data file is parsed."""
    d = cfg.in_dir
    man_p = d / "stage06_08_manifest.json"
    if not man_p.is_file():
        raise InputError(f"Stage 06-08 manifest not found: {man_p}")
    if cfg.expect_manifest_sha256 is None:
        raise InputError("--expect-stage06-08-manifest-sha256 is required")
    man_sha = sha256_file(man_p)
    if man_sha != cfg.expect_manifest_sha256:
        raise InputError("stage06_08_manifest.json SHA-256 differs from --expect-stage06-08-manifest-sha256")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    if man.get("script") != PINS["stage06_08_script"]:
        raise InputError(f"Stage 06-08 manifest script is {man.get('script')!r}, expected {PINS['stage06_08_script']}")
    if man.get("script_sha256") != PINS["stage06_08_script_sha256"]:
        raise InputError("Stage 06-08 manifest script SHA-256 differs from the pinned v1.2.1 builder")
    if man.get("release_gate") != "ok":
        raise InputError("Stage 06-08 release gate is not ok")
    ac = man.get("acceptance_checks") or {}
    bad = [k for k, v in ac.items() if v != "pass"]
    if bad or not ac:
        raise InputError(f"Stage 06-08 acceptance checks not all pass: {bad}")
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
    absent = [n for n in need if n not in files]
    if absent:
        raise InputError(f"Stage 06-08 manifest lacks required files: {absent[:5]}")
    hashes = {"stage06_08_manifest.json": man_sha}
    for i, (name, rec) in enumerate(sorted(files.items())):
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            say(f"hashing Stage 06-08 files: {i + 1}/{len(files)}")
        h = sha256_file(d / name)
        if h != rec.get("sha256"):
            raise InputError(f"Stage 06-08 file {name} SHA-256 differs from its manifest entry")
        hashes[name] = h
    ih = man.get("input_hashes", {})
    if not cfg.canonical.is_file():
        raise InputError(f"canonical encounter file not found: {cfg.canonical}")
    hc = sha256_file(cfg.canonical)
    if hc != ih.get("canonical"):
        raise InputError("canonical file SHA-256 differs from the hash recorded in the Stage 06-08 manifest")
    if PINS.get("canonical") and hc != PINS["canonical"]:
        raise InputError("canonical file SHA-256 differs from the plan pin")
    hashes["canonical"] = hc
    if cfg.out_dir.exists() and any(cfg.out_dir.iterdir()):
        raise InputError(f"output folder exists and is not empty: {cfg.out_dir.name}")
    par = cfg.out_dir.parent
    pend = [p.name for p in par.glob(cfg.out_dir.name + ".pending-*")] if par.exists() else []
    if pend:
        raise InputError(f"a pending folder from an earlier failed run exists: {pend[0]} (remove it unopened first)")
    return {"hashes": hashes, "manifest": man}


# ------------------------------------------------------------------------------------------------------- loading
def read_npz(path: Path, keys) -> dict:
    """Read the named arrays fully into memory and close the archive at once (no handle stays open on Windows)."""
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            raise InputError(f"{path.name} lacks arrays {missing}")
        return {k: np.array(z[k]) for k in keys}


def load_inputs(cfg: Config) -> dict:
    d = cfg.in_dir
    lv = OrderedDict()
    eid0 = split0 = elig24 = None
    for k in LEVELS:
        z = read_npz(d / f"stage08_{k}.npz", ("encounter_id", "split", "eligible_24h", "X_z", "X_full_z", "columns",
                                              "kinds", "columns_full"))
        eid, split = z["encounter_id"].astype(str), z["split"].astype(str)
        if eid0 is None:
            eid0, split0 = eid, split
            elig24 = z["eligible_24h"].astype(bool)
        elif not (np.array_equal(eid, eid0) and np.array_equal(split, split0)):
            raise InputError(f"level {k}: encounter order or split differs from level a0")
        X, Xf = z["X_z"].astype(np.float64), z["X_full_z"].astype(np.float64)
        if not (np.isfinite(X).all() and np.isfinite(Xf).all()):
            raise InputError(f"level {k}: non-finite values in X_z / X_full_z")
        lv[k] = dict(X=X, cols=[str(c) for c in z["columns"]], kinds=[str(c) for c in z["kinds"]],
                     Xf=Xf, cols_full=[str(c) for c in z["columns_full"]])
        if len(lv[k]["cols"]) != X.shape[1] or len(lv[k]["kinds"]) != X.shape[1] or \
                len(lv[k]["cols_full"]) != Xf.shape[1]:
            raise InputError(f"level {k}: column lists do not match X_z / X_full_z")
    if not set(np.unique(split0)) <= {"train", "validation", "test"}:
        raise InputError("unexpected split labels")
    can = pd.read_csv(cfg.canonical, usecols=["encounter_id", "SUBJECT_ID"], dtype=str, keep_default_na=False)
    can["encounter_id"] = can["encounter_id"].str.strip()
    if can["encounter_id"].duplicated().any():
        raise InputError("canonical file has duplicated encounter ids")
    smap = dict(zip(can["encounter_id"], can["SUBJECT_ID"].str.strip()))
    if any(e not in smap for e in eid0):
        raise InputError("some Stage 08 encounters are not in the canonical file (count withheld)")
    subj = np.array([smap[e] for e in eid0])
    sp = pd.DataFrame({"s": subj, "sp": split0}).drop_duplicates()
    if sp["s"].duplicated().any():
        raise InputError("a subject appears in more than one split")
    lc = OrderedDict()
    emap = {e: i for i, e in enumerate(eid0)}
    for L in LANDMARKS:
        z = read_npz(d / f"stage08_layer_c_L{L}.npz", ("encounter_id", "split", "X_filled_z", "channels"))
        e = z["encounter_id"].astype(str)
        s = z["split"].astype(str)
        idx = np.array([emap.get(x, -1) for x in e])
        if (idx < 0).any() or not np.array_equal(split0[idx], s):
            raise InputError(f"Layer C L{L}: encounters or split do not match the tabular levels")
        Xf = z["X_filled_z"]
        if Xf.ndim != 3 or Xf.shape[1] != L // 6:
            raise InputError(f"Layer C L{L}: X_filled_z is not (encounters, {L // 6}, channels)")
        tr = s == "train"
        if not np.isfinite(Xf[tr]).all():
            raise InputError(f"Layer C L{L}: non-finite train values in X_filled_z")
        lc[L] = dict(X=Xf[tr].astype(np.float64), channels=[str(c) for c in z["channels"]], subj=subj[idx][tr])
    sizes = [lc[L]["X"].shape[0] for L in LANDMARKS]
    if any(b > a for a, b in zip(sizes, sizes[1:])):
        raise InputError("Layer C train landmark sizes are not non-increasing with L")
    tr = split0 == "train"
    if int((elig24 & tr).sum()) != lc[24]["X"].shape[0]:
        raise InputError("eligible_24h train rows do not match the Layer C L24 train rows")
    return dict(levels=lv, eid=eid0, split=split0, subj=subj, train=tr, elig24=elig24, lc=lc)


def sensitive_numbers(D: dict) -> set:
    """Sizes that must never appear in a TEAM file (cohort, splits, subjects, landmark sets, stacked rows)."""
    out = set()
    sp, subj = D["split"], D["subj"]
    out.update({len(sp), len(np.unique(subj)), int(D["elig24"].sum())})
    for s in ("train", "validation", "test"):
        m = sp == s
        out.update({int(m.sum()), len(np.unique(subj[m])), int((D["elig24"] & m).sum())})
    for L, r in D["lc"].items():
        n = r["X"].shape[0]
        out.update({n, n * (L // 6), len(np.unique(r["subj"]))})
    return {x for x in out if x >= 100}


# ---------------------------------------------------------------------------------------------- planted self-test
def planted_selftest() -> dict:
    """Built-in synthetic gate (plan 8, 13.3): planted rank 10 recovered by k95 (sigma^2) and the permutation null;
    a planted range column (max - min) named by Belsley as an exact dependency."""
    rng = rng_for("planted")
    n, p, r = 1500, 40, 10
    Qu, _ = np.linalg.qr(rng.standard_normal((n, r)))
    Qv, _ = np.linalg.qr(rng.standard_normal((p, r)))
    sig = math.sqrt(n) * np.linspace(1.2, 0.8, r)          # comparable energies: k95 must reach the planted rank
    A = (Qu * sig) @ Qv.T + 0.01 * rng.standard_normal((n, p))
    s = svdvals(A)
    null = perm_null(A, 30, rng_for("planted-null"))
    cnt = int((s > np.percentile(null, 95, axis=0)).sum())
    k95 = k_energy(s, 2)
    B = rng.standard_normal((n, 6))
    B = np.c_[B, B[:, 0] - B[:, 1]]                  # "range" = max - min, an exact dependency
    B = (B - B.mean(0)) / B.std(0)
    _, sb, Vt = thin_svd(B)
    rows = belsley(sb, Vt, ["max", "min", "c2", "c3", "c4", "c5", "range"], B.shape, 30.0, 0.5)
    named = any(r_["kind"] == "exact_null" and "range" in r_["columns"].split(";") for r_ in rows)
    return dict(planted_rank=r, k95_sigma2=k95, perm_null_count=cnt, belsley_names_range=named,
                ok=bool(k95 == r and cnt == r and named))


# ---------------------------------------------------------------------------------------------------- stage 09p
def stage09p(D: dict, checks: dict, details: dict) -> dict:
    tr = D["train"]
    dmap_rows, widths = [], OrderedDict()
    for k, L in D["levels"].items():
        ex = (lambda c: DUP_TOKEN in c) if k == "b2d" else (lambda c: False)
        keep, groups = dedup_columns(L["X"][tr], L["cols"], ex)
        L["keep"] = keep
        L["Xd"] = L["X"][:, keep]
        L["cols_d"] = [L["cols"][j] for j in keep]
        L["kinds_d"] = [L["kinds"][j] for j in keep]
        dmap_rows += [dict(matrix=k, kept_column=kept, dropped_column=dn) for kept, dr in groups for dn in dr]
        widths[k] = dict(width_predup=len(L["cols"]), width_dedup=len(keep), width_full_onehot=len(L["cols_full"]),
                         n_dropped=len(L["cols"]) - len(keep), n_dup_kept=sum(DUP_TOKEN in c for c in L["cols_d"]))
    checks["09p_b2d_dup_copies_never_dropped"] = (
        widths["b2d"]["n_dup_kept"] == sum(DUP_TOKEN in c for c in D["levels"]["b2d"]["cols"]))
    s_names = [c[4:] for c in D["levels"]["b2s"]["cols_d"]]
    d_names = [c[4:] for c in D["levels"]["b2d"]["cols_d"] if DUP_TOKEN not in c]
    checks["09p_b2d_nondup_equals_b2s_after_dedup"] = s_names == d_names
    if s_names != d_names:
        details["09p_b2d_nondup_equals_b2s_after_dedup"] = sorted(set(s_names) ^ set(d_names))[:20]
    ok = True
    for row in dmap_rows:
        L = D["levels"][row["matrix"]]
        a, b = L["cols"].index(row["kept_column"]), L["cols"].index(row["dropped_column"])
        ok &= bool(np.array_equal(L["X"][tr, a].astype(np.float32), L["X"][tr, b].astype(np.float32)))
        ok &= not (row["matrix"] == "b2d" and DUP_TOKEN in row["dropped_column"])
    checks["09p_dropped_copies_identical_on_train"] = bool(ok)
    ab = OrderedDict()
    for a, b in AB_PAIRS:
        La, Lb = D["levels"][a], D["levels"][b]
        X = np.c_[La["Xd"], Lb["Xd"]]
        cols, kinds = La["cols_d"] + Lb["cols_d"], La["kinds_d"] + Lb["kinds_d"]
        ex = (lambda c: DUP_TOKEN in c) if b == "b2d" else (lambda c: False)
        keep, groups = dedup_columns(X[tr], cols, ex)
        dmap_rows += [dict(matrix=f"{a}+{b}", kept_column=kept, dropped_column=dn) for kept, dr in groups for dn in dr]
        ab[f"{a}+{b}"] = dict(X=X[:, keep], cols=[cols[j] for j in keep], kinds=[kinds[j] for j in keep])
        widths[f"{a}+{b}"] = dict(width_predup=len(cols), width_dedup=len(keep), n_dropped=len(cols) - len(keep))
    for Lh, r in D["lc"].items():
        n, b, q = r["X"].shape
        keep, groups = dedup_columns(r["X"].reshape(n * b, q), r["channels"])
        dmap_rows += [dict(matrix=f"C_L{Lh}", kept_column=kept, dropped_column=dn) for kept, dr in groups for dn in dr]
        r["keep"] = keep
        r["chan_d"] = [r["channels"][j] for j in keep]
        widths[f"C_L{Lh}"] = dict(width_predup=q, width_dedup=len(keep), n_dropped=q - len(keep))
    return dict(dmap_rows=dmap_rows, widths=widths, ab=ab)


# ------------------------------------------------------------------------------------------- stage 09 (tabular)
def stage09_points(D: dict, checks: dict, details: dict) -> dict:
    tr = D["train"]
    summ, spectra, bel_rows, base = [], [], [], OrderedDict()
    recon_bad = []
    for k, L in D["levels"].items():
        nm = [j for j, kd in enumerate(L["kinds_d"]) if kd != "mask"]
        mats = OrderedDict([("dedup", (L["Xd"][tr], L["cols_d"])), ("predup", (L["X"][tr], L["cols"])),
                            ("full_onehot_predup", (L["Xf"][tr], L["cols_full"])),
                            ("dedup_no_masks", (L["Xd"][tr][:, nm], [L["cols_d"][j] for j in nm]))])
        for var, (A, cols) in mats.items():
            U, s, Vt = thin_svd(A)
            if np.linalg.norm(A - (U * s) @ Vt) > 1e-6 * max(np.linalg.norm(A), 1e-300):
                recon_bad.append(f"{k}:{var}")
            row = dict(level=k, variant=var, **spectral_summary(s, A.shape), **decay_fit(s, A.shape))
            row["n_above_mp_edge"] = int((s > mp_edge(*A.shape)).sum())
            summ.append(row)
            spectra += [dict(level=k, variant=var, index=i + 1, sigma_rel=v) for i, v in enumerate(s / s[0])]
            if var != "dedup_no_masks":
                bel_rows += [dict(level=k, variant=var, **b_) for b_ in
                             belsley(s, Vt, cols, A.shape, THRESH["belsley_eta"], THRESH["belsley_prop"])]
            if var == "dedup":
                base[k] = dict(U=U, s=s, Vt=Vt)
            check_time()
        say(f"stage 09: level {k} point estimates done")
    checks["09_svd_reconstruction"] = not recon_bad
    if recon_bad:
        details["09_svd_reconstruction"] = recon_bad
    sm = {(r_["level"], r_["variant"]): r_ for r_ in summ}
    nd_pre = sum(DUP_TOKEN in c for c in D["levels"]["b2d"]["cols"])
    nd_ded = sum(DUP_TOKEN in c for c in D["levels"]["b2d"]["cols_d"])
    h1_pre = sm[("b2d", "predup")]["rank_deficiency"] - sm[("b2s", "predup")]["rank_deficiency"] == nd_pre
    h1_ded = sm[("b2d", "dedup")]["rank_deficiency"] - sm[("b2s", "dedup")]["rank_deficiency"] == nd_ded
    checks["09_H1_rank_identity_predup"] = bool(h1_pre)
    checks["09_H1_rank_identity_dedup"] = bool(h1_ded)
    return dict(summ=summ, sm=sm, spectra=spectra, bel_rows=bel_rows, base=base, h1_pre=h1_pre, h1_ded=h1_ded)


def stage09_null(D: dict, P: dict, cfg: Config) -> dict:
    tr = D["train"]
    null95 = {}
    for k, L in D["levels"].items():
        t = time.time()
        nl = perm_null(L["Xd"][tr], cfg.n_perm, rng_for("perm", k))
        q95 = np.percentile(nl, 95, axis=0)
        null95[k] = q95
        s = P["base"][k]["s"]
        P["sm"][(k, "dedup")]["n_above_perm_null"] = int((s > q95[:len(s)]).sum())
        P["spectra"] += [dict(level=k, variant="perm_null_p95", index=i + 1, sigma_rel=v)
                         for i, v in enumerate(q95 / s[0])]
        say(f"stage 09: permutation null {k} done ({time.time() - t:.0f} s)")
    return null95


def stage09_bootstrap(D: dict, P: dict, null95: dict, cfg: Config) -> dict:
    """Shared subject bootstrap (row weights = number of draws of the row's subject): spectra and decay family
    (09), Davis-Kahan / Wedin (09), principal-angle intervals (10)."""
    tr = D["train"]
    subj_tr = D["subj"][tr]
    usub, inv = np.unique(subj_tr, return_inverse=True)
    boot = {k: dict(stable=[], er2=[], ers=[], k95=[], k95s=[], r=[], r3=[], r2=[], cnt=[], fam=[], gaps=[],
                    dk_obs={kk: [] for kk in K_GRID}, dk_bnd={kk: [] for kk in K_GRID}) for k in LEVELS}
    ang_boot = {f"{a}|{b}": dict(nzero=[], med=[]) for a, b in ANGLE_PAIRS}
    A_tr = {k: D["levels"][k]["Xd"][tr] for k in LEVELS}
    t0 = time.time()
    for bi in range(cfg.n_boot):
        rg = rng_for("boot", bi)
        cnt_s = np.bincount(rg.integers(0, len(usub), len(usub)), minlength=len(usub))
        w = cnt_s[inv].astype(np.float64)
        pos = w > 0
        sw = np.sqrt(w)
        Ub = {}
        need_u = bi < cfg.n_boot_angles      # left vectors (row-space angles between levels) on the first replicates
        for k in LEVELS:
            A = A_tr[k]
            Ab = A[pos] * sw[pos, None]          # float64: the numerical-rank tolerance is a float64 one
            if need_u:
                U, s, Vt = thin_svd(Ab)
            else:
                U = None
                s, Vt = r_svd(Ab)
            s = s.astype(np.float64)
            r = numerical_rank(s, Ab.shape)
            b = boot[k]
            b["stable"].append(stable_rank(s)); b["er2"].append(erank_sigma2(s)); b["ers"].append(erank_sigma(s))
            b["k95"].append(k_energy(s, 2)); b["k95s"].append(k_energy(s, 1)); b["r"].append(r)
            b["r3"].append(int((s > 1e-3 * s[0]).sum())); b["r2"].append(int((s > 1e-2 * s[0]).sum()))
            q = null95[k]
            m = min(len(s), len(q))
            b["cnt"].append(int((s[:m] > q[:m]).sum()))
            b["fam"].append(decay_fit(s, Ab.shape)["decay_family"])
            b["gaps"].append([(s[kk - 1] - s[kk]) / s[kk - 1] if kk < len(s) and s[kk - 1] > 0 else np.nan
                              for kk in K_GRID])
            s0, V0 = P["base"][k]["s"], P["base"][k]["Vt"]
            r0 = numerical_rank(s0, A.shape)
            nE = None
            for kk in K_GRID:
                if kk >= min(len(s0), r0, r):
                    continue
                if nE is None:
                    nE = spec_norm((sw - 1.0)[:, None] * A)
                ang = principal_angles(V0[:kk].T, Vt[:kk].T.astype(np.float64))
                b["dk_obs"][kk].append(float(np.sin(ang[-1])) if ang.size else 0.0)
                gap = s0[kk - 1] - s0[kk]
                b["dk_bnd"][kk].append(nE / gap if gap > 0 else float("inf"))
            if need_u:
                Ub[k] = U[:, :r]
        for a, b2 in (ANGLE_PAIRS if need_u else ()):
            ang = principal_angles(Ub[a], Ub[b2])
            ang_boot[f"{a}|{b2}"]["nzero"].append(int((np.cos(ang) >= 1 - 1e-5).sum()))
            ang_boot[f"{a}|{b2}"]["med"].append(float(np.degrees(np.median(ang))) if ang.size else float("nan"))
        check_time()
        if bi in (0, 4) or (bi + 1) % 20 == 0 or bi + 1 == cfg.n_boot:
            el = time.time() - t0
            say(f"bootstrap {bi + 1}/{cfg.n_boot} ({el:.0f} s; about {el / (bi + 1) * (cfg.n_boot - bi - 1):.0f} s left)")
    boot_rows, dk_rows = [], []
    for k in LEVELS:
        b = boot[k]
        row = dict(level=k, variant="dedup", B=cfg.n_boot)
        for key, nm_ in (("stable", "stable_rank"), ("er2", "erank_sigma2"), ("ers", "erank_sigma"),
                         ("k95", "k95_sigma2"), ("k95s", "k95_sigma"), ("r", "rank_numerical"),
                         ("r3", "rank_1e3"), ("r2", "rank_1e2"), ("cnt", "n_above_perm_null")):
            row[f"{nm_}_p2_5"] = pct(b[key], 2.5)
            row[f"{nm_}_p97_5"] = pct(b[key], 97.5)
        fams = pd.Series(b["fam"]).value_counts(normalize=True)
        row["decay_family_modal"] = str(fams.index[0]) if len(fams) else "na"
        row["decay_family_modal_share"] = float(fams.iloc[0]) if len(fams) else float("nan")
        boot_rows.append(row)
        g = np.array(b["gaps"], dtype=float)
        s0 = P["base"][k]["s"]
        for j, kk in enumerate(K_GRID):
            obs, bnd = np.array(b["dk_obs"][kk]), np.array(b["dk_bnd"][kk])
            if kk >= len(s0) or not obs.size:
                continue
            dk_rows.append(dict(level=k, k=kk, rel_gap=(s0[kk - 1] - s0[kk]) / s0[kk - 1],
                                rel_gap_p2_5=pct(g[:, j], 2.5), rel_gap_p97_5=pct(g[:, j], 97.5),
                                sin_angle_obs_p50=pct(obs, 50), sin_angle_obs_p90=pct(obs, 90),
                                bound_p50=pct(bnd, 50), bound_p90=pct(bnd, 90),
                                share_bound_vacuous=float(np.mean(bnd >= 1.0)),
                                share_bound_holds=float(np.mean(obs <= bnd + 1e-9))))
    return dict(boot_rows=boot_rows, dk_rows=dk_rows, ang_boot=ang_boot)


# ------------------------------------------------------------------------------------------- stage 09 (Layer C)
def stage09_layer_c(D: dict, cfg: Config) -> dict:
    lc_rows, lc_boot_rows = [], []
    for Lh, r in D["lc"].items():
        X = r["X"][:, :, r["keep"]]
        n, b, q = X.shape
        S = X.reshape(n * b, q)
        U, s, Vt = thin_svd(S)
        row = dict(landmark=Lh, view="stacked", bin="all", **spectral_summary(s, S.shape), **decay_fit(s, S.shape))
        if Lh in CORE_LANDMARKS:
            Sb = X.transpose(1, 0, 2).reshape(b * n, q)           # bin-major rows: one block per bin
            nl = perm_null(Sb, cfg.n_perm, rng_for("lcperm", Lh), blocks=[(i * n, (i + 1) * n) for i in range(b)])
            q95 = np.percentile(nl, 95, axis=0)
            row["n_above_perm_null"] = int((s > q95[:len(s)]).sum())
            us, inv_c = np.unique(r["subj"], return_inverse=True)
            bb = dict(er2=[], k95=[], stable=[])
            for bi in range(cfg.n_boot):
                rg = rng_for("lcboot", Lh, bi)
                wr = np.bincount(rg.integers(0, len(us), len(us)), minlength=len(us))[inv_c]
                pos = wr > 0
                Sw = (X[pos] * np.sqrt(wr[pos])[:, None, None]).reshape(-1, q).astype(np.float32)
                sb = svdvals(Sw).astype(np.float64)
                bb["er2"].append(erank_sigma2(sb)); bb["k95"].append(k_energy(sb, 2))
                bb["stable"].append(stable_rank(sb))
                if bi % 50 == 49:
                    check_time()
            lc_boot_rows.append(dict(landmark=Lh, B=cfg.n_boot,
                                     erank_sigma2_p2_5=pct(bb["er2"], 2.5), erank_sigma2_p97_5=pct(bb["er2"], 97.5),
                                     k95_sigma2_p2_5=pct(bb["k95"], 2.5), k95_sigma2_p97_5=pct(bb["k95"], 97.5),
                                     stable_rank_p2_5=pct(bb["stable"], 2.5),
                                     stable_rank_p97_5=pct(bb["stable"], 97.5)))
            V5 = Vt[:5].T
            for j in range(b):
                Mj = X[:, j, :]
                _, sj, Vtj = thin_svd(Mj)
                rj = dict(landmark=Lh, view="per_bin", bin=j, **spectral_summary(sj, Mj.shape))
                ang = principal_angles(V5, Vtj[:5].T) if min(Vtj.shape[0], V5.shape[1]) >= 5 else np.zeros(0)
                rj["angle_top5_vs_stacked_p50_deg"] = float(np.degrees(np.median(ang))) if ang.size else float("nan")
                rj["angle_top5_vs_stacked_largest_deg"] = float(np.degrees(ang[-1])) if ang.size else float("nan")
                lc_rows.append(rj)
        lc_rows.append(row)
        r["V"] = Vt.T
        r["cpqr"] = cpqr_order(S)
        say(f"stage 09: Layer C L{Lh} done")
        check_time()
    return dict(lc_rows=lc_rows, lc_boot_rows=lc_boot_rows)


# ---------------------------------------------------------------------------------- stage 10: low-rank and CUR
def stage10_lowrank(D: dict, P: dict, S9p: dict, cfg: Config, checks: dict) -> dict:
    tr = D["train"]
    ey_rows, rs_rows, cur_rows, cneed_rows, cpqr_rows = [], [], [], [], []
    ey_ok = cur_ok = rs_ok = True
    bases = OrderedDict()
    for k, L in D["levels"].items():
        A = L["Xd"][tr]
        n, p = A.shape
        U, s, Vt = P["base"][k]["U"], P["base"][k]["s"], P["base"][k]["Vt"]
        rk = numerical_rank(s, A.shape)
        piv = cpqr_order(A)
        cpqr_rows += [dict(matrix=k, order=i + 1, column=L["cols_d"][j]) for i, j in enumerate(piv[:59])]
        nA = float(np.linalg.norm(A))
        G_A = A.T @ A                    # for residual norms only (top eigenvalue / trace), never for the spectrum
        nA2 = float((A * A).sum())
        for kk in K_GRID:
            if kk >= min(p, rk):
                continue
            Ak = (U[:, :kk] * s[:kk]) @ Vt[:kk]
            e2, ef = spec_norm(A - Ak), float(np.linalg.norm(A - Ak))
            t2, tf = float(s[kk]), math.sqrt(float((s[kk:] ** 2).sum()))
            ok = abs(e2 - t2) <= 1e-5 * s[0] and abs(ef - tf) <= 1e-6 * max(nA, 1.0)
            ey_ok &= ok
            ey_rows.append(dict(level=k, k=kk, spec_err_over_sigma_k1=e2 / t2 if t2 > 0 else float("nan"),
                                fro_err_over_tail=ef / tf if tf > 0 else float("nan"), identity_holds=bool(ok)))
            for sp_ in (0, 1, 2):
                proj, rkk = [], []
                for rep in range(cfg.n_rsvd_rep):
                    rg = rng_for("rsvd", k, kk, sp_, rep)
                    Y = A @ rg.standard_normal((p, kk + 10))
                    for _ in range(sp_):
                        Qy, _r = np.linalg.qr(Y)
                        Z, _r = np.linalg.qr(A.T @ Qy)
                        Y = A @ Z
                    Q, _r = np.linalg.qr(Y)
                    B_ = Q.T @ A
                    proj.append(math.sqrt(_top_eig(G_A - B_.T @ B_)) / t2)          # ||A - Q Q^T A||_2
                    Ub_, sb_, Vb_ = thin_svd(B_)
                    Bk = (Ub_[:, :kk] * sb_[:kk]) @ Vb_[:kk]
                    X1 = B_.T @ Bk
                    rkk.append(math.sqrt(_top_eig(G_A - X1 - X1.T + Bk.T @ Bk)) / t2)  # ||A - Q B_k||_2
                rs_ok &= min(rkk) >= 1 - 1e-5
                rs_rows.append(dict(level=k, k=kk, power_iterations=sp_, oversampling=10, repeats=cfg.n_rsvd_rep,
                                    proj_err_over_sigma_k1_p50=pct(proj, 50),
                                    rankk_err_over_sigma_k1_p50=pct(rkk, 50), rankk_err_over_sigma_k1_p90=pct(rkk, 90)))
            Vk, Uk = Vt[:kk].T, U[:, :kk]
            lev_c = leverage(Vk) + 1e-12
            lev_c /= lev_c.sum()
            lev_r = leverage(Uk) + 1e-12
            lev_r /= lev_r.sum()
            det_rules = OrderedDict([
                ("top_leverage", (list(np.argsort(-lev_c, kind="stable")[:kk]),
                                  list(np.argsort(-lev_r, kind="stable")[:kk]))),
                ("cpqr", (list(piv[:kk]), list(cpqr_order(Uk.T)[:kk]))),
                ("deim", (deim(Vk), deim(Uk)))])
            for rule, (J, I) in det_rules.items():
                for (mid, nrm), v in cur_errors(A, J, I, tf, t2, G=G_A, nA2=nA2).items():
                    cur_ok &= v >= 1 - 1e-6
                    cur_rows.append(dict(level=k, k=kk, c=kk, rule=rule, approx=mid, norm=nrm, repeats=1,
                                         ratio_p50=v, ratio_p25=v, ratio_p75=v))
            for rule in ("uniform", "leverage_sampling"):
                acc = OrderedDict()
                for rep in range(cfg.n_cur_rep):
                    rg = rng_for("cur", k, kk, rule, rep)
                    if rule == "uniform":
                        J, I = rg.choice(p, kk, replace=False), rg.choice(n, kk, replace=False)
                    else:
                        J, I = rg.choice(p, kk, replace=False, p=lev_c), rg.choice(n, kk, replace=False, p=lev_r)
                    for key, v in cur_errors(A, list(J), list(I), tf, t2, do_spec=rep < cfg.n_cur_spec_rep,
                                             G=G_A, nA2=nA2).items():
                        acc.setdefault(key, []).append(v)
                for (mid, nrm), vs in acc.items():
                    cur_ok &= min(vs) >= 1 - 1e-6
                    cur_rows.append(dict(level=k, k=kk, c=kk, rule=rule, approx=mid, norm=nrm, repeats=len(vs),
                                         ratio_p50=pct(vs, 50), ratio_p25=pct(vs, 25), ratio_p75=pct(vs, 75)))
            cgrid = sorted({c for c in (kk, int(1.5 * kk), 2 * kk, 3 * kk, 4 * kk, 6 * kk, 8 * kk, 12 * kk, p)
                            if kk <= c <= p})
            med = {}
            for c in cgrid:
                vs = []
                for rep in range(5):
                    rg = rng_for("cneed", k, kk, c, rep)
                    J = rg.choice(p, c, replace=False, p=lev_c) if c < p else np.arange(p)
                    Qc, _r = np.linalg.qr(A[:, J])
                    Bq = Qc.T @ A
                    vs.append(math.sqrt(max(nA2 - float((Bq * Bq).sum()), 0.0)) / tf if tf > 0 else float("nan"))
                med[c] = pct(vs, 50)
            for eps in (0.1, 0.5):
                hit = [c for c in cgrid if med[c] <= 1 + eps]
                cneed_rows.append(dict(level=k, k=kk, eps=eps, c_needed_observed=hit[0] if hit else "not_reached",
                                       c_bound_order_klnk_over_eps2=int(math.ceil(kk * math.log(kk) / eps ** 2)),
                                       p=p))
            check_time()
        bases[k] = dict(V=Vt[:min(59, rk)].T.astype(np.float32), cpqr=piv[:59], cols=L["cols_d"])
        say(f"stage 10: level {k} low-rank / CUR done")
    for nm_, M in S9p["ab"].items():
        A = M["X"][tr]
        _, s, Vt = thin_svd(A)
        rk = numerical_rank(s, A.shape)
        piv = cpqr_order(A)
        bases[nm_] = dict(V=Vt[:min(59, rk)].T.astype(np.float32), cpqr=piv[:59], cols=M["cols"])
        cpqr_rows += [dict(matrix=nm_, order=i + 1, column=M["cols"][j]) for i, j in enumerate(piv[:59])]
    for Lh, r in D["lc"].items():
        cpqr_rows += [dict(matrix=f"C_L{Lh}", order=i + 1, column=r["chan_d"][j]) for i, j in enumerate(r["cpqr"])]
    checks["10_eckart_young_identities"] = bool(ey_ok)
    checks["10_cur_ratio_at_least_1"] = bool(cur_ok)
    checks["10_rsvd_rank_k_error_at_least_sigma_k1"] = bool(rs_ok)
    return dict(ey_rows=ey_rows, rs_rows=rs_rows, cur_rows=cur_rows, cneed_rows=cneed_rows, cpqr_rows=cpqr_rows,
                bases=bases)


# ---------------------------------------------------------------------------------------- stage 10: angles
def stage10_angles(D: dict, P: dict, B9: dict, cfg: Config, checks: dict) -> dict:
    tr = D["train"]
    Q = {k: P["base"][k]["U"][:, :numerical_rank(P["base"][k]["s"], D["levels"][k]["Xd"][tr].shape)]
         for k in LEVELS}
    checks["10_angles_self_zero"] = bool(all(np.allclose(principal_angles(Q[k], Q[k]), 0, atol=1e-4)
                                             for k in LEVELS))
    zero_deg = float(np.degrees(np.arccos(1 - 1e-5)))
    ang_rows = []
    for a, b in ANGLE_PAIRS:
        small, big = (a, b) if Q[a].shape[1] <= Q[b].shape[1] else (b, a)
        QU, QV = Q[small], Q[big]
        ang = np.degrees(principal_angles(QU, QV))
        n = QU.shape[0]
        rg = rng_for("angnull", a, b)
        G1 = np.linalg.qr(rg.standard_normal((n, QU.shape[1])))[0]
        G2 = np.linalg.qr(rg.standard_normal((n, QV.shape[1])))[0]
        gnull = np.degrees(principal_angles(G1, G2))
        Xb = D["levels"][big]["Xd"][tr]
        idx = np.argsort(rg.random(Xb.shape, dtype=np.float32), axis=0)
        pnull = np.degrees(principal_angles(QU, orth_basis(np.take_along_axis(Xb, idx, axis=0))))
        p5 = pct(pnull, 5)
        bt = B9["ang_boot"][f"{a}|{b}"]
        nz = int((ang <= zero_deg).sum())
        ang_rows.append(dict(pair=f"{a}|{b}", smaller=small, larger=big, dim_smaller=QU.shape[1],
                             dim_larger=QV.shape[1], n_zero_angles=nz, n_below_1deg=int((ang < 1).sum()),
                             n_below_10deg=int((ang < 10).sum()), n_below_perm_null_p5=int((ang < p5).sum()),
                             new_directions_larger=int(QV.shape[1] - nz),
                             angle_p10=pct(ang, 10), angle_p25=pct(ang, 25), angle_p50=pct(ang, 50),
                             angle_p75=pct(ang, 75), angle_p90=pct(ang, 90),
                             gauss_null_p5=pct(gnull, 5), gauss_null_p50=pct(gnull, 50),
                             perm_null_p5=p5, perm_null_p50=pct(pnull, 50),
                             B_angles=len(bt["nzero"]), boot_n_zero_p2_5=pct(bt["nzero"], 2.5), boot_n_zero_p97_5=pct(bt["nzero"], 97.5),
                             boot_angle_p50_p2_5=pct(bt["med"], 2.5), boot_angle_p50_p97_5=pct(bt["med"], 97.5)))
        check_time()
    sh_rows = []
    subj_tr = D["subj"][tr]
    usub = np.unique(subj_tr)
    for k in LEVELS:
        A = D["levels"][k]["Xd"][tr]
        acc = {kk: [] for kk in (5, 10, 20)}
        for rep in range(cfg.n_splithalf):
            rg = rng_for("split", k, rep)
            half = set(rg.choice(usub, len(usub) // 2, replace=False).tolist())
            m = np.array([x in half for x in subj_tr])
            _, _, V1 = thin_svd(A[m].astype(np.float32))
            _, _, V2 = thin_svd(A[~m].astype(np.float32))
            for kk in acc:
                if kk < min(V1.shape[0], V2.shape[0]):
                    acc[kk].append(float(np.degrees(principal_angles(V1[:kk].T.astype(np.float64),
                                                                     V2[:kk].T.astype(np.float64))[-1])))
        sh_rows += [dict(level=k, k=kk, repeats=len(v), largest_angle_p50=pct(v, 50), largest_angle_p90=pct(v, 90))
                    for kk, v in acc.items() if v]
        check_time()
    return dict(ang_rows=ang_rows, sh_rows=sh_rows)


# -------------------------------------------------------------------------------------- stage 10: recovery
def stage10_recovery(D: dict, cfg: Config, checks: dict) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    tr = D["train"]
    subj_tr = D["subj"][tr]
    usub = np.unique(subj_tr)
    frac = min(1.0, cfg.recovery_max_rows / max(len(subj_tr), 1))
    pick = set(rng_for("recovery-rows").permutation(usub)[:max(10, int(round(len(usub) * frac)))].tolist())
    rmask = np.array([x in pick for x in subj_tr])
    g = subj_tr[rmask]
    folds = grouped_folds(g, 5, rng_for("recovery-folds"))
    checks["10_recovery_folds_grouped_by_subject"] = not any(set(g[a]) & set(g[b]) for a, b in folds)
    srcA = D["levels"]["a0"]["Xd"][tr][rmask]
    srcB = np.c_[srcA, D["levels"]["b0"]["Xd"][tr][rmask]]
    rows = []
    for k, prev in RECOVERY_PREV.items():
        L, Pv = D["levels"][k], D["levels"][prev]
        Xs = srcA if k.startswith("a") else srcB
        Pt = Pv["Xd"][tr].astype(np.float32)
        ph = {hashlib.sha256(np.ascontiguousarray(Pt[:, j]).tobytes()).hexdigest() for j in range(Pt.shape[1])}
        Lt = L["Xd"][tr].astype(np.float32)
        cand = OrderedDict()
        for j, (c, kd) in enumerate(zip(L["cols_d"], L["kinds_d"])):
            if DUP_TOKEN in c or hashlib.sha256(np.ascontiguousarray(Lt[:, j]).tobytes()).hexdigest() in ph:
                continue
            cand.setdefault(family_of(c, kd), []).append(j)
        t = time.time()
        for fam, js in cand.items():
            sel = sorted(rng_for("recpick", k, fam).permutation(js)[:cfg.recovery_max_per_family].tolist())
            for j in sel:
                y = L["Xd"][tr][rmask, j]
                vals = np.unique(y)
                if vals.size < 2:
                    continue
                binary = vals.size == 2
                yb = (y == vals[1]).astype(int) if binary else None
                yl, pl = np.zeros(len(y)), np.zeros(len(y))
                for f_tr, f_te in folds:
                    if binary:
                        if yb[f_tr].min() == yb[f_tr].max():
                            yl[f_te] = pl[f_te] = yb[f_tr][0]
                            continue
                        m1 = LogisticRegression(C=1.0, max_iter=300).fit(Xs[f_tr], yb[f_tr])
                        m2 = HistGradientBoostingClassifier(max_iter=60, early_stopping=False,
                                                            random_state=SEED).fit(Xs[f_tr], yb[f_tr])
                    else:
                        m1 = Ridge(alpha=1.0).fit(Xs[f_tr], y[f_tr])
                        m2 = HistGradientBoostingRegressor(max_iter=60, early_stopping=False,
                                                           random_state=SEED).fit(Xs[f_tr], y[f_tr])
                    yl[f_te], pl[f_te] = m1.predict(Xs[f_te]), m2.predict(Xs[f_te])
                if binary:
                    lin, non, metric = balanced_accuracy(yb, yl), balanced_accuracy(yb, pl), "balanced_accuracy"
                else:
                    lin, non, metric = r2(y, yl), r2(y, pl), "r2"
                rows.append(dict(level=k, source="a0" if k.startswith("a") else "a0+b0", family=fam,
                                 column=L["cols_d"][j], metric=metric, linear=lin, nonlinear=non,
                                 gap_nonlinear_minus_linear=non - lin))
                check_time()
        say(f"stage 10: recovery {k} done ({time.time() - t:.0f} s)")
    rec = pd.DataFrame(rows, columns=["level", "source", "family", "column", "metric", "linear", "nonlinear",
                                      "gap_nonlinear_minus_linear"])
    fam_rows = []
    for (lvl, fam, met), gg in rec.groupby(["level", "family", "metric"], sort=False):
        fam_rows.append(dict(level=lvl, family=fam, metric=met, n_columns_evaluated=len(gg),
                             linear_p50=gg["linear"].median(), nonlinear_p50=gg["nonlinear"].median(),
                             gap_p50=gg["gap_nonlinear_minus_linear"].median()))
    return dict(rec=rec, fam_rows=fam_rows)


# ------------------------------------------------------------------------------------------------- hypotheses
def hypotheses(sm, boot_rows, cur_rows, fam_rows, h1_pre, h1_ded) -> list:
    out = []
    for k in ("a0", "a1", "b0", "b1"):
        v = sm[(k, "dedup")]["kappa2_over_rank"]
        out.append(dict(hypothesis="H1", item=f"{k}: kappa2 over numerical rank < {THRESH['H1_kappa']:g} (dedup)",
                        value=v, verdict="supported" if v < THRESH["H1_kappa"] else "not_supported"))
    out.append(dict(hypothesis="H1", item="rank deficiency b2d minus b2s equals the retained __dup__ copies "
                    "(before and after dedup)", value=float(h1_pre and h1_ded),
                    verdict="supported" if (h1_pre and h1_ded) else "not_supported"))
    for k in LEVELS:
        r = sm[(k, "dedup")]
        ok = r["n_above_perm_null"] < THRESH["H2_frac_p"] * r["p"]
        out.append(dict(hypothesis="H2", item=f"{k}: count above permutation null below p/3 (value = count / p)",
                        value=r["n_above_perm_null"] / r["p"], verdict="supported" if ok else "not_supported"))
    for lay, levs in (("A", LEVELS_A), ("B", LEVELS_B)):
        c0 = sm[(levs[0], "dedup")]["n_above_perm_null"]
        rise = max(sm[(k, "dedup")]["n_above_perm_null"] for k in levs) - c0
        out.append(dict(hypothesis="H2", item=f"layer {lay}: rise of the null count from phi0 to the richest level "
                        f"at most {THRESH['H2_rise']}", value=rise,
                        verdict="supported" if rise <= THRESH["H2_rise"] else "not_supported"))
    fams = [r["decay_family_modal"] for r in boot_rows]
    shares = [r["decay_family_modal_share"] for r in boot_rows]
    ok = len(set(fams)) == 1 and min(shares) >= THRESH["H3_freq"]
    out.append(dict(hypothesis="H3", item="same BIC decay family at every level in at least 80% of resamples "
                    "(value = smallest modal share)", value=min(shares), verdict="supported" if ok else "not_supported"))
    fr = pd.DataFrame(fam_rows)
    if len(fr):
        z = fr[(fr["family"] == "zscore") & (fr["metric"] == "r2")]["gap_p50"]
        f = fr[fr["family"].isin(["flag", "aki_stage"])]["gap_p50"]
        if len(z):
            out.append(dict(hypothesis="H4", item="recovery gap (nonlinear minus linear) for z-scores below 0.05",
                            value=float(z.median()),
                            verdict="supported" if z.median() < THRESH["H4_gap_z"] else "not_supported"))
        if len(f):
            out.append(dict(hypothesis="H4", item="recovery gap for flags and AKI stage at least 0.3",
                            value=float(f.median()),
                            verdict="supported" if f.median() >= THRESH["H4_gap_flag"] else "not_supported"))
    out.append(dict(hypothesis="H4", item="principal-angle part: see stage10_angles_TEAM.csv (for review)",
                    value=float("nan"), verdict="review"))
    cr = pd.DataFrame(cur_rows)
    good = None
    if len(cr):
        cr = cr[(cr["approx"] == "CUR_CpARp") & (cr["norm"] == "fro")]
        good = True
        for _, gg in cr.groupby(["level", "k"]):
            m = dict(zip(gg["rule"], gg["ratio_p50"]))
            if all(r_ in m for r_ in ("cpqr", "deim", "leverage_sampling", "uniform")):
                good &= max(m["cpqr"], m["deim"]) < min(m["leverage_sampling"], m["uniform"])
    out.append(dict(hypothesis="H5", item="CPQR and DEIM beat leverage and uniform sampling at c = k at every level "
                    "and k (CUR with U = C+AR+, Frobenius, medians)", value=float(bool(good)),
                    verdict="supported" if good else "not_supported"))
    return out


# --------------------------------------------------------------------------------------------------------- build
def build(cfg: Config) -> dict:
    _STATE["deadline"] = time.time() + cfg.max_seconds
    say(f"{SCRIPT_VERSION}: precheck (BLAS threads {os.environ.get('OPENBLAS_NUM_THREADS')})")
    pre = precheck(cfg)
    if cfg.precheck_only:
        return dict(mode="precheck-only", hashes=pre["hashes"])
    say("loading Stage 08 matrices and Layer C landmark tensors")
    D = load_inputs(cfg)
    checks, details = OrderedDict(), OrderedDict()
    st = planted_selftest()
    checks["09_planted_selftest"] = st["ok"]
    say(f"planted self-test: {'pass' if st['ok'] else 'FAIL'}")
    say("stage 09p: exact-duplicate collapse on train rows (D15/D16)")
    S9p = stage09p(D, checks, details)
    check_time()
    say("stage 09: SVD of every level and variant (train rows)")
    P = stage09_points(D, checks, details)
    say(f"stage 09: permutation null ({cfg.n_perm} draws per level)")
    null95 = stage09_null(D, P, cfg)
    say(f"stage 09/10: subject bootstrap, B = {cfg.n_boot} (spectra, Davis-Kahan, principal angles)")
    B9 = stage09_bootstrap(D, P, null95, cfg)
    checks["09_bootstrap_subject_clustered"] = True
    say("stage 09: Layer C per-bin and stacked channel spectra (train prefixes; never flattened)")
    C9 = stage09_layer_c(D, cfg)
    say("stage 10: Eckart-Young checks, randomized SVD, CUR")
    R10 = stage10_lowrank(D, P, S9p, cfg, checks)
    say("stage 10: principal angles between level column spaces, with nulls")
    A10 = stage10_angles(D, P, B9, cfg, checks)
    say("stage 10: linear vs nonlinear recovery of added columns from phi0 (grouped 5-fold CV)")
    V10 = stage10_recovery(D, cfg, checks)
    hyp = hypotheses(P["sm"], B9["boot_rows"], R10["cur_rows"], V10["fam_rows"], P["h1_pre"], P["h1_ded"])
    T = OrderedDict()
    T["stage09p_dedup_map_TEAM.csv"] = pd.DataFrame(S9p["dmap_rows"], columns=["matrix", "kept_column",
                                                                              "dropped_column"])
    T["stage09p_widths_TEAM.csv"] = pd.DataFrame([dict(matrix=k, **v) for k, v in S9p["widths"].items()])
    T["stage09_summary_TEAM.csv"] = pd.DataFrame(P["summ"])
    T["stage09_bootstrap_TEAM.csv"] = pd.DataFrame(B9["boot_rows"])
    T["stage09_spectra_TEAM.csv"] = pd.DataFrame(P["spectra"])
    T["stage09_belsley_TEAM.csv"] = pd.DataFrame(P["bel_rows"], columns=["level", "variant", "direction", "eta",
                                                                          "kind", "columns", "weights"])
    T["stage09_daviskahan_TEAM.csv"] = pd.DataFrame(B9["dk_rows"])
    T["stage09_layer_c_TEAM.csv"] = pd.DataFrame(C9["lc_rows"])
    T["stage09_layer_c_bootstrap_TEAM.csv"] = pd.DataFrame(C9["lc_boot_rows"])
    T["stage10_eckart_young_TEAM.csv"] = pd.DataFrame(R10["ey_rows"])
    T["stage10_rsvd_TEAM.csv"] = pd.DataFrame(R10["rs_rows"])
    T["stage10_cur_TEAM.csv"] = pd.DataFrame(R10["cur_rows"])
    T["stage10_cur_c_needed_TEAM.csv"] = pd.DataFrame(R10["cneed_rows"])
    T["stage10_cpqr_columns_TEAM.csv"] = pd.DataFrame(R10["cpqr_rows"])
    T["stage10_angles_TEAM.csv"] = pd.DataFrame(A10["ang_rows"])
    T["stage10_angles_splithalf_TEAM.csv"] = pd.DataFrame(A10["sh_rows"])
    T["stage10_recovery_TEAM.csv"] = V10["rec"]
    T["stage10_recovery_by_family_TEAM.csv"] = pd.DataFrame(V10["fam_rows"])
    T["stage09_10_hypotheses_TEAM.csv"] = pd.DataFrame(hyp)
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise BuildError("acceptance checks failed: " + "; ".join(
            f"{k} ({', '.join(map(str, details[k][:10]))})" if details.get(k) else k for k in failed))
    return publish(cfg, D, pre, T, R10["bases"], checks, S9p["widths"], P["summ"], B9["boot_rows"], hyp, st)


# ------------------------------------------------------------------------------------------------- release gate
AUDIT_NUM_KEYS = ("width", "n_dropped", "n_dup", "rank", "kappa", "erank", "k95", "stable", "p", "B", "n_boot",
                  "n_perm", "n_cur", "n_splithalf", "n_rsvd", "recovery_max", "k_grid", "landmarks", "core_landmarks",
                  "seed", "threshold", "value", "planted", "perm_null_count", "decay", "n_above", "n_files")
_FORBID_COL = re.compile(r"(rows|subject|encounter|patient|^n$|n_train|n_obs|support|eligible|cohort)", re.I)


def audit_nums_ok(o, key="", inherited=False) -> bool:
    """Shareable audit: every number sits under a structural key (widths, ranks, spectral summaries,
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


def _size_hit(text: str, forbidden_nums: set) -> bool:
    """A forbidden size as a whole number: also as a CSV cell next to commas, comma-grouped ("4,646") or written as a
    float ("1234.0"); never the digits of a fraction or exponent, nor inside a hex digest or a name (fail closed: a
    match inside a thousands-grouped number also counts)."""
    for n in forbidden_nums:
        for form in {str(n), f"{n:,}"}:
            if re.search(rf"(?<![\w.])(?<![eE][+-]){re.escape(form)}(?!\w|\.0*[1-9])", text):
                return True
    return False


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
# identifier: both the size scan and the identifier scan skip them (v1.0.1: sizes; v1.0.2: identifiers too), so a
# rank of 151 never collides with a landmark subset of 151 members or with encounter id 151.
#   width_*, n_dropped, n_dup_kept, p, dim_*            column counts of a matrix (bounded by its width)
#   rank_*, k95_*, decay_m, rank_noise_r                ranks / indices into the spectrum (bounded by the width)
#   n_above_*, n_below_*, n_zero_angles, boot_n_zero_*,
#   new_directions_larger                               counts of singular values / angles (bounded by the width)
#   index, direction, order                             positions in a spectrum or a CPQR order (bounded by width)
#   k, c, c_needed_observed, c_bound_*                  the fixed k / c grid and its DMM order k ln k / eps^2
#   repeats, B, B_angles, power_iterations, oversampling configuration constants
#   landmark, bin                                       landmark hours (24..336) and bin index (< 56)
#   n_columns_evaluated                                 recovery targets per family (capped at 6 by config)
#   value                                               hypothesis values: ratios, ranks or shares (H1-H5)
#   weights                                             Belsley proportions / null-vector loadings, all in [0, 1]
#   eta                                                 Belsley condition index sigma_1 / sigma_j (or "inf")
_STRUCT_COL = re.compile(
    r"^(width_\w+|n_dropped|n_dup_kept|p|rank_\w+|k95_\w+|decay_m|rank_noise_r|n_above_\w+|index|direction|k|c|order|"
    r"repeats|B|B_angles|dim_\w+|n_zero_angles|n_below_\w+|new_directions_larger|c_needed_observed|c_bound_\w+|"
    r"power_iterations|oversampling|landmark|bin|n_columns_evaluated|boot_n_zero_\w+|value|weights|eta)$")
# Text columns written only from the script's fixed sentences, level names and D18 thresholds (hypothesis items):
# numeric tokens there are the script's own constants ("< 100", "80%", "p/3"); word tokens are still scanned.
_SCRIPT_TEXT_COL = re.compile(r"^(item)$")
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
    txt = " ".join(list(DECISIONS.values()) + list(DEVIATIONS))
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
    aj = pend / "stage09_10_r1_audit_TEAM.json"
    audit = json.loads(aj.read_text(encoding="utf-8")) if aj.exists() else None
    audit_struct = _audit_ints(audit, set()) if audit is not None else set()
    allowed = audit_struct | script_text_ints(audit)
    if audit is not None:
        # clock digits of the generation timestamp, and the integers inside the hypothesis sentences, which the
        # script composes only from fixed text, level names and the D18 thresholds (see hypotheses())
        allowed |= {int(t) for t in re.findall(r"\d+", str(audit.get("generated_utc", "")))}
        for h in audit.get("hypotheses", []) or []:
            allowed |= {int(t) for t in id_tokens(str(h.get("item", ""))) if t.isdigit()}
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
                    if not cell:
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


def figures(T: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {}
    sp = T["stage09_spectra_TEAM.csv"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, levs, ttl in ((axes[0], LEVELS_A, "Layer A"), (axes[1], LEVELS_B, "Layer B")):
        for k in levs:
            for var, ls in (("dedup", "-"), ("predup", ":")):
                g = sp[(sp["level"] == k) & (sp["variant"] == var)]
                if len(g):
                    ax.semilogy(g["index"], np.maximum(g["sigma_rel"].astype(float), 1e-12), ls, lw=1.1,
                                label=k if var == "dedup" else None)
        ax.set_title(f"{ttl}: sigma_i / sigma_1 (solid dedup, dotted before dedup)", fontsize=9)
        ax.set_xlabel("index i")
        ax.legend(fontsize=7)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=130)
    plt.close(fig)
    out["fig09_1_spectra_TEAM.png"] = b.getvalue()
    lc = T["stage09_layer_c_TEAM.csv"]
    st = lc[lc["view"] == "stacked"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(st["landmark"], st["erank_sigma2"], "o-", label="effective rank (sigma^2)")
    ax.plot(st["landmark"], st["k95_sigma2"], "s-", label="k95 (sigma^2)")
    ax.set_xlabel("landmark (h)")
    ax.set_title("Layer C stacked channel matrix (train prefixes)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=130)
    plt.close(fig)
    out["fig09_2_layer_c_stacked_TEAM.png"] = b.getvalue()
    return out


def fmt_df(df: pd.DataFrame) -> bytes:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(r6)
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def publish(cfg, D, pre, T, bases, checks, widths, summ, boot_rows, hyp, st) -> dict:
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
        arrs = {}
        for k, b in bases.items():
            key = k.replace("+", "_")
            arrs[f"{key}__V"] = b["V"]
            arrs[f"{key}__cpqr"] = np.asarray(b["cpqr"], dtype=np.int32)
            arrs[f"{key}__columns"] = np.array(b["cols"])
        for Lh, r in D["lc"].items():
            arrs[f"C_L{Lh}__V"] = r["V"].astype(np.float32)
            arrs[f"C_L{Lh}__cpqr"] = np.asarray(r["cpqr"], dtype=np.int32)
            arrs[f"C_L{Lh}__channels"] = np.array(r["chan_d"])
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrs)
        nm = "stage10_matched_width_bases_INTERNAL.npz"
        files[nm] = write_bytes_atomic(pend / nm, buf.getvalue())
        audit = make_audit(cfg, pre, checks, widths, summ, boot_rows, hyp, st, script_sha, len(files))
        if not audit_nums_ok(audit):
            raise BuildError("release gate: the shareable audit contains a number outside the structural whitelist")
        aj = json.dumps(audit, indent=1, ensure_ascii=False).encode("utf-8")
        files["stage09_10_r1_audit_TEAM.json"] = write_bytes_atomic(pend / "stage09_10_r1_audit_TEAM.json", aj)
        md = audit_md(audit).encode("utf-8")
        files["stage09_10_r1_audit_TEAM.md"] = write_bytes_atomic(pend / "stage09_10_r1_audit_TEAM.md", md)
        team += ["stage09_10_r1_audit_TEAM.json", "stage09_10_r1_audit_TEAM.md"]
        back = json.loads((pend / "stage09_10_r1_audit_TEAM.json").read_text(encoding="utf-8"))
        if back != json.loads(aj.decode("utf-8")) or not audit_nums_ok(back):
            raise BuildError("release gate: disk reload of the audit differs")
        release_gate(pend, team, sensitive_numbers(D), set(map(str, D["eid"])) | set(map(str, D["subj"])))
        for nm_, h in files.items():
            if sha256_file(pend / nm_) != h:
                raise BuildError(f"release gate: {nm_} hash differs after writing")
        man = dict(script=SCRIPT_VERSION, script_sha256=script_sha, spec=SPEC_DOC, plan=PLAN, contract=CONTRACT,
                   generated_utc=utc_now(), input_hashes=pre["hashes"],
                   files={k: {"sha256": v, "bytes": (pend / k).stat().st_size} for k, v in files.items()},
                   team_files=team, internal_files=[k for k in files if k not in team],
                   acceptance_checks={k: ("pass" if v else "fail") for k, v in checks.items()}, release_gate="ok")
        write_bytes_atomic(pend / "stage09_10_r1_manifest.json", json.dumps(man, indent=1).encode("utf-8"))
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


def make_audit(cfg, pre, checks, widths, summ, boot_rows, hyp, st, script_sha, n_written) -> dict:
    bmap = {r["level"]: r for r in boot_rows}
    head = []
    for r in summ:
        if r["variant"] != "dedup":
            continue
        b = bmap[r["level"]]
        head.append(OrderedDict(level=r["level"], p=r["p"], rank_numerical=r["rank_numerical"],
                                kappa2=r6(r["kappa2"]), kappa2_over_rank=r6(r["kappa2_over_rank"]),
                                erank_sigma2=r6(r["erank_sigma2"]),
                                erank_sigma2_ci=[r6(b["erank_sigma2_p2_5"]), r6(b["erank_sigma2_p97_5"])],
                                k95_sigma2=r["k95_sigma2"],
                                k95_sigma2_ci=[r6(b["k95_sigma2_p2_5"]), r6(b["k95_sigma2_p97_5"])],
                                n_above_perm_null=r.get("n_above_perm_null"), decay_family=r["decay_family"],
                                decay_family_modal_share=r6(b["decay_family_modal_share"])))
    return OrderedDict(
        script=SCRIPT_VERSION, script_sha256=script_sha, spec=SPEC_DOC, plan=PLAN, contract=CONTRACT,
        generated_utc=utc_now(), input_stage06_08_manifest_sha256=pre["hashes"]["stage06_08_manifest.json"],
        decisions=DECISIONS, deviations=DEVIATIONS,
        config=OrderedDict(seed=SEED, n_boot=cfg.n_boot, n_boot_angles=min(cfg.n_boot_angles, cfg.n_boot), n_perm=cfg.n_perm, n_cur_rep=cfg.n_cur_rep,
                           n_cur_spec_rep=cfg.n_cur_spec_rep, n_splithalf=cfg.n_splithalf, n_rsvd_rep=cfg.n_rsvd_rep,
                           recovery_max_per_family=cfg.recovery_max_per_family, k_grid=list(K_GRID),
                           landmarks=list(LANDMARKS), core_landmarks=list(CORE_LANDMARKS),
                           thresholds_D18=OrderedDict((f"threshold_{k}", v) for k, v in THRESH.items())),
        planted_selftest=OrderedDict(planted_rank=st["planted_rank"], planted_k95_sigma2=st["k95_sigma2"],
                                     perm_null_count=st["perm_null_count"],
                                     belsley_names_range=st["belsley_names_range"]),
        widths=OrderedDict((k, v) for k, v in widths.items()),
        rank_deficiency_predup=OrderedDict((r["level"], r["rank_deficiency"]) for r in summ if r["variant"] == "predup"),
        rank_deficiency_dedup=OrderedDict((r["level"], r["rank_deficiency"]) for r in summ if r["variant"] == "dedup"),
        headline=head,
        hypotheses=[OrderedDict(H=h["hypothesis"], item=h["item"], value=r6(h["value"]), verdict=h["verdict"])
                    for h in hyp],
        acceptance_checks=OrderedDict((k, "pass" if v else "fail") for k, v in checks.items()),
        release_gate="ok", n_files=n_written + 3)


def audit_md(a: dict) -> str:
    L = ["# Stages 09p, 09 and 10 (Run 1) audit - TEAM", "",
         f"- script: {a['script']} (`{a['script_sha256']}`); spec {a['spec']}; plan {a['plan']}, contract {a['contract']}",
         f"- generated (UTC): {a['generated_utc']}; Stage 06-08 manifest `{a['input_stage06_08_manifest_sha256']}`",
         "- aggregate only: no patient, row, encounter or subject counts and no identifiers; the release gate re-read "
         "every TEAM file from disk", "", "## Decisions applied", ""]
    L += [f"- **{k}**: {v}" for k, v in a["decisions"].items()]
    L += ["", "## Deviations from the plan (logged)", ""] + [f"- {d}" for d in a["deviations"]]
    c = a["config"]
    L += ["", "## Configuration", "",
          f"- seed {c['seed']}; subject bootstrap B = {c['n_boot']} (angle intervals on {c['n_boot_angles']}); permutation draws {c['n_perm']}; CUR random "
          f"repeats {c['n_cur_rep']} (spectral norm on {c['n_cur_spec_rep']}); split-half repeats {c['n_splithalf']}; "
          f"randomized SVD repeats {c['n_rsvd_rep']}; recovery cap per (level, family) {c['recovery_max_per_family']}",
          f"- k grid {c['k_grid']}; landmarks {c['landmarks']} (core {c['core_landmarks']})",
          "- D18 thresholds: " + ", ".join(f"{k.replace('threshold_', '')} {v:.4g}"
                                          for k, v in c["thresholds_D18"].items())]
    st = a["planted_selftest"]
    L += ["", "## Planted self-test (synthetic gate inside the script)", "",
          f"- planted rank {st['planted_rank']}: k95 (sigma^2) {st['planted_k95_sigma2']}, permutation-null count "
          f"{st['perm_null_count']}; Belsley names the planted range column: {st['belsley_names_range']}"]
    L += ["", "## 09p widths (columns) before and after the exact-duplicate collapse", "",
          "| matrix | before | after | dropped | __dup__ kept |", "|---|---|---|---|---|"]
    for k, v in a["widths"].items():
        L.append(f"| {k} | {v['width_predup']} | {v['width_dedup']} | {v['n_dropped']} | {v.get('n_dup_kept', '')} |")
    L += ["", "## 09 rank deficiency (p minus numerical rank), before / after dedup", "",
          "| level | before | after |", "|---|---|---|"]
    L += [f"| {k} | {a['rank_deficiency_predup'][k]} | {a['rank_deficiency_dedup'][k]} |"
          for k in a["rank_deficiency_predup"]]
    L += ["", "## 09 headline (deduplicated X_z, train; 95% subject-bootstrap intervals)", "",
          "| level | p | rank | kappa2 | kappa2 over rank | erank sigma^2 [CI] | k95 sigma^2 [CI] | above perm null | "
          "decay family (modal share) |", "|---|---|---|---|---|---|---|---|---|"]
    for h in a["headline"]:
        L.append(f"| {h['level']} | {h['p']} | {h['rank_numerical']} | {h['kappa2']} | {h['kappa2_over_rank']} | "
                 f"{h['erank_sigma2']} [{h['erank_sigma2_ci'][0]}, {h['erank_sigma2_ci'][1]}] | {h['k95_sigma2']} "
                 f"[{h['k95_sigma2_ci'][0]}, {h['k95_sigma2_ci'][1]}] | {h['n_above_perm_null']} | "
                 f"{h['decay_family']} ({h['decay_family_modal_share']}) |")
    L += ["", "## Hypotheses (D18 thresholds; descriptive verdicts for review)", "",
          "| H | item | value | verdict |", "|---|---|---|---|"]
    L += [f"| {h['H']} | {h['item']} | {h['value']} | {h['verdict']} |" for h in a["hypotheses"]]
    L += ["", "## Acceptance checks", ""] + [f"- {k}: {v}" for k, v in a["acceptance_checks"].items()]
    L += ["", f"Release gate: {a['release_gate']}. Detail tables: stage09_*_TEAM.csv and stage10_*_TEAM.csv; "
          "stage10_matched_width_bases_INTERNAL.npz is team-internal (never opened, pasted or shared).", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------ runtime environment (v1.0.1)
CACHE_NAME = "_cache_stage09_10_r1"     # next to the output folder; never a leftover, never scanned by the gate
_ENV_DIRS = OrderedDict([
    ("MPLCONFIGDIR", "matplotlib"),         # matplotlib config + font cache (default ~/.matplotlib)
    ("JOBLIB_TEMP_FOLDER", "joblib"),       # joblib / loky memmapping folder
    ("SCIKIT_LEARN_DATA", "sklearn_data"),  # sklearn data home (default ~/scikit_learn_data)
    ("XDG_CACHE_HOME", "xdg_cache"),        # generic caches (fontconfig etc.)
    ("XDG_CONFIG_HOME", "xdg_config"),
    ("TMP", "tmp"), ("TEMP", "tmp"), ("TMPDIR", "tmp"),   # temp files of this process and any subprocess (loky)
])
HEAVY_IMPORTS = ("sklearn", "sklearn.ensemble", "sklearn.linear_model", "matplotlib", "matplotlib.pyplot")
_PIP = {"sklearn": "scikit-learn", "matplotlib": "matplotlib"}


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
    ap.add_argument("--canonical", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--expect-stage06-08-manifest-sha256", default=None)
    ap.add_argument("--precheck-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-boot-angles", type=int, default=50)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    ap.add_argument("--heartbeat-seconds", type=float, default=30.0)
    a = ap.parse_args(argv)
    root = Path(a.project_root)
    lad = root / "data" / "processed" / "ladder"
    return Config(project_root=root, in_dir=Path(a.in_dir) if a.in_dir else lad / "stage06_08",
                  canonical=Path(a.canonical) if a.canonical else root / "data" / "interim" /
                  "canonical_icu_encounters.csv",
                  out_dir=Path(a.out_dir) if a.out_dir else lad / OUT_NAME,
                  expect_manifest_sha256=(a.expect_stage06_08_manifest_sha256 or "").strip().lower() or None,
                  precheck_only=a.precheck_only, n_boot=a.n_boot, n_boot_angles=a.n_boot_angles, n_perm=a.n_perm, max_seconds=a.max_seconds,
                  heartbeat_seconds=a.heartbeat_seconds)


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
        print("result: PRECHECK OK (Stage 06-08 manifest, every file it lists, canonical file, packages); "
              "nothing written (library cache folder only)")
        print(f"stage06_08_manifest.json sha256 {h['stage06_08_manifest.json']}")
        print(f"canonical sha256 {h['canonical']}")
        return 0
    print(res["audit_md"])
    print(f"elapsed {time.time() - t0:.0f} s")
    print("result: RUN 1 (09p, 09, 10) OK; RELEASE GATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
