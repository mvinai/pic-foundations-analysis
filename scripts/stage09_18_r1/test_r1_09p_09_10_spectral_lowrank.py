"""Tests for r1_09p_09_10_spectral_lowrank v1.0.0 (Run 1: stages 09p, 09, 10).

Self-contained: every test builds its own tiny SYNTHETIC Stage 06-08 release in a pytest tmp folder (no real data,
no network). Run with:  python -m pytest -q test_r1_09p_09_10_spectral_lowrank.py
"""
from __future__ import annotations

import hashlib
import os
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "r1_09p_09_10_spectral_lowrank.py"


def _load():
    spec = importlib.util.spec_from_file_location("r1_mod_under_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["r1_mod_under_test"] = m
    spec.loader.exec_module(m)
    return m


M = _load()
FAST = ["--n-boot", "6", "--n-boot-angles", "3", "--n-perm", "6", "--heartbeat-seconds", "600"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------ synthetic fixture
def make_release(root: Path, n_subj: int = 260, seed: int = 11, small_ids: bool = False) -> str:
    """Write a tiny synthetic Stage 06-08 v1.2.1-shaped release + canonical file under root; return manifest sha."""
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n_subj), rng.integers(1, 4, n_subj))
    n = len(subj)
    eid = np.array([f"S{100000 + s}_{i}" for i, s in enumerate(subj)])
    if small_ids:                       # v1.0.2: real PIC-like ids, small integers 1..N (maximal collisions)
        eid = np.array([str(i + 1) for i in range(n)])
    u = rng.random(n_subj)
    split = np.where(u[subj] < 0.7, "train", np.where(u[subj] < 0.85, "validation", "test"))
    los = rng.integers(6, 400, n)
    elig24 = los >= 24
    Z = rng.standard_normal((n, 6))

    def block(prefix, width, extra_kinds=True):
        W = rng.standard_normal((6, width))
        X = Z @ W + 0.5 * rng.standard_normal((n, width))
        cols = [f"{prefix}_v{j:03d}" for j in range(width)]
        kinds = ["cont"] * width
        if extra_kinds:
            m = (rng.random(n) < 0.3).astype(float)
            X = np.c_[X, m, m]                      # an exact duplicate pair (mask + its copy) -> 09p drops one
            cols += [f"{prefix}_lab_missing", f"{prefix}_lab_missing_copy"]
            kinds += ["mask", "bin"]
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
        return X, cols, kinds

    lvl = {}
    base_a, ca, ka = block("a", 14)
    add1, c1, k1 = block("x1", 4, False)
    add2, c2, k2 = block("x2", 3, False)
    A = {"a0": (base_a, ca, ka), "a1": (np.c_[base_a, add1], ca + c1, ka + k1),
         "a2": (np.c_[base_a, add1, add2], ca + c1 + c2, ka + k1 + k2)}
    for k, (X, c, kd) in A.items():
        cols = [f"{k}_{x}" for x in c]
        oh = (rng.random(n) < 0.5).astype(float)
        oh = (oh - oh.mean()) / oh.std()
        lvl[k] = dict(X=X, cols=cols, kinds=kd, Xf=np.c_[X, -oh], cols_full=cols + [f"{k}_unit__other"])
    base_b, cb, kb = block("b", 16)
    addb, cb1, kb1 = block("y1", 5, False)
    addb2, cb2, kb2 = block("y2", 4, False)
    B = {"b0": (base_b, cb, kb), "b1": (np.c_[base_b, addb], cb + cb1, kb + kb1),
         "b2s": (np.c_[base_b, addb, addb2], cb + cb1 + cb2, kb + kb1 + kb2)}
    Xs, cs, ks = B["b2s"]
    dup_idx = [0, 3, 5]
    B["b2d"] = (np.c_[Xs, Xs[:, dup_idx]], cs + [f"organ__dup__{cs[j]}" for j in dup_idx], ks + [ks[j] for j in dup_idx])
    pp, cp, kp = block("p", 9, False)
    B["b2p"] = (pp, cp, kp)
    for k, (X, c, kd) in B.items():
        cols = [f"{k}_{x}" for x in c]
        lvl[k] = dict(X=X, cols=cols, kinds=kd, Xf=X, cols_full=cols)
    d = root / "data" / "processed" / "ladder" / "stage06_08"
    d.mkdir(parents=True)
    files = {}
    for k, L in lvl.items():
        p = d / f"stage08_{k}.npz"
        np.savez(p, encounter_id=eid, split=split, eligible_24h=elig24, columns=np.array(L["cols"]),
                 kinds=np.array(L["kinds"]), X_imputed=L["X"], X_z=L["X"], columns_full=np.array(L["cols_full"]),
                 X_full_z=L["Xf"])
    q = 5
    for Lh in M.LANDMARKS:
        m = los >= Lh
        b = Lh // 6
        T = rng.standard_normal((m.sum(), b, q)) + Z[m][:, None, :q]
        np.savez(d / f"stage08_layer_c_L{Lh}.npz", encounter_id=eid[m], split=split[m],
                 channels=np.array([f"ch{j}" for j in range(q)]), X_raw_z=T, X_filled_z=T,
                 missing=np.zeros_like(T, bool), carried_forward=np.zeros_like(T, bool), imputed=np.zeros_like(T, bool))
    for j in range(M.PINS["stage06_08_files"] - 22):
        (d / f"filler_{j:02d}.txt").write_text(f"synthetic filler {j}\n")
    for p in sorted(d.iterdir()):
        files[p.name] = {"sha256": sha(p), "bytes": p.stat().st_size}
    can = root / "data" / "interim" / "canonical_icu_encounters.csv"
    can.parent.mkdir(parents=True)
    pd.DataFrame({"encounter_id": eid, "SUBJECT_ID": [str(s + 1) if small_ids else str(100000 + s) for s in subj], "Died": 0}).to_csv(can, index=False)
    man = {"script": M.PINS["stage06_08_script"], "script_sha256": M.PINS["stage06_08_script_sha256"],
           "release_gate": "ok", "acceptance_checks": {"synthetic": "pass"}, "files": files,
           "input_hashes": {"canonical": sha(can)}}
    mp = d / "stage06_08_manifest.json"
    mp.write_text(json.dumps(man, indent=1))
    return sha(mp)


@pytest.fixture(autouse=True)
def _no_canonical_pin(monkeypatch):
    monkeypatch.setitem(M.PINS, "canonical", None)      # synthetic canonical file; the real pin is tested below


@pytest.fixture
def rel(tmp_path):
    return tmp_path, make_release(tmp_path)


def run(root, man, *extra):
    return M.main(["--project-root", str(root), "--expect-stage06-08-manifest-sha256", man, *FAST, *extra])


OUT = Path("data") / "processed" / "ladder" / "stage09_10_r1"


# ------------------------------------------------------------------------------------------------- hash gate
def test_precheck_ok(rel, capsys):
    root, man = rel
    assert run(root, man, "--precheck-only") == 0
    assert "PRECHECK OK" in capsys.readouterr().out
    assert not (root / OUT).exists()


def test_manifest_sha_mismatch_stops(rel):
    root, man = rel
    assert run(root, "0" * 64, "--precheck-only") == 2


def test_manifest_sha_required(rel):
    root, _ = rel
    assert M.main(["--project-root", str(root), "--precheck-only"]) == 2


def test_tampered_listed_file_stops(rel):
    root, man = rel
    p = root / "data/processed/ladder/stage06_08/filler_03.txt"
    p.write_text("tampered\n")
    assert run(root, man, "--precheck-only") == 2


def test_extra_file_in_input_folder_stops(rel):
    root, man = rel
    (root / "data/processed/ladder/stage06_08/extra.csv").write_text("x\n")
    assert run(root, man, "--precheck-only") == 2


def test_desktop_ini_ignored(rel):
    root, man = rel
    (root / "data/processed/ladder/stage06_08/desktop.ini").write_text("[x]\n")
    assert run(root, man, "--precheck-only") == 0


def test_canonical_hash_mismatch_stops(rel):
    root, man = rel
    can = root / "data/interim/canonical_icu_encounters.csv"
    can.write_text(can.read_text() + "\n")
    assert run(root, man, "--precheck-only") == 2


def test_canonical_plan_pin_enforced(rel, monkeypatch):
    root, man = rel
    monkeypatch.setitem(M.PINS, "canonical", "f" * 64)
    assert run(root, man, "--precheck-only") == 2


def test_nonempty_output_folder_refused(rel):
    root, man = rel
    (root / OUT).mkdir(parents=True)
    (root / OUT / "old.txt").write_text("x")
    assert run(root, man, "--precheck-only") == 2


def test_pending_folder_refused(rel):
    root, man = rel
    (root / OUT.parent / "stage09_10_r1.pending-123").mkdir(parents=True)
    assert run(root, man, "--precheck-only") == 2


# --------------------------------------------------------------------------------------------- numerical units
def test_planted_selftest():
    st = M.planted_selftest()
    assert st["ok"] and st["k95_sigma2"] == 10 and st["perm_null_count"] == 10 and st["belsley_names_range"]


def test_belsley_names_range_column():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((500, 4))
    B = np.c_[B, B[:, 0] - B[:, 1]]
    U, s, Vt = M.thin_svd(B)
    rows = M.belsley(s, Vt, ["mx", "mn", "c", "d", "range"], B.shape, 30.0, 0.5)
    hit = [r for r in rows if r["kind"] == "exact_null"]
    assert hit and {"mx", "mn", "range"} <= set(hit[0]["columns"].split(";"))


def test_dedup_and_b2d_exemption():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, 3))
    X = np.c_[x, x[:, 0], x[:, 1]]
    cols = ["a", "b", "c", "a_copy", "o__dup__b"]
    keep, groups = M.dedup_columns(X, cols, exempt=lambda c: "__dup__" in c)
    assert keep == [0, 1, 2, 4]
    assert groups == [("a", ["a_copy"])]
    keep2, _ = M.dedup_columns(X, cols)
    assert keep2 == [0, 1, 2]


def test_eckart_young_and_rsvd_bound():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((300, 40)) @ np.diag(np.linspace(5, 0.1, 40))
    U, s, Vt = M.thin_svd(A)
    for k in (5, 10, 20):
        Ak = (U[:, :k] * s[:k]) @ Vt[:k]
        assert np.isclose(np.linalg.norm(A - Ak, 2), s[k], rtol=1e-8)
        assert np.isclose(np.linalg.norm(A - Ak), np.sqrt((s[k:] ** 2).sum()), rtol=1e-8)


def test_cur_ratio_at_least_one():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((200, 30)) @ rng.standard_normal((30, 30))
    U, s, Vt = M.thin_svd(A)
    k = 5
    J = list(M.cpqr_order(A)[:k])
    I = list(M.deim(U[:, :k]))
    e = M.cur_errors(A, J, I, float(np.sqrt((s[k:] ** 2).sum())), float(s[k]))
    assert all(v >= 1 - 1e-9 for v in e.values() if np.isfinite(v))


def test_spec_norm_matches_svd():
    A = np.random.default_rng(4).standard_normal((400, 120))
    assert np.isclose(M.spec_norm(A), np.linalg.svd(A, compute_uv=False)[0], rtol=1e-10)


def test_r_svd_matches_svd():
    A = np.random.default_rng(5).standard_normal((400, 50))
    s, Vt = M.r_svd(A)
    assert np.allclose(s, np.linalg.svd(A, compute_uv=False), rtol=1e-10)


def test_principal_angles_identity_and_orthogonal():
    Q = np.linalg.qr(np.random.default_rng(6).standard_normal((100, 8)))[0]
    assert np.allclose(M.principal_angles(Q[:, :4], Q[:, :4]), 0, atol=1e-6)
    assert np.allclose(M.principal_angles(Q[:, :4], Q[:, 4:]), np.pi / 2, atol=1e-6)


# ------------------------------------------------------------------------------------------------ release gate
def test_release_gate_catches_sizes_and_ids():
    f = M.team_text_problems
    ids = {"100001", "S100001_0"}
    assert f("x,4646,y", {4646}, set())
    assert f("x,4,646", {4646}, set())
    assert f("x,4646.0", {4646}, set())
    assert not f("x,0.4646,y", {4646}, set())
    assert f("subject,100001\n", set(), ids)
    assert f("S100001_0", set(), ids)
    assert not f("v,0.100001,1.00001e+05", set(), ids)
    assert not f("sha 9af4646e0", {4646}, set())          # never inside a hex digest


def test_release_gate_structural_columns_and_audit(tmp_path):
    (tmp_path / "x_TEAM.csv").write_text("level,rank_numerical,ratio\na0,291,0.5\n")
    M.release_gate(tmp_path, ["x_TEAM.csv"], {291}, set())          # a rank equal to a size is structural
    (tmp_path / "y_TEAM.csv").write_text("level,ratio\na0,291\n")
    with pytest.raises(M.BuildError):
        M.release_gate(tmp_path, ["y_TEAM.csv"], {291}, set())
    (tmp_path / "z_TEAM.csv").write_text("level,n_train\na0,0.5\n")
    with pytest.raises(M.BuildError):
        M.release_gate(tmp_path, ["z_TEAM.csv"], set(), set())
    (tmp_path / "w_TEAM.csv").write_text("level,column\nS100001_0,x\n")
    with pytest.raises(M.BuildError):
        M.release_gate(tmp_path, ["w_TEAM.csv"], set(), {"S100001_0"})


def test_audit_number_whitelist():
    assert M.audit_nums_ok({"widths": {"a0": {"width_dedup": 12}}})
    assert not M.audit_nums_ok({"train_count": 4646})


# ------------------------------------------------------------------------------------------- end to end (synthetic)
@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setitem(M.PINS, "canonical", None)
    outs = []
    for rep in range(2):
        root = tmp_path_factory.mktemp(f"e2e{rep}")
        man = make_release(root)
        rc = run(root, man)
        outs.append((root, rc))
    mp.undo()
    return outs


def test_e2e_runs_and_publishes(e2e):
    root, rc = e2e[0]
    assert rc == 0
    out = root / OUT
    man = json.loads((out / "stage09_10_r1_manifest.json").read_text())
    assert not list(out.parent.glob("stage09_10_r1.pending-*"))
    for nm, rec in man["files"].items():
        assert sha(out / nm) == rec["sha256"]
    assert all(n.endswith("_TEAM.csv") or n.endswith("_TEAM.png") or n.endswith("_TEAM.json") or n.endswith("_TEAM.md")
               for n in man["team_files"])
    assert all("_INTERNAL" in n for n in man["internal_files"])
    a = json.loads((out / "stage09_10_r1_audit_TEAM.json").read_text())
    assert a["release_gate"] == "ok" and all(v == "pass" for v in a["acceptance_checks"].values())
    assert a["acceptance_checks"]["09p_b2d_dup_copies_never_dropped"] == "pass"
    assert a["acceptance_checks"]["09_H1_rank_identity_dedup"] == "pass"


def test_e2e_team_files_have_no_sizes_or_ids(e2e):
    root, _ = e2e[0]
    out = root / OUT
    can = pd.read_csv(root / "data/interim/canonical_icu_encounters.csv", dtype=str)
    ids = set(can["encounter_id"]) | set(can["SUBJECT_ID"])
    z = np.load(root / "data/processed/ladder/stage06_08/stage08_a0.npz")
    sp = z["split"].astype(str)
    sizes = {len(sp)} | {int((sp == s).sum()) for s in ("train", "validation", "test")}
    sizes |= {len(set(can["SUBJECT_ID"]))}
    man = json.loads((out / "stage09_10_r1_manifest.json").read_text())
    for nm in man["team_files"]:
        if nm.endswith(".png"):
            continue
        text = (out / nm).read_text(encoding="utf-8")
        assert not M.team_text_problems(text, {s for s in sizes if s >= 100}, ids), nm
        if nm.endswith(".csv"):
            for c in pd.read_csv(out / nm, nrows=0).columns:
                assert not re.search(r"(rows|subject|encounter|patient|^n$|cohort)", c, re.I), (nm, c)


def test_e2e_deterministic(e2e):
    (r0, rc0), (r1, rc1) = e2e
    assert rc0 == 0 and rc1 == 0
    m0 = json.loads((r0 / OUT / "stage09_10_r1_manifest.json").read_text())
    m1 = json.loads((r1 / OUT / "stage09_10_r1_manifest.json").read_text())
    for nm in m0["team_files"]:
        if nm.endswith((".png", "_audit_TEAM.json", "_audit_TEAM.md")):
            continue                                  # figures and audits carry timestamps / renderer metadata
        assert m0["files"][nm]["sha256"] == m1["files"][nm]["sha256"], nm


def test_e2e_rerun_into_existing_output_refused(e2e):
    root, _ = e2e[0]
    man = sha(root / "data/processed/ladder/stage06_08/stage06_08_manifest.json")
    assert run(root, man) == 2


# ----------------------------------------------------------------------------------------- v1.0.1 additions
def _boom_lib():
    json.loads("{not json")                      # raises inside the json library (library frames)


def _boom_outer():
    _boom_lib()


def test_describe_exception_frames_only():
    try:
        _boom_outer()
    except Exception as e:  # noqa: BLE001
        lines = M.describe_exception(e, None)
    assert lines[0] == "unexpected JSONDecodeError"
    frames = [ln for ln in lines if ln.startswith("  frame ")]
    assert any(ln.startswith("  frame test_r1_09p_09_10_spectral_lowrank.py:") and ln.endswith("_boom_outer [r1_test]")
               or "_boom_outer" in ln for ln in frames)
    assert any("[json.decoder]" in ln or "[json]" in ln for ln in frames)       # library frames name their module
    for ln in frames:
        loc = ln.split()[1]
        assert "/" not in loc and "\\" not in loc                               # basenames only
        assert re.fullmatch(r"[^:]+:\d+:[\w.<>]+", loc), loc
    assert not any("not json" in ln or "Expecting" in ln for ln in lines)       # no exception messages


def test_describe_exception_chain_and_oserror_redaction(tmp_path):
    root = tmp_path / "proj"
    (root / "data").mkdir(parents=True)
    secret = root / "data" / "interim" / "x.csv"
    try:
        try:
            raise PermissionError(13, "Permission denied", str(secret))
        except PermissionError as inner:
            raise RuntimeError("wrapper with a message that must not print") from inner
    except RuntimeError as e:
        lines = M.describe_exception(e, root)
    text = "\n".join(lines)
    assert "caused by PermissionError" in text
    assert "must not print" not in text and "Permission denied" not in text
    assert str(tmp_path) not in text
    assert re.search(r"file data[\\/]interim[\\/]x\.csv", text)
    home_file = Path.home() / ".matplotlib" / "deep" / "fontlist.json"
    red = M.redact_path(home_file, root)
    assert red.startswith("~") and red.endswith("fontlist.json") and "deep" not in red


def test_main_unexpected_error_prints_frames(rel, monkeypatch, capsys):
    root, man = rel

    def bad_build(cfg):
        raise PermissionError(13, "denied", str(Path.home() / "x" / "secret.bin"))
    monkeypatch.setattr(M, "build", bad_build)
    assert run(root, man) == 3
    err = capsys.readouterr().err
    assert "unexpected PermissionError" in err and "frame " in err and "bad_build" in err
    assert "secret.bin" in err and str(Path.home()) not in err and "denied" not in err


def test_runtime_env_setup_and_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", "/somewhere/else")
    monkeypatch.delenv("JOBLIB_TEMP_FOLDER", raising=False)
    old_tmp = tempfile_gettempdir()
    cache = tmp_path / "data/processed/ladder/_cache_stage09_10_r1"
    prev = M.setup_runtime_env(cache)
    try:
        for k in ("MPLCONFIGDIR", "JOBLIB_TEMP_FOLDER", "SCIKIT_LEARN_DATA", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
                  "TMP", "TEMP", "TMPDIR"):
            assert Path(os.environ[k]).is_dir() and Path(os.environ[k]).resolve().is_relative_to(cache.resolve()), k
        assert Path(tempfile_gettempdir()).resolve().is_relative_to(cache.resolve())
    finally:
        M.restore_runtime_env(prev)
    assert os.environ["MPLCONFIGDIR"] == "/somewhere/else"
    assert "JOBLIB_TEMP_FOLDER" not in os.environ
    assert tempfile_gettempdir() == old_tmp


def tempfile_gettempdir():
    import tempfile
    return tempfile.gettempdir()


def test_import_failure_is_reported(rel, monkeypatch, capsys):
    root, man = rel
    monkeypatch.setattr(M, "HEAVY_IMPORTS", ("sklearn", "sklearn_not_a_real_module_xyz"))
    monkeypatch.setitem(M._PIP, "sklearn_not_a_real_module_xyz", "nothing")
    assert run(root, man, "--precheck-only") == 2
    assert "import of sklearn_not_a_real_module_xyz failed" in capsys.readouterr().err


def test_cache_folder_is_not_a_leftover(rel):
    root, man = rel
    assert run(root, man, "--precheck-only") == 0
    cache = root / OUT.parent / "_cache_stage09_10_r1"
    assert cache.is_dir() and (cache / "matplotlib").is_dir()
    assert run(root, man, "--precheck-only") == 0                      # a second precheck is not blocked by it
    assert not list((cache / "tmp").iterdir())                         # temp emptied after the run


def test_read_npz_closes_handle(tmp_path):
    p = tmp_path / "a.npz"
    np.savez(p, x=np.arange(3))
    d = M.read_npz(p, ("x",))
    os.replace(p, tmp_path / "b.npz")                                  # would fail on Windows if left open
    assert d["x"].tolist() == [0, 1, 2]


def test_rmtree_robust_read_only(tmp_path):
    d = tmp_path / "pend"
    d.mkdir()
    f = d / "x.txt"
    f.write_text("x")
    os.chmod(f, 0o400)
    M.rmtree_robust(d)
    assert not d.exists()


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="POSIX non-root only: simulates a read-only user profile")
def test_precheck_with_read_only_home(tmp_path):
    import subprocess
    root = tmp_path / "proj"
    root.mkdir()
    man = make_release(root)
    home = tmp_path / "rohome"
    home.mkdir()
    os.chmod(home, 0o500)
    drv = tmp_path / "drv.py"
    drv.write_text(
        "import sys, importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('r1x', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['r1x'] = m; spec.loader.exec_module(m)\n"
        "m.PINS['canonical'] = None\n"
        "sys.exit(m.main(sys.argv[1:]))\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XDG_", "MPL", "JOBLIB", "SCIKIT"))}
    import numpy, scipy, pandas, sklearn, matplotlib                 # keep the same packages visible: with a new
    sites = {str(Path(m.__file__).resolve().parents[1]) for m in (numpy, scipy, pandas, sklearn, matplotlib)}
    env.update(HOME=str(home), USERPROFILE=str(home), TMPDIR=str(home / "no_tmp"),   # HOME the user site is gone
               PYTHONPATH=os.pathsep.join(sorted(sites)), PYTHONNOUSERSITE="1")
    try:
        r = subprocess.run([sys.executable, str(drv), "--project-root", str(root),
                            "--expect-stage06-08-manifest-sha256", man, "--precheck-only"],
                           env=env, capture_output=True, text=True, timeout=300)
    finally:
        os.chmod(home, 0o700)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "PRECHECK OK" in r.stdout
    assert not any(home.iterdir())                                     # nothing written to the profile


# ----------------------------------------------------------------------------------------- v1.0.2 additions
@pytest.fixture(scope="module")
def e2e_small(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setitem(M.PINS, "canonical", None)
    root = tmp_path_factory.mktemp("e2e_small_ids")
    man = make_release(root, small_ids=True)
    rc = run(root, man)
    mp.undo()
    can = pd.read_csv(root / "data/interim/canonical_icu_encounters.csv", dtype=str)
    ids = set(can["encounter_id"]) | set(can["SUBJECT_ID"])
    return root, rc, ids


def test_e2e_small_integer_ids_pass_gate(e2e_small):
    root, rc, ids = e2e_small
    assert rc == 0                                                  # gate passes although ids are 1..N
    out = root / OUT
    man = json.loads((out / "stage09_10_r1_manifest.json").read_text())
    assert man["release_gate"] == "ok"
    assert "151" in ids and "5" in ids                              # collisions with widths / k are certain
    assert not M.gate_problems(out, man["team_files"], set(), ids)


def _leak_copy(src: Path, dst: Path):
    import shutil
    shutil.copytree(src, dst)
    return dst


def test_planted_leaks_reported_without_values(e2e_small, tmp_path):
    root, _, ids = e2e_small
    out = _leak_copy(root / OUT, tmp_path / "leak")
    man = json.loads((out / "stage09_10_r1_manifest.json").read_text())
    a = json.loads((out / "stage09_10_r1_audit_TEAM.json").read_text())
    allowed = M._audit_ints(a, set()) | M.script_text_ints(a)
    leak = max((i for i in ids if i.isdigit() and int(i) not in allowed and len(i) >= 3), key=int)
    rec = out / "stage10_recovery_TEAM.csv"
    df = pd.read_csv(rec, dtype=str, keep_default_na=False)
    df.loc[0, "column"] = leak                                      # an id in a non-structural text column
    rec.write_text(df.to_csv(index=False, lineterminator="\n"))
    md = out / "stage09_10_r1_audit_TEAM.md"
    md.write_text(md.read_text() + f"\nnote: encounter {leak} reviewed\n")   # an id in md free text
    probs = M.gate_problems(out, man["team_files"], set(), ids)
    assert ("stage10_recovery_TEAM.csv", "column column", M._ID_MSG) in probs
    assert ("stage09_10_r1_audit_TEAM.md", "md text", M._ID_MSG) in probs
    with pytest.raises(M.BuildError) as ei:
        M.release_gate(out, man["team_files"], set(), ids)
    msg = str(ei.value)
    assert "stage10_recovery_TEAM.csv [column column]: an identifier" in msg
    assert "stage09_10_r1_audit_TEAM.md [md text]: an identifier" in msg
    assert leak not in msg                                          # the value is never printed


def test_integer_cell_vs_float_cell_rule(tmp_path):
    ids = {"3", "151"}
    (tmp_path / "a_TEAM.csv").write_text("level,ratio,note\na0,3.0,x\na1,151.0,y\n")
    assert not M.gate_problems(tmp_path, ["a_TEAM.csv"], set(), ids)        # floats are never id candidates
    (tmp_path / "b_TEAM.csv").write_text("level,ratio,note\na0,3,x\n")
    assert M.gate_problems(tmp_path, ["b_TEAM.csv"], set(), ids) == [("b_TEAM.csv", "column ratio", M._ID_MSG)]
    (tmp_path / "c_TEAM.csv").write_text("level,width_dedup,rank_numerical,k\na0,151,3,3\n")
    assert not M.gate_problems(tmp_path, ["c_TEAM.csv"], set(), ids)        # structural columns exempt


def test_gate_collects_all_problems_and_sizes_located(tmp_path):
    (tmp_path / "a_TEAM.csv").write_text("level,ratio\na0,4646\n")
    (tmp_path / "b_TEAM.csv").write_text("level,n_train\n7,0.5\n")
    probs = M.gate_problems(tmp_path, ["a_TEAM.csv", "b_TEAM.csv"], {4646}, {"7"})
    assert ("a_TEAM.csv", "column ratio", M._SIZE_MSG) in probs
    assert ("b_TEAM.csv", "column names", "forbidden column name (count or identifier)") in probs
    assert ("b_TEAM.csv", "column level", M._ID_MSG) in probs
    with pytest.raises(M.BuildError) as ei:
        M.release_gate(tmp_path, ["a_TEAM.csv", "b_TEAM.csv"], {4646}, {"7"})
    assert "3 problem(s)" in str(ei.value) and "4646" not in str(ei.value)


def test_json_key_path_reported(tmp_path):
    (tmp_path / "stage09_10_r1_audit_TEAM.json").write_text(json.dumps({"widths": {"a0": {"width_dedup": 151}},
                                                                         "note": "encounter 98765"}))
    probs = M.gate_problems(tmp_path, ["stage09_10_r1_audit_TEAM.json"], set(), {"151", "98765"})
    assert probs == [("stage09_10_r1_audit_TEAM.json", "json key $.note", M._ID_MSG)]
