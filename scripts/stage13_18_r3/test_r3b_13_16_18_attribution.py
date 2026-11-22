"""Tests for r3b_13_16_18_attribution v1.0.0 (Run 3b: stage 13 attribution stability, stage 18 sequence attribution,
stage 16 Layer C tensor SVD).

Self-contained and SYNTHETIC only: each fixture builds a tiny Stage 06-08 / Run 1-shaped project with the Run 2 test
helper (make_release from test_r2_11p_11_12_17_models.py, hash-checked), runs the unmodified Run 2 script on it with
the real pins nulled, then runs Run 3b. No real data, no network. Run with:
    python -s -m pytest -q -p no:cacheprovider test_r3b_13_16_18_attribution.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import itertools
import json
import math
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
R2DIR = next(p for p in (HERE.parent / "stage11_18_r2", HERE.parent / "r2") if (p / "r2_11p_11_12_17_models.py").is_file())
R2_SCRIPT = R2DIR / "r2_11p_11_12_17_models.py"
R2_TEST = R2DIR / "test_r2_11p_11_12_17_models.py"
R2_TEST_SHA = "a26ca8810994e50cc8bdaacfa4bded4d3ca7f63fc6675f52dcdf50b02f6d8d7b"
FAST2 = ["--n-boot", "20", "--gru-max-epochs", "2", "--heartbeat-seconds", "600"]
FAST3 = ["--s13-refits", "3", "--s13-tree-refits", "2", "--s18-refits", "2", "--s16-boot", "1",
         "--s16-cp-starts", "1", "--gru-max-epochs", "2", "--heartbeat-seconds", "600"]
NULL2 = ("canonical", "harmonization", "split", "ICUSTAYS", "ADMISSIONS", "run1_manifest", "run1_bases",
         "raw_rows", "valid_stays")
LAD = Path("data") / "processed" / "ladder"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


assert sha(R2_TEST) == R2_TEST_SHA, "the Run 2 test helper differs from Run 2 v1.0.0"
M3 = _load("r3b_mod_under_test", HERE / "r3b_13_16_18_attribution.py")
R2 = M3.load_run2(R2_SCRIPT, M3.PINS3["run2_script_sha256"])[0]
MT = _load("r2_test_helper_for_r3b", R2_TEST)          # make_release only
OUTN = {p: M3.PARTS[p]["out"] for p in M3.PARTS}      # part -> output folder name
OUTP = {p: LAD / OUTN[p] for p in M3.PARTS}
OUT = OUTP["1"]
OUT2 = LAD / M3.RUN2_NAME


@pytest.fixture(autouse=True)
def _synthetic_pins(monkeypatch):
    for k in NULL2:
        monkeypatch.setitem(R2.PINS, k, None)
    monkeypatch.setitem(M3.PINS3, "run2_manifest", None)


def _null_pins():
    mp = pytest.MonkeyPatch()
    for k in NULL2:
        mp.setitem(R2.PINS, k, None)
    mp.setitem(M3.PINS3, "run2_manifest", None)
    return mp


def make_project(base: Path, small_ids: bool = False) -> dict:
    """Synthetic project with a published Run 2 (unmodified script, pins nulled) and the laptop layout: split file
    and raw tables copied under data/interim/r2_inputs (the Copilot sandbox reads only the project)."""
    r = MT.make_release(base, small_ids=small_ids)
    root = r["root"]
    rc = R2.main(["--project-root", str(root), "--expect-stage06-08-manifest-sha256", r["man"],
                  "--expect-run1-manifest-sha256", r["r1"], *FAST2])
    assert rc == 0, "Run 2 failed on the synthetic project"
    ri = root / "data" / "interim" / "r2_inputs"
    (ri / "raw").mkdir(parents=True)
    shutil.copy2(base / "PIC_Analysis_3_0" / "data" / "processed" / "subject_splits.csv", ri / "subject_splits.csv")
    for t in ("ICUSTAYS", "ADMISSIONS"):
        shutil.copy2(base / "PICDataset" / f"{t}.csv", ri / "raw" / f"{t}.csv")
    r["r2"] = sha(root / OUT2 / f"{M3.RUN2_NAME}_manifest.json")
    return r


def copy_project(r: dict, dest: Path) -> dict:
    shutil.copytree(r["root"], dest / "proj", ignore=shutil.ignore_patterns("_cache_*"))
    return dict(r, root=dest / "proj")


def args3(r, *extra, r2=None, script=None):
    root = r["root"]
    return ["--project-root", str(root), "--expect-stage06-08-manifest-sha256", r["man"],
            "--expect-run1-manifest-sha256", r["r1"], "--expect-run2-manifest-sha256", r2 or r["r2"],
            "--run2-script", str(script or R2_SCRIPT),
            "--split-file", str(root / "data/interim/r2_inputs/subject_splits.csv"),
            "--raw-dir", str(root / "data/interim/r2_inputs/raw"), *FAST3, *extra]


def run3(r, *extra, part="1", **kw) -> int:
    return M3.main(["--part", part, *args3(r, *extra, **kw)])


class RecordingFitLog(M3.FitLog):
    made = []

    def __init__(self, split):
        super().__init__(split)
        RecordingFitLog.made.append(self)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    mp = _null_pins()
    r = make_project(tmp_path_factory.mktemp("r3b_e2e"))
    pristine = copy_project(r, tmp_path_factory.mktemp("r3b_pristine"))
    RecordingFitLog.made.clear()
    mp.setattr(M3, "FitLog", RecordingFitLog)
    rc = {"2": run3(r, part="2")}                       # part 2 first: it needs nothing from part 1
    rc["1"] = run3(r, part="1")
    logs = list(RecordingFitLog.made)
    mp.undo()
    return dict(r=r, rc=rc, logs=logs, pristine=pristine)


@pytest.fixture(scope="module")
def e2e_small(tmp_path_factory):
    mp = _null_pins()
    r = make_project(tmp_path_factory.mktemp("r3b_small"), small_ids=True)
    rc = {p: run3(r, part=p) for p in ("1", "2")}
    mp.undo()
    return r, rc, set(map(str, r["eid"])) | set(map(str, r["sid"]))


def fresh(e2e, tmp_path) -> dict:
    return copy_project(e2e["pristine"], tmp_path)


# ------------------------------------------------------------------------------- home-made SHAP: exactness
def _shapley(v, p):
    phi = np.zeros(p)
    for k in range(p):
        oth = [e for e in range(p) if e != k]
        for r in range(p):
            for S in itertools.combinations(oth, r):
                w = math.factorial(r) * math.factorial(p - r - 1) / math.factorial(p)
                phi[k] += w * (v(set(S) | {k}) - v(set(S)))
    return phi


@pytest.fixture(scope="module")
def small_gb():
    rng = np.random.default_rng(0)
    n, p = 500, 5
    X = rng.normal(size=(n, p))
    y = np.clip((X[:, 0] + 0.5 * X[:, 1] - 0.4 * X[:, 2] * X[:, 3] + 0.5 * rng.normal(size=n) + 2).round(), 0, 4)
    fh = R2.FrankHall(5, "gb", dict(max_iter=15, learning_rate=0.1, max_leaf_nodes=7), 6310).fit(X, y.astype(int))
    return fh, X, M3.TreeTable(fh, p)


def test_tree_margins_match_sklearn(small_gb):
    fh, X, T = small_gb
    ref = np.column_stack([m.decision_function(X[:20]) for m in fh.models])
    assert np.abs(T.margins(X[:20]) - ref).max() < 1e-9


def test_path_dependent_treeshap_is_exact(small_gb):
    fh, X, T = small_gb
    S = M3.tree_shap_margins(T, X[:3])

    def es(pred, x, S_):
        nd = pred.nodes

        def rec(i):
            if nd["is_leaf"][i]:
                return nd["value"][i]
            f, lft, rgt = nd["feature_idx"][i], nd["left"][i], nd["right"][i]
            if f in S_:
                return rec(lft) if x[f] <= nd["num_threshold"][i] else rec(rgt)
            return (nd["count"][lft] * rec(lft) + nd["count"][rgt] * rec(rgt)) / nd["count"][i]
        return rec(0)
    for j in range(2):
        for i in range(3):
            v = lambda s, j=j, i=i: sum(es(it[0], X[i], s) for it in fh.models[j]._predictors)   # noqa: E731
            assert np.abs(S[j, i] - _shapley(v, X.shape[1])).max() < 1e-9


def test_interventional_treeshap_is_exact(small_gb):
    fh, X, T = small_gb
    Z = X[100:120]
    S = M3.tree_shap_interventional(T, X[:2], Z)
    for j in (0, 3):
        m = fh.models[j]
        for i in range(2):
            def v(s, i=i):
                Zm = Z.copy()
                Zm[:, list(s)] = X[i, list(s)]
                return m.decision_function(Zm).mean()
            assert np.abs(S[j, i] - _shapley(v, X.shape[1])).max() < 1e-9


def test_expected_score_shap_is_additive(small_gb):
    fh, X, T = small_gb
    e, res = M3.fh_shap_ey(T, X[:30])
    f = fh.predict_proba(X[:30]) @ np.arange(5)
    assert res < 1e-9
    assert np.abs(e.sum(1) - (f - M3._sig(T.mean_margin).sum())).max() < 1e-9


def test_linear_shap_matches_brute_force():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    y = np.clip((X[:, 0] - X[:, 1] + rng.normal(size=300) + 2).round(), 0, 4).astype(int)
    m = M3.polr_fit_warm(5, X, y, None)
    mu = X.mean(0)
    e = M3.linear_shap_ey(m, X[:5], mu)
    f = lambda Z: m.predict_proba(Z) @ np.arange(5)                                   # noqa: E731
    assert np.abs(e.sum(1) - (f(X[:5]) - f(mu[None, :]))).max() < 1e-9
    assert np.all(np.sign(e[:, 2:]) == 0) or np.abs(e[:, :2]).mean() > np.abs(e[:, 2:]).mean()


def test_planted_redundant_group_gate():
    g = M3.planted_gate()
    assert g and all(g.values()), g


def test_planted_tensor_gate():
    g = M3.planted_tensor_gate()
    assert g and all(g.values()), g


def test_cpqr_pivots_match_scipy():
    from scipy.linalg import qr
    Y = np.random.default_rng(2).normal(size=(200, 90))
    assert M3.cpqr_pivots(Y, 12) == list(qr(Y, pivoting=True, mode="r")[1][:12])


def test_cp_rank_rule_needs_corcondia_and_congruence():
    rows = {1: (100.0, 1.0), 2: (95.0, 0.99), 3: (90.0, 0.95), 4: (85.0, 0.30), 5: (40.0, 0.9)}
    assert M3.select_cp_rank(rows) == 3
    assert M3.select_cp_rank({1: (50.0, 0.1)}) == 1


def test_compressed_cp_keeps_factors():
    rng = np.random.default_rng(4)
    A, B, C = rng.normal(size=(300, 2)), rng.normal(size=(6, 2)), rng.normal(size=(7, 2))
    X = np.einsum("nr,br,qr->nbq", A, B, C) + 0.01 * rng.normal(size=(300, 6, 7))
    sols, Xc = M3.cp_path(X, 2, 2, rng)
    assert sols[0][0] < 0.05 and M3.congruence((B, C), sols[0][2][1:]) > 0.99


def test_kendall_and_boot_rows():
    M = np.array([[1.0, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]])
    t = M3.kendall_pairs(M)
    assert np.allclose(sorted(t), [-1, -1, 1])
    g = np.array([5, 5, 7, 9, 9, 9])
    jb = M3.boot_rows(g, np.random.default_rng(0))
    assert len(jb) >= 3 and all(np.sum(g[jb] == s) in (0, (g == s).sum(), 2 * (g == s).sum(), 3 * (g == s).sum())
                                for s in (5, 7, 9))


def test_fitlog_fails_closed_on_test_rows():
    F = M3.FitLog(np.array(["train", "validation", "test"]))
    assert F.rows(np.array([0, 1]), "ok").tolist() == [0, 1]
    with pytest.raises(M3.BuildError):
        F.rows(np.array([0, 2]), "with a test row")


def test_unpickler_maps_main_and_refuses_other_modules():
    obj = R2.OrdinalLinear.__new__(R2.OrdinalLinear)
    obj.__dict__.update(coef=np.arange(3.0))
    raw = pickle.dumps(obj, protocol=2).replace(f"c{M3.RUN2_MODULE}\n".encode(), b"c__main__\n")
    back = M3.R2Unpickler(io.BytesIO(raw)).load()
    assert isinstance(back, R2.OrdinalLinear) and back.coef.tolist() == [0.0, 1.0, 2.0]
    for mod, name in (("os", "system"), ("builtins", "eval"), ("builtins", "getattr"), ("__main__", "Heartbeat")):
        with pytest.raises(M3.BuildError):
            M3.R2Unpickler(io.BytesIO(b"")).find_class(mod, name)


def test_gate_collects_all_problems_without_values(tmp_path):
    (tmp_path / "a_TEAM.csv").write_text("landmark,share_median\n72,4646\n")
    (tmp_path / "b_TEAM.csv").write_text("item,n_patients\n777,3\n")
    (tmp_path / "c_TEAM.md").write_text("note: encounter 777 reviewed\n")
    probs = M3.gate_problems(tmp_path, ["a_TEAM.csv", "b_TEAM.csv", "c_TEAM.md"], {4646}, {"777"})
    assert ("a_TEAM.csv", "column share_median", R2._SIZE_MSG) in probs
    assert ("b_TEAM.csv", "column names", "forbidden column name (count or identifier)") in probs
    assert ("c_TEAM.md", "md text", R2._ID_MSG) in probs
    with pytest.raises(M3.BuildError) as ei:
        M3.release_gate(tmp_path, ["a_TEAM.csv", "b_TEAM.csv", "c_TEAM.md"], {4646}, {"777"})
    assert "4646" not in str(ei.value) and "777" not in str(ei.value)


def test_audit_whitelist_names_keys_not_values():
    good = {"s13_tau": [{"landmark": 24, "tau_median": 0.5, "refits": 50}], "configuration": {"s13_refits": 50}}
    assert M3.audit_nums_ok(good) and not M3.bad_num_keys(good)
    bad = {"stage13": {"train_size": 4321}}
    assert not M3.audit_nums_ok(bad)
    keys = M3.bad_num_keys(bad)
    assert keys and not any("4321" in k for k in keys)


def _boom():
    json.loads("{secret value")


def test_describe_exception_frames_only():
    try:
        _boom()
    except Exception as e:  # noqa: BLE001
        lines = M3.describe_exception(e, None)
    assert lines[0] == "unexpected JSONDecodeError" and any("_boom" in ln for ln in lines)
    assert not any("secret" in ln for ln in lines)


# ------------------------------------------------------------------------------------------- end to end
PART_TABLES = {"1": ("stage13_tau_TEAM.csv", "stage13_h7_TEAM.csv"),
               "2": ("stage18_seq_tau_TEAM.csv", "stage16_mode_spectra_TEAM.csv", "stage16_cp_TEAM.csv",
                     "stage16_angles_TEAM.csv")}


@pytest.mark.parametrize("part", ["1", "2"])
def test_e2e_publishes_with_all_checks(e2e, part):
    r = e2e["r"]
    assert e2e["rc"][part] == 0
    out = r["root"] / OUTP[part]
    man = json.loads((out / f"{OUTN[part]}_manifest.json").read_text())
    assert man["release_gate"] == "ok" and all(v == "pass" for v in man["acceptance_checks"].values())
    for nm, rec in man["files"].items():
        assert sha(out / nm) == rec["sha256"], nm
    assert man["internal_files"] == [f"{OUTN[part]}_importances_factors_INTERNAL.npz"]
    assert all(n.endswith(("_TEAM.csv", "_TEAM.png", "_TEAM.json", "_TEAM.md")) for n in man["team_files"])
    assert man["input_hashes"]["stage11_12_17_r2_manifest.json"] == r["r2"]
    assert not list(out.parent.glob(f"{OUTN[part]}.pending-*"))
    other = "2" if part == "1" else "1"
    for nm in PART_TABLES[part]:
        assert nm in man["team_files"] and len(pd.read_csv(out / nm)) > 0, nm
    for nm in PART_TABLES[other]:
        assert nm not in man["team_files"], nm
    md = (out / f"{OUTN[part]}_audit_TEAM.md").read_text()
    assert "Release gate: ok" in md and f"part {part} of 2" in md
    a = json.loads((out / f"{OUTN[part]}_audit_TEAM.json").read_text())
    assert a["configuration"]["s13_refits"] == 3 and a["configuration"]["s18_refits"] == 2      # FAST3
    if part == "2":
        s18 = pd.read_csv(out / "stage18_seq_tau_TEAM.csv")
        assert set(s18["landmark"]) <= set(M3.S18_LANDMARKS)
        ang = pd.read_csv(out / "stage16_angles_TEAM.csv")
        assert len(ang) == 2 * (len(R2.LANDMARKS) - 1)
        assert "## Stage 13" not in md and "## Stage 16" in md and "## Stage 18" in md
    else:
        assert "## Stage 13" in md and "## Stage 16" not in md and "## Stage 18" not in md


def test_defaults_are_the_budgeted_settings():
    a = M3.build_config(["--part", "1"])
    assert (a.s13_refits, a.s13_tree_refits, a.s18_refits, a.s16_boot, a.s16_cp_starts) == (50, 10, 10, 50, 10)
    assert tuple(M3.S18_LANDMARKS) == (72, 120, 168, 240, 336) and a.max_seconds == 3600.0
    assert M3.OUT_NAME == "stage13_r3b1"
    M3.build_config(["--part", "2"])
    assert M3.OUT_NAME == "stage16_18_r3b2" and M3.CACHE_NAME == "_cache_stage16_18_r3b2"
    with pytest.raises(SystemExit):
        M3.build_config([])                          # --part is required


def test_e2e_never_touches_test_rows(e2e):
    assert e2e["rc"] == {"1": 0, "2": 0} and len(e2e["logs"]) >= 2
    for F in e2e["logs"]:
        assert "test" not in set(F.split.tolist())                     # dropped at load
        assert F.fits and all("test" not in sp for _, sp in F.fits)
    for p in ("1", "2"):
        man = json.loads((e2e["r"]["root"] / OUTP[p] / f"{OUTN[p]}_manifest.json").read_text())
        assert man["acceptance_checks"]["r3b_no_test_rows_loaded"] == "pass"
        assert man["acceptance_checks"]["r3b_no_test_rows_in_any_fit_or_sample"] == "pass"


@pytest.mark.parametrize("part", ["1", "2"])
def test_e2e_no_patient_loadings_or_per_row_shap(e2e, part):
    out = e2e["r"]["root"] / OUTP[part]
    man = json.loads((out / f"{OUTN[part]}_manifest.json").read_text())
    for nm in man["team_files"]:
        if nm.endswith(".csv"):
            if not (out / nm).read_text(encoding="utf-8").strip():
                continue                     # a table with no rows on this fixture (e.g. no gradient-boosting job)
            df = pd.read_csv(out / nm)
            if "mode" in df.columns:
                assert "patients" not in set(df.loc[df.get("factor", pd.Series(dtype=float)).notna(), "mode"]
                                             if "factor" in df.columns else []), nm
            for c in df.columns:
                assert not R2._FORBID_COL.search(c), (nm, c)
    z = np.load(out / f"{OUTN[part]}_importances_factors_INTERNAL.npz")
    assert not any(k.endswith("__A") or "patient" in k for k in z.files)


@pytest.mark.parametrize("part", ["1", "2"])
def test_e2e_team_files_no_sizes_or_ids(e2e, part):
    r = e2e["r"]
    out = r["root"] / OUTP[part]
    man = json.loads((out / f"{OUTN[part]}_manifest.json").read_text())
    z = np.load(r["root"] / LAD / "stage06_08" / "stage08_a0.npz")
    sp = z["split"].astype(str)
    sizes = {len(sp), len(set(r["sid"]))} | {int((sp == s).sum()) for s in ("train", "validation", "test")}
    ids = set(map(str, r["eid"])) | set(map(str, r["sid"]))
    assert not M3.gate_problems(out, man["team_files"], {s for s in sizes if s >= 100}, ids)


@pytest.mark.parametrize("part", ["1", "2"])
def test_rerun_refused(e2e, part):
    assert run3(e2e["r"], part=part) == 2                                 # output folder exists


def test_run2_manifest_hash_mismatch_stops(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    assert run3(r, r2="0" * 64) == 2
    assert "differs from --expect-run2-manifest-sha256" in capsys.readouterr().err
    assert not (r["root"] / OUT).exists()


def test_run2_pin_enforced(e2e, tmp_path, monkeypatch):
    r = fresh(e2e, tmp_path)
    monkeypatch.setitem(M3.PINS3, "run2_manifest", "f" * 64)
    assert run3(r) == 2


@pytest.mark.parametrize("victim", list(M3.RUN2_INTERNAL) + [M3.RUN2_AUDIT] + list(M3.RUN2_TEAM_USED))
def test_tampered_run2_file_stops(e2e, tmp_path, capsys, victim):
    r = fresh(e2e, tmp_path)
    p = r["root"] / OUT2 / victim
    p.write_bytes(p.read_bytes() + b"\0")
    assert run3(r) == 2
    assert "SHA-256 differs from the Run 2 manifest" in capsys.readouterr().err


def test_tampered_run2_script_stops(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    bad = tmp_path / "r2_11p_11_12_17_models.py"
    bad.write_text(R2_SCRIPT.read_text() + "\n# edited\n")
    assert run3(r, script=bad) == 2
    assert "Run 2 script SHA-256 differs from the pin" in capsys.readouterr().err


def test_stage06_08_hash_mismatch_stops(e2e, tmp_path):
    r = fresh(e2e, tmp_path)
    assert run3(dict(r, man="0" * 64)) == 2


def test_failure_publishes_nothing(e2e, tmp_path, monkeypatch, capsys):
    r = fresh(e2e, tmp_path)

    def boom(*a, **k):
        raise M3.BuildError("planted failure in stage 16")
    monkeypatch.setattr(M3, "stage16", boom)
    assert run3(r, part="2") == 3
    assert "result: FAILED" in capsys.readouterr().out
    assert not (r["root"] / OUTP["2"]).exists() and not list((r["root"] / LAD).glob(f"{OUTN['2']}.pending-*"))


def test_gate_failure_publishes_nothing_exit_3(e2e, tmp_path, monkeypatch, capsys):
    r = fresh(e2e, tmp_path)

    def gate(*a, **k):
        raise M3.BuildError("release gate: planted problem")
    monkeypatch.setattr(M3, "release_gate", gate)
    assert run3(r) == 3
    assert "result: FAILED; nothing left behind" in capsys.readouterr().out
    assert not (r["root"] / OUT).exists() and not list((r["root"] / LAD).glob(f"{OUTN['1']}.pending-*"))


@pytest.mark.parametrize("part", ["1", "2"])
def test_time_limit_stops_cleanly(e2e, tmp_path, capsys, part):
    r = fresh(e2e, tmp_path)
    assert run3(r, "--max-seconds", "0.01", part=part) == 3
    assert "result: FAILED" in capsys.readouterr().out
    assert not (r["root"] / OUTP[part]).exists()
    assert not list((r["root"] / LAD).glob(f"{OUTN[part]}.pending-*"))


def test_precheck_only_prints_hashes(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    assert run3(r, "--precheck-only") == 0
    out = capsys.readouterr().out
    assert "result: PRECHECK OK" in out and f"stage11_12_17_r2_manifest.json sha256 {r['r2']}" in out
    assert not (r["root"] / OUT).exists()


# ----------------------------------------------------------------------------------- small integer ids, leaks
def test_small_integer_ids_pass_gate(e2e_small):
    r, rc, ids = e2e_small
    assert rc == {"1": 0, "2": 0}
    assert {"1", "5", "10", "24", "72", "100"} <= ids
    for p in ("1", "2"):
        out = r["root"] / OUTP[p]
        man = json.loads((out / f"{OUTN[p]}_manifest.json").read_text())
        assert man["release_gate"] == "ok"
        assert not M3.gate_problems(out, man["team_files"], set(), ids)


def test_planted_leaks_fail_gate_without_values(e2e_small, tmp_path):
    r, _, ids = e2e_small
    outs, mans, allowed = {}, {}, set()
    for p in ("1", "2"):
        outs[p] = tmp_path / f"leak{p}"
        shutil.copytree(r["root"] / OUTP[p], outs[p])
        mans[p] = json.loads((outs[p] / f"{OUTN[p]}_manifest.json").read_text())
        a = json.loads((outs[p] / f"{OUTN[p]}_audit_TEAM.json").read_text())
        allowed |= R2._audit_ints(a, set()) | M3.script_text_ints(a)
    leak = max((i for i in ids if i.isdigit() and int(i) not in allowed and len(i) >= 3), key=int)
    o1, o2 = outs["1"], outs["2"]
    p = o1 / "stage13_h7_TEAM.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df.loc[0, "matrix"] = leak
    p.write_text(df.to_csv(index=False, lineterminator="\n"))
    md = o1 / f"{OUTN['1']}_audit_TEAM.md"
    md.write_text(md.read_text() + f"\nnote: patient {leak} checked\n")
    q = o2 / "stage16_cp_TEAM.csv"
    q.write_text(q.read_text().rstrip("\n") + "\n" + ",".join([leak] * len(pd.read_csv(q, nrows=0).columns)) + "\n")
    s = o2 / "stage18_seq_tau_TEAM.csv"
    df = pd.read_csv(s, dtype=str, keep_default_na=False)
    df.insert(0, "n_rows", "5")
    s.write_text(df.to_csv(index=False, lineterminator="\n"))
    probs1 = M3.gate_problems(o1, mans["1"]["team_files"], set(), ids)
    probs2 = M3.gate_problems(o2, mans["2"]["team_files"], set(), ids)
    assert ("stage13_h7_TEAM.csv", "column matrix", R2._ID_MSG) in probs1
    assert (md.name, "md text", R2._ID_MSG) in probs1
    assert any(f == "stage16_cp_TEAM.csv" and m == R2._ID_MSG for f, _, m in probs2)
    assert ("stage18_seq_tau_TEAM.csv", "column names", "forbidden column name (count or identifier)") in probs2
    for o, p in ((o1, "1"), (o2, "2")):
        with pytest.raises(M3.BuildError) as ei:
            M3.release_gate(o, mans[p]["team_files"], set(), ids)
        assert leak not in str(ei.value)
