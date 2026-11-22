"""Tests for r3a_desktop_full v2.0.1-desktop (Run 3a desktop full settings: stage 14 and stage 18 test scoring,
stacked sensitivity). Outputs must stay under data/processed/ladder/DESKTOP_FULL_PLAN/.

Self-contained and SYNTHETIC only: each fixture builds a tiny Stage 06-08 / Run 1-shaped project with the Run 2 test
helper (make_release from test_r2_11p_11_12_17_models.py, hash-checked), runs the unmodified Run 2 script on it with
the real pins nulled, then runs Run 3a. No real data, no network. Run with:
    python -s -m pytest -q -p no:cacheprovider test_r3a_desktop_full.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import pickle
import re
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
FAST3 = ["--n-boot", "20", "--n-boot-family-a", "60", "--gru-max-epochs", "2",
         "--stacked-max-epochs", "2", "--heartbeat-seconds", "600"]
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
M3 = _load("r3a_mod_under_test", HERE / "r3a_desktop_full.py")
R2 = M3.load_run2(R2_SCRIPT, M3.PINS3["run2_script_sha256"])[0]
MT = _load("r2_test_helper_for_r3", R2_TEST)          # make_release only
DESK = LAD / M3.DESKTOP_ROOT_NAME
OUT = DESK / M3.OUT_NAME
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
    """Synthetic project with a published Run 2 (unmodified script, pins nulled) and the laptop layout of Run 3:
    split file and raw tables copied under data/interim/r2_inputs (the Copilot sandbox reads only the project)."""
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


def run3(r, *extra, **kw) -> int:
    return M3.main(args3(r, *extra, **kw))


def log_of(r) -> list:
    p = r["root"] / DESK / M3.TEST_LOG_NAME
    return json.loads(p.read_text())["attempts"] if p.exists() else []


class RecordingGuard(M3.TestGuard):
    made = []

    def __init__(self, split):
        super().__init__(split)
        self.te_phases = []
        RecordingGuard.made.append(self)

    def te(self):
        self.te_phases.append(self.phase)
        return super().te()


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    mp = _null_pins()
    r = make_project(tmp_path_factory.mktemp("r3_e2e"))
    pristine = copy_project(r, tmp_path_factory.mktemp("r3_pristine"))
    files_before = _rel_files(r["root"])
    RecordingGuard.made.clear()
    mp.setattr(M3, "TestGuard", RecordingGuard)
    rc = run3(r)
    guards = list(RecordingGuard.made)
    mp.undo()
    return dict(r=r, rc=rc, guards=guards, pristine=pristine, files_before=files_before)


@pytest.fixture(scope="module")
def e2e_small(tmp_path_factory):
    mp = _null_pins()
    r = make_project(tmp_path_factory.mktemp("r3_small"), small_ids=True)
    rc = run3(r)
    mp.undo()
    return r, rc, set(map(str, r["eid"])) | set(map(str, r["sid"]))


def fresh(e2e, tmp_path) -> dict:
    return copy_project(e2e["pristine"], tmp_path)


# ------------------------------------------------------------------------------------------------- unit tests
def test_guard_blocks_test_rows_outside_scoring():
    G = M3.TestGuard(np.array(["train", "validation", "test", "train"]))
    with pytest.raises(M3.BuildError):
        G.te()
    with pytest.raises(M3.BuildError):
        G.score("x", lambda: 1)
    with pytest.raises(M3.BuildError):
        G.fit_rows(np.array([0, 2]), "fit with a test row")
    assert G.fit_rows(np.array([0, 1, 3]), "ok").tolist() == [0, 1, 3]
    G.phase = "score"
    assert G.te().tolist() == [2] and G.score("x", lambda: 7) == 7
    with pytest.raises(M3.BuildError):
        G.score("x", lambda: 7)


def test_holm_and_bootstrap_p_values():
    adj, rej = M3.holm(np.array([0.01, 0.04, 0.03, 0.2]))
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.2]) and rej.tolist() == [True, False, False, False]
    d = np.r_[np.full(95, 0.05), np.full(5, -0.01)]
    assert M3.p_gain(d) == pytest.approx(6 / 101)
    assert M3.p_tost(np.zeros(100), 0.02) == pytest.approx(1 / 101)
    assert M3.p_tost(np.full(100, 0.05), 0.02) == pytest.approx(1.0)


def test_family_a_pvalues_can_pass_holm_over_131_tests():
    rng = np.random.default_rng(5)
    subj = np.repeat(np.arange(150), 2)
    y = rng.integers(0, 9, subj.size)
    same = {("polr", "a", "k"): y.copy(), ("polr", "b", "k"): y.copy()}
    far = np.clip(y + 3, 0, 8)
    B = 3000
    p = M3.family_a_pvalues(subj, y, 9, same, [(("polr", "b", "k"), ("polr", "a", "k"))], (far, y, np.arange(subj.size)),
                            B, 0.02, chunk=700)
    assert p[0] == pytest.approx(1 / (B + 1)) and p[0] < 0.05 / 131          # identical models: equivalent
    assert p[1] == pytest.approx(1.0)                                       # a far worse model: not equivalent
    assert p == M3.family_a_pvalues(subj, y, 9, same, [(("polr", "b", "k"), ("polr", "a", "k"))],
                                    (far, y, np.arange(subj.size)), B, 0.02, chunk=1000)   # chunking is invisible


def test_boot_auc_matches_unweighted_auc():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    s, y = rng.random(300), rng.random(300) < 0.3
    W = np.ones((2, 300))
    W[1, :150] = 2.0
    a = M3.boot_auc(W, s, y)
    assert a[0] == pytest.approx(roc_auc_score(y, s))
    assert a[1] == pytest.approx(roc_auc_score(y, s, sample_weight=W[1]))


def test_unpickler_maps_main_and_refuses_other_modules():
    obj = R2.OrdinalLinear.__new__(R2.OrdinalLinear)
    obj.__dict__.update(coef=np.arange(3.0))
    raw = pickle.dumps(obj, protocol=2).replace(f"c{M3.RUN2_MODULE}\n".encode(), b"c__main__\n")
    assert b"c__main__\nOrdinalLinear" in raw                    # as pickled when Run 2 ran as a script
    back = M3.R2Unpickler(io.BytesIO(raw)).load()
    assert isinstance(back, R2.OrdinalLinear) and back.coef.tolist() == [0.0, 1.0, 2.0]
    for mod, name in (("os", "system"), ("builtins", "eval"), ("builtins", "getattr"), ("__main__", "Heartbeat")):
        with pytest.raises(M3.BuildError):
            M3.R2Unpickler(io.BytesIO(b"")).find_class(mod, name)


def test_unpickler_loads_gradient_boosting_models():
    """The real Run 2 winner is a Frank-Hall HistGradientBoosting model; its Cython loss pickles as module _loss."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(0)
    X, y = rng.random((80, 3)), rng.random(80) < 0.4
    gb = HistGradientBoostingClassifier(max_iter=5, random_state=0).fit(X, y)
    iso = IsotonicRegression(out_of_bounds="clip").fit(X[:, 0], y)
    raw = pickle.dumps({"gb": gb, "iso": iso})
    assert b"_loss" in raw
    back = M3.R2Unpickler(io.BytesIO(raw)).load()
    assert np.allclose(back["gb"].predict_proba(X), gb.predict_proba(X))
    assert np.allclose(back["iso"].predict(X[:, 0]), iso.predict(X[:, 0]))
    with pytest.raises(M3.BuildError):
        M3.R2Unpickler(io.BytesIO(b"")).find_class("_loss", "system")


def test_gate_collects_all_problems_without_values(tmp_path):
    (tmp_path / "a_TEAM.csv").write_text("landmark,qwk\n72,4646\n")
    (tmp_path / "b_TEAM.csv").write_text("contrast,h9_supported\n777,True\n")
    (tmp_path / "c_TEAM.md").write_text("note: encounter 777 reviewed\n")
    probs = M3.gate_problems(tmp_path, ["a_TEAM.csv", "b_TEAM.csv", "c_TEAM.md"], {4646}, {"777"})
    assert ("a_TEAM.csv", "column qwk", R2._SIZE_MSG) in probs
    assert ("b_TEAM.csv", "column names", "forbidden column name (count or identifier)") in probs
    assert ("b_TEAM.csv", "column contrast", R2._ID_MSG) in probs
    assert ("c_TEAM.md", "md text", R2._ID_MSG) in probs
    with pytest.raises(M3.BuildError) as ei:
        M3.release_gate(tmp_path, ["a_TEAM.csv", "b_TEAM.csv", "c_TEAM.md"], {4646}, {"777"})
    assert "4646" not in str(ei.value) and "777" not in str(ei.value)


def test_audit_whitelist_names_keys_not_values():
    good = {"h9_family_b": [{"landmark": 72, "delta_qwk": 0.01, "p_holm": 0.2}], "configuration": {"n_boot": 1000}}
    assert M3.audit_nums_ok(good) and not M3.bad_num_keys(good)
    bad = {"stage18": {"test_size": 4321}, "h9_family_b": [{"landmark": 72, "eligible": 987}]}
    assert not M3.audit_nums_ok(bad)
    keys = M3.bad_num_keys(bad)
    assert keys and not any("4321" in k or "987" in k for k in keys)


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
def test_e2e_publishes_with_all_checks(e2e):
    r = e2e["r"]
    assert e2e["rc"] == 0
    out = r["root"] / OUT
    man = json.loads((out / f"{M3.OUT_NAME}_manifest.json").read_text())
    assert man["release_gate"] == "ok" and all(v == "pass" for v in man["acceptance_checks"].values())
    for nm, rec in man["files"].items():
        assert sha(out / nm) == rec["sha256"], nm
    assert man["internal_files"] == ["stage14_18_test_predictions_INTERNAL.npz"]
    assert all(re.search(r"_TEAM\.(csv|png|json|md)$", n) for n in man["team_files"])
    assert man["input_hashes"]["stage11_12_17_r2_manifest.json"] == r["r2"]
    assert not list(out.parent.glob(f"{M3.OUT_NAME}.pending-*"))
    lg = log_of(r)
    assert [a["status"] for a in lg] == ["published"] and lg[0]["manifest_sha256"] == sha(
        out / f"{M3.OUT_NAME}_manifest.json")
    h = pd.read_csv(out / "stage18_holm_core_TEAM.csv")
    h1 = h[h["outcome"] == "outcome1"]
    assert h1["landmark"].tolist()[:5] == list(M3.H9_FOCUS)                 # Vinai's five lead the table
    assert set(h1["landmark"]) == set(R2.CORE_LANDMARKS)                     # Holm family: contract core landmarks
    assert np.all(np.diff(np.sort(h1["p_holm"].to_numpy())) >= -1e-12)
    cur = pd.read_csv(out / "stage18_curve_TEAM.csv")
    assert set(cur["landmark"]) == set(R2.LANDMARKS) and cur["label"].str.startswith("descriptive").all()
    ct = pd.read_csv(out / "stage18_contrast_TEAM.csv")
    assert ct.loc[~(ct["core"] & (ct["contrast"] == "ABC_minus_AB")), "p_one_sided_le_0"].isna().all()
    fa = pd.read_csv(out / "stage14_holm_family_a_TEAM.csv")
    assert len(fa) > 0 and (fa["p_tost_holm"] >= fa["p_tost"] - 1e-12).all()
    assert set(fa["model"]) <= {M3.FAMILY_A_MODEL, "gru"}
    assert "stage18_stacked_sensitivity_TEAM.csv" in man["team_files"]
    md = (out / f"{M3.OUT_NAME}_audit_TEAM.md").read_text()
    assert "test" in md and "Release gate: ok" in md


def test_e2e_test_rows_touched_only_in_scoring(e2e):
    assert e2e["rc"] == 0 and len(e2e["guards"]) == 1
    G = e2e["guards"][0]
    assert G.fits and all(ph == "verify" and "test" not in sp for _, ph, sp in G.fits)
    assert G.te_phases and set(G.te_phases) == {"score"}
    assert len(G.scored) == len(set(G.scored)) > 0 and G.phase == "done"


def test_verification_ignores_test_rows(e2e, tmp_path):
    """Replacing every test row of every Stage 08 level with noise leaves all verification results unchanged."""
    r = fresh(e2e, tmp_path)

    def verify(poison: bool):
        cfg = M3.build_config(args3(r))
        cfg.run2_script_sha = sha(R2_SCRIPT)
        checks, info = {}, {}
        D = M3.load_all(cfg, checks)
        if poison:
            te = D["rows"]["split"] == "test"
            rng = np.random.default_rng(99)
            for v in D["levels"].values():
                v["X"][te] = rng.normal(0, 5, (int(te.sum()), v["X"].shape[1]))
        mats = R2.build_matrices(D, checks)
        G = M3.TestGuard(D["rows"]["split"])
        V = M3.verify14(D, mats, G, cfg, checks, info)
        M3.verify18(D, mats, G, V, cfg, checks, info)
        return checks, info

    c0, i0 = verify(False)
    c1, i1 = verify(True)
    assert c0 == c1 and all(c0.values())
    assert i0.keys() == i1.keys()
    for k in i0:
        if isinstance(i0[k], float):
            assert i1[k] == pytest.approx(i0[k], rel=1e-6, abs=1e-9), k
        else:
            assert i1[k] == i0[k], k


def test_e2e_team_files_no_sizes_or_ids(e2e):
    r = e2e["r"]
    out = r["root"] / OUT
    man = json.loads((out / f"{M3.OUT_NAME}_manifest.json").read_text())
    z = np.load(r["root"] / LAD / "stage06_08" / "stage08_a0.npz")
    sp = z["split"].astype(str)
    sizes = {len(sp), len(set(r["sid"]))} | {int((sp == s).sum()) for s in ("train", "validation", "test")}
    ids = set(map(str, r["eid"])) | set(map(str, r["sid"]))
    assert not M3.gate_problems(out, man["team_files"], {s for s in sizes if s >= 100}, ids)
    for nm in man["team_files"]:
        if nm.endswith(".csv"):
            for c in pd.read_csv(out / nm, nrows=0).columns:
                assert not R2._FORBID_COL.search(c), (nm, c)


def test_rerun_refused(e2e, tmp_path, capsys):
    r = e2e["r"]
    assert run3(r) == 2                                                     # output folder exists
    other = r["root"] / DESK / "other_out"
    assert run3(r, "--out-dir", str(other)) == 2                            # test-use log: published once already
    assert "scored once only" in capsys.readouterr().err
    assert [a["status"] for a in log_of(r)] == ["published"]


def test_run2_manifest_hash_mismatch_stops(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    assert run3(r, r2="0" * 64) == 2
    assert "differs from --expect-run2-manifest-sha256" in capsys.readouterr().err
    assert not log_of(r) and not (r["root"] / OUT).exists()


def test_run2_pin_enforced(e2e, tmp_path, monkeypatch):
    r = fresh(e2e, tmp_path)
    monkeypatch.setitem(M3.PINS3, "run2_manifest", "f" * 64)
    assert run3(r) == 2


@pytest.mark.parametrize("victim", list(M3.RUN2_INTERNAL) + [M3.RUN2_AUDIT])
def test_tampered_run2_file_stops(e2e, tmp_path, capsys, victim):
    r = fresh(e2e, tmp_path)
    p = r["root"] / OUT2 / victim
    p.write_bytes(p.read_bytes() + b"\0")
    assert run3(r) == 2
    assert "SHA-256 differs from the Run 2 manifest" in capsys.readouterr().err
    assert not log_of(r)


def test_tampered_run2_script_stops(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    bad = tmp_path / "r2_11p_11_12_17_models.py"
    bad.write_text(R2_SCRIPT.read_text() + "\n# edited\n")
    assert run3(r, script=bad) == 2
    assert "Run 2 script SHA-256 differs from the pin" in capsys.readouterr().err


def test_stage06_08_hash_mismatch_stops(e2e, tmp_path):
    r = fresh(e2e, tmp_path)
    assert run3(dict(r, man="0" * 64)) == 2


def test_verification_failure_reads_no_test_rows(e2e, tmp_path, monkeypatch, capsys):
    r = fresh(e2e, tmp_path)
    real = M3.build_config

    def strict(argv=None):
        cfg = real(argv)
        cfg.repro_tol_lr_qwk = -1.0         # no refit can reproduce: verification must fail before scoring
        return cfg
    monkeypatch.setattr(M3, "build_config", strict)
    assert run3(r, "--no-stacked") == 3
    err = capsys.readouterr().err
    assert "before any test row was read" in err
    assert "14_ordinal_lr_refits_reproduce_run2_validation_qwk" in err
    assert "max abs QWK diff=" in err and "tol=-1" in err
    assert not log_of(r) and not (r["root"] / OUT).exists()


def test_failed_scoring_is_logged_and_publishes_nothing(e2e, tmp_path, monkeypatch):
    r = fresh(e2e, tmp_path)

    def boom(*a, **k):
        raise M3.BuildError("planted failure in stage 18 scoring")
    monkeypatch.setattr(M3, "score18", boom)
    assert run3(r, "--no-stacked") == 3
    assert [a["status"] for a in log_of(r)] == ["failed (nothing published)"]
    assert not (r["root"] / OUT).exists() and not list((r["root"] / DESK).glob(f"{M3.OUT_NAME}.pending-*"))


def test_time_limit_stops_cleanly(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    assert run3(r, "--max-seconds", "3") == 3
    assert "result: FAILED" in capsys.readouterr().out
    assert not (r["root"] / OUT).exists()


def test_precheck_only_prints_hashes(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    assert run3(r, "--precheck-only") == 0
    out = capsys.readouterr().out
    assert "result: PRECHECK OK" in out and f"stage11_12_17_r2_manifest.json sha256 {r['r2']}" in out
    assert not log_of(r) and not (r["root"] / OUT).exists()


# ----------------------------------------------------------------------------------- small integer ids, leaks
def test_small_integer_ids_pass_gate(e2e_small):
    r, rc, ids = e2e_small
    assert rc == 0
    assert {"1", "5", "10", "24", "72", "100"} <= ids
    out = r["root"] / OUT
    man = json.loads((out / f"{M3.OUT_NAME}_manifest.json").read_text())
    assert man["release_gate"] == "ok"
    assert not M3.gate_problems(out, man["team_files"], set(), ids)


def test_planted_leaks_fail_gate_without_values(e2e_small, tmp_path):
    r, _, ids = e2e_small
    out = tmp_path / "leak"
    shutil.copytree(r["root"] / OUT, out)
    man = json.loads((out / f"{M3.OUT_NAME}_manifest.json").read_text())
    a = json.loads((out / f"{M3.OUT_NAME}_audit_TEAM.json").read_text())
    allowed = R2._audit_ints(a, set()) | M3.script_text_ints(a)
    leak = max((i for i in ids if i.isdigit() and int(i) not in allowed and len(i) >= 3), key=int)
    p = out / "stage18_holm_core_TEAM.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df.loc[0, "contrast"] = leak
    p.write_text(df.to_csv(index=False, lineterminator="\n"))
    md = out / f"{M3.OUT_NAME}_audit_TEAM.md"
    md.write_text(md.read_text() + f"\nnote: patient {leak} checked\n")
    q = out / "stage14_test_TEAM.csv"
    q.write_text(q.read_text().rstrip("\n") + "\n" + ",".join([leak] * len(pd.read_csv(q, nrows=0).columns)) + "\n")
    s = out / "stage18_curve_TEAM.csv"
    df = pd.read_csv(s, dtype=str, keep_default_na=False)
    df.insert(0, "n_rows", "5")
    s.write_text(df.to_csv(index=False, lineterminator="\n"))
    probs = M3.gate_problems(out, man["team_files"], set(), ids)
    assert ("stage18_holm_core_TEAM.csv", "column contrast", R2._ID_MSG) in probs
    assert (md.name, "md text", R2._ID_MSG) in probs
    assert any(f == "stage14_test_TEAM.csv" and m == R2._ID_MSG for f, _, m in probs)
    assert ("stage18_curve_TEAM.csv", "column names", "forbidden column name (count or identifier)") in probs
    with pytest.raises(M3.BuildError) as ei:
        M3.release_gate(out, man["team_files"], set(), ids)
    assert leak not in str(ei.value)



# -------------------------------------------------------------------------------- desktop full-settings checks
def test_defaults_are_the_desktop_full_settings():
    a = M3.build_config([])
    assert a.max_seconds == 0.0
    assert a.stacked_max_epochs == 200 and M3.STACK_MAX_EPOCHS == 200
    assert M3.OUT_NAME == "stage14_18_r3a" and M3.DESKTOP_ROOT_NAME == "DESKTOP_FULL_PLAN"
    assert a.out_dir == Path("data") / "processed" / "ladder" / "DESKTOP_FULL_PLAN" / "stage14_18_r3a"
    assert a.test_log == Path("data") / "processed" / "ladder" / "DESKTOP_FULL_PLAN" / "test_use_log_stage14_18_r3a.json"
    assert M3.SCRIPT_VERSION == "r3a_desktop_full v2.0.1-desktop"
    assert a.repro_tol_lr_qwk == 0.005 and a.repro_tol_gru_qwk == 0.005
    assert a.repro_tol_prob == 1e-4


def test_out_dir_outside_desktop_root_is_refused(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    outside = tmp_path / "outside_out"
    assert run3(r, "--out-dir", str(outside)) == 2
    err = capsys.readouterr().err
    assert "DESKTOP_FULL_PLAN" in err and "STOPPED (input)" in err
    assert not outside.exists()


def test_laptop_r3a_published_does_not_block_desktop(e2e, tmp_path):
    """A published laptop Run 3a (folder + test-use log under ladder/) must not refuse the desktop run."""
    r = fresh(e2e, tmp_path)
    lad = r["root"] / LAD
    laptop_out = lad / "stage14_18_r3a"
    laptop_out.mkdir(parents=True)
    (laptop_out / "stage14_18_r3a_manifest.json").write_text('{"release_gate": "ok"}\n')
    laptop_log = lad / "test_use_log_stage14_18_r3a.json"
    laptop_log.write_text(
        '{"run": "stage14_18_r3a", "attempts": [{"script": "r3a_14_18_test_scoring v1.0.0",'
        ' "status": "published", "started_utc": "2026-11-19T00:00:00Z"}]}\n'
    )
    assert run3(r) == 0
    assert (r["root"] / OUT).is_dir()
    assert [a["status"] for a in log_of(r)] == ["published"]
    # laptop artefacts left in place and unused
    assert laptop_log.is_file() and laptop_out.is_dir()


def _rel_files(root: Path) -> set:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


def test_writes_stay_in_desktop_full_plan(e2e):
    """Every file the desktop script added to the project is under data/processed/ladder/DESKTOP_FULL_PLAN/."""
    r = e2e["r"]
    before = e2e["files_before"]
    after = _rel_files(r["root"])
    new = after - before
    assert new, "the desktop run should have written outputs"
    desk = DESK
    for p in sorted(new):
        assert desk in p.parents, f"wrote outside DESKTOP_FULL_PLAN: {p}"
    readme = r["root"] / desk / M3.README_NAME
    assert readme.is_file()
    txt = readme.read_text(encoding="utf-8").lower()
    assert "post-primary" in txt and "sensitivity" in txt and "full-settings desktop run" in txt
    assert (r["root"] / OUT / f"{M3.OUT_NAME}_manifest.json").is_file()
    assert (r["root"] / DESK / M3.TEST_LOG_NAME).is_file()
    md = (r["root"] / OUT / f"{M3.OUT_NAME}_audit_TEAM.md").read_text()
    assert "post-primary sensitivity analysis" in md
    assert "second look" in md


def test_verify_device_is_cpu_even_if_cuda_reported():
    dev = M3.verify_device()
    assert str(dev) == "cpu" or getattr(dev, "type", None) == "cpu"


def test_verify18_frozen_forward_passes_use_cpu(e2e, tmp_path, monkeypatch):
    """Reproduction forward passes must request the CPU device, even if torch_device would be CUDA."""
    r = fresh(e2e, tmp_path)
    used = []
    real = M3.gru_predict

    def wrapped(state, seq, st, off, device=None):
        used.append(device)
        return real(state, seq, st, off, device=device)

    monkeypatch.setattr(M3, "gru_predict", wrapped)
    cfg = M3.build_config(args3(r, "--no-stacked"))
    cfg.run2_script_sha = sha(R2_SCRIPT)
    checks, info = {}, {}
    D = M3.load_all(cfg, checks)
    mats = R2.build_matrices(D, checks)
    G = M3.TestGuard(D["rows"]["split"])
    V = M3.verify14(D, mats, G, cfg, checks, info)
    M3.verify18(D, mats, G, V, cfg, checks, info)
    assert used, "verify18 should call gru_predict"
    torch = R2._torch()
    for d in used:
        assert d is not None, "verify18 must pass an explicit device, not the default training device"
        assert torch.device(d).type == "cpu"
    assert checks["18_frozen_landmark_models_reproduce_run2_validation"]
    assert "repro_s17_max_abs_prob_diff" in info
    assert info["repro_lr_max_abs_qwk_diff"] <= cfg.repro_tol_lr_qwk
    # the default training device is not used for these frozen checks
    assert checks["14_ordinal_lr_refits_reproduce_run2_validation_qwk"]


def test_audit_logs_environment_repro_and_failed_v2_attempt(e2e):
    md = (e2e["r"]["root"] / OUT / f"{M3.OUT_NAME}_audit_TEAM.md").read_text()
    assert "environment-reproducibility tolerance" in md
    assert "not required to be bit-identical" in md
    assert "v2.0.0-desktop attempt" in md
    assert "before any test row was read" in md
    assert "test-use log is unchanged" in md
    assert "not a second test scoring" in md
    a = json.loads((e2e["r"]["root"] / OUT / f"{M3.OUT_NAME}_audit_TEAM.json").read_text())
    assert a["script"] == "r3a_desktop_full v2.0.1-desktop"
    assert "repro_lr_max_abs_qwk_diff" in a["reproduction"]
    assert "repro_s17_max_abs_prob_diff" in a["reproduction"]
