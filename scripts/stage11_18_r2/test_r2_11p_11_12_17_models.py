"""Tests for r2_11p_11_12_17_models v1.0.0 (Run 2: label build 11p, stages 11, 12 and 17; validation only).

Self-contained: every test builds its own tiny SYNTHETIC project in a pytest tmp folder (Stage 06-08-shaped release,
canonical file, harmonization file, subject split, raw ICUSTAYS / ADMISSIONS and a Run 1-shaped output folder with
matched-width bases). No real data, no network. Run with:  python -s -m pytest -q test_r2_11p_11_12_17_models.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "r2_11p_11_12_17_models.py"


def _load():
    spec = importlib.util.spec_from_file_location("r2_mod_under_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["r2_mod_under_test"] = m
    spec.loader.exec_module(m)
    return m


M = _load()
FAST = ["--n-boot", "20", "--gru-max-epochs", "2", "--heartbeat-seconds", "600"]
SYN_NULL_PINS = ("canonical", "harmonization", "split", "ICUSTAYS", "ADMISSIONS", "run1_manifest", "run1_bases",
                 "raw_rows", "valid_stays")
OUT = Path("data") / "processed" / "ladder" / "stage11_12_17_r2"
FMT = "%Y-%m-%d %H:%M:%S"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------ synthetic fixture
def make_release(base: Path, n_subj: int = 200, seed: int = 11, small_ids: bool = False,
                 ret24: bool = False) -> dict:
    """Tiny synthetic project under base/proj, raw tables under base/PICDataset, split under base/PIC_Analysis_3_0.
    Planted label cases (eligible at 24 h, survivors): a readmission (Outcome 2 -> 3), an index-HADM tie broken by the
    latest discharge (-> 2), an unmatched index stay (-> unassignable), a missing DISCHTIME (-> unassignable); with
    ret24 an ICU return 10 h after OUT (-> the run must stop)."""
    rng = np.random.default_rng(seed)
    root = base / "proj"
    subj = np.repeat(np.arange(n_subj), rng.integers(1, 3, n_subj))
    n = len(subj)
    sid = np.array([str(s + 1) if small_ids else str(100000 + s) for s in subj])
    eid = np.array([str(i + 1) if small_ids else f"E{100000 + s}_{i}" for i, s in enumerate(subj)])
    hadm = np.array([str(700000 + i) for i in range(n)])
    u = rng.random(n_subj)
    split = np.where(u[subj] < 0.6, "train", np.where(u[subj] < 0.85, "validation", "test"))
    los = rng.uniform(6, 400, n) + 0.37
    died = rng.random(n) < 0.12
    t0 = pd.Timestamp("2016-01-01 08:00:00") + pd.to_timedelta(np.arange(n) * 40, unit="D")
    out = t0 + pd.to_timedelta(los, unit="h")
    elig24 = los > 24
    sig = (np.log(los) - np.log(los).mean()) / np.log(los).std()
    Z = np.c_[sig, rng.standard_normal((n, 3))] + 0.6 * rng.standard_normal((n, 4))
    widths = {"a0": 12, "a1": 15, "a2": 18, "b0": 14, "b1": 17, "b2s": 21, "b2d": 23, "b2p": 9}
    d = root / "data" / "processed" / "ladder" / "stage06_08"
    d.mkdir(parents=True)
    lv = {}
    for k, w in widths.items():
        X = Z @ rng.standard_normal((4, w)) + 0.7 * rng.standard_normal((n, w))
        X = (X - X.mean(0)) / X.std(0)
        cols = [f"{k}_v{j:03d}" for j in range(w)]
        lv[k] = (X, cols)
        np.savez(d / f"stage08_{k}.npz", encounter_id=eid, split=split, eligible_24h=elig24, columns=np.array(cols),
                 kinds=np.array(["cont"] * w), X_imputed=X, X_z=X, columns_full=np.array(cols), X_full_z=X)
    q = 5
    chans = np.array([f"ch{j}" for j in range(q)])
    for L in M.LANDMARKS:
        m = los > L
        T = rng.standard_normal((m.sum(), L // 6, q)) + Z[m][:, None, :1]
        miss = rng.random(T.shape) < 0.2
        np.savez(d / f"stage08_layer_c_L{L}.npz", encounter_id=eid[m], split=split[m], channels=chans, X_raw_z=T,
                 X_filled_z=T, missing=miss, carried_forward=miss, imputed=np.zeros_like(miss))
    for j in range(M.PINS["stage06_08_files"] - 22):
        (d / f"filler_{j:02d}.txt").write_text(f"synthetic filler {j}\n")
    files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(d.iterdir())}
    # planted cases among eligible survivors
    cand = np.flatnonzero(elig24 & ~died)
    i_readm, i_tie, i_unm, i_dmiss, i_ret = cand[:5]
    post = rng.uniform(0.5, 30, n)
    disch = out + pd.to_timedelta(post * 24, unit="h")
    admit = t0 - pd.to_timedelta(rng.uniform(1, 30, n), unit="h")
    icu_h = los.copy()
    can = pd.DataFrame({"encounter_id": eid, "SUBJECT_ID": sid, "HADM_ID": hadm, "hadm_id_list": [f"[{h}]" for h in hadm],
                        "INTIME": t0.strftime(FMT), "OUTTIME": out.strftime(FMT), "Died": died.astype(int),
                        "icu_los_hours": np.round(icu_h, 4),
                        "hospital_los_hours": np.round((disch - admit).total_seconds() / 3600, 4)})
    tie_h = str(790000)
    can.loc[i_tie, "hadm_id_list"] = f"[{hadm[i_tie]}, {tie_h}]"
    can_p = root / "data" / "interim" / "canonical_icu_encounters.csv"
    can_p.parent.mkdir(parents=True)
    can.to_csv(can_p, index=False, lineterminator="\n")
    pd.DataFrame({"encounter_id": eid, "t0": t0.strftime(FMT), "OUTTIME": out.strftime(FMT)}).to_csv(
        root / "data" / "interim" / "encounter_harmonization_fields.csv", index=False, lineterminator="\n")
    spl = base / "PIC_Analysis_3_0" / "data" / "processed"
    spl.mkdir(parents=True)
    pd.DataFrame({"SUBJECT_ID": sid, "split": split}).drop_duplicates().to_csv(spl / "subject_splits.csv", index=False)
    raw = base / "PICDataset"
    raw.mkdir(parents=True)
    st = pd.DataFrame({"SUBJECT_ID": sid, "HADM_ID": hadm, "ICUSTAY_ID": np.arange(n) + 1,
                       "INTIME": t0.strftime(FMT), "OUTTIME": out.strftime(FMT)})
    st.loc[i_unm, "OUTTIME"] = (out[i_unm] + pd.Timedelta(hours=1)).strftime(FMT)
    extra = [dict(SUBJECT_ID=sid[i_readm], HADM_ID=hadm[i_readm], INTIME=(out[i_readm] + pd.Timedelta(hours=50)),
                  OUTTIME=(out[i_readm] + pd.Timedelta(hours=98))),
             dict(SUBJECT_ID=sid[i_tie], HADM_ID=tie_h, INTIME=t0[i_tie], OUTTIME=out[i_tie]),
             dict(SUBJECT_ID=sid[0], HADM_ID=hadm[0], INTIME=t0[0], OUTTIME=t0[0])]          # invalid stay
    if ret24:
        extra.append(dict(SUBJECT_ID=sid[i_ret], HADM_ID=hadm[i_ret], INTIME=(out[i_ret] + pd.Timedelta(hours=10)),
                          OUTTIME=(out[i_ret] + pd.Timedelta(hours=40))))
    ex = pd.DataFrame(extra)
    ex["INTIME"] = pd.to_datetime(ex["INTIME"]).dt.strftime(FMT)
    ex["OUTTIME"] = pd.to_datetime(ex["OUTTIME"]).dt.strftime(FMT)
    ex["ICUSTAY_ID"] = np.arange(len(ex)) + 900001
    pd.concat([st, ex], ignore_index=True).to_csv(raw / "ICUSTAYS.csv", index=False)
    dis = disch.strftime(FMT).to_numpy().astype(object)
    dis[i_tie] = (out[i_tie] + pd.Timedelta(days=2)).strftime(FMT)
    dis[i_dmiss] = ""
    dis[i_readm] = (out[i_readm] + pd.Timedelta(days=9)).strftime(FMT)
    adm = pd.DataFrame({"SUBJECT_ID": sid, "HADM_ID": hadm, "ADMITTIME": admit.strftime(FMT), "DISCHTIME": dis})
    adm = pd.concat([adm, pd.DataFrame([{"SUBJECT_ID": sid[i_tie], "HADM_ID": tie_h,
                                         "ADMITTIME": admit[i_tie].strftime(FMT),
                                         "DISCHTIME": (out[i_tie] + pd.Timedelta(days=20)).strftime(FMT)}])])
    adm.to_csv(raw / "ADMISSIONS.csv", index=False)
    man = {"script": M.PINS["stage06_08_script"], "script_sha256": M.PINS["stage06_08_script_sha256"],
           "release_gate": "ok", "acceptance_checks": {"synthetic": "pass"}, "files": files,
           "input_hashes": {"canonical": sha(can_p)}}
    mp = d / "stage06_08_manifest.json"
    mp.write_text(json.dumps(man, indent=1))
    # Run 1-shaped output: matched-width bases fitted on all train rows (SVD V up to 59, CPQR order)
    from scipy.linalg import qr
    r1 = root / "data" / "processed" / "ladder" / "stage09_10_r1"
    r1.mkdir(parents=True)
    tr = split == "train"
    arrs = {}
    mats = {k: lv[k] for k in widths}
    for a, b in M.AB_PAIRS:
        mats[f"{a}+{b}"] = (np.c_[lv[a][0], lv[b][0]], lv[a][1] + lv[b][1])
    for k, (X, cols) in mats.items():
        _, _, Vt = np.linalg.svd(X[tr], full_matrices=False)
        _, _, piv = qr(X[tr], pivoting=True, mode="economic")
        key = k.replace("+", "_")
        arrs[f"{key}__V"] = Vt[:59].T
        arrs[f"{key}__cpqr"] = piv.astype(np.int32)
        arrs[f"{key}__columns"] = np.array(cols)
    for L in M.LANDMARKS:
        arrs[f"C_L{L}__V"] = np.eye(q, dtype=np.float32)
        arrs[f"C_L{L}__cpqr"] = np.array([2, 0, 4, 1, 3], dtype=np.int32)
        arrs[f"C_L{L}__channels"] = chans
    np.savez_compressed(r1 / "stage10_matched_width_bases_INTERNAL.npz", **arrs)
    (r1 / "stage09_summary_TEAM.csv").write_text("level,p\na0,12\n")
    r1files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(r1.iterdir())}
    m1 = {"script": M.PINS["run1_script"], "script_sha256": M.PINS["run1_script_sha256"], "release_gate": "ok",
          "acceptance_checks": {"synthetic": "pass"}, "input_hashes": {"stage06_08_manifest.json": sha(mp)},
          "files": r1files}
    r1m = r1 / "stage09_10_r1_manifest.json"
    r1m.write_text(json.dumps(m1, indent=1))
    return dict(root=root, man=sha(mp), r1=sha(r1m), eid=eid, sid=sid,
                planted=dict(readm=eid[i_readm], tie=eid[i_tie], unm=eid[i_unm], dmiss=eid[i_dmiss]))


@pytest.fixture(autouse=True)
def _no_real_pins(monkeypatch):
    for k in SYN_NULL_PINS:
        monkeypatch.setitem(M.PINS, k, None)      # synthetic inputs; the real pins are tested below


@pytest.fixture
def rel(tmp_path):
    return make_release(tmp_path)


def run(r, *extra, man=None, r1=None):
    return M.main(["--project-root", str(r["root"]), "--expect-stage06-08-manifest-sha256", man or r["man"],
                   "--expect-run1-manifest-sha256", r1 or r["r1"], *FAST, *extra])


def _cfg(r):
    return M.build_config(["--project-root", str(r["root"]), "--expect-stage06-08-manifest-sha256", r["man"],
                           "--expect-run1-manifest-sha256", r["r1"]])


# ------------------------------------------------------------------------------------------------ label units
def test_label_selftest_and_outcome1_bands():
    assert M.label_selftest()
    r = np.array([0.01, 24, 24.01, 48, 48.01, 72, 72.01, 120, 120.01, 168, 168.01, 240, 240.01, 336, 336.01])
    assert M.outcome1_level(r, np.zeros(len(r), bool)).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
    assert M.outcome1_level(np.array([1.0, 500.0]), np.array([True, True])).tolist() == [8, 8]   # death overrides


def test_outcome2_and_ordinal8_boundaries():
    assert M.outcome2_stay_level(np.array([0.0, 6.9999, 7.0, 13.9999, 14.0])).tolist() == [0, 0, 1, 1, 2]
    lv = M.ordinal8_level([71.99, 72.0, 167.99, 168.0, 336.0, 336.001, np.nan, 50],
                          [167.99, 335.99, 336.0, 10, 10, 10, 10, np.nan], np.zeros(8, bool))
    assert lv.tolist() == [0, 3, 4, 5, 5, 6, -1, -1]
    assert M.ordinal8_level([np.nan], [np.nan], np.array([True])).tolist() == [7]


def test_build_labels_planted_cases(rel):
    cfg = _cfg(rel)
    D = M.load_inputs(cfg)
    checks = {}
    Lb = M.build_labels(cfg, D, checks)
    pos = {e: i for i, e in enumerate(D["eid_all"])}
    p = rel["planted"]
    assert Lb["O2"][pos[p["readm"]]] == 3                        # readmission (gap 50 h) -> 3
    assert Lb["O2"][pos[p["tie"]]] == 2                          # tie: latest discharge (20 d) wins over 2 d
    assert Lb["O2"][pos[p["unm"]]] == -1 and Lb["O2"][pos[p["dmiss"]]] == -1
    info = Lb["info"]
    assert info["outcome2_unassignable"]["index_hadm_unmatched"] == "<10"
    assert info["outcome2_unassignable"]["dischtime_missing"] == "<10"
    assert info["outcome2_unassignable"]["dischtime_before_out"] == "none"
    assert info["index_hadm_ambiguous"] == "<10"
    d = Lb["died"]
    for L in M.LANDMARKS:
        el = Lb["out_h"] > L
        assert ((Lb["O1"][L][el] == 8) == d[el]).all() and (Lb["O1"][L][~el] == -1).all()
    assert ((Lb["O2"] == 4) == d).all() and ((Lb["O8"] == 7) == d).all()
    assert all(checks.values()), checks


def test_return_within_24h_stops(tmp_path, capsys):
    r = make_release(tmp_path, ret24=True)
    assert run(r) == 3
    cap = capsys.readouterr()
    assert "ICU return within 24 h" in cap.err and "nothing left behind" in cap.out
    assert not (r["root"] / OUT).exists()


def test_class_table_complementary_suppression():
    rng = np.random.default_rng(3)
    n = 3000
    out_h = rng.uniform(25, 400, n)
    died = rng.random(n) < 0.004                                  # rare deaths -> death cells suppressed
    O1 = {L: np.where(out_h > L, M.outcome1_level(out_h - L, died), -1) for L in M.LANDMARKS}
    O2 = np.where(died, 4, rng.choice([0, 1, 2, 3], n, p=[0.7, 0.2, 0.097, 0.003]))
    O8 = np.where(died, 7, rng.choice(7, n))
    split = rng.choice(["train", "validation", "test"], n, p=[0.6, 0.2, 0.2])
    checks = {}
    T = M.label_class_table(dict(out_h=out_h, O1=O1, O2=O2, O8=O8), split, checks)
    assert checks["11p_class_table_no_suppressed_cell_recoverable"]
    assert set(T.columns) == {"outcome", "landmark", "split", "class", "percent"}
    for _, g in T.groupby(["outcome", "landmark", "split"]):
        ns = (g["percent"] == "<10").sum()
        assert ns == 0 or ns >= 2                                   # never a single suppressed cell in a row
    death = {"outcome1": 8, "outcome2": 4, "ordinal8": 7}
    for (L, s), g in T.groupby(["landmark", "split"]):
        flags = {(g[(g["outcome"] == o) & (g["class"] == c)]["percent"] == "<10").any()
                 for o, c in death.items() if (g["outcome"] == o).any()}
        assert len(flags) == 1                                      # death cells linked across outcomes
    assert "test" not in set(T["split"])


# ---------------------------------------------------------------------------------------------- model units
def test_ordinal_linear_gradients():
    from scipy.optimize import check_grad
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4))
    y = np.clip((X[:, 0] * 2 + rng.standard_normal(80) + 4).astype(int), 0, 8)
    off = rng.standard_normal(80) * 0.3
    for kind in ("polr", "allthr"):
        m = M.OrdinalLinear(9, 0.5, kind)
        par = rng.standard_normal(4 + 8) * 0.3
        err = check_grad(lambda p: m._loss(p, X, y, off)[0], lambda p: m._loss(p, X, y, off)[1], par)
        assert err < 1e-4 * max(1.0, abs(m._loss(par, X, y, off)[0])), (kind, err)
        P = m.fit(X, y).predict_proba(X)
        assert np.allclose(P.sum(1), 1) and (P >= 0).all() and (np.diff(m.theta) > 0).all()


def test_frank_hall_monotone_and_metrics():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 5))
    y = np.clip((X[:, 0] * 2 + 4 + rng.standard_normal(200)).astype(int), 0, 8)
    for name in ("gb_fh", "rf_fh"):
        _, P = M.fit_predict(name, 9, X, y, X, hp={"max_iter": 20} if name == "gb_fh" else {"n_estimators": 10})
        assert np.allclose(P.sum(1), 1) and (P >= -1e-12).all()
    assert M.qwk(y, y, 9) == pytest.approx(1.0)
    W = np.ones((3, len(y)), dtype=np.float32)
    yh = np.clip(y + rng.integers(-1, 2, len(y)), 0, 8)
    assert np.allclose(M.boot_qwk(W, y, yh, 9), M.qwk(y, yh, 9))
    P = np.eye(9)[y]
    met = M.metrics(P, y, 9, y == 8)
    assert met["emd"] == 0 and met["rps"] == 0 and met["mae"] == 0
    assert M.median_class(np.array([[0.2, 0.3, 0.5]])).tolist() == [1]
    G = np.array([[0.9, 0.95, 0.2]])                                 # non-monotone exceedance -> rearranged
    Q = M.exceed_to_probs(G)
    assert np.allclose(Q.sum(1), 1) and (Q >= 0).all()


def test_gru_fit_predict_valid_probabilities():
    rng = np.random.default_rng(2)
    S = rng.standard_normal((120, 4, 7)).astype(np.float32)
    st = rng.standard_normal((120, 3))
    y = rng.integers(0, 5, 120)
    g = np.arange(120) // 2
    state, P, ep = M.gru_fit_predict(S[:90], st[:90], np.zeros(90), y[:90], g[:90], S[90:], st[90:], np.zeros(30),
                                     5, 8, 0, gamma0=np.linspace(-1, 1, 4), max_epochs=3)
    assert P.shape == (30, 5) and np.allclose(P.sum(1), 1, atol=1e-6) and (P >= 0).all() and 1 <= ep <= 3
    assert state["d_seq"] == 7 and state["d_static"] == 3


def test_calibration_monotone_and_crossfit():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 4, 300)
    P = rng.dirichlet(np.ones(4), 300)
    cals = M.fit_calibrators(P, y, 4)
    Pc = M.apply_calibrators(cals, P)
    assert np.allclose(Pc.sum(1), 1) and (Pc >= -1e-12).all()
    Px = M.crossfit_calibration(P, y, np.arange(300) // 3, 4, ("t",))
    assert np.allclose(Px.sum(1), 1)


# ------------------------------------------------------------------------------------------------- hash gate
def test_precheck_ok_prints_hashes(rel, capsys):
    assert run(rel, "--precheck-only") == 0
    o = capsys.readouterr().out
    assert "PRECHECK OK" in o and "stage09_10_r1_manifest.json sha256 " + rel["r1"] in o
    assert not (rel["root"] / OUT).exists()


def test_run1_manifest_sha_required_and_mismatch(rel):
    assert M.main(["--project-root", str(rel["root"]), "--expect-stage06-08-manifest-sha256", rel["man"],
                   "--precheck-only"]) == 2
    assert run(rel, "--precheck-only", r1="0" * 64) == 2


def test_run1_pin_enforced(rel, monkeypatch):
    monkeypatch.setitem(M.PINS, "run1_manifest", "f" * 64)
    assert run(rel, "--precheck-only") == 2


def test_run1_tampered_file_stops(rel, capsys):
    p = rel["root"] / "data/processed/ladder/stage09_10_r1/stage09_summary_TEAM.csv"
    p.write_text(p.read_text() + "a1,13\n")
    assert run(rel, "--precheck-only") == 2
    assert "Run 1 file stage09_summary_TEAM.csv" in capsys.readouterr().err


def test_run1_bases_pin_enforced(rel, monkeypatch):
    monkeypatch.setitem(M.PINS, "run1_bases", "e" * 64)
    assert run(rel, "--precheck-only") == 2


def test_run1_from_other_stage06_08_stops(rel):
    r1m = rel["root"] / "data/processed/ladder/stage09_10_r1/stage09_10_r1_manifest.json"
    m = json.loads(r1m.read_text())
    m["input_hashes"]["stage06_08_manifest.json"] = "1" * 64
    r1m.write_text(json.dumps(m))
    assert run(rel, "--precheck-only", r1=sha(r1m)) == 2


def test_stage06_08_manifest_mismatch_stops(rel):
    assert run(rel, "--precheck-only", man="0" * 64) == 2


@pytest.mark.parametrize("pin", ["harmonization", "split", "ICUSTAYS", "ADMISSIONS", "canonical"])
def test_input_pins_enforced(rel, monkeypatch, pin):
    monkeypatch.setitem(M.PINS, pin, "a" * 64)
    assert run(rel, "--precheck-only") == 2


def test_canonical_vs_stage06_08_manifest(rel):
    can = rel["root"] / "data/interim/canonical_icu_encounters.csv"
    can.write_text(can.read_text() + "\n")
    assert run(rel, "--precheck-only") == 2


def test_raw_local_path_refused(rel, tmp_path):
    bad = tmp_path / "x_raw_LOCAL"
    shutil.copytree(rel["root"].parent / "PICDataset", bad)
    assert run(rel, "--precheck-only", "--raw-dir", str(bad)) == 2


def test_nonempty_output_and_pending_refused(rel):
    (rel["root"] / OUT.parent / "stage11_12_17_r2.pending-7").mkdir(parents=True)
    assert run(rel, "--precheck-only") == 2
    (rel["root"] / OUT.parent / "stage11_12_17_r2.pending-7").rmdir()
    (rel["root"] / OUT).mkdir(parents=True)
    (rel["root"] / OUT / "old.txt").write_text("x")
    assert run(rel, "--precheck-only") == 2


# --------------------------------------------------------------------------------------- release gate (units)
def test_suppression_marker_and_hyperparameters_not_ids(tmp_path):
    ids = {"10", "5", "100", "31"}
    (tmp_path / "a_TEAM.csv").write_text("outcome,landmark,split,class,percent\noutcome1,24,train,5,<10\n"
                                         "outcome1,24,train,6,<10\n")
    (tmp_path / "b_TEAM.csv").write_text("config_id,model,lam,max_leaf_nodes,max_iter,qwk\n5,gb_fh,,31,100,0.4\n")
    assert not M.gate_problems(tmp_path, ["a_TEAM.csv", "b_TEAM.csv"], set(), ids)
    (tmp_path / "c_TEAM.csv").write_text("model,note\npolr,10\n")          # an integer in a free column is flagged
    assert M.gate_problems(tmp_path, ["c_TEAM.csv"], set(), ids) == [("c_TEAM.csv", "column note", M._ID_MSG)]


def test_gate_collects_all_problems_without_values(tmp_path):
    (tmp_path / "a_TEAM.csv").write_text("model,qwk\npolr,4646\n")
    (tmp_path / "b_TEAM.csv").write_text("model,n_train\n777,0.5\n")
    probs = M.gate_problems(tmp_path, ["a_TEAM.csv", "b_TEAM.csv"], {4646}, {"777"})
    assert ("a_TEAM.csv", "column qwk", M._SIZE_MSG) in probs
    assert ("b_TEAM.csv", "column names", "forbidden column name (count or identifier)") in probs
    assert ("b_TEAM.csv", "column model", M._ID_MSG) in probs
    with pytest.raises(M.BuildError) as ei:
        M.release_gate(tmp_path, ["a_TEAM.csv", "b_TEAM.csv"], {4646}, {"777"})
    assert "3 problem(s)" in str(ei.value) and "4646" not in str(ei.value) and "777" not in str(ei.value)


def test_audit_number_whitelist():
    assert M.audit_nums_ok({"stage11_winner": {"qwk": 0.4}, "config": {"n_boot": 1000, "landmarks": [24, 48]}})
    assert not M.audit_nums_ok({"labels": {"unassignable": 7}})
    assert not M.audit_nums_ok({"validation_size": 1234})


# ------------------------------------------------------------------------------------------- end to end (synthetic)
@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    for k in SYN_NULL_PINS:
        mp.setitem(M.PINS, k, None)
    outs = []
    for rep in range(2):
        r = make_release(tmp_path_factory.mktemp(f"e2e{rep}"))
        outs.append((r, run(r)))
    mp.undo()
    return outs


def _man(r):
    return json.loads((r["root"] / OUT / "stage11_12_17_r2_manifest.json").read_text())


def test_e2e_runs_and_publishes(e2e):
    r, rc = e2e[0]
    assert rc == 0
    out = r["root"] / OUT
    man = _man(r)
    assert not list(out.parent.glob("stage11_12_17_r2.pending-*"))
    for nm, rec in man["files"].items():
        assert sha(out / nm) == rec["sha256"]
    assert all(re.search(r"_TEAM\.(csv|png|json|md)$", n) for n in man["team_files"])
    assert set(man["internal_files"]) == {"stage11p_labels_landmark_INTERNAL.npz",
                                          "stage11_12_17_validation_predictions_INTERNAL.npz",
                                          "stage12_tuned_model_calibrators_INTERNAL.pkl",
                                          "stage17_models_INTERNAL.pt", "stage17_tabular_models_INTERNAL.pkl"}
    a = json.loads((out / "stage11_12_17_r2_audit_TEAM.json").read_text())
    assert a["release_gate"] == "ok" and all(v == "pass" for v in a["acceptance_checks"].values())
    assert a["stage11_winner"]["model"] != "multinom"
    sc = pd.read_csv(out / "stage11_validation_scores_TEAM.csv")
    assert set(sc["matrix"]) == set(M.MATRICES)
    ct = pd.read_csv(out / "stage17_contrast_TEAM.csv")
    assert set(ct[ct.outcome == "outcome1"]["landmark"]) == set(M.LANDMARKS)
    assert set(ct[ct.outcome == "outcome2"]["landmark"]) == set(M.CORE_LANDMARKS)
    assert set(ct[ct.outcome == "ordinal8"]["landmark"]) == set(M.ORD8_LANDMARKS)
    assert man["input_hashes"]["stage09_10_r1_manifest.json"] == r["r1"]
    assert not (out.parent / "_cache_stage11_12_17_r2" / "tmp").exists() or \
        not list((out.parent / "_cache_stage11_12_17_r2" / "tmp").iterdir())


def test_e2e_team_files_no_sizes_ids_or_counts(e2e):
    r, _ = e2e[0]
    out = r["root"] / OUT
    ids = set(r["eid"]) | set(r["sid"])
    z = np.load(r["root"] / "data/processed/ladder/stage06_08/stage08_a0.npz")
    sp = z["split"].astype(str)
    sizes = {len(sp), len(set(r["sid"]))} | {int((sp == s).sum()) for s in ("train", "validation", "test")}
    man = _man(r)
    assert not M.gate_problems(out, man["team_files"], {s for s in sizes if s >= 100}, ids)
    for nm in man["team_files"]:
        if nm.endswith(".csv"):
            for c in pd.read_csv(out / nm, nrows=0).columns:
                assert not re.search(r"(rows|subject|encounter|patient|^n$|cohort|count)", c, re.I), (nm, c)
    cl = pd.read_csv(out / "stage11p_label_classes_TEAM.csv", dtype=str)
    assert set(cl["split"]) <= {"train", "validation"}
    vals = cl["percent"][cl["percent"] != "<10"].astype(float)
    assert ((vals >= 0) & (vals <= 100)).all()
    text = (out / "stage11_12_17_r2_audit_TEAM.md").read_text()
    assert "test" not in " ".join(pd.read_csv(out / "stage17_validation_TEAM.csv", dtype=str).columns)
    assert "validation split only" in text


def test_e2e_deterministic(e2e):
    (r0, rc0), (r1, rc1) = e2e
    assert rc0 == 0 and rc1 == 0
    m0, m1 = _man(r0), _man(r1)
    for nm in m0["team_files"]:
        if nm.endswith((".png", "_audit_TEAM.json", "_audit_TEAM.md")):
            continue
        assert m0["files"][nm]["sha256"] == m1["files"][nm]["sha256"], nm


def test_e2e_rerun_into_existing_output_refused(e2e):
    r, _ = e2e[0]
    assert run(r) == 2


@pytest.fixture(scope="module")
def e2e_small(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    for k in SYN_NULL_PINS:
        mp.setitem(M.PINS, k, None)
    r = make_release(tmp_path_factory.mktemp("e2e_small_ids"), small_ids=True)
    rc = run(r)
    mp.undo()
    return r, rc, set(r["eid"]) | set(r["sid"])


def test_e2e_small_integer_ids_pass_gate(e2e_small):
    r, rc, ids = e2e_small
    assert rc == 0
    man = _man(r)
    assert man["release_gate"] == "ok"
    assert {"1", "5", "10", "24", "59", "100"} <= ids                 # collisions with k, class, landmark, <10
    assert not M.gate_problems(r["root"] / OUT, man["team_files"], set(), ids)


def test_planted_leaks_reported_without_values(e2e_small, tmp_path):
    r, _, ids = e2e_small
    out = tmp_path / "leak"
    shutil.copytree(r["root"] / OUT, out)
    man = json.loads((out / "stage11_12_17_r2_manifest.json").read_text())
    a = json.loads((out / "stage11_12_17_r2_audit_TEAM.json").read_text())
    allowed = M._audit_ints(a, set()) | M.script_text_ints(a)
    leak = max((i for i in ids if i.isdigit() and int(i) not in allowed and len(i) >= 3), key=int)
    p = out / "stage11_winner_TEAM.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df.loc[0, "model"] = leak
    p.write_text(df.to_csv(index=False, lineterminator="\n"))
    md = out / "stage11_12_17_r2_audit_TEAM.md"
    md.write_text(md.read_text() + f"\nnote: encounter {leak} reviewed\n")
    cl = out / "stage11p_label_classes_TEAM.csv"
    cl.write_text(cl.read_text() + f"outcome1,24,train,0,{leak}\n")
    probs = M.gate_problems(out, man["team_files"], set(), ids)
    assert ("stage11_winner_TEAM.csv", "column model", M._ID_MSG) in probs
    assert ("stage11_12_17_r2_audit_TEAM.md", "md text", M._ID_MSG) in probs
    assert ("stage11p_label_classes_TEAM.csv", "column percent", M._ID_MSG) in probs
    with pytest.raises(M.BuildError) as ei:
        M.release_gate(out, man["team_files"], set(), ids)
    assert leak not in str(ei.value)


# ------------------------------------------------------------------------------------- runtime environment
def _boom_lib():
    json.loads("{not json")


def _boom_outer():
    _boom_lib()


def test_describe_exception_frames_only():
    try:
        _boom_outer()
    except Exception as e:  # noqa: BLE001
        lines = M.describe_exception(e, None)
    assert lines[0] == "unexpected JSONDecodeError"
    frames = [ln for ln in lines if ln.startswith("  frame ")]
    assert any("_boom_outer" in ln for ln in frames)
    assert any("[json.decoder]" in ln or "[json]" in ln for ln in frames)
    for ln in frames:
        loc = ln.split()[1]
        assert "/" not in loc and "\\" not in loc and re.fullmatch(r"[^:]+:\d+:[\w.<>]+", loc), loc
    assert not any("not json" in ln or "Expecting" in ln for ln in lines)


def test_main_unexpected_error_prints_frames(rel, monkeypatch, capsys):
    def bad_build(cfg):
        raise PermissionError(13, "denied", str(Path.home() / "x" / "secret.bin"))
    monkeypatch.setattr(M, "build", bad_build)
    assert run(rel) == 3
    err = capsys.readouterr().err
    assert "unexpected PermissionError" in err and "bad_build" in err
    assert "secret.bin" in err and str(Path.home()) not in err and "denied" not in err


def test_runtime_env_setup_and_restore(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setenv("MPLCONFIGDIR", "/somewhere/else")
    old_tmp = tempfile.gettempdir()
    cache = tmp_path / "data/processed/ladder/_cache_stage11_12_17_r2"
    prev = M.setup_runtime_env(cache)
    try:
        for k in ("MPLCONFIGDIR", "JOBLIB_TEMP_FOLDER", "SCIKIT_LEARN_DATA", "XDG_CACHE_HOME", "TMP", "TEMP", "TMPDIR"):
            assert Path(os.environ[k]).resolve().is_relative_to(cache.resolve()), k
    finally:
        M.restore_runtime_env(prev)
    assert os.environ["MPLCONFIGDIR"] == "/somewhere/else" and tempfile.gettempdir() == old_tmp


def test_import_failure_is_reported(rel, monkeypatch, capsys):
    monkeypatch.setattr(M, "HEAVY_IMPORTS", ("sklearn", "torch_not_a_real_module_xyz"))
    monkeypatch.setitem(M._PIP, "torch_not_a_real_module_xyz", "nothing")
    assert run(rel, "--precheck-only") == 2
    assert "import of torch_not_a_real_module_xyz failed" in capsys.readouterr().err


def test_cache_folder_is_not_a_leftover(rel):
    assert run(rel, "--precheck-only") == 0
    cache = rel["root"] / OUT.parent / "_cache_stage11_12_17_r2"
    assert cache.is_dir() and (cache / "matplotlib").is_dir()
    assert run(rel, "--precheck-only") == 0
    assert not list((cache / "tmp").iterdir())


def test_time_limit_stops_cleanly(rel, capsys):
    assert run(rel, "--max-seconds", "4") == 3
    cap = capsys.readouterr()
    assert "result: FAILED" in cap.out
    assert not list((rel["root"] / OUT.parent).glob("stage11_12_17_r2.pending-*"))
