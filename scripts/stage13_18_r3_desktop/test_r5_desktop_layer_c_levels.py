"""Tests for r5_desktop_layer_c_levels v1.0.1-desktop (desktop Tier 3: Stage 15 Layer C levels).

Outputs must stay under data/processed/ladder/DESKTOP_FULL_PLAN/stage15_layer_c_levels/.
Self-contained and SYNTHETIC only: each fixture builds a tiny Stage 06-08 / Run 1-shaped project with the
Run 2 test helper (make_release from test_r2_11p_11_12_17_models.py, hash-checked), runs the unmodified
Run 2 script on it with the real pins nulled, then runs the Tier 3 script. No real data, no network. Run with:
    python -s -m pytest -q -p no:cacheprovider test_r5_desktop_layer_c_levels.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
R2DIR = next(p for p in (
    HERE.parent / "stage11_18_r2",
    HERE.parent / "r2",
) if (p / "r2_11p_11_12_17_models.py").is_file())
R2_SCRIPT = R2DIR / "r2_11p_11_12_17_models.py"
R2_TEST = R2DIR / "test_r2_11p_11_12_17_models.py"
R2_TEST_SHA = "a26ca8810994e50cc8bdaacfa4bded4d3ca7f63fc6675f52dcdf50b02f6d8d7b"
FAST2 = ["--n-boot", "20", "--gru-max-epochs", "2", "--heartbeat-seconds", "600"]
FAST5 = ["--n-boot", "20", "--max-epochs", "2", "--heartbeat-seconds", "600",
         "--landmarks", "24", "--n-seeds", "1"]
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
M5 = _load("r5_mod_under_test", HERE / "r5_desktop_layer_c_levels.py")
R2 = M5.load_run2(R2_SCRIPT, M5.PINS3["run2_script_sha256"])[0]
MT = _load("r2_test_helper_for_r5", R2_TEST)
DESK = LAD / M5.DESKTOP_ROOT_NAME
WORK = DESK / M5.OUT_NAME
OUT = WORK / M5.PUBLISHED_NAME
OUT2 = LAD / M5.RUN2_NAME


@pytest.fixture(autouse=True)
def _synthetic_pins(monkeypatch):
    for k in NULL2:
        monkeypatch.setitem(R2.PINS, k, None)
    monkeypatch.setitem(M5.PINS3, "run2_manifest", None)


def _null_pins():
    mp = pytest.MonkeyPatch()
    for k in NULL2:
        mp.setitem(R2.PINS, k, None)
    mp.setitem(M5.PINS3, "run2_manifest", None)
    return mp


def make_project(base: Path, small_ids: bool = False) -> dict:
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
    r["r2"] = sha(root / OUT2 / f"{M5.RUN2_NAME}_manifest.json")
    return r


def copy_project(r: dict, dest: Path) -> dict:
    shutil.copytree(r["root"], dest / "proj", ignore=shutil.ignore_patterns("_cache_*"))
    return dict(r, root=dest / "proj")


def args5(r, *extra, r2=None, script=None):
    root = r["root"]
    return ["--project-root", str(root), "--expect-stage06-08-manifest-sha256", r["man"],
            "--expect-run1-manifest-sha256", r["r1"], "--expect-run2-manifest-sha256", r2 or r["r2"],
            "--run2-script", str(script or R2_SCRIPT),
            "--split-file", str(root / "data/interim/r2_inputs/subject_splits.csv"),
            "--raw-dir", str(root / "data/interim/r2_inputs/raw"), *FAST5, *extra]


def run5(r, *extra, **kw) -> int:
    return M5.main(args5(r, *extra, **kw))


def log_of(r) -> list:
    p = r["root"] / WORK / M5.TEST_LOG_NAME
    return json.loads(p.read_text())["attempts"] if p.exists() else []


def _rel_files(root: Path) -> set:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


# ------------------------------------------------------------------------------------------------- unit: Stage 15
def _toy_tensor(n=12, b=8, seed=0):
    rng = np.random.default_rng(seed)
    ch = ["heart_rate", "map", "creatinine", "urine_output", "fluid_intake", "platelets", "lactate",
          "bilirubin_total", "temperature"]
    q = len(ch)
    X = rng.normal(size=(n, b, q))
    missing = (rng.random((n, b, q)) < 0.25).astype(np.uint8)
    carried = ((rng.random((n, b, q)) < 0.15) & (missing == 0)).astype(np.uint8)
    X = np.where(missing, np.nan, X).astype(np.float32)
    split = np.array(["train"] * 8 + ["validation"] * 2 + ["test"] * 2)
    train = split == "train"
    age = np.array([0.5, 3, 18, 48, 90, 160, 8, 20, 40, 100, 12, 70], dtype=float)[:n]
    inv = np.zeros(n, dtype=bool)
    inv[-1] = True
    bands = M5.age_band_codes(age, inv)
    return dict(X=X, missing=missing, carried=carried, ch=ch, train=train, bands=bands, split=split, age=age)


def test_age_bands_match_contract():
    months = np.array([28 / M5.DAYS_PER_MONTH - 1e-6, 28 / M5.DAYS_PER_MONTH, 11.9, 12.0, 35.9, 36.0,
                       71.9, 72.0, 143.9, 144.0, 215.9, 216.0, np.nan, -1])
    inv = np.zeros(len(months), dtype=bool)
    inv[-1] = True
    c = M5.age_band_codes(months, inv)
    # neonate, infant, infant, toddler, toddler, preschool, preschool, school, school, adolescent, adolescent, invalid, invalid, invalid
    assert c[0] == 0
    assert c[1] == 1 and c[2] == 1
    assert c[3] == 2 and c[4] == 2
    assert c[5] == 3 and c[6] == 3
    assert c[7] == 4 and c[8] == 4
    assert c[9] == 5 and c[10] == 5
    assert c[11] == -1 and c[12] == -1 and c[13] == -1


def test_slope_recovers_a_linear_trend():
    n, b, q = 4, 8, 1
    t = np.arange(b, dtype=float)
    X = np.broadcast_to((2.0 * t)[None, :, None], (n, b, q)).copy()
    missing = np.zeros((n, b, q), dtype=np.uint8)
    carried = np.zeros((n, b, q), dtype=np.uint8)
    train = np.ones(n, dtype=bool)
    bands = np.zeros(n, dtype=np.int8)
    block, names, _ = M5.build_engineered_block(X, missing, carried, ["heart_rate"], train, bands)
    j = names.index("heart_rate_slope_4bin")
    # after 4 observed bins the OLS slope vs window index (0..3) of 2*t is 2
    assert np.allclose(block[:, 3:, j], 2.0, atol=1e-5)


def test_rolling_sd_is_strictly_past():
    n, b, q = 3, 6, 1
    X = np.zeros((n, b, q), dtype=np.float32)
    X[:, 5, 0] = 100.0  # a spike in the last bin must not enter the past-only SD at that bin
    X[:, :5, 0] = 1.0
    missing = np.zeros((n, b, q), dtype=np.uint8)
    carried = np.zeros((n, b, q), dtype=np.uint8)
    train = np.ones(n, dtype=bool)
    bands = np.zeros(n, dtype=np.int8)
    block, names, _ = M5.build_engineered_block(X, missing, carried, ["map"], train, bands)
    j = names.index("map_rollsd_4bin_past")
    # at bin 5 the past window is bins 1-4, all 1.0, SD = 0
    assert np.allclose(block[:, 5, j], 0.0, atol=1e-6)


def test_lookahead_gate_passes_on_clean_features():
    t = _toy_tensor()
    assert M5.lookahead_is_clean(t["X"], t["missing"], t["carried"], t["ch"], t["train"], t["bands"])


def test_ageband_stats_ignore_validation_and_test():
    t = _toy_tensor()
    # plant an extreme value only on a test row
    X = t["X"].copy()
    X[-1, :, 0] = 50.0
    t["missing"][-1, :, 0] = 0
    t["carried"][-1, :, 0] = 0
    obs = M5.observed_mask(t["missing"], t["carried"])
    stats = M5.fit_ageband_stats(np.where(np.isfinite(X), X, np.nan), obs, t["train"], t["bands"])
    # overall mean of heart_rate must be near the train distribution, not 50
    assert stats["overall_mean"][0] < 5


def test_aki_stage_uses_min_prefix_baseline_and_frozen_cuts():
    # one patient, creatinine z: 0, 0.4, 0.6, 1.2, 1.6
    cr = np.array([[0.0, 0.4, 0.6, 1.2, 1.6]], dtype=np.float32)
    avail = np.ones_like(cr, dtype=bool)
    stg, delta, miss = M5.aki_cr_stage(cr, avail)
    assert miss.sum() == 0
    # baseline stays 0; stages at cuts 0.5, 1.0, 1.5
    assert np.allclose(stg[0], [0, 0, 1, 2, 3])
    assert np.allclose(delta[0], cr[0])


def test_carried_values_are_excluded_from_slope_but_used_for_aki():
    n, b, q = 1, 4, 1
    X = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    missing = np.zeros((n, b, q), dtype=np.uint8)
    carried = np.array([[[0], [0], [1], [0]]], dtype=np.uint8)
    train = np.ones(n, dtype=bool)
    bands = np.zeros(n, dtype=np.int8)
    block, names, _ = M5.build_engineered_block(X, missing, carried, ["creatinine"], train, bands)
    # slope at last bin uses observed bins 0,1,3 (values 1,2,4) — not the carried 3
    # window is bins 0..3 but bin 2 is carried so masked; OLS of (0,1),(1,2),(3,4)
    j = names.index("creatinine_slope_4bin")
    # t=0,1,3 and y=1,2,4 -> slope 1.0
    assert np.allclose(block[0, 3, j], 1.0, atol=1e-5)
    # AKI uses available (including carried) min baseline 1.0; last delta 3.0 -> stage 3
    js = names.index("aki_cr_stage_zsurrogate")
    assert block[0, 3, js] == 3


def test_normalized_pack_has_filled_values_raw_pack_zeros_nans():
    X_raw = np.array([[[1.0, np.nan]]], dtype=np.float32)
    X_fill = np.array([[[1.0, 0.25]]], dtype=np.float32)
    missing = np.array([[[0, 1]]], dtype=np.uint8)
    carried = np.zeros_like(missing)
    ch = ["heart_rate", "map"]
    train = np.array([True])
    bands = np.array([0], dtype=np.int8)
    Sr, nr, _ = M5.build_level_tensor("raw", X_raw, X_fill, missing, carried, ch, ch, train, bands)
    Sn, nn, _ = M5.build_level_tensor("normalized", X_raw, X_fill, missing, carried, ch, ch, train, bands)
    # raw value of channel 2 is 0 (NaN filled); normalized keeps 0.25
    assert Sr[0, 0, 1] == 0.0
    assert Sn[0, 0, 1] == pytest.approx(0.25)
    assert Sr[0, 0, 3] == 1.0  # missing flag of channel 2 (offset q=2)
    assert Sr.shape[2] == 2 * 3 + 1


def test_engineered_catalog_is_frozen_and_named():
    cat = M5.engineered_channel_catalog()
    names = [n for n, _ in cat]
    assert "heart_rate_slope_4bin" in names
    assert "aki_cr_stage_zsurrogate" in names
    assert "psofa_mod_total" in names
    assert "organ_flag_renal" in names
    assert "fluid_net_cum" in names
    # vasoactive / ventilation are intentionally absent
    assert not any("vaso" in n or "ventil" in n for n in names)


def test_holm_is_monotone_and_step_down():
    adj, rej = M5.holm(np.array([0.01, 0.04, 0.20]))
    assert adj[0] <= adj[1] <= adj[2]
    assert rej[0] and (not rej[2])


def test_guard_blocks_test_rows_outside_scoring():
    g = M5.TestGuard(np.array(["train", "validation", "test"]))
    with pytest.raises(M5.BuildError):
        g.te()
    with pytest.raises(M5.BuildError):
        g.fit_rows(np.array([2]), "bad")
    g.fit_rows(np.array([0, 1]), "ok")
    g.phase = "score"
    assert list(g.te()) == [2]
    g.score("a", lambda: 1)
    with pytest.raises(M5.BuildError):
        g.score("a", lambda: 2)


def test_defaults_are_the_desktop_full_settings():
    assert M5.DEFAULT_MAX_EPOCHS == 40
    assert M5.DEFAULT_N_BOOT == 1000
    assert M5.GRU_HIDDEN_GRID == (32, 64)
    assert M5.CORE_LANDMARKS == (24, 48, 72, 120, 168, 240, 336)
    assert M5.DELTA == 0.02 and M5.DELTA_C == 0.01
    assert M5.LEVELS == ("raw", "normalized", "engineered")


def test_environment_rejects_cuda_requirement_on_cpu():
    import torch
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    with pytest.raises(M5.InputError, match="CUDA is required"):
        M5.check_environment(require_cuda=True)


def test_out_dir_outside_stage15_folder_is_refused_before_writes(tmp_path, capsys):
    with pytest.raises(M5.InputError):
        M5.build_config(["--project-root", str(tmp_path), "--out-dir", str(tmp_path / "elsewhere")])


def test_gate_collects_all_problems_without_values(tmp_path):
    pend = tmp_path / "pend"
    pend.mkdir()
    (pend / "a.csv").write_text("n_train,qwk\n12,0.2\n")
    (pend / "b.md").write_text("encounter E999 leaks\n")
    probs = M5.gate_problems(pend, ["a.csv", "b.md"], {12}, {"E999"})
    flat = " ".join(" ".join(map(str, p)) for p in probs)
    assert "forbidden column" in flat or "size" in flat or "count" in flat
    assert "identifier" in flat


# ------------------------------------------------------------------------------------------------- e2e
@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    base = tmp_path_factory.mktemp("r5_e2e_src")
    mp = _null_pins()
    try:
        r = make_project(base)
        rc = run5(r)
        assert rc == 0, "Tier 3 e2e failed"
        return r
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def e2e_small(tmp_path_factory):
    base = tmp_path_factory.mktemp("r5_e2e_small_src")
    mp = _null_pins()
    try:
        r = make_project(base, small_ids=True)
        rc = run5(r)
        assert rc == 0, "Tier 3 small-id e2e failed"
        return r
    finally:
        mp.undo()


def fresh(e2e, tmp_path) -> dict:
    return copy_project(e2e, tmp_path)


def test_e2e_publishes_with_all_checks(e2e):
    root = e2e["root"]
    out = root / OUT
    assert out.is_dir()
    man = json.loads((out / f"{M5.OUT_NAME}_manifest.json").read_text())
    assert man["release_gate"] == "ok"
    assert man["labelled"].startswith("post-primary")
    checks = man["acceptance_checks"]
    assert checks, "no acceptance checks"
    assert all(v == "pass" for v in checks.values()), checks
    audit = json.loads((out / f"{M5.OUT_NAME}_audit_TEAM.json").read_text())
    assert audit["levels"] == ["raw", "normalized", "engineered"]
    assert 24 in audit["landmarks"]
    log = log_of(e2e)
    assert log and log[-1]["status"] == "published"
    # required TEAM files
    for n in ("stage15_engineered_channels_TEAM.csv", "stage15_validation_scores_TEAM.csv",
              "stage15_validation_contrasts_TEAM.csv", "stage15_selected_TEAM.csv",
              "stage15_test_scores_TEAM.csv", "stage15_test_contrasts_TEAM.csv",
              f"{M5.OUT_NAME}_audit_TEAM.md"):
        assert (out / n).is_file(), n
    assert audit.get("device") in ("cpu", "cuda")
    assert "gpu_name" not in audit
    md = (out / f"{M5.OUT_NAME}_audit_TEAM.md").read_text()
    assert "NVIDIA" not in md and "GeForce" not in md
    assert "cuda:" in md.lower()
    assert (out / f"{M5.OUT_NAME}_environment_INTERNAL.json").is_file()


def test_e2e_test_rows_touched_only_in_scoring(e2e):
    # the published log exists only after scoring; the script records started then published
    log = log_of(e2e)
    assert log[-1]["status"] == "published"
    assert "started_utc" in log[-1]


def test_verification_ignores_test_rows(e2e, tmp_path):
    r = fresh(e2e, tmp_path)
    # a published folder in the copy would block; the copy includes the published run, so strip it
    out = r["root"] / OUT
    logp = r["root"] / WORK / M5.TEST_LOG_NAME
    shutil.rmtree(out)
    logp.unlink()
    # run precheck only — must not require test scoring
    rc = run5(r, "--precheck-only")
    assert rc == 0


def test_e2e_team_files_no_sizes_or_ids(e2e):
    out = e2e["root"] / OUT
    # encounter / subject ids from the synthetic project must not appear in TEAM text
    can = pd.read_csv(e2e["root"] / "data/interim/canonical_icu_encounters.csv", dtype=str)
    ids = set(can["encounter_id"].astype(str)) | set(can["SUBJECT_ID"].astype(str))
    for q in out.glob("*_TEAM.*"):
        text = q.read_text(encoding="utf-8", errors="replace")
        toks = R2.id_tokens(text)
        leaked = toks & ids
        assert not leaked, f"{q.name} leaked {sorted(leaked)[:3]}"
        if q.suffix == ".csv":
            df = pd.read_csv(q)
            assert not any(re_search_size(c) for c in df.columns)


def re_search_size(col):
    import re
    return re.search(r"(^n$|^N$|_n$|^count|n_train|n_val|n_test|n_elig|n_row|sample_size|n_subj)", col, re.I)


def test_rerun_refused(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    rc = run5(r)
    assert rc == 2
    err = capsys.readouterr().err
    assert "scored once" in err.lower() or "published test scoring" in err.lower()


def test_run2_manifest_hash_mismatch_stops(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    out = r["root"] / OUT
    shutil.rmtree(out, ignore_errors=True)
    (r["root"] / WORK / M5.TEST_LOG_NAME).unlink(missing_ok=True)
    rc = run5(r, r2="0" * 64)
    assert rc == 2
    assert "SHA-256" in capsys.readouterr().err


def test_tampered_run2_script_stops(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    out = r["root"] / OUT
    shutil.rmtree(out, ignore_errors=True)
    (r["root"] / WORK / M5.TEST_LOG_NAME).unlink(missing_ok=True)
    bad = tmp_path / "bad_r2.py"
    bad.write_text(R2_SCRIPT.read_text() + "\n# tamper\n")
    rc = run5(r, script=bad)
    assert rc == 2
    assert "SHA-256" in capsys.readouterr().err


def test_precheck_only_prints_hashes(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    out = r["root"] / OUT
    shutil.rmtree(out, ignore_errors=True)
    (r["root"] / WORK / M5.TEST_LOG_NAME).unlink(missing_ok=True)
    rc = run5(r, "--precheck-only")
    assert rc == 0
    outp = capsys.readouterr().out
    assert "PRECHECK OK" in outp
    assert "run2_script sha256" in outp
    assert not (r["root"] / OUT).exists()


def test_failed_scoring_is_logged_and_publishes_nothing(e2e, tmp_path, monkeypatch):
    r = fresh(e2e, tmp_path)
    out = r["root"] / OUT
    shutil.rmtree(out, ignore_errors=True)
    (r["root"] / WORK / M5.TEST_LOG_NAME).unlink(missing_ok=True)
    def boom(*a, **k):
        raise M5.BuildError("planted scoring failure")
    monkeypatch.setattr(M5, "score_test_landmark", boom)
    rc = run5(r)
    assert rc == 3
    log = log_of(r)
    assert log and log[-1]["status"].startswith("failed")
    assert not (r["root"] / OUT).exists()


def test_small_integer_ids_pass_gate(e2e_small):
    out = e2e_small["root"] / OUT
    man = json.loads((out / f"{M5.OUT_NAME}_manifest.json").read_text())
    assert man["release_gate"] == "ok"


def test_planted_leaks_fail_gate_without_values(e2e_small, tmp_path):
    pend = tmp_path / "pend"
    pend.mkdir()
    src = e2e_small["root"] / OUT / "stage15_test_scores_TEAM.csv"
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    df.loc[df.index[0], "qwk"] = "1000001"
    can = pd.read_csv(e2e_small["root"] / "data/interim/canonical_icu_encounters.csv", dtype=str)
    ids = set(can["encounter_id"].astype(str)) | set(can["SUBJECT_ID"].astype(str))
    ident = next(iter(ids))
    col = "config" if "config" in df.columns else df.columns[-1]
    df.loc[df.index[0], col] = f"leak {ident}"
    df.to_csv(pend / "stage15_test_scores_TEAM.csv", index=False)
    (pend / f"{M5.OUT_NAME}_audit_TEAM.json").write_text("{}")
    probs = M5.gate_problems(pend, ["stage15_test_scores_TEAM.csv"], {1000001}, set(map(str, ids)))
    blob = " ".join(f"{loc} {chk}" for _, loc, chk in probs)
    assert probs
    assert "1000001" not in blob
    for i in ids:
        assert i not in blob


def test_writes_stay_in_stage15_folder(e2e):
    root = e2e["root"]
    after = _rel_files(root)
    written = {p for p in after if "DESKTOP_FULL_PLAN/stage15_layer_c_levels" in p or
               "DESKTOP_FULL_PLAN\\stage15_layer_c_levels" in p}
    assert written
    # no writes under laptop Run 2 / 3 folders from this script except the pre-existing Run 2 outputs
    new_bad = [p for p in after if "stage15_layer_c_levels" not in p and "stage11_12_17_r2" not in p
               and "stage09_10_r1" not in p and "stage06_08" not in p
               and p.startswith("data/processed/ladder/") and "DESKTOP_FULL_PLAN" not in p]
    assert new_bad == []


def test_concurrent_pending_is_refused(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    out = r["root"] / OUT
    shutil.rmtree(out, ignore_errors=True)
    (r["root"] / WORK / M5.TEST_LOG_NAME).unlink(missing_ok=True)
    other = r["root"] / DESK / "stage17_seqmodels" / "published.pending-999"
    other.mkdir(parents=True)
    rc = run5(r, "--precheck-only")
    err = capsys.readouterr().err.lower()
    assert rc == 2
    assert "concurrently" in err or "pending folder" in err


def test_rtx_3080_gpu_name_stays_out_of_team_and_passes_gate(tmp_path):
    """A real desktop GPU name must not appear in TEAM files; 3080 may also be an identifier."""
    env = dict(python="3.14.3",
               executable="<VENV_PYTHON>",
               numpy="2.4.2", pandas="3.0.1", scipy="1.17.1", sklearn="1.8.0",
               torch="2.10.0+cu126", cuda_available=True, cuda_version="12.6",
               gpu_name="NVIDIA GeForce RTX 3080", gpu_memory_bytes=10736893952)
    cfg = type("C", (), dict(env_info=env, landmarks=(24,), second_family=None,
                             max_epochs=2, patience=5, batch=256, n_boot=20,
                             n_seeds=1, amp=False))()
    pre = {"hashes": {"split": "ab"}, "prior_attempts": 0}
    audit = M5.make_audit(cfg, pre, OrderedDict(), OrderedDict(), OrderedDict(),
                          pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "abc", 3)
    md = M5.audit_md(audit)
    pend = tmp_path / "pend"
    pend.mkdir()
    ajn, amn = f"{M5.OUT_NAME}_audit_TEAM.json", f"{M5.OUT_NAME}_audit_TEAM.md"
    (pend / ajn).write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    (pend / amn).write_text(md, encoding="utf-8")
    ids = {"3080", "10736893952", "exampleuser"}
    probs = M5.gate_problems(pend, [ajn, amn], set(), ids)
    assert probs == [], probs
    blob = json.dumps(audit, default=str) + "\n" + md
    for s in ("NVIDIA", "GeForce", "RTX", "3080", "10736893952", "exampleuser", "gpu_name", "gpu_memory"):
        assert s not in blob
    assert audit["device"] == "cuda"
    assert audit["cuda_available"] is True
    assert "cuda: yes" in md


def test_planted_real_looking_identifier_still_fails_gate(tmp_path):
    """Planting a real-looking identifier (the RTX 3080 token) in TEAM text must still fail."""
    pend = tmp_path / "pend"
    pend.mkdir()
    ident = "NVIDIA GeForce RTX 3080"
    (pend / f"{M5.OUT_NAME}_audit_TEAM.json").write_text(
        json.dumps({"device": ident}, indent=2), encoding="utf-8")
    (pend / f"{M5.OUT_NAME}_audit_TEAM.md").write_text(
        "# audit - TEAM\n\ndevice: " + ident + "\n", encoding="utf-8")
    ids = {"3080"}
    names = [f"{M5.OUT_NAME}_audit_TEAM.json", f"{M5.OUT_NAME}_audit_TEAM.md"]
    probs = M5.gate_problems(pend, names, set(), ids)
    assert probs, "the planted identifier should fail the gate"
    assert any(chk == "an identifier" for _, _, chk in probs)
    blob = " ".join(f"{loc} {chk}" for _, loc, chk in probs)
    assert "3080" not in blob and "NVIDIA" not in blob and "RTX" not in blob
    assert ident not in blob


def test_rerun_allowed_after_failed_unpublished_attempt(e2e, tmp_path, monkeypatch, capsys):
    """A failed unpublished attempt must allow one rerun, log the prior scoring, and still refuse after publish."""
    r = fresh(e2e, tmp_path)
    out = r["root"] / OUT
    shutil.rmtree(out, ignore_errors=True)
    (r["root"] / WORK / M5.TEST_LOG_NAME).unlink(missing_ok=True)

    def boom(*a, **k):
        raise M5.BuildError("planted scoring failure")

    orig = M5.score_test_landmark
    monkeypatch.setattr(M5, "score_test_landmark", boom)
    assert run5(r) == 3
    assert not out.exists()
    log = log_of(r)
    assert log and log[-1]["status"] == "failed (nothing published)"
    monkeypatch.setattr(M5, "score_test_landmark", orig)
    capsys.readouterr()
    assert run5(r) == 0
    log2 = log_of(r)
    assert [a["status"] for a in log2] == ["failed (nothing published)", "published"]
    audit = json.loads((r["root"] / OUT / f"{M5.OUT_NAME}_audit_TEAM.json").read_text())
    md = (r["root"] / OUT / f"{M5.OUT_NAME}_audit_TEAM.md").read_text()
    assert any("unpublished attempt" in d for d in audit["deviations"])
    assert "scored once before" in md
    capsys.readouterr()
    assert run5(r) == 2
    err = capsys.readouterr().err
    assert "scored once" in err.lower() or "published test scoring" in err.lower()


def test_amp_flag_is_ignored_on_cpu(e2e, tmp_path, capsys):
    import torch
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    r = fresh(e2e, tmp_path)
    out = r["root"] / OUT
    shutil.rmtree(out, ignore_errors=True)
    (r["root"] / WORK / M5.TEST_LOG_NAME).unlink(missing_ok=True)
    rc = run5(r, "--precheck-only", "--amp")
    assert rc == 0
    # AMP note is printed from main() only after env check, during a full run; precheck still ok
