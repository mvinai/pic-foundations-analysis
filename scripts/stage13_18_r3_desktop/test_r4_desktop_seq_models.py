"""Tests for r4_desktop_seq_models v1.0.1-desktop (desktop Tier 2: LSTM, LSTM-attention, transformer).

Outputs must stay under data/processed/ladder/DESKTOP_FULL_PLAN/stage17_seqmodels/.
Self-contained and SYNTHETIC only: each fixture builds a tiny Stage 06-08 / Run 1-shaped project with the
Run 2 test helper (make_release from test_r2_11p_11_12_17_models.py, hash-checked), runs the unmodified
Run 2 script on it with the real pins nulled, then runs the Tier 2 script. No real data, no network. Run with:
    python -s -m pytest -q -p no:cacheprovider test_r4_desktop_seq_models.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import pickle
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
FAST4 = ["--n-boot", "20", "--max-epochs", "2", "--heartbeat-seconds", "600",
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
M4 = _load("r4_mod_under_test", HERE / "r4_desktop_seq_models.py")
R2 = M4.load_run2(R2_SCRIPT, M4.PINS3["run2_script_sha256"])[0]
MT = _load("r2_test_helper_for_r4", R2_TEST)
DESK = LAD / M4.DESKTOP_ROOT_NAME
WORK = DESK / M4.OUT_NAME
OUT = WORK / M4.PUBLISHED_NAME
OUT2 = LAD / M4.RUN2_NAME


@pytest.fixture(autouse=True)
def _synthetic_pins(monkeypatch):
    for k in NULL2:
        monkeypatch.setitem(R2.PINS, k, None)
    monkeypatch.setitem(M4.PINS3, "run2_manifest", None)


def _null_pins():
    mp = pytest.MonkeyPatch()
    for k in NULL2:
        mp.setitem(R2.PINS, k, None)
    mp.setitem(M4.PINS3, "run2_manifest", None)
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
    r["r2"] = sha(root / OUT2 / f"{M4.RUN2_NAME}_manifest.json")
    return r


def copy_project(r: dict, dest: Path) -> dict:
    shutil.copytree(r["root"], dest / "proj", ignore=shutil.ignore_patterns("_cache_*"))
    return dict(r, root=dest / "proj")


def args4(r, *extra, r2=None, script=None):
    root = r["root"]
    return ["--project-root", str(root), "--expect-stage06-08-manifest-sha256", r["man"],
            "--expect-run1-manifest-sha256", r["r1"], "--expect-run2-manifest-sha256", r2 or r["r2"],
            "--run2-script", str(script or R2_SCRIPT),
            "--split-file", str(root / "data/interim/r2_inputs/subject_splits.csv"),
            "--raw-dir", str(root / "data/interim/r2_inputs/raw"), *FAST4, *extra]


def run4(r, *extra, **kw) -> int:
    return M4.main(args4(r, *extra, **kw))


def log_of(r) -> list:
    p = r["root"] / WORK / M4.TEST_LOG_NAME
    return json.loads(p.read_text())["attempts"] if p.exists() else []


class RecordingGuard(M4.TestGuard):
    made = []

    def __init__(self, split):
        super().__init__(split)
        self.te_phases = []
        RecordingGuard.made.append(self)

    def te(self):
        self.te_phases.append(self.phase)
        return super().te()


def _rel_files(root: Path) -> set:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    mp = _null_pins()
    r = make_project(tmp_path_factory.mktemp("r4_e2e"))
    pristine = copy_project(r, tmp_path_factory.mktemp("r4_pristine"))
    files_before = _rel_files(r["root"])
    RecordingGuard.made.clear()
    mp.setattr(M4, "TestGuard", RecordingGuard)
    rc = run4(r)
    guards = list(RecordingGuard.made)
    mp.undo()
    return dict(r=r, rc=rc, guards=guards, pristine=pristine, files_before=files_before)


@pytest.fixture(scope="module")
def e2e_small(tmp_path_factory):
    mp = _null_pins()
    r = make_project(tmp_path_factory.mktemp("r4_small"), small_ids=True)
    rc = run4(r)
    mp.undo()
    return r, rc, set(map(str, r["eid"])) | set(map(str, r["sid"]))


def fresh(e2e, tmp_path) -> dict:
    return copy_project(e2e["pristine"], tmp_path)


# ------------------------------------------------------------------------------------------------- unit tests
def test_guard_blocks_test_rows_outside_scoring():
    G = M4.TestGuard(np.array(["train", "validation", "test", "train"]))
    with pytest.raises(M4.BuildError):
        G.te()
    with pytest.raises(M4.BuildError):
        G.fit_rows(np.array([0, 2]), "bad")
    G.fit_rows(np.array([0, 1, 3]), "ok")
    G.phase = "score"
    te = G.te()
    assert set(te.tolist()) == {2}
    G.score("once", lambda: 1)
    with pytest.raises(M4.BuildError):
        G.score("once", lambda: 2)


def test_default_grids_are_small_and_fixed():
    lstm = M4.default_grid("lstm")
    attn = M4.default_grid("lstm_attn")
    tr = M4.default_grid("transformer")
    assert len(lstm) == 4 and {c["hidden"] for c in lstm} == {32, 64}
    assert {c["n_layers"] for c in lstm} == {1, 2}
    assert len(attn) == 4
    assert len(tr) == 8
    assert {c["d_model"] for c in tr} == {64, 128}
    assert {c["n_layers"] for c in tr} == {2, 4}
    assert {c["pe"] for c in tr} == {"sinusoidal", "learned"}
    assert all(c["n_heads"] == 4 for c in tr)


def test_seq_fit_is_deterministic_on_cpu():
    """Two fits with the same seed and CPU tensors must match class probabilities."""
    rng = np.random.default_rng(6310)
    n, t, d, ds, K = 24, 8, 6, 4, 5
    seq = rng.normal(size=(n, t, d)).astype(np.float32)
    st = rng.normal(size=(n, ds)).astype(np.float32)
    y = rng.integers(0, K, size=n).astype(np.int64)
    g = np.array([i // 2 for i in range(n)])
    off = np.zeros(n, dtype=np.float32)
    split = 16
    hp = dict(hidden=16, n_layers=1)
    kw = dict(max_epochs=3, patience=3, batch=8, amp=False)
    s1, p1, e1 = M4.seq_fit_predict("lstm", hp, seq[:split], st[:split], off[:split], y[:split], g[:split],
                                    seq[split:], st[split:], off[split:], K, 99, **kw)
    s2, p2, e2 = M4.seq_fit_predict("lstm", hp, seq[:split], st[:split], off[:split], y[:split], g[:split],
                                    seq[split:], st[split:], off[split:], K, 99, **kw)
    assert e1 == e2
    assert np.allclose(p1, p2, atol=1e-6)
    p3 = M4.seq_predict(s1, seq[split:], st[split:], off[split:])
    assert np.allclose(p1, p3, atol=1e-6)
    # a different family also trains and is deterministic
    hpt = dict(d_model=32, n_layers=2, n_heads=4, pe="sinusoidal")
    t1, q1, _ = M4.seq_fit_predict("transformer", hpt, seq[:split], st[:split], off[:split], y[:split], g[:split],
                                   seq[split:], st[split:], off[split:], K, 7, **kw)
    t2, q2, _ = M4.seq_fit_predict("transformer", hpt, seq[:split], st[:split], off[:split], y[:split], g[:split],
                                   seq[split:], st[split:], off[split:], K, 7, **kw)
    assert np.allclose(q1, q2, atol=1e-5)
    # attention pooling with a mask
    mask = np.ones((n, t), dtype=bool)
    mask[:, -2:] = False
    ha = dict(hidden=16, n_layers=1)
    a1, u1, _ = M4.seq_fit_predict("lstm_attn", ha, seq[:split], st[:split], off[:split], y[:split], g[:split],
                                   seq[split:], st[split:], off[split:], K, 3, mask_tr=mask[:split],
                                   mask_va=mask[split:], **kw)
    a2, u2, _ = M4.seq_fit_predict("lstm_attn", ha, seq[:split], st[:split], off[:split], y[:split], g[:split],
                                   seq[split:], st[split:], off[split:], K, 3, mask_tr=mask[:split],
                                   mask_va=mask[split:], **kw)
    assert np.allclose(u1, u2, atol=1e-6)
    assert np.allclose(u1.sum(1), 1.0, atol=1e-5)


def test_out_dir_outside_stage17_folder_is_refused_before_writes(tmp_path, capsys):
    outside = tmp_path / "outside_out"
    # build_config itself refuses a path outside the work root (no project needed)
    with pytest.raises(M4.InputError):
        M4.build_config(["--project-root", str(tmp_path / "proj"), "--out-dir", str(outside)])
    assert not outside.exists()


def test_defaults_are_the_desktop_full_settings():
    a = M4.build_config([])
    assert a.max_seconds == 0.0
    assert a.max_epochs == 40 and M4.DEFAULT_MAX_EPOCHS == 40
    assert a.n_boot == 1000 and a.n_seeds == 1
    assert a.landmarks == (24, 48, 72, 120, 168, 240, 336)
    assert a.families == ("lstm", "lstm_attn", "transformer")
    assert a.amp is False and a.require_cuda is False
    assert a.out_dir == Path("data") / "processed" / "ladder" / "DESKTOP_FULL_PLAN" / "stage17_seqmodels" / "published"
    assert a.test_log == Path("data") / "processed" / "ladder" / "DESKTOP_FULL_PLAN" / "stage17_seqmodels" / M4.TEST_LOG_NAME
    assert a.cache_dir == Path("data") / "processed" / "ladder" / "DESKTOP_FULL_PLAN" / "stage17_seqmodels" / M4.CACHE_NAME
    assert M4.SCRIPT_VERSION == "r4_desktop_seq_models v1.0.1-desktop"
    assert M4.PINS3["run2_script_sha256"] == "0262df9482c360baeb091f9bfcb4107e19d45712fa9f1aa8fd9db8f5c270ffbb"


def test_gate_collects_all_problems_without_values(tmp_path):
    pend = tmp_path / "pend"
    pend.mkdir()
    # two problems in one file: forbidden size and a forbidden column
    csv = "qwk,n_train\n0.2,12345\n"
    (pend / "stage17_seq_tuning_TEAM.csv").write_text(csv)
    (pend / f"{M4.OUT_NAME}_audit_TEAM.json").write_text("{}")
    ids = {"999"}
    forbidden = {12345}
    probs = M4.gate_problems(pend, ["stage17_seq_tuning_TEAM.csv"], forbidden, ids)
    kinds = {(f, loc, chk) for f, loc, chk in probs}
    assert any(loc == "column names" for _, loc, _ in probs)
    assert any("n_train" in loc or "column names" in loc for _, loc, _ in probs)
    # values themselves must not appear in the problem strings
    blob = " ".join(chk for _, _, chk in probs)
    assert "12345" not in blob and "999" not in blob
    assert len(probs) >= 2


def test_environment_rejects_cuda_requirement_on_cpu():
    import torch
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    with pytest.raises(M4.InputError, match="CUDA is required"):
        M4.check_environment(require_cuda=True)
    info = M4.check_environment(require_cuda=False)
    assert info["cuda_available"] is False
    assert info["python"].startswith("3.14")
    assert info["torch"].startswith("2.10")


# ------------------------------------------------------------------------------------------------- e2e
def test_e2e_publishes_with_all_checks(e2e):
    assert e2e["rc"] == 0, "synthetic e2e failed"
    r = e2e["r"]
    out = r["root"] / OUT
    man = json.loads((out / f"{M4.OUT_NAME}_manifest.json").read_text())
    assert man["release_gate"] == "ok"
    assert man["script"] == M4.SCRIPT_VERSION
    checks = man["acceptance_checks"]
    assert all(v == "pass" for v in checks.values()), checks
    for nm in ("stage17_seq_tuning_TEAM.csv", "stage17_seq_selected_TEAM.csv",
               "stage17_seq_val_contrast_TEAM.csv", "stage17_seq_test_TEAM.csv",
               "stage17_seq_test_contrast_TEAM.csv", f"{M4.OUT_NAME}_audit_TEAM.md",
               f"{M4.OUT_NAME}_audit_TEAM.json"):
        assert (out / nm).is_file(), nm
    md = (out / f"{M4.OUT_NAME}_audit_TEAM.md").read_text()
    assert "post-primary sensitivity analysis" in md
    assert "second look" in md
    assert "stage17_seqmodels" in md
    # selected one config per family at the requested landmark
    sel = pd.read_csv(out / "stage17_seq_selected_TEAM.csv")
    fams = set(sel.loc[sel["family"].isin(M4.FAMILIES), "family"])
    assert fams == set(M4.FAMILIES)
    log = log_of(r)
    assert log and log[-1]["status"] == "published"
    audit = json.loads((out / f"{M4.OUT_NAME}_audit_TEAM.json").read_text())
    cfg = audit["configuration"]
    assert "gpu_name" not in cfg and "gpu_memory_bytes" not in cfg and "executable" not in cfg
    assert "cuda_available" in cfg
    assert "gpu" not in md.lower() or "cuda:" in md.lower()
    assert (out / f"{M4.OUT_NAME}_environment_INTERNAL.json").is_file()
    env_rec = json.loads((out / f"{M4.OUT_NAME}_environment_INTERNAL.json").read_text())
    assert "environment" in env_rec


def test_e2e_test_rows_touched_only_in_scoring(e2e):
    assert e2e["guards"], "RecordingGuard was not used"
    g = e2e["guards"][0]
    assert g.phase == "done"
    assert g.te_phases and all(p == "score" for p in g.te_phases)
    assert all(ph == "verify" for _, ph, _ in g.fits)
    assert all("test" not in sp for _, _, sp in g.fits)


def test_verification_ignores_test_rows(e2e, tmp_path):
    r = fresh(e2e, tmp_path)

    class BoomGuard(M4.TestGuard):
        def te(self):
            if self.phase != "score":
                raise AssertionError("test rows requested during tuning")
            return super().te()

        def fit_rows(self, idx, what):
            out = super().fit_rows(idx, what)
            if "test" in set(self.split[idx]):
                raise AssertionError("fit saw test rows")
            return out

    mp = pytest.MonkeyPatch()
    mp.setattr(M4, "TestGuard", BoomGuard)
    try:
        assert run4(r) == 0
    finally:
        mp.undo()


def test_e2e_team_files_no_sizes_or_ids(e2e):
    r = e2e["r"]
    out = r["root"] / OUT
    man = json.loads((out / f"{M4.OUT_NAME}_manifest.json").read_text())
    ids = set(map(str, r["eid"])) | set(map(str, r["sid"]))
    for nm in man["team_files"]:
        if nm.endswith(".png"):
            continue
        text = (out / nm).read_text(encoding="utf-8")
        for ident in ids:
            # short numeric tokens may appear as structural hyperparameters (24, 32, 64, ...);
            # only flag identifiers of length >= 4, or any non-numeric id
            if ident.isdigit() and len(ident) < 4:
                continue
            assert ident not in text, f"{nm} leaked identifier"


def test_rerun_refused(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    assert run4(r) == 0
    capsys.readouterr()
    assert run4(r) == 2
    err = capsys.readouterr().err
    assert "scored once" in err or "published test scoring" in err
    assert "STOPPED (input)" in err


def test_run2_manifest_hash_mismatch_stops(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    assert run4(r, r2="0" * 64) == 2
    err = capsys.readouterr().err
    assert "STOPPED (input)" in err
    assert not (r["root"] / OUT).exists() or not any((r["root"] / OUT).iterdir())


def test_tampered_run2_script_stops(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    bad = tmp_path / "bad_r2.py"
    bad.write_text(R2_SCRIPT.read_text() + "\n# tamper\n")
    assert run4(r, script=bad) == 2
    err = capsys.readouterr().err
    assert "STOPPED (input)" in err
    assert "SHA-256" in err or "pin" in err


def test_precheck_only_prints_hashes(e2e, tmp_path, capsys):
    r = fresh(e2e, tmp_path)
    assert run4(r, "--precheck-only") == 0
    out = capsys.readouterr().out
    assert "PRECHECK OK" in out
    assert "sha256" in out
    assert not (r["root"] / OUT).exists()
    # README / cache may exist under the work root, but nothing published and no test-use published attempt
    assert not any(a.get("status") == "published" for a in log_of(r))


def test_failed_scoring_is_logged_and_publishes_nothing(e2e, tmp_path, monkeypatch):
    r = fresh(e2e, tmp_path)

    def boom(*a, **k):
        raise M4.BuildError("planted scoring failure")

    orig = M4.score_test_landmark
    monkeypatch.setattr(M4, "score_test_landmark", boom)
    assert run4(r) == 3
    assert not (r["root"] / OUT).exists()
    log = log_of(r)
    assert log and log[-1]["status"].startswith("failed")


def test_small_integer_ids_pass_gate(e2e_small):
    r, rc, ids = e2e_small
    assert rc == 0
    out = r["root"] / OUT
    assert (out / f"{M4.OUT_NAME}_manifest.json").is_file()
    man = json.loads((out / f"{M4.OUT_NAME}_manifest.json").read_text())
    assert man["release_gate"] == "ok"


def test_planted_leaks_fail_gate_without_values(e2e_small, tmp_path):
    r, rc, ids = e2e_small
    assert rc == 0
    pend = tmp_path / "pend"
    pend.mkdir()
    # copy a real TEAM csv and plant a size and an id
    src = r["root"] / OUT / "stage17_seq_test_TEAM.csv"
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    # use a number that is large enough to be in the forbidden set (>=100)
    df.loc[df.index[0], "qwk"] = "1000001"
    col = "label" if "label" in df.columns else df.columns[-1]
    ident = next(iter(ids))
    df.loc[df.index[0], col] = f"leak {ident}"
    df.to_csv(pend / "stage17_seq_test_TEAM.csv", index=False)
    (pend / f"{M4.OUT_NAME}_audit_TEAM.json").write_text("{}")
    # reload D-like forbidden set: just plant 1000001
    probs = M4.gate_problems(pend, ["stage17_seq_test_TEAM.csv"], {1000001}, set(map(str, ids)))
    blob = " ".join(f"{loc} {chk}" for _, loc, chk in probs)
    assert probs
    assert "1000001" not in blob
    for ident in ids:
        assert ident not in blob


def test_writes_stay_in_stage17_seqmodels(e2e):
    """Every file the desktop script added to the project is under DESKTOP_FULL_PLAN/stage17_seqmodels/."""
    r = e2e["r"]
    before = e2e["files_before"]
    after = _rel_files(r["root"])
    new = after - before
    assert new, "the desktop run should have written outputs"
    work = WORK
    for p in sorted(new):
        assert work in p.parents or p == work, f"wrote outside stage17_seqmodels: {p}"
    readme = r["root"] / work / M4.README_NAME
    assert readme.is_file()
    txt = readme.read_text(encoding="utf-8").lower()
    assert "post-primary" in txt and "sensitivity" in txt
    assert "lstm" in txt and "transformer" in txt
    assert (r["root"] / OUT / f"{M4.OUT_NAME}_manifest.json").is_file()
    assert (r["root"] / work / M4.TEST_LOG_NAME).is_file()
    md = (r["root"] / OUT / f"{M4.OUT_NAME}_audit_TEAM.md").read_text()
    assert "post-primary sensitivity analysis" in md
    assert "second look" in md


def test_laptop_r3a_published_does_not_block_desktop(e2e, tmp_path):
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
    assert run4(r) == 0
    assert (r["root"] / OUT).is_dir()
    assert [a["status"] for a in log_of(r)] == ["published"]
    assert laptop_log.is_file() and laptop_out.is_dir()


def test_rtx_3080_gpu_name_stays_out_of_team_and_passes_gate(tmp_path, monkeypatch):
    """A real desktop GPU name (NVIDIA GeForce RTX 3080) must not appear in TEAM files, even when
    3080 is also a subject/encounter identifier; the release gate must still pass."""
    monkeypatch.setattr(M4, "torch_device", lambda: "cuda")
    env = dict(python="3.14.3",
               executable="<VENV_PYTHON>",
               numpy="2.4.2", pandas="3.0.1", scipy="1.17.1", sklearn="1.8.0",
               torch="2.10.0+cu126", cuda_available=True, cuda_version="12.6",
               gpu_name="NVIDIA GeForce RTX 3080", gpu_memory_bytes=10736893952)
    cfg = type("C", (), dict(env_info=env, n_boot=20, amp=False, max_seconds=0,
                             max_epochs=2, patience=5, n_seeds=1,
                             landmarks=(24,), families=("lstm",),
                             test_log=tmp_path / "no_log.json"))()
    pre = {"hashes": {"split": "ab"}, "prior_attempts": 0}
    audit = M4.make_audit(cfg, pre, OrderedDict(), OrderedDict(), OrderedDict(),
                          pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                          "abc", 3)
    md = M4.audit_md(audit)
    pend = tmp_path / "pend"
    pend.mkdir()
    ajn, amn = f"{M4.OUT_NAME}_audit_TEAM.json", f"{M4.OUT_NAME}_audit_TEAM.md"
    (pend / ajn).write_text(json.dumps(audit, indent=1, default=str), encoding="utf-8")
    (pend / amn).write_text(md, encoding="utf-8")
    ids = {"3080", "10736893952", "exampleuser"}
    probs = M4.gate_problems(pend, [ajn, amn], set(), ids)
    assert probs == [], probs
    for text in (json.dumps(audit, default=str), md):
        assert "NVIDIA" not in text
        assert "GeForce" not in text
        assert "RTX" not in text
        assert "3080" not in text
        assert "10736893952" not in text
        assert "exampleuser" not in text
        assert "gpu_name" not in text
        assert "gpu_memory" not in text
        assert "executable" not in text
    assert audit["configuration"].get("cuda_available") is True
    assert "cuda: yes" in md
    # INTERNAL record may (and should) hold the GPU name
    rec = json.loads(M4.environment_internal_bytes(cfg, pre).decode("utf-8"))
    assert rec["environment"]["gpu_name"] == "NVIDIA GeForce RTX 3080"
    assert rec["environment"]["gpu_memory_bytes"] == 10736893952


def test_planted_real_looking_identifier_still_fails_gate(tmp_path):
    """The identifier gate is unchanged: planting a real-looking id (the RTX 3080 token) in TEAM
    text must still fail, and the problem report must not include the value."""
    pend = tmp_path / "pend"
    pend.mkdir()
    ident = "NVIDIA GeForce RTX 3080"
    (pend / f"{M4.OUT_NAME}_audit_TEAM.json").write_text(
        json.dumps({"configuration": {"gpu_name": ident}}, indent=1), encoding="utf-8")
    (pend / f"{M4.OUT_NAME}_audit_TEAM.md").write_text(
        f"# audit - TEAM\n\ngpu {ident}\n", encoding="utf-8")
    ids = {"3080"}
    names = [f"{M4.OUT_NAME}_audit_TEAM.json", f"{M4.OUT_NAME}_audit_TEAM.md"]
    probs = M4.gate_problems(pend, names, set(), ids)
    assert probs, "the planted identifier should fail the gate"
    assert any(chk == "an identifier" for _, _, chk in probs)
    blob = " ".join(f"{loc} {chk}" for _, loc, chk in probs)
    assert "3080" not in blob
    assert "NVIDIA" not in blob
    assert "RTX" not in blob
    assert ident not in blob


def test_rerun_allowed_after_failed_unpublished_attempt(e2e, tmp_path, monkeypatch, capsys):
    """v1.0.1 still allows one rerun after a failed, unpublished attempt, records that the test
    set was scored once before, and still refuses after a published attempt."""
    r = fresh(e2e, tmp_path)

    def boom(*a, **k):
        raise M4.BuildError("planted scoring failure")

    orig = M4.score_test_landmark
    monkeypatch.setattr(M4, "score_test_landmark", boom)
    assert run4(r) == 3
    assert not (r["root"] / OUT).exists()
    log = log_of(r)
    assert log and log[-1]["status"] == "failed (nothing published)"
    monkeypatch.setattr(M4, "score_test_landmark", orig)
    capsys.readouterr()
    assert run4(r) == 0
    log2 = log_of(r)
    assert [a["status"] for a in log2] == ["failed (nothing published)", "published"]
    md = (r["root"] / OUT / f"{M4.OUT_NAME}_audit_TEAM.md").read_text()
    audit = json.loads((r["root"] / OUT / f"{M4.OUT_NAME}_audit_TEAM.json").read_text())
    assert any("unpublished attempt" in d for d in audit["deviations"])
    assert "scored once before" in md
    assert "unpublished attempt" in md
    capsys.readouterr()
    assert run4(r) == 2
    err = capsys.readouterr().err
    assert "scored once" in err or "published test scoring" in err
    assert "STOPPED (input)" in err


def test_amp_flag_is_ignored_on_cpu():
    """AMP may be requested; the trainer must still run on CPU without using mixed precision."""
    rng = np.random.default_rng(1)
    n, t, d, K = 12, 4, 3, 4
    seq = rng.normal(size=(n, t, d)).astype(np.float32)
    y = rng.integers(0, K, size=n)
    g = np.arange(n)
    off = np.zeros(n, np.float32)
    st = np.zeros((n, 0), np.float32)
    _, P, _ = M4.seq_fit_predict("lstm", dict(hidden=8, n_layers=1), seq[:8], st[:8], off[:8], y[:8], g[:8],
                                 seq[8:], st[8:], off[8:], K, 5, max_epochs=2, patience=2, batch=4, amp=True)
    assert P.shape == (4, K) and np.allclose(P.sum(1), 1.0, atol=1e-5)
