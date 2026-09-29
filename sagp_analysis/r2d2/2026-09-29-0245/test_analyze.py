"""Tests for the R2-D2 study analysis (`analyze.py` beside this file).

Run with the analysis environment (numpy, scipy, pandas, matplotlib; the study's `saasbo` env has no
pandas), from the repository root:

    python -m pytest -q sagp_analysis/r2d2/<stamp>/test_analyze.py            # the fast tests
    python -m pytest -q -m slow sagp_analysis/r2d2/<stamp>/test_analyze.py    # the fixture pipeline

The slow tests read the stored half-Cauchy runs under `runs/` (read-only) and write only under
pytest's tmp_path. `test_method_table_matches_the_library` runs the study's `saasbo` interpreter
(`SAASBO_PY`, default /opt/anaconda3/envs/saasbo/bin/python) in a subprocess and is skipped where it
does not exist.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import fixture  # noqa: E402

REPO = analyze.repo_root(HERE)
R2_CELLS = [m.dir for m in analyze.METHOD_TABLE if m.prior_family == "R2D2"]
HC_CELLS = [m.dir for m in analyze.METHOD_TABLE if m.prior_family == "HC"]
PILOT_FAMILIES = ["aligned10", "decoupled"]


# ----------------------------------------------------------------------------- the method table
def test_the_method_table_has_eleven_methods_and_each_r2d2_cell_a_twin():
    table = analyze.METHOD_TABLE
    assert len(table) == 11
    assert len({m.dir for m in table}) == 11
    by_dir = {m.dir: m for m in table}
    assert len(R2_CELLS) == 4 and len(HC_CELLS) == 4
    for d in R2_CELLS:
        m, twin = by_dir[d], by_dir[by_dir[d].twin]
        assert twin.prior_family == "HC"
        assert (m.structure, m.parameterization) == (twin.structure, twin.parameterization)
    assert {(by_dir[d].structure, by_dir[d].parameterization) for d in HC_CELLS} == {
        (s, p) for s in ("additive", "product") for p in ("amplitude", "lengthscale")}
    assert by_dir["product-amplitude_r2d2"].method == "product/amplitude_r2d2"
    assert by_dir["dsp_map"].method == "dsp_map"


@pytest.mark.slow
def test_method_table_matches_the_library():
    """The table is literal (the analysis env has no torch); this pins it to sagp/experiments."""
    py = os.environ.get("SAASBO_PY", "/opt/anaconda3/envs/saasbo/bin/python")
    if not Path(py).exists():
        pytest.skip(f"no saasbo interpreter at {py}")
    code = (
        "import json, numpy as np\n"
        "from sagp.gp import CELLS\n"
        "from experiments.run_bo import METHODS\n"
        "from experiments.replay import TWINS\n"
        "from sagp.readouts import r2d2_r2, first_order_r2\n"
        "rng = np.random.default_rng(3); a = rng.gamma(0.3, 0.05, (16, 100)); s2 = rng.uniform(1e-5, 0.3, 16)\n"
        "print(json.dumps(dict(methods=METHODS, twins=TWINS,\n"
        "  cells={'/'.join(k): [c.structure, c.native_site] for k, c in CELLS.items()},\n"
        "  a=a.tolist(), s2=s2.tolist(), r2d2=np.asarray(r2d2_r2(a)).tolist(),\n"
        "  first=np.asarray(first_order_r2(a, s2)).tolist())))\n"
    )
    out = subprocess.run([py, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    lib = json.loads(out.stdout.strip().splitlines()[-1])
    table = {m.method: m for m in analyze.METHOD_TABLE}
    by_dir = {m.dir: m for m in analyze.METHOD_TABLE}
    assert sorted(table) == sorted(lib["methods"])
    assert {m.method: by_dir[m.twin].method for m in analyze.METHOD_TABLE if m.twin} == lib["twins"]
    native = {"a_sq": "amplitude", "kernel_inv_length_sq": "lengthscale"}
    for method, (structure, native_site) in lib["cells"].items():
        assert (table[method].structure, table[method].parameterization) == (structure, native[native_site])
        assert table[method].dir == method.replace("/", "-")
    a, s2 = np.array(lib["a"]), np.array(lib["s2"])
    np.testing.assert_allclose(analyze.r2d2_r2(a), lib["r2d2"], rtol=1e-12)
    # the library clamps nothing here (it takes the noise prediction uses); s2 >= 1e-5 < 1e-4, so
    # compare the additive formula without the npz clamp
    np.testing.assert_allclose(analyze.first_order_r2(a, s2, "additive"), lib["first"], rtol=1e-12)


# ----------------------------------------------------------------------------- small helpers
def test_parse_seeds():
    assert analyze.parse_seeds("0-4") == ["00", "01", "02", "03", "04"]
    assert analyze.parse_seeds("0,3,9") == ["00", "03", "09"]
    assert analyze.parse_seeds("0-1,7") == ["00", "01", "07"]


def test_log_regret_floors_at_1e_8():
    np.testing.assert_array_equal(analyze.log_regret(np.array([1.0, 1e-3, 0.0, -1e-12, 1e-9])),
                                  [0.0, -3.0, -8.0, -8.0, -8.0])


def test_holm_matches_a_hand_computed_example():
    np.testing.assert_allclose(analyze.holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])


def test_wilcoxon_p_edge_cases():
    assert analyze.wilcoxon_p(np.zeros(5))[1] == 1.0
    assert np.isnan(analyze.wilcoxon_p(np.array([0.3]))[1])
    # five positive pairs: the smallest two-sided exact p is 2/32
    assert analyze.wilcoxon_p(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))[1] == pytest.approx(0.0625)
    assert analyze.wilcoxon_p(np.arange(1, 11) / 10)[1] == pytest.approx(2 / 1024)


# ----------------------------------------------------------------------------- R2 readouts
def test_r2_readouts_follow_from_the_draws():
    rng = np.random.default_rng(0)
    a = rng.gamma(0.5, 0.02, (16, 100))
    noise = rng.uniform(0, 0.2, 16)
    noise[0] = 1e-7                                   # below the prediction floor
    S = a.sum(1)
    s2 = np.maximum(noise, 1e-4)
    np.testing.assert_allclose(analyze.r2d2_r2(a), S / (1 + S))
    np.testing.assert_allclose(analyze.first_order_r2(a, s2, "additive"), S / (S + s2))
    prod = np.prod(1 + a, axis=1) - 1
    np.testing.assert_allclose(analyze.first_order_r2(a, s2, "product"), S / (prod + s2))
    draws = {"a_sq": a, "noise": noise}
    r = analyze.npz_r2_medians(draws, "additive", fixed_noise=None)
    assert r == pytest.approx((np.median(S / (1 + S)), np.median(S / (S + s2))))
    r = analyze.npz_r2_medians(draws, "product", fixed_noise=None)
    assert r[1] == pytest.approx(np.median(S / (prod + s2)))
    # every draw below the floor: the clamp is what prediction uses
    r = analyze.npz_r2_medians({"a_sq": a, "noise": np.full(16, 1e-7)}, "additive", fixed_noise=None)
    assert r[1] == pytest.approx(np.median(S / (S + 1e-4)), rel=1e-12)
    # fixed-noise runs have no noise site: the manifest's value is the noise
    r = analyze.npz_r2_medians({"a_sq": a}, "additive", fixed_noise=0.01)
    assert r[1] == pytest.approx(np.median(S / (S + 0.01)))


# ----------------------------------------------------------------------------- effects
def _final_from_deltas(deltas: dict[str, np.ndarray], rng_seed=0):
    """HC regrets drawn at random; each R2-D2 cell's regret = twin's * 10**delta. (fam, seed) = u."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for u in range(10):
        fam, seed = PILOT_FAMILIES[u // 5], f"{u % 5:02d}"
        for d in HC_CELLS:
            r = float(10 ** rng.uniform(-2.5, 0))
            rows.append(dict(family=fam, method=d, seed=seed, regret=r))
            r2 = [m.dir for m in analyze.METHOD_TABLE if m.twin == d][0]
            rows.append(dict(family=fam, method=r2, seed=seed, regret=r * 10 ** deltas[r2][u]))
        rows.append(dict(family=fam, method="sobol", seed=seed, regret=1.0))
    return pd.DataFrame(rows)


def test_unit_effects_are_the_defined_contrasts():
    fam, seed = "aligned10", "00"
    y = {"additive-amplitude": -1.0, "product-amplitude": -0.4, "additive-lengthscale": -0.7,
         "product-lengthscale": -0.2, "additive-amplitude_r2d2": -1.3, "product-amplitude_r2d2": -0.1,
         "additive-lengthscale_r2d2": -0.6, "product-lengthscale_r2d2": -0.5}
    final = pd.DataFrame([dict(family=fam, seed=seed, method=m, regret=10 ** v) for m, v in y.items()])
    e = analyze.unit_effects(final).set_index("effect").value
    S_hc = 0.5 * ((-1.0 + 0.4) + (-0.7 + 0.2))
    P_hc = 0.5 * ((-1.0 + 0.7) + (-0.4 + 0.2))
    S_r2 = 0.5 * ((-1.3 + 0.1) + (-0.6 + 0.5))
    P_r2 = 0.5 * ((-1.3 + 0.6) + (-0.1 + 0.5))
    assert e["S_HC"] == pytest.approx(S_hc) and e["P_HC"] == pytest.approx(P_hc)
    assert e["S_R2D2"] == pytest.approx(S_r2) and e["P_R2D2"] == pytest.approx(P_r2)
    assert e["dS"] == pytest.approx(S_r2 - S_hc) and e["dP"] == pytest.approx(P_r2 - P_hc)
    assert e["Delta_additive-amplitude_r2d2"] == pytest.approx(-0.3)
    assert e["Delta_product-lengthscale_r2d2"] == pytest.approx(-0.3)
    kinds = analyze.unit_effects(final).set_index("effect").kind
    assert kinds["S_HC"] == "main_HC" and kinds["P_R2D2"] == "main_R2D2"
    assert kinds["dS"] == "interaction" and kinds["Delta_product-amplitude_r2d2"] == "twin"


def test_unit_effects_skip_a_unit_missing_a_cell():
    final = _final_from_deltas({d: np.zeros(10) for d in R2_CELLS})
    final = final[~((final.method == "product-amplitude_r2d2") & (final.seed == "00")
                    & (final.family == "aligned10"))]
    e = analyze.unit_effects(final)
    u0 = e[(e.family == "aligned10") & (e.seed == "00")]
    assert "S_R2D2" not in set(u0.effect) and "dS" not in set(u0.effect)
    assert "Delta_product-amplitude_r2d2" not in set(u0.effect)
    assert "S_HC" in set(u0.effect) and "Delta_additive-amplitude_r2d2" in set(u0.effect)
    assert len(e[e.effect == "dS"]) == 9


# ----------------------------------------------------------------------------- bootstrap
def test_cluster_bootstrap_resamples_clusters_not_rows():
    # every cluster's mean is 0.1, with large spread inside: resampling clusters cannot move the mean
    vals, clus = [], []
    for k in range(10):
        vals += [0.1 + 2.0, 0.1 - 2.0, 0.1 + (k - 4.5), 0.1 - (k - 4.5)]
        clus += [k] * 4
    b = analyze.cluster_bootstrap_mean(np.array(vals), np.array(clus))
    assert b["mean"] == pytest.approx(0.1)
    for key in ("lo90", "hi90", "lo95", "hi95"):
        assert b[key] == pytest.approx(0.1, abs=1e-12)


def test_cluster_bootstrap_is_seeded_and_nested():
    rng = np.random.default_rng(5)
    v = rng.normal(0.2, 1.0, 40)
    c = np.repeat(np.arange(10), 4)
    b1 = analyze.cluster_bootstrap_mean(v, c)
    b2 = analyze.cluster_bootstrap_mean(v, c)
    assert b1 == b2
    assert b1["n_boot"] == 10_000 and b1["n_clusters"] == 10
    assert b1["lo95"] <= b1["lo90"] <= b1["mean"] <= b1["hi90"] <= b1["hi95"]
    # one value per cluster: the ordinary bootstrap's spread, sd/sqrt(n) within 15 %
    b = analyze.cluster_bootstrap_mean(v, np.arange(40))
    half = (b["hi95"] - b["lo95"]) / 2
    assert half == pytest.approx(1.96 * v.std() / np.sqrt(40), rel=0.15)


# ----------------------------------------------------------------------------- G3
W = np.array([0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2, 1.5, -1.5])   # symmetric seed noise
A = np.array([0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 1.2, 1.2, 1.5, 1.5])
SGN = np.array([1, -1] * 5)
U = np.arange(10)
AA, AL, PA, PL = ("additive-amplitude_r2d2", "additive-lengthscale_r2d2",
                  "product-amplitude_r2d2", "product-lengthscale_r2d2")

SCENARIOS = {
    "stop": ({AA: W, AL: W, PA: W, PL: W}, dict(a=False, b=False, c=False, d=False)),
    "a_only": ({AA: 0.35 + W, AL: W, PA: W, PL: W}, dict(a=True, b=False, c=False, d=False)),
    "b_only": ({AA: 0.1 + A * SGN, AL: 0.1 - A * SGN, PA: 0.1 + A * SGN, PL: 0.1 - A * SGN},
               dict(a=False, b=True, c=False, d=False)),
    "c_only": ({AA: 0.1 + 0.001 * U, AL: W, PA: W, PL: W}, dict(a=False, b=False, c=True, d=False)),
    "d_only": ({AA: 0.2 + W, AL: 0.2 + W, PA: -0.2 + W, PL: -0.2 + W},
               dict(a=False, b=False, c=False, d=True)),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_g3_each_criterion_fires_alone(name):
    deltas, expected = SCENARIOS[name]
    stats = analyze.g3_statistics(analyze.unit_effects(_final_from_deltas(deltas)))
    assert stats["n_pairs"] == 40 and stats["n_units"] == 10
    crit = analyze.g3_criteria(stats)
    assert {k: crit[k] for k in "abcd"} == expected
    assert crit["go"] == any(expected.values())
    # the statistics themselves, against numpy on the injected deltas
    allv = np.concatenate([deltas[c] for c in R2_CELLS])
    assert stats["median_pooled"] == pytest.approx(np.median(allv), abs=1e-9)
    for c in R2_CELLS:
        assert stats["median_cell"][c] == pytest.approx(np.median(deltas[c]), abs=1e-9)
    assert stats["mean_pooled"] == pytest.approx(allv.mean(), abs=1e-9)
    dS = 0.5 * ((deltas[AA] - deltas[PA]) + (deltas[AL] - deltas[PL]))
    dP = 0.5 * ((deltas[AA] - deltas[AL]) + (deltas[PA] - deltas[PL]))
    assert stats["median_dS"] == pytest.approx(np.median(dS), abs=1e-9)
    assert stats["median_dP"] == pytest.approx(np.median(dP), abs=1e-9)


def test_g3_criteria_boundaries():
    base = dict(median_pooled=0.0, median_cell={c: 0.0 for c in R2_CELLS}, lo95=-0.1, hi95=0.1,
                p_holm_cell={c: 1.0 for c in R2_CELLS}, median_dS=0.0, median_dP=0.0)
    assert not analyze.g3_criteria(base)["go"]
    assert analyze.g3_criteria(base | dict(median_pooled=-0.3))["a"]          # >= 0.3 in size
    assert not analyze.g3_criteria(base | dict(median_pooled=0.2999))["a"]
    assert analyze.g3_criteria(base | dict(median_cell={**base["median_cell"], PL: 0.3}))["a"]
    assert not analyze.g3_criteria(base | dict(lo95=0.0))["b"]                # touching 0 includes it
    assert analyze.g3_criteria(base | dict(lo95=1e-9))["b"]
    assert analyze.g3_criteria(base | dict(hi95=-1e-9, lo95=-0.2))["b"]
    assert not analyze.g3_criteria(base | dict(p_holm_cell={**base["p_holm_cell"], AA: 0.05}))["c"]
    assert analyze.g3_criteria(base | dict(p_holm_cell={**base["p_holm_cell"], AA: 0.0499}))["c"]
    assert analyze.g3_criteria(base | dict(median_dP=0.3))["d"]
    assert analyze.g3_criteria(base | dict(median_dS=-0.31))["d"]


def test_g3_verdict_is_read_only_on_the_complete_pilot():
    stats = analyze.g3_statistics(analyze.unit_effects(_final_from_deltas(SCENARIOS["d_only"][0])))
    ok = analyze.g3_reading(stats, families=PILOT_FAMILIES, seeds=analyze.parse_seeds("0-4"),
                            sanity_failures=[])
    assert ok["verdict"] == "GO" and ok["read"]
    other = analyze.g3_reading(stats, families=["aligned10"], seeds=analyze.parse_seeds("0-4"),
                               sanity_failures=[])
    assert not other["read"] and other["verdict"] == "NOT READ"
    bad = analyze.g3_reading(stats, families=PILOT_FAMILIES, seeds=analyze.parse_seeds("0-4"),
                             sanity_failures=["prior_code_ok"])
    assert not bad["read"] and bad["verdict"] == "INCONCLUSIVE" and bad["would_read"] == "GO"
    stop = analyze.g3_statistics(analyze.unit_effects(_final_from_deltas(SCENARIOS["stop"][0])))
    assert analyze.g3_reading(stop, families=PILOT_FAMILIES, seeds=analyze.parse_seeds("0-4"),
                              sanity_failures=[])["verdict"] == "STOP"


# ----------------------------------------------------------------------------- prereg readings
def test_pr2_reading():
    assert analyze.pr2_reading(-0.29, 0.29) == "equivalent"
    assert analyze.pr2_reading(0.3, 0.9) == "prior family changes the effect"
    assert analyze.pr2_reading(-1.0, -0.3) == "prior family changes the effect"
    assert analyze.pr2_reading(-0.3, 0.1) == "inconclusive"
    assert analyze.pr2_reading(0.1, 0.5) == "inconclusive"


# ----------------------------------------------------------------------------- sanity: pairing
def _ck(over=None):
    rows = []
    for d in HC_CELLS + R2_CELLS:
        rows.append(dict(family="aligned10", method=d, seed="00", f_star=5.0, S=(1, 2, 3),
                         y_mean_t20=0.1234567890123, y_std_t20=0.98765))
    ck = pd.DataFrame(rows)
    for (method, col), v in (over or {}).items():
        ck.loc[ck.method == method, col] = v
    return ck


def test_twin_pairing_passes_on_identical_rows():
    tp = analyze.twin_pairing_check(_ck())
    assert len(tp) == 4 and tp.twin_paired.all()


@pytest.mark.parametrize("col,val", [("y_mean_t20", np.nextafter(0.1234567890123, 1)),
                                     ("y_std_t20", 0.98766), ("f_star", 5.0 + 1e-12)])
def test_twin_pairing_fails_on_one_ulp(col, val):
    tp = analyze.twin_pairing_check(_ck({("additive-amplitude", col): val})).set_index("method")
    assert not tp.loc["additive-amplitude_r2d2", "twin_paired"]
    assert tp.drop("additive-amplitude_r2d2").twin_paired.all()


def test_twin_pairing_fails_on_other_S_or_missing_twin():
    ck = _ck()
    ck.loc[ck.method == "product-lengthscale", "S"] = pd.Series([(1, 2, 4)], index=ck.index[ck.method == "product-lengthscale"])
    tp = analyze.twin_pairing_check(ck).set_index("method")
    assert not tp.loc["product-lengthscale_r2d2", "twin_paired"]
    tp = analyze.twin_pairing_check(_ck()[lambda d: d.method != "product-amplitude"]).set_index("method")
    assert not tp.loc["product-amplitude_r2d2", "twin_exists"]
    assert not tp.loc["product-amplitude_r2d2", "twin_paired"]


# ----------------------------------------------------------------------------- sanity: prior code
def _man(commit, resumes=(), dirty=False):
    return {"git": {"commit": commit, "dirty": dirty}, "resumed": [{"commit": c} for c in resumes]}


def test_prior_code_check_with_a_fake_repository():
    blobs = {("c1", "sagp/gp.py"): "g1", ("c1", "sagp/r2d2.py"): "r1",
             ("c2", "sagp/gp.py"): "g1", ("c2", "sagp/r2d2.py"): "r1",
             ("c3", "sagp/gp.py"): "g2", ("c3", "sagp/r2d2.py"): "r1"}
    resolve = lambda commit, path: blobs.get((commit, path))  # noqa: E731
    mans = {("aligned10", AA, "00"): _man("c1", ["c2", "c2"]),
            ("aligned10", AL, "00"): _man("c2")}
    pc = analyze.prior_code_check(mans, resolve)
    assert pc.prior_code_ok.all() and set(pc.n_sessions) == {3, 1}
    # a resume at a commit whose gp.py differs
    pc = analyze.prior_code_check(mans | {("decoupled", PA, "01"): _man("c1", ["c3"])}, resolve)
    assert pc.set_index("method").loc[PA, "prior_code_ok"] == False  # noqa: E712
    assert pc[pc.method != PA].prior_code_ok.all()
    # an unknown commit, and a dirty creation
    pc = analyze.prior_code_check({("aligned10", AA, "00"): _man("unknown")}, resolve)
    assert not pc.prior_code_ok.any() and not pc.blobs_resolved.any()
    pc = analyze.prior_code_check({("aligned10", AA, "00"): _man("c1", dirty=True)}, resolve)
    assert not pc.prior_code_ok.any() and not pc.created_clean.any()
    # only R2-D2 runs are checked
    pc = analyze.prior_code_check({("aligned10", "additive-amplitude", "00"): _man("c3")}, resolve)
    assert len(pc) == 0


def test_prior_code_check_against_this_repository():
    resolve = analyze.git_blob_resolver(REPO)
    # the smoke runs' commit and the pilot's launch commit share both blobs
    mans = {("aligned10", AA, "00"): _man("76cbb7994ff0c82405a9462d037ec8674fd8ac9c", ["97eda925b"])}
    assert analyze.prior_code_check(mans, resolve).prior_code_ok.all()
    # the first half-Cauchy launch predates sagp/r2d2.py
    mans = {("aligned10", AA, "00"): _man("97eda925b", ["a51a4b9"])}
    pc = analyze.prior_code_check(mans, resolve)
    assert not pc.prior_code_ok.any() and not pc.blobs_resolved.any()


# ----------------------------------------------------------------------------- the pipeline (slow)
def _run_fixture(tmp_path, deltas, name):
    r2root = tmp_path / f"runs_r2d2_{name}"
    fixture.build(REPO / "runs", r2root, deltas, families=PILOT_FAMILIES, seeds=range(5))
    out = tmp_path / f"out_{name}"
    out.mkdir()
    subprocess.run([sys.executable, str(HERE / "analyze.py"), "--runs", str(REPO / "runs"),
                    "--runs", str(r2root), "--out", str(out), "--families", ",".join(PILOT_FAMILIES),
                    "--seeds", "0-4", "--asof", "fixture"], check=True, cwd=tmp_path)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("name", ["stop", "d_only"])
def test_fixture_pipeline_reads_g3_as_constructed(tmp_path, name):
    deltas, expected = SCENARIOS[name]
    out = _run_fixture(tmp_path, deltas, name)
    san = pd.read_csv(out / "tables" / "sanity_summary.csv")
    assert (san.n_fail == 0).all(), san[san.n_fail > 0]
    assert {"prior_code_ok", "twin_paired"} <= set(san.check)
    g3 = json.loads((out / "g3.json").read_text())
    assert g3["reading"]["read"] and g3["reading"]["verdict"] == ("GO" if any(expected.values()) else "STOP")
    assert {k: g3["criteria"][k] for k in "abcd"} == expected
    # every pair's Delta is the injected one
    pu = pd.read_csv(out / "tables" / "effects_per_unit.csv", dtype={"seed": str})
    for c in R2_CELLS:
        for u in range(10):
            fam, seed = PILOT_FAMILIES[u // 5], f"{u % 5:02d}"
            v = pu[(pu.family == fam) & (pu.seed == seed) & (pu.effect == f"Delta_{c}")].value
            assert len(v) == 1 and v.iloc[0] == pytest.approx(deltas[c][u], abs=1e-9)
    report = (out / "REPORT.md").read_text()
    assert ("G3: GO to stage 4" if any(expected.values()) else "G3: STOP") in report
