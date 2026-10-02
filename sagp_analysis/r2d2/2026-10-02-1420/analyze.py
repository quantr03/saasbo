#!/usr/bin/env python
"""Read-only analysis of the sagp study runs under both prior families: half-Cauchy (HC) and R2-D2.

Derived from `sagp_analysis/sagp2/2026-09-25-1242/analyze.py` (the `--families` version). Writes
figures/, tables/, REPORT.md, summary.json and g3.json into --out. What is new against sagp2 (plan
D11): `--runs` is repeatable (`runs/` holds the HC cells and the references, `runs_r2d2/` the R2-D2
cells), `--seeds` selects seeds, and a method table (`METHOD_TABLE`) replaces the hard-coded lists,
so every existing table covers 11 methods; the within-prior-family Wilcoxon pairs run under both
prior families; the effects S, P (per prior family), Delta_c (each R2-D2 cell against its HC
twin), dS and dP, with Wilcoxon + Holm per objective family and a (family, seed)-cluster bootstrap
pooled; gate G3; the preregistered readings PR1-PR4; the npz R2 readouts by t; identification by
prior family; cost of each R2-D2 cell against its twin by GPU type; and two new sanity checks, the
prior-code check (Review Focus 3) and the twin pairing check (D10).

"Family" alone is ambiguous here: the code says `family` for the objective family (aligned10, ...)
and `prior_family` for HC / R2D2.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALL_FAMILIES = ["aligned3", "aligned10", "anti_aligned", "decoupled",
                "interaction_g0.00", "interaction_g0.25", "interaction_g0.50", "interaction_g0.75"]
FAMILIES = list(ALL_FAMILIES)                 # set from --families
SEEDS = [f"{s:02d}" for s in range(10)]       # set from --seeds
T0, T1, N_ROWS = 20, 199, 180

# The eleven methods. `dir` is the run directory name (run_dir_for: the method's "/" becomes "-"),
# `method` the manifest's spelling. The analysis environment has no torch, so the table is literal;
# test_analyze.py::test_method_table_matches_the_library pins it to sagp.gp.CELLS (structure and
# native site), experiments.run_bo.METHODS and experiments.replay.TWINS. The HC cells keep sagp2's
# order, so the within-prior-family pairs and their Holm groups are sagp2's.
Method = namedtuple("Method", "dir method structure parameterization prior_family twin short color ls")
METHOD_TABLE = (
    Method("product-lengthscale", "product/lengthscale", "product", "lengthscale", "HC", None, "PL", "#2a78d6", "-"),
    Method("additive-lengthscale", "additive/lengthscale", "additive", "lengthscale", "HC", None, "AL", "#eb6834", "-"),
    Method("product-amplitude", "product/amplitude", "product", "amplitude", "HC", None, "PA", "#1baf7a", "-"),
    Method("additive-amplitude", "additive/amplitude", "additive", "amplitude", "HC", None, "AA", "#eda100", "-"),
    Method("product-lengthscale_r2d2", "product/lengthscale_r2d2", "product", "lengthscale", "R2D2",
           "product-lengthscale", "PL-R2", "#2a78d6", (0, (5, 1.5))),
    Method("additive-lengthscale_r2d2", "additive/lengthscale_r2d2", "additive", "lengthscale", "R2D2",
           "additive-lengthscale", "AL-R2", "#eb6834", (0, (5, 1.5))),
    Method("product-amplitude_r2d2", "product/amplitude_r2d2", "product", "amplitude", "R2D2",
           "product-amplitude", "PA-R2", "#1baf7a", (0, (5, 1.5))),
    Method("additive-amplitude_r2d2", "additive/amplitude_r2d2", "additive", "amplitude", "R2D2",
           "additive-amplitude", "AA-R2", "#eda100", (0, (5, 1.5))),
    Method("sobol", "sobol", None, None, None, None, "sobol", "#e87ba4", ":"),
    Method("dsp_map", "dsp_map", None, None, None, None, "dsp_map", "#008300", "--"),
    Method("oracle_S", "oracle_S", None, None, None, None, "oracle_S", "#4a3aa7", "-."),
)
BY_DIR = {m.dir: m for m in METHOD_TABLE}
PRIOR_FAMILIES = ("HC", "R2D2")
HC_CELLS = [m.dir for m in METHOD_TABLE if m.prior_family == "HC"]
R2_CELLS = [m.dir for m in METHOD_TABLE if m.prior_family == "R2D2"]
CELLS = HC_CELLS + R2_CELLS
REFS = [m.dir for m in METHOD_TABLE if m.prior_family is None]
METHODS = [m.dir for m in METHOD_TABLE]
COLOR = {m.dir: m.color for m in METHOD_TABLE}
LINESTYLE = {m.dir: m.ls for m in METHOD_TABLE}

# Preregistered constants (docs/superpowers/specs/2026-09-27-r2d2-prereg.md) and G3's.
REGRET_FLOOR = 1e-8          # y = log10(max(r, 1e-8))
SESOI = 0.3                  # smallest effect of interest, log10 regret at T = 200
ALPHA = 0.05
N_BOOT = 10_000
BOOT_SEED = 0
PR4_MARGIN = 0.05            # AP within 0.05 of the twin's
PILOT_FAMILIES = ("aligned10", "decoupled")
PILOT_SEEDS = [f"{s:02d}" for s in range(5)]
# BoTorch's MIN_INFERRED_NOISE_LEVEL: prediction clamps a learned noise draw at it (ruling R20).
NOISE_FLOOR = 1e-4
# The files whose blobs every R2-D2 session must share (Review Focus 3).
PRIOR_CODE_PATHS = ("sagp/gp.py", "sagp/r2d2.py")
# The t at which the npz R2 readouts are read: the identification-vs-t grid.
R2_T = [T0] + list(range(25, T1, 25)) + [T1]

ITER_COLS = ["t", "method", "family", "seed", "y", "f", "best_obs", "best_f", "regret", "acq_value",
             "fit_wall_s", "acq_wall_s", "fit_calls", "nuts_attempts", "status", "reason",
             "r_hat_max", "r_hat_median", "frac_r_hat_below_1_05", "n_eff_min", "divergences",
             "num_steps_mean", "r_hat_max_native", "n_eff_min_native", "r_hat_max_ell",
             "n_eff_min_ell", "r_hat_max_global", "n_eff_min_global", "y_mean", "y_std",
             "sobol_computed"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#6f6e6a",
    "xtick.color": "#1a1a1a", "ytick.color": "#1a1a1a", "axes.labelcolor": "#1a1a1a",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "lines.linewidth": 1.8,
})


# ----------------------------------------------------------------------------- small helpers
def repo_root(start: Path) -> Path:
    out = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def parse_seeds(spec: str) -> list[str]:
    """"0-4" -> 00..04; "0,3" -> 00, 03; ranges and lists combine."""
    seeds = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds += list(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return [f"{s:02d}" for s in seeds]


def log_regret(r) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(r, float), REGRET_FLOOR))


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx]); adj[idx] = min(1.0, running)
    return adj


def wilcoxon_p(d) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank (scipy's default: exact without ties/zeros for n <= 50)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan, np.nan
    if np.all(d == 0):
        return np.nan, 1.0
    res = stats.wilcoxon(d, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def cluster_bootstrap_mean(values, clusters, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                           stat: str = "mean") -> dict:
    """Percentile bootstrap of the mean (or, with stat="median", the median) of `values`,
    resampling whole clusters with replacement.

    A resample draws as many clusters as there are, with replacement, and takes the mean or median
    of every value in the drawn clusters (a cluster drawn twice counts twice). With one value per
    cluster this is the ordinary bootstrap. Both statistics use the same resamples.
    """
    values = np.asarray(values, float)
    labels, inv = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    k = len(labels)
    idx = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    if stat == "mean":
        sums = np.bincount(inv, weights=values, minlength=k)
        counts = np.bincount(inv, minlength=k).astype(float)
        boot = sums[idx].sum(1) / counts[idx].sum(1)
    elif stat == "median":
        # each resample's median, exactly as np.median of the drawn values: the values sorted once,
        # weighted by how often their cluster was drawn, and read at the middle position(s)
        order = np.argsort(values, kind="stable")
        draws = np.zeros((n_boot, k))
        np.add.at(draws, (np.arange(n_boot)[:, None], idx), 1)
        cum = np.cumsum(draws[:, inv[order]], axis=1)
        n = cum[:, -1]
        lo = (cum > ((n - 1) // 2)[:, None]).argmax(1)
        hi = (cum > (n // 2)[:, None]).argmax(1)
        boot = 0.5 * (values[order][lo] + values[order][hi])
    else:
        raise ValueError(f"stat must be 'mean' or 'median', not {stat!r}")
    lo90, hi90, lo95, hi95 = np.quantile(boot, [0.05, 0.95, 0.025, 0.975])
    out = dict(mean=float(values.mean()), lo90=float(lo90), hi90=float(hi90), lo95=float(lo95),
               hi95=float(hi95), n=int(len(values)), n_clusters=int(k), n_boot=int(n_boot), stat=stat)
    if stat == "median":
        out["median"] = float(np.median(values))
    return out


def pr2_reading(lo90: float, hi90: float) -> str:
    """PR2's three-way reading of a 90 % interval against (-SESOI, SESOI)."""
    if lo90 > -SESOI and hi90 < SESOI:
        return "equivalent"
    if hi90 <= -SESOI or lo90 >= SESOI:
        return "prior family changes the effect"
    return "inconclusive"


def pr4_reading(lo90: float, hi90: float) -> str:
    """PR4's three-way reading (prereg clarification, 2026-09-29) of the 90 % interval of the median
    paired AP difference against (-PR4_MARGIN, PR4_MARGIN), parallel to PR2's."""
    if lo90 > -PR4_MARGIN and hi90 < PR4_MARGIN:
        return "equivalent"
    if hi90 <= -PR4_MARGIN or lo90 >= PR4_MARGIN:
        return "differs"
    return "inconclusive"


def pr4_table(idc: pd.DataFrame) -> pd.DataFrame:
    """PR4: per R2-D2 cell and AP kind, the paired differences AP(R2-D2) - AP(twin) over (family,
    seed), with the (family, seed)-cluster bootstrap 90 % interval of their median and its reading.
    `idc`: family, method, seed, ap_native, ap_sobol (complete runs)."""
    pr4 = []
    idk = idc.set_index(["family", "method", "seed"])
    for d in R2_CELLS:
        tw = BY_DIR[d].twin
        for met in ("ap_native", "ap_sobol"):
            diffs = []
            for (fam, m, s), r in idk.iterrows():
                if m == d and (fam, tw, s) in idk.index:
                    diffs.append((fam, s, r[met] - idk.loc[(fam, tw, s), met]))
            dd = pd.DataFrame(diffs, columns=["family", "seed", "diff"]).dropna()
            if dd.empty:
                continue
            b = cluster_bootstrap_mean(dd["diff"].values, (dd.family + "/" + dd.seed).values, stat="median")
            pr4.append(dict(cell=d, twin=tw, metric=met, n_pairs=len(dd), mean_diff=b["mean"],
                            median_diff=b["median"], lo90=b["lo90"], hi90=b["hi90"],
                            reading=pr4_reading(b["lo90"], b["hi90"])))
    return pd.DataFrame(pr4, columns=["cell", "twin", "metric", "n_pairs", "mean_diff", "median_diff", "lo90", "hi90",
                                      "reading"])


# ----------------------------------------------------------------------------- R2 readouts
def r2d2_r2(a_sq) -> np.ndarray:
    """S/(1 + S), S = sum_i a_sq_i, per draw: what the R2-D2 prior's Beta is on (sagp.readouts)."""
    s = np.asarray(a_sq, float).sum(-1)
    return s / (1.0 + s)


def first_order_r2(a_sq, noise, structure: str) -> np.ndarray:
    """The fitted model's first-order R^2 per draw (ruling R20).

    additive: S/(S + sigma^2); product: S/(prod_i(1 + a_sq_i) - 1 + sigma^2), whose signal variance
    prod_i(1 + a_sq_i) - 1 >= S also counts the interactions the product kernel carries.
    """
    a = np.asarray(a_sq, float)
    s = a.sum(-1)
    signal = s if structure == "additive" else np.expm1(np.log1p(a).sum(-1))
    return s / (signal + np.reshape(np.asarray(noise, float), (-1,)))


def npz_r2_medians(draws: dict, structure: str, fixed_noise) -> tuple[float, float]:
    """Medians over one fit's retained draws of `r2d2_r2` and `first_order_r2`.

    The noise is the raw `noise` draw clamped at NOISE_FLOOR (FittedGP.noises() needs a model; the
    npz holds the draw); a fixed-noise run has no `noise` site and takes the manifest's value.
    """
    a = np.asarray(draws["a_sq"], float)
    if "noise" in draws:
        s2 = np.maximum(np.reshape(np.asarray(draws["noise"], float), (-1,)), NOISE_FLOOR)
    elif fixed_noise is not None:
        s2 = np.full(a.shape[0], float(fixed_noise))
    else:
        s2 = np.full(a.shape[0], np.nan)
    return float(np.median(r2d2_r2(a))), float(np.median(first_order_r2(a, s2, structure)))


def r2_readouts(keys, run_dirs, manifests) -> pd.DataFrame:
    rows = []
    for fam, m, s in keys:
        if BY_DIR[m].parameterization != "amplitude":
            continue
        for t in R2_T:
            p = run_dirs[(fam, m, s)] / "samples" / f"t{t:03d}.npz"
            if not p.exists():
                continue
            with np.load(p) as z:
                draws = {k: z[k] for k in ("a_sq", "noise") if k in z.files}
            r2d2, first = npz_r2_medians(draws, BY_DIR[m].structure, manifests[(fam, m, s)].get("fixed_noise"))
            rows.append(dict(family=fam, method=m, seed=s, t=t, prior_family=BY_DIR[m].prior_family,
                             r2d2_r2_median=r2d2, first_order_r2_median=first))
    return pd.DataFrame(rows, columns=["family", "method", "seed", "t", "prior_family",
                                       "r2d2_r2_median", "first_order_r2_median"])


# ----------------------------------------------------------------------------- effects
EFFECT_KIND = {"S_HC": "main_HC", "P_HC": "main_HC", "S_R2D2": "main_R2D2", "P_R2D2": "main_R2D2",
               "dS": "interaction", "dP": "interaction"} | {f"Delta_{d}": "twin" for d in R2_CELLS}
EFFECT_ORDER = list(EFFECT_KIND)


def unit_effects(final: pd.DataFrame) -> pd.DataFrame:
    """Per (objective family, seed): the D11 effects on y = log10(max(r_199, 1e-8)), long form.

    S = 1/2[(y_AA - y_PA) + (y_AL - y_PL)] (additive minus product) and P = 1/2[(y_AA - y_AL) +
    (y_PA - y_PL)] (amplitude minus lengthscale) within each prior family; Delta_c = y_c(R2-D2) -
    y_c(HC twin); dS = S(R2D2) - S(HC), dP likewise. A unit missing a run lacks exactly the effects
    that need it. `final`: family, seed, method (run dir name), regret at t = 199.
    """
    y = final.assign(y=log_regret(final.regret.values)).pivot_table(
        index=["family", "seed"], columns="method", values="y", aggfunc="first")
    col = lambda d: y[d] if d in y.columns else pd.Series(np.nan, index=y.index)  # noqa: E731
    out = pd.DataFrame(index=y.index)
    for pf in PRIOR_FAMILIES:
        c = {(m.structure, m.parameterization): m.dir for m in METHOD_TABLE if m.prior_family == pf}
        AA, PA = col(c[("additive", "amplitude")]), col(c[("product", "amplitude")])
        AL, PL = col(c[("additive", "lengthscale")]), col(c[("product", "lengthscale")])
        out[f"S_{pf}"] = 0.5 * ((AA - PA) + (AL - PL))
        out[f"P_{pf}"] = 0.5 * ((AA - AL) + (PA - PL))
    out["dS"] = out["S_R2D2"] - out["S_HC"]
    out["dP"] = out["P_R2D2"] - out["P_HC"]
    for d in R2_CELLS:
        out[f"Delta_{d}"] = col(d) - col(BY_DIR[d].twin)
    long = out.reset_index().melt(id_vars=["family", "seed"], var_name="effect", value_name="value")
    long = long.dropna(subset=["value"])
    long["kind"] = long.effect.map(EFFECT_KIND)
    long["effect_order"] = long.effect.map({e: i for i, e in enumerate(EFFECT_ORDER)})
    return long.sort_values(["effect_order", "family", "seed"]).drop(columns="effect_order").reset_index(drop=True)


def effect_tables(effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per objective family (Wilcoxon over seeds, Holm within family and kind, seed bootstrap) and
    pooled over families ((family, seed)-cluster bootstrap).

    Holm's four kinds per objective family are the prereg's clarification (2026-09-29): {S(HC),
    P(HC)}, {S(R2-D2), P(R2-D2)}, {dS, dP} and the four Delta_c (EFFECT_KIND).
    """
    rows = []
    for (fam, eff), g in effects.groupby(["family", "effect"], sort=False):
        v = g.value.values
        stat, p = wilcoxon_p(v)
        b = cluster_bootstrap_mean(v, g.seed.values)
        rows.append(dict(family=fam, effect=eff, kind=EFFECT_KIND[eff], n=len(v), median=float(np.median(v)),
                         mean=b["mean"], n_positive=int((v > 0).sum()), n_negative=int((v < 0).sum()),
                         wilcoxon_stat=stat, p_value=p, lo90=b["lo90"], hi90=b["hi90"], lo95=b["lo95"],
                         hi95=b["hi95"], pr2_reading=pr2_reading(b["lo90"], b["hi90"]) if EFFECT_KIND[eff] == "interaction" else ""))
    fam_tab = pd.DataFrame(rows)
    if len(fam_tab):
        fam_tab["p_holm_within_family_kind"] = np.nan
        for (fam, kind), g in fam_tab.groupby(["family", "kind"]):
            sel = g.index[g.p_value.notna()]
            if len(sel):
                fam_tab.loc[sel, "p_holm_within_family_kind"] = holm(fam_tab.loc[sel, "p_value"].values)
        fam_tab["fo"] = fam_tab.family.map({f: i for i, f in enumerate(FAMILIES)})
        fam_tab["eo"] = fam_tab.effect.map({e: i for i, e in enumerate(EFFECT_ORDER)})
        fam_tab = fam_tab.sort_values(["fo", "eo"]).drop(columns=["fo", "eo"]).reset_index(drop=True)
    rows = []
    for eff in EFFECT_ORDER:
        g = effects[effects.effect == eff]
        if g.empty:
            continue
        b = cluster_bootstrap_mean(g.value.values, (g.family + "/" + g.seed).values)
        rows.append(dict(effect=eff, kind=EFFECT_KIND[eff], n_units=len(g), median=float(np.median(g.value)),
                         mean=b["mean"], lo90=b["lo90"], hi90=b["hi90"], lo95=b["lo95"], hi95=b["hi95"],
                         n_boot=b["n_boot"],
                         pr2_reading=pr2_reading(b["lo90"], b["hi90"]) if EFFECT_KIND[eff] == "interaction" else ""))
    return fam_tab, pd.DataFrame(rows)


def package_versions() -> dict:
    """The interpreter and the four analysis packages this run used (recorded in summary.json)."""
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, matplotlib=matplotlib.__version__)


# ----------------------------------------------------------------------------- G3
def g3_statistics(effects: pd.DataFrame, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """G3's four statistics over the twin pairs Delta_c and the units' dS, dP (plan, Task 14)."""
    tw = effects[effects.kind == "twin"]
    st = dict(n_pairs=int(len(tw)), n_units=int(len(tw[["family", "seed"]].drop_duplicates())))
    st["median_pooled"] = float(np.median(tw.value)) if len(tw) else np.nan
    b = cluster_bootstrap_mean(tw.value.values, (tw.family + "/" + tw.seed).values, n_boot, seed) if len(tw) else {}
    st["mean_pooled"] = b.get("mean", np.nan)
    st["lo95"], st["hi95"] = b.get("lo95", np.nan), b.get("hi95", np.nan)
    st["n_boot"] = n_boot
    st["median_cell"], st["n_cell"], st["p_cell"] = {}, {}, {}
    for d in R2_CELLS:
        v = tw[tw.effect == f"Delta_{d}"].value.values
        st["n_cell"][d] = int(len(v))
        st["median_cell"][d] = float(np.median(v)) if len(v) else np.nan
        st["p_cell"][d] = wilcoxon_p(v)[1]
    # Holm over the fixed four cells: a cell without a p (fewer than 2 pairs) counts as p = 1
    adj = holm([1.0 if np.isnan(st["p_cell"][d]) else st["p_cell"][d] for d in R2_CELLS])
    st["p_holm_cell"] = dict(zip(R2_CELLS, map(float, adj)))
    for e in ("dS", "dP"):
        v = effects[effects.effect == e].value.values
        st[f"n_{e}"] = int(len(v))
        st[f"median_{e}"] = float(np.median(v)) if len(v) else np.nan
    return st


def g3_criteria(st: dict) -> dict:
    """GO if any of (a)-(d); a NaN statistic fires nothing."""
    ge = lambda x: bool(abs(x) >= SESOI)  # noqa: E731
    c = dict(a=ge(st["median_pooled"]) or any(ge(v) for v in st["median_cell"].values()),
             b=bool(st["lo95"] > 0 or st["hi95"] < 0),
             c=any(bool(p < ALPHA) for p in st["p_holm_cell"].values()),
             d=ge(st["median_dS"]) or ge(st["median_dP"]))
    c["go"] = any(c.values())
    return c


def g3_interval_caveat(n_units: int) -> str:
    """REPORT's caveat on criterion (b), over the run's (family, seed) clusters (10 in the pilot, 80 in stage 4)."""
    return (f"For (b): a percentile bootstrap over {n_units} clusters tends to under-cover, and the prereg does not "
            "fix the interval type.")


def g3_reading(st: dict, families, seeds, sanity_failures) -> dict:
    """G3 is read only on the pilot's scope, with all 40 pairs and every sanity check passing."""
    crit = g3_criteria(st)
    would = "GO" if crit["go"] else "STOP"
    if sorted(families) != sorted(PILOT_FAMILIES) or list(seeds) != PILOT_SEEDS:
        return dict(read=False, verdict="NOT READ", would_read=would,
                    reasons=[f"G3 is the pilot gate (families {', '.join(PILOT_FAMILIES)}, seeds 0-4); this "
                             f"run's scope is families {', '.join(families)}, seeds {', '.join(seeds)}"])
    reasons = []
    n_expected = len(R2_CELLS) * len(PILOT_FAMILIES) * len(PILOT_SEEDS)
    if st["n_pairs"] != n_expected:
        reasons.append(f"{st['n_pairs']} of {n_expected} twin pairs have both runs complete")
    if st.get("n_dS", 0) != len(PILOT_FAMILIES) * len(PILOT_SEEDS) or st.get("n_dP", 0) != len(PILOT_FAMILIES) * len(PILOT_SEEDS):
        reasons.append(f"dS/dP over {st.get('n_dS', 0)}/{st.get('n_dP', 0)} of 10 units")
    if sanity_failures:
        reasons.append("sanity checks failing: " + ", ".join(sanity_failures))
    if reasons:
        return dict(read=False, verdict="INCONCLUSIVE", would_read=would, reasons=reasons)
    return dict(read=True, verdict=would, would_read=would, reasons=[])


# ----------------------------------------------------------------------------- new sanity checks
def twin_pairing_check(ck: pd.DataFrame) -> pd.DataFrame:
    """Each R2-D2 run against its twin (D10): t = 20's y_mean/y_std bit-identical, f_star and S equal."""
    idx = {(r.family, r.method, r.seed): r for r in ck.itertuples()}
    rows = []
    for r in ck.itertuples():
        if r.method not in R2_CELLS:
            continue
        tw = idx.get((r.family, BY_DIR[r.method].twin, r.seed))
        row = dict(family=r.family, method=r.method, seed=r.seed, twin=BY_DIR[r.method].twin,
                   twin_exists=tw is not None)
        for c in ("f_star", "y_mean_t20", "y_std_t20", "S"):
            row[f"{c}_eq"] = tw is not None and bool(getattr(r, c) == getattr(tw, c))
        row["twin_paired"] = all(row[f"{c}_eq"] for c in ("f_star", "y_mean_t20", "y_std_t20", "S"))
        rows.append(row)
    return pd.DataFrame(rows, columns=["family", "method", "seed", "twin", "twin_exists", "f_star_eq",
                                       "y_mean_t20_eq", "y_std_t20_eq", "S_eq", "twin_paired"])


def git_blob_resolver(repo: Path):
    """(commit, path) -> the blob id of `path` at `commit` in `repo`, or None; cached."""
    cache = {}

    def resolve(commit: str, path: str):
        if not commit or commit == "unknown":
            return None
        if (commit, path) not in cache:
            out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{commit}:{path}"],
                                 capture_output=True, text=True)
            cache[(commit, path)] = out.stdout.strip() if out.returncode == 0 else None
        return cache[(commit, path)]
    return resolve


def blob_pair_at(resolve, commit):
    """The blob ids of PRIOR_CODE_PATHS at `commit`, or None if any does not resolve."""
    blobs = tuple(resolve(commit, p) for p in PRIOR_CODE_PATHS)
    return None if any(b is None for b in blobs) else blobs


def prior_code_check(manifests: dict, resolve, launch_commit: str | None = None) -> pd.DataFrame:
    """Review Focus 3: every R2-D2 run's creation and resume commits carry one pair of blobs of
    sagp/gp.py and sagp/r2d2.py, the same pair for every run, and the creation checkout was clean.
    With `launch_commit`, the pair most runs share must also be the pair at that commit.

    config_hash does not see module constants, so a resume after the prior's code changed would
    splice two priors with no error; this is where that is caught. A run passes if its sessions
    resolve to a single blob pair equal to the pair most runs share; any run that fails fails the
    check. Resume entries record no `dirty` flag, so only the creation's cleanliness is checked.
    """
    rows = []
    for (fam, m, s), man in manifests.items():
        if BY_DIR[m].prior_family != "R2D2":
            continue
        git = man.get("git", {}) or {}
        commits = [git.get("commit", "unknown")] + [r.get("commit", "unknown") for r in man.get("resumed", [])]
        pairs, resolved = set(), True
        for c in commits:
            blobs = tuple(resolve(c, p) for p in PRIOR_CODE_PATHS)
            if any(b is None for b in blobs):
                resolved = False
            else:
                pairs.add(blobs)
        rows.append(dict(family=fam, method=m, seed=s, n_sessions=len(commits),
                         commits=";".join(dict.fromkeys(c[:9] for c in commits)),
                         created_clean=git.get("dirty") is False, blobs_resolved=resolved,
                         single_blob_pair=len(pairs) == 1,
                         blob_pair="/".join(b[:10] for b in next(iter(pairs))) if len(pairs) == 1 else
                         ("MIXED" if pairs else "")))
    pc = pd.DataFrame(rows, columns=["family", "method", "seed", "n_sessions", "commits", "created_clean",
                                     "blobs_resolved", "single_blob_pair", "blob_pair"])
    good = pc[pc.blobs_resolved & pc.single_blob_pair]
    modal = good.blob_pair.value_counts().index[0] if len(good) else None
    pc["prior_code_ok"] = pc.created_clean & pc.blobs_resolved & pc.single_blob_pair & (pc.blob_pair == modal)
    if launch_commit is not None:
        anchor = blob_pair_at(resolve, launch_commit)
        pc["prior_code_ok"] &= anchor is not None and modal == "/".join(b[:10] for b in anchor)
    return pc


# ----------------------------------------------------------------------------- loading
def find_run(roots, fam, m, s):
    hits = [r / fam / m / f"seed{s}" for r in roots if (r / fam / m / f"seed{s}" / "iterations.csv").exists()]
    if len(hits) > 1:
        raise SystemExit(f"{fam}/{m}/seed{s} exists under more than one --runs root: {[str(h) for h in hits]}")
    return hits[0] if hits else None


def load_runs(roots):
    """iterations rows (no x_*), manifests, coords, per-iteration device, run dirs; one pass."""
    iters, manifests, coords, log_devices, inventory, run_dirs = [], {}, [], [], [], {}
    for fam in FAMILIES:
        for m in METHODS:
            for s in SEEDS:
                d = find_run(roots, fam, m, s)
                key = (fam, m, s)
                if d is None:
                    inventory.append(dict(family=fam, method=m, seed=s, rows=0, exists=False))
                    continue
                run_dirs[key] = d
                df = pd.read_csv(d / "iterations.csv", usecols=ITER_COLS, float_precision="round_trip")
                df["reason"] = df["reason"].fillna("")
                df["method"] = m  # canonical dir name (the CSV says product/lengthscale)
                df["seed"] = s
                iters.append(df)
                man = json.loads((d / "manifest.json").read_text())
                manifests[key] = man
                inventory.append(dict(family=fam, method=m, seed=s, rows=len(df), exists=True,
                                      complete=len(df) == N_ROWS, t_min=df.t.min(), t_max=df.t.max(),
                                      n_resumed=len(man.get("resumed", [])),
                                      created=man.get("created"),
                                      csv_mtime=datetime.fromtimestamp(
                                          (d / "iterations.csv").stat().st_mtime).isoformat(),
                                      device_created=man["env"].get("jax_device"),
                                      commit=man.get("git", {}).get("commit", "")[:7],
                                      config_hash=man.get("config_hash", "")[:8], root=str(d.parents[2])))
                if (d / "coords.csv").exists():
                    c = pd.read_csv(d / "coords.csv")
                    coords.append(c[c.t == c.t.max()].assign(family=fam, method=m, seed=s))
                    # last readout with a sobol_hat (the exact/QMC ones at t = 25k and 199)
                    cs = c[c.sobol_hat.notna()]
                    if len(cs):
                        coords.append(cs[cs.t == cs.t.max()].assign(family=fam, method=m, seed=s,
                                                                    which="sobol_last"))
                # per-iteration device from log order: segment 0 = creation env, k = resumed[k-1]
                devs = [man["env"].get("jax_device")] + [r["env"].get("jax_device")
                                                        for r in man.get("resumed", [])]
                seg, t2dev, n_res = 0, {}, 0
                for line in (d / "log.txt").read_text().splitlines():
                    if line.startswith("resume at"):
                        seg += 1
                        n_res += 1
                    elif (mt := re.match(r"t=(\d+) method=", line)):
                        t2dev[int(mt.group(1))] = devs[min(seg, len(devs) - 1)]
                log_devices.append(pd.DataFrame(dict(family=fam, method=m, seed=s,
                                                     t=list(t2dev), device=list(t2dev.values()),
                                                     log_resumes=n_res, man_resumes=len(devs) - 1)))
    it = pd.concat(iters, ignore_index=True)
    inv = pd.DataFrame(inventory)
    co = pd.concat(coords, ignore_index=True)
    if "which" not in co:
        co["which"] = np.nan
    co["which"] = co["which"].fillna("last")
    dev = pd.concat(log_devices, ignore_index=True)
    return it, inv, manifests, co, dev, run_dirs


# ----------------------------------------------------------------------------- checks
def sanity_checks(it, inv, manifests, dev):
    rows = []
    for (fam, m, s), g in it.groupby(["family", "method", "seed"], sort=False):
        g = g.sort_values("t")
        man = manifests[(fam, m, s)]
        f_star = man["objective"]["f_star"]
        r = dict(family=fam, method=m, seed=s)
        r["rows_180"] = len(g) == N_ROWS
        r["no_dup_t"] = g.t.is_unique
        r["t_contiguous"] = bool((g.t.values == np.arange(T0, T0 + len(g))).all())
        r["regret_nonincreasing"] = bool((np.diff(g.regret.values) <= 1e-12).all())
        r["best_f_nondecreasing"] = bool((np.diff(g.best_f.values) >= -1e-12).all())
        # best_f at t counts the initial design too, so cummax over rows alone is a lower bound:
        r["best_f_ge_cummax_f"] = bool((g.best_f.values + 1e-9 >= np.maximum.accumulate(g.f.values)).all())
        r["regret_eq_fstar_minus_bestf"] = bool(np.allclose(g.regret.values, f_star - g.best_f.values,
                                                             rtol=0, atol=1e-9))
        r["regret_nonnegative"] = bool((g.regret.values >= -1e-9).all())
        dv = dev[(dev.family == fam) & (dev.method == m) & (dev.seed == s)]
        r["log_resumes_eq_manifest"] = bool((dv.log_resumes == dv.man_resumes).all())
        r["log_t_set_eq_csv"] = set(dv.t) == set(g.t)
        r["manifest_method_eq_dir"] = man.get("method") == BY_DIR[m].method
        r["f_star"] = f_star
        r["y_mean_t20"] = float(g.loc[g.t == T0, "y_mean"].iloc[0]) if (g.t == T0).any() else np.nan
        r["y_std_t20"] = float(g.loc[g.t == T0, "y_std"].iloc[0]) if (g.t == T0).any() else np.nan
        r["S"] = tuple(man["objective"]["S"])
        rows.append(r)
    ck = pd.DataFrame(rows)
    # cross-method checks within (family, seed)
    x = []
    for (fam, s), g in ck.groupby(["family", "seed"]):
        x.append(dict(family=fam, seed=s, n_methods=len(g),
                      f_star_identical=g.f_star.nunique() == 1,
                      S_identical=g.S.nunique() == 1,
                      y_mean_t20_identical=g.y_mean_t20.nunique() == 1,
                      y_std_t20_identical=g.y_std_t20.nunique() == 1,
                      y_mean_t20_range=float(g.y_mean_t20.max() - g.y_mean_t20.min()),
                      y_std_t20_range=float(g.y_std_t20.max() - g.y_std_t20.min())))
    cross = pd.DataFrame(x)
    return ck, cross


# ----------------------------------------------------------------------------- metrics
def average_precision(scores, truth):
    """AP as sum_n (R_n - R_{n-1}) P_n over the ranking by score desc (NaN scores rank last)."""
    scores = np.where(np.isnan(scores), -np.inf, scores)
    order = np.argsort(-scores, kind="stable")
    hits = truth[order].astype(float)
    if hits.sum() == 0:
        return np.nan
    prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((prec * hits).sum() / hits.sum())


def recall_at_k(scores, truth):
    k = int(truth.sum())
    scores = np.where(np.isnan(scores), -np.inf, scores)
    order = np.argsort(-scores, kind="stable")[:k]
    return float(truth[order].sum() / k)


def prf(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * p * r / (p + r) if (tp + fp) and (tp + fn) and (p + r) > 0 else (0.0 if (tp + fp) else np.nan)
    return p, r, f1, tp, fp, fn


def identification(co, manifests, it):
    rows = []
    last_status = it.sort_values("t").groupby(["family", "method", "seed"]).tail(1).set_index(
        ["family", "method", "seed"])
    for (fam, m, s), g in co[co.which == "last"].groupby(["family", "method", "seed"], sort=False):
        g = g.sort_values("i")
        D = len(g)
        truth = np.zeros(D, bool); truth[list(manifests[(fam, m, s)]["objective"]["S"])] = True
        gs = co[(co.which == "sobol_last") & (co.family == fam) & (co.method == m) & (co.seed == s)]
        gs = gs.sort_values("i")
        p, r, f1, tp, fp, fn = prf(g.p_active.values > 0.5, truth)
        st = last_status.loc[(fam, m, s)]
        rows.append(dict(
            family=fam, method=m, seed=s, t_readout=int(g.t.iloc[0]),
            t_sobol_readout=int(gs.t.iloc[0]) if len(gs) else np.nan,
            n_true=int(truth.sum()),
            ap_native=average_precision(g.native_median.values, truth),
            recall_at_S_native=recall_at_k(g.native_median.values, truth),
            ap_sobol=average_precision(gs.sobol_hat.values, truth) if len(gs) else np.nan,
            recall_at_S_sobol=recall_at_k(gs.sobol_hat.values, truth) if len(gs) else np.nan,
            precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn,
            n_pred_active=int((g.p_active.values > 0.5).sum()),
            status_at_readout=st.status, reason_at_readout=st.reason[:60],
            native_median_true_min=float(g.native_median.values[truth].min()),
            native_median_false_max=float(g.native_median.values[~truth].max()),
        ))
    return pd.DataFrame(rows)


def paired_tests(final):
    """final: family, method, seed, regret at t=199 (complete runs). Wilcoxon signed-rank over seeds.

    sagp2's pairs (cell vs reference, cell vs cell, on raw regret), run within each prior family;
    Holm within (objective family, prior family, kind), so the HC rows are sagp2's.
    """
    rows = []
    wide = final.pivot_table(index=["family", "seed"], columns="method", values="regret")
    for pf in PRIOR_FAMILIES:
        cells = [m.dir for m in METHOD_TABLE if m.prior_family == pf]
        pairs = [(c, r) for c in cells for r in REFS] + [(a, b) for i, a in enumerate(cells) for b in cells[i + 1:]]
        for fam in FAMILIES:
            if fam not in wide.index.get_level_values(0):
                continue
            w = wide.loc[fam]
            for a, b in pairs:
                if a not in w or b not in w:
                    continue
                d = (w[a] - w[b]).dropna()
                if len(d) == 0:
                    continue
                stat, p = wilcoxon_p(d.values)
                rows.append(dict(family=fam, prior_family=pf, cell=a, reference=b,
                                 kind="cell_vs_ref" if b in REFS else "cell_vs_cell",
                                 n_seeds=len(d), median_diff=float(d.median()), mean_diff=float(d.mean()),
                                 n_cell_better=int((d < 0).sum()), n_ref_better=int((d > 0).sum()),
                                 wilcoxon_stat=stat, p_value=p))
    df = pd.DataFrame(rows, columns=["family", "prior_family", "cell", "reference", "kind", "n_seeds",
                                     "median_diff", "mean_diff", "n_cell_better", "n_ref_better",
                                     "wilcoxon_stat", "p_value"])
    df["p_holm_within_family_kind"] = np.nan
    for (fam, pf, kind), g in df.groupby(["family", "prior_family", "kind"]):
        sel = g.index[g.p_value.notna()]
        if len(sel):
            df.loc[sel, "p_holm_within_family_kind"] = holm(df.loc[sel, "p_value"].values)
    return df


# ----------------------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, floatfmt="{:.3g}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(out)


def save_table(df, out: Path, name: str, floatfmt="{:.3g}"):
    df.to_csv(out / "tables" / f"{name}.csv", index=False)
    (out / "tables" / f"{name}.md").write_text(md_table(df, floatfmt) + "\n")


def med_iqr(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return "nan"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.3g} [{q1:.3g}, {q3:.3g}]"


def grid(n):
    """(rows, cols) for n family panels: sagp2's 2 x 2 for four, else at most four per row."""
    if n == 4:
        return 2, 2
    nc = min(n, 4)
    return -(-n // nc), nc


def order_methods(df, col="method"):
    return df.assign(_o=df[col].map({m: i for i, m in enumerate(METHODS)})).sort_values("_o").drop(columns="_o")


# ----------------------------------------------------------------------------- main
def main():
    global FAMILIES, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, action="append", required=True,
                    help="a runs root (<family>/<method>/seedNN); repeat for runs/ and runs_r2d2/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--asof", default="")
    ap.add_argument("--families", default=",".join(ALL_FAMILIES), help="comma-separated objective families")
    ap.add_argument("--seeds", default="0-9", help='seeds, e.g. "0-4" or "0,3,7"')
    ap.add_argument("--repo", type=Path, default=None,
                    help="git repository holding every run's commits (default: the one this script is in)")
    ap.add_argument("--launch-commit", default=None,
                    help="the R2-D2 launch commit (the pilot's: 97eda925b); prior_code_ok then also requires "
                         "the runs' modal (gp.py, r2d2.py) blob pair to be the pair at this commit")
    a = ap.parse_args()
    FAMILIES = [f for f in a.families.split(",") if f]
    SEEDS = parse_seeds(a.seeds)
    repo = a.repo or repo_root(Path(__file__).resolve().parent)
    out = a.out
    asof = a.asof or datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    (out / "figures").mkdir(exist_ok=True); (out / "tables").mkdir(exist_ok=True)
    notes = []  # anything that looks wrong

    it, inv, manifests, co, dev, run_dirs = load_runs(a.runs)
    it = it.merge(dev[["family", "method", "seed", "t", "device"]], on=["family", "method", "seed", "t"], how="left")
    it["is_exception"] = it.reason.str.startswith("exception:")
    it["gate_excluded"] = (it.status == "excluded") & ~it.is_exception
    it["is_cell"] = it.method.isin(CELLS)

    # 1. inventory --------------------------------------------------------------------------
    inv["complete"] = inv["complete"].eq(True) if "complete" in inv else False
    complete_keys = set(map(tuple, inv.loc[inv.complete, ["family", "method", "seed"]].values))
    it["complete"] = [k in complete_keys for k in zip(it.family, it.method, it.seed)]
    save_table(inv.sort_values(["family", "method", "seed"]), out, "inventory")
    incomplete = inv[~inv.complete].sort_values(["family", "method", "seed"])
    inv_summary = inv.groupby("method").agg(runs=("rows", "size"), complete=("complete", "sum"),
                                            rows=("rows", "sum")).reindex(METHODS).reset_index()
    save_table(inv_summary, out, "inventory_summary")

    # 2. sanity ------------------------------------------------------------------------------
    ck, cross = sanity_checks(it, inv, manifests, dev)
    tp = twin_pairing_check(ck)
    resolver = git_blob_resolver(repo)
    pc = prior_code_check(manifests, resolver, a.launch_commit)
    launch_pair = blob_pair_at(resolver, a.launch_commit) if a.launch_commit else None
    save_table(ck.assign(S=ck.S.astype(str)), out, "sanity_per_run")
    save_table(cross, out, "sanity_cross_method")
    save_table(tp, out, "sanity_twin_pairing")
    save_table(pc, out, "sanity_prior_code")
    check_cols = ["rows_180", "no_dup_t", "t_contiguous", "regret_nonincreasing", "best_f_nondecreasing",
                  "best_f_ge_cummax_f", "regret_eq_fstar_minus_bestf", "regret_nonnegative",
                  "log_resumes_eq_manifest", "log_t_set_eq_csv", "manifest_method_eq_dir"]
    runs_of = lambda df, c: "; ".join(f"{r.family}/{r.method}/seed{r.seed}" for r in df[~df[c]].itertuples())  # noqa: E731
    sanity_summary = pd.DataFrame(
        [dict(check=c, n_runs=len(ck), n_pass=int(ck[c].sum()), n_fail=int((~ck[c]).sum()), failing_runs=runs_of(ck, c))
         for c in check_cols] +
        [dict(check=c, n_runs=len(cross), n_pass=int(cross[c].sum()), n_fail=int((~cross[c]).sum()),
              failing_runs="; ".join(f"{r.family}/seed{r.seed}" for r in cross[~cross[c]].itertuples()))
         for c in ["f_star_identical", "S_identical", "y_mean_t20_identical", "y_std_t20_identical"]] +
        [dict(check="twin_paired", n_runs=len(tp), n_pass=int(tp.twin_paired.sum()),
              n_fail=int((~tp.twin_paired.astype(bool)).sum()), failing_runs=runs_of(tp.astype({"twin_paired": bool}), "twin_paired")),
         dict(check="prior_code_ok", n_runs=len(pc), n_pass=int(pc.prior_code_ok.sum()),
              n_fail=int((~pc.prior_code_ok.astype(bool)).sum()), failing_runs=runs_of(pc.astype({"prior_code_ok": bool}), "prior_code_ok"))])
    save_table(sanity_summary, out, "sanity_summary")
    sanity_failures = [r.check for r in sanity_summary.itertuples() if r.n_fail]
    for r in sanity_summary.itertuples():
        if r.n_fail:
            expected = r.check == "rows_180" and set(r.failing_runs.split("; ")) == set(
                f"{x.family}/{x.method}/seed{x.seed}" for x in incomplete[incomplete.exists].itertuples())
            if not expected:
                notes.append(f"Sanity check `{r.check}` FAILED for {r.n_fail} run(s): {r.failing_runs}")

    # 3. regret curves + final regret ----------------------------------------------------------
    itc = it[it.complete]
    curves = itc.groupby(["family", "method", "t"]).regret.agg(
        median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75), n="size").reset_index()
    curves.to_csv(out / "tables" / "regret_curves_median_iqr.csv", index=False)
    floor = max(1e-6, float(itc.regret[itc.regret > 0].min()) / 2) if (itc.regret > 0).any() else 1e-6
    nf = len(FAMILIES); nr, nc = grid(nf)
    fig, axes = plt.subplots(nr, nc, figsize=(11, 7.5) if nf == 4 else (3.4 * nc, 3.4 * nr + 1.2), sharex=True, squeeze=False)
    for ax in axes.ravel()[nf:]:
        ax.set_visible(False)
    for ax, fam in zip(axes.ravel(), FAMILIES):
        for m in METHODS:
            c = curves[(curves.family == fam) & (curves.method == m)]
            if c.empty:
                continue
            ax.plot(c.t, np.maximum(c["median"], floor), color=COLOR[m], ls=LINESTYLE[m], label=f"{m} (n={int(c.n.iloc[-1])})")
            ax.fill_between(c.t, np.maximum(c.q1, floor), np.maximum(c.q3, floor), color=COLOR[m], alpha=0.10, lw=0)
        ax.set_yscale("log"); ax.set_title(fam); ax.set_xlabel("iteration t"); ax.set_ylabel("regret = f* - best_f")
    hd, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03 if nf == 4 else -0.16 / nr))
    fig.suptitle(f"Median regret across seeds (band = IQR), complete runs only; log y, floored at {floor:.1e}; "
                 "dashed = R2-D2 (colour = its half-Cauchy twin)", y=1.0)
    fig.tight_layout(); fig.savefig(out / "figures" / "regret_curves.png", bbox_inches="tight"); plt.close(fig)

    final = itc[itc.t == T1][["family", "method", "seed", "regret", "best_f"]].copy()
    final["y"] = log_regret(final.regret.values)
    final.to_csv(out / "tables" / "final_regret_per_run.csv", index=False)
    rows = []
    for fam in FAMILIES:
        for m in METHODS:
            x = final[(final.family == fam) & (final.method == m)].regret.values
            if len(x) == 0:
                continue
            rows.append(dict(family=fam, method=m, prior_family=BY_DIR[m].prior_family or "reference",
                             n=len(x), median=np.median(x), q1=np.percentile(x, 25),
                             q3=np.percentile(x, 75), mean=x.mean(), min=x.min(), max=x.max(),
                             median_iqr=med_iqr(x), median_y=float(np.median(log_regret(x)))))
    fr = pd.DataFrame(rows, columns=["family", "method", "prior_family", "n", "median", "q1", "q3", "mean",
                                     "min", "max", "median_iqr", "median_y"])
    fr["rank_in_family"] = fr.groupby("family")["median"].rank(method="min").astype(int)
    fr = fr.sort_values(["family", "rank_in_family"])
    # per-seed ranks (a rank-based ranking that is robust to one wild seed)
    final["rank_within_seed"] = final.groupby(["family", "seed"]).regret.rank(method="average")
    mean_rank = final.groupby(["family", "method"]).rank_within_seed.mean().reset_index(name="mean_rank_within_seed")
    fr = fr.merge(mean_rank, on=["family", "method"])
    save_table(fr, out, "final_regret_t199")
    # figure: final regret per family, strip + median
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.9 * nr), squeeze=False)
    for ax in axes.ravel()[nf:]:
        ax.set_visible(False)
    for ax, fam in zip(axes.ravel(), FAMILIES):
        for j, m in enumerate(METHODS):
            x = final[(final.family == fam) & (final.method == m)].regret.values
            if len(x) == 0:
                continue
            jit = (np.random.default_rng(0).uniform(-0.18, 0.18, len(x)))
            ax.scatter(j + jit, np.maximum(x, floor), s=14, color=COLOR[m], alpha=0.75, lw=0,
                       marker="o" if BY_DIR[m].prior_family != "R2D2" else "D")
            ax.hlines(np.median(x), j - 0.3, j + 0.3, color="#0b0b0b", lw=1.6)
        ax.set_yscale("log"); ax.set_title(fam); ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([BY_DIR[m].short for m in METHODS], fontsize=8, rotation=40, ha="right")
        ax.set_ylabel("final regret (t=199)")
    fig.suptitle("Final regret per seed (dots; diamonds = R2-D2) and median (bar); complete runs only, log y. "
                 "PL/AL = product/additive lengthscale, PA/AA = product/additive amplitude, -R2 = R2-D2 prior", y=1.02)
    fig.tight_layout(); fig.savefig(out / "figures" / "final_regret_strip.png", bbox_inches="tight"); plt.close(fig)

    # 4. paired tests within each prior family --------------------------------------------------
    pt = paired_tests(final)
    save_table(pt, out, "paired_wilcoxon_final_regret")

    # 4b. effects, twin pairs, G3, PR1-PR3 -------------------------------------------------------
    effects = unit_effects(final)
    effects.to_csv(out / "tables" / "effects_per_unit.csv", index=False)
    eff_fam, eff_pool = effect_tables(effects)
    save_table(eff_fam, out, "effects_by_family")
    save_table(eff_pool, out, "effects_pooled")
    g3s = g3_statistics(effects)
    g3c = g3_criteria(g3s)
    g3r = g3_reading(g3s, FAMILIES, SEEDS, sanity_failures)
    (out / "g3.json").write_text(json.dumps(dict(statistics=g3s, criteria=g3c, reading=g3r), indent=2, default=float) + "\n")
    g3_rows = [dict(criterion="(a)", statistic="median Delta, pooled", n=g3s["n_pairs"], value=g3s["median_pooled"],
                    rule=f"abs(value) >= {SESOI}", fires=bool(abs(g3s["median_pooled"]) >= SESOI))]
    g3_rows += [dict(criterion="(a)", statistic=f"median Delta, {d}", n=g3s["n_cell"][d], value=g3s["median_cell"][d],
                     rule=f"abs(value) >= {SESOI}", fires=bool(abs(g3s["median_cell"][d]) >= SESOI)) for d in R2_CELLS]
    g3_rows += [dict(criterion="(b)", statistic=f"mean Delta, pooled; cluster bootstrap 95 % [{g3s['lo95']:.3g}, {g3s['hi95']:.3g}]",
                     n=g3s["n_pairs"], value=g3s["mean_pooled"], rule="interval excludes 0", fires=g3c["b"])]
    g3_rows += [dict(criterion="(c)", statistic=f"Wilcoxon p, {d} (raw {g3s['p_cell'][d]:.3g}); Holm over 4 cells",
                     n=g3s["n_cell"][d], value=g3s["p_holm_cell"][d], rule=f"value < {ALPHA}",
                     fires=bool(g3s["p_holm_cell"][d] < ALPHA)) for d in R2_CELLS]
    g3_rows += [dict(criterion="(d)", statistic=f"median {e} over units", n=g3s[f"n_{e}"], value=g3s[f"median_{e}"],
                     rule=f"abs(value) >= {SESOI}", fires=bool(abs(g3s[f"median_{e}"]) >= SESOI)) for e in ("dS", "dP")]
    g3_tab = pd.DataFrame(g3_rows)
    save_table(g3_tab, out, "g3")

    # PR1: wherever S(HC) (P(HC)) is significant after Holm, S(R2D2) (P(R2D2)) has the same sign
    pr1 = []
    for fam in FAMILIES:
        for e in ("S", "P"):
            h = eff_fam[(eff_fam.family == fam) & (eff_fam.effect == f"{e}_HC")] if len(eff_fam) else eff_fam
            r = eff_fam[(eff_fam.family == fam) & (eff_fam.effect == f"{e}_R2D2")] if len(eff_fam) else eff_fam
            if h.empty:
                continue
            ph = float(h.p_holm_within_family_kind.iloc[0])
            mh = float(h["median"].iloc[0]); mr = float(r["median"].iloc[0]) if len(r) else np.nan
            sig = bool(ph < ALPHA)
            pr1.append(dict(family=fam, effect=e, n_HC=int(h.n.iloc[0]), median_HC=mh, p_holm_HC=ph, significant_HC=sig,
                            n_R2D2=int(r.n.iloc[0]) if len(r) else 0, median_R2D2=mr,
                            holds=("yes" if np.sign(mr) == np.sign(mh) else "no") if sig and not np.isnan(mr) else
                            ("no R2-D2 data" if sig else "not tested")))
    pr1 = pd.DataFrame(pr1, columns=["family", "effect", "n_HC", "median_HC", "p_holm_HC", "significant_HC",
                                     "n_R2D2", "median_R2D2", "holds"])
    save_table(pr1, out, "pr1_sign_agreement")
    # PR3: |median Delta_c| < SESOI for every cell and objective family
    pr3 = eff_fam[eff_fam.kind == "twin"][["family", "effect", "n", "median", "mean", "lo90", "hi90", "p_value",
                                          "p_holm_within_family_kind"]].copy() if len(eff_fam) else pd.DataFrame(
        columns=["family", "effect", "n", "median"])
    pr3["holds"] = pr3["median"].abs() < SESOI
    save_table(pr3, out, "pr3_twin_medians")

    # 5. identification ------------------------------------------------------------------------
    idf = identification(co, manifests, it)
    idf["complete"] = [k in complete_keys for k in zip(idf.family, idf.method, idf.seed)]
    save_table(idf, out, "identification_per_run")
    idc = idf[idf.complete]
    metrics = ["ap_native", "recall_at_S_native", "ap_sobol", "recall_at_S_sobol", "precision", "recall", "f1", "n_pred_active"]
    by_cf = idc.groupby(["method", "family"])[metrics].agg(["median", "mean"])
    by_cf.columns = [f"{a_}_{b_}" for a_, b_ in by_cf.columns]
    by_cf = by_cf.reset_index()
    by_cf["n_runs"] = idc.groupby(["method", "family"]).size().values
    by_cf["method_order"] = by_cf.method.map({m: i for i, m in enumerate(METHODS)})
    by_cf = by_cf.sort_values(["method_order", "family"]).drop(columns="method_order")
    save_table(by_cf, out, "identification_by_cell_family")
    by_c = idc.groupby("method")[metrics].agg(["median", "mean"])
    by_c.columns = [f"{a_}_{b_}" for a_, b_ in by_c.columns]
    by_c = by_c.reindex([m for m in METHODS if m in by_c.index]).reset_index()
    by_c["n_runs"] = idc.groupby("method").size().reindex(by_c.method).values
    save_table(by_c, out, "identification_by_cell")
    # by prior family: each cell (structure x parameterization) under HC and R2-D2, pooled over families
    idp = idc[idc.method.isin(CELLS)].assign(
        prior_family=lambda d: d.method.map(lambda m: BY_DIR[m].prior_family),
        structure=lambda d: d.method.map(lambda m: BY_DIR[m].structure),
        parameterization=lambda d: d.method.map(lambda m: BY_DIR[m].parameterization))
    id_pf = idp.groupby(["structure", "parameterization", "prior_family"]).agg(
        n_runs=("ap_native", "size"), ap_native_mean=("ap_native", "mean"), ap_native_median=("ap_native", "median"),
        ap_sobol_mean=("ap_sobol", "mean"), ap_sobol_median=("ap_sobol", "median"), f1_mean=("f1", "mean"),
        n_pred_active_median=("n_pred_active", "median")).reset_index()
    save_table(id_pf, out, "identification_by_prior_family")
    # PR4: AP by native_median and by sobol_hat within PR4_MARGIN of the twin's (prereg clarification:
    # the cluster bootstrap 90 % interval of the median paired difference, read as PR2 is)
    pr4 = pr4_table(idc)
    save_table(pr4, out, "pr4_identification_twin_pairs")
    # per-family ranking of the cells on mean AP(native) and mean F1
    rank_rows = []
    for fam in FAMILIES:
        sub = by_cf[(by_cf.family == fam) & by_cf.method.isin(CELLS)]
        if sub.empty:
            continue
        rank_rows.append(dict(family=fam,
                              best_ap_native=sub.sort_values("ap_native_mean", ascending=False).method.iloc[0],
                              best_ap_sobol=sub.sort_values("ap_sobol_mean", ascending=False).method.iloc[0],
                              best_f1=sub.sort_values("f1_mean", ascending=False).method.iloc[0],
                              order_by_ap_native=" > ".join(sub.sort_values("ap_native_mean", ascending=False).method)))
    save_table(pd.DataFrame(rank_rows), out, "identification_ranking")
    # figure: boxes of AP(native), AP(sobol), F1 per cell per family
    id_methods = CELLS + ["dsp_map"]
    stride = len(id_methods) + 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, met, ttl in zip(axes, ["ap_native", "ap_sobol", "f1"],
                            ["average precision, score = native_median", "average precision, score = sobol_hat",
                             "F1 at p_active > 0.5"]):
        for fi, fam in enumerate(FAMILIES):
            for j, m in enumerate(id_methods):
                x = idc[(idc.family == fam) & (idc.method == m)][met].dropna().values
                if len(x) == 0:
                    continue
                pos = fi * stride + j
                jit = np.random.default_rng(1).uniform(-0.2, 0.2, len(x))
                ax.scatter(pos + jit, x, s=10, color=COLOR[m], alpha=0.7, lw=0,
                           marker="o" if BY_DIR[m].prior_family != "R2D2" else "D")
                ax.hlines(np.median(x), pos - 0.35, pos + 0.35, color="#0b0b0b", lw=1.3)
        ax.set_xticks([fi * stride + stride / 2 - 1 for fi in range(len(FAMILIES))]); ax.set_xticklabels(FAMILIES, fontsize=7, rotation=20)
        ax.set_title(ttl); ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("score at last readout (1 = perfect)")
    handles = [plt.Line2D([], [], marker="o" if BY_DIR[m].prior_family != "R2D2" else "D", ls="", color=COLOR[m], label=m)
               for m in id_methods]
    axes[2].legend(handles=handles, loc="lower right", frameon=False, fontsize=6)
    fig.suptitle("Active-set identification at the last readout (t=199), per seed; bar = median; diamonds = R2-D2", y=1.02)
    fig.tight_layout(); fig.savefig(out / "figures" / "identification.png", bbox_inches="tight"); plt.close(fig)

    # identification over time (AP native at each readout) for the cells: cheap and informative
    ap_t = []
    for fam in FAMILIES:
        for m in CELLS + ["dsp_map"]:
            for s in SEEDS:
                if (fam, m, s) not in complete_keys:
                    continue
                c = pd.read_csv(run_dirs[(fam, m, s)] / "coords.csv", usecols=["t", "i", "native_median", "p_active"])
                truth = np.zeros(100, bool); truth[list(manifests[(fam, m, s)]["objective"]["S"])] = True
                for t, g in c.groupby("t"):
                    if t % 25 == 0 or t == T1 or t == T0:
                        g = g.sort_values("i")
                        ap_t.append(dict(family=fam, method=m, seed=s, t=t,
                                         ap_native=average_precision(g.native_median.values, truth),
                                         f1=prf(g.p_active.values > 0.5, truth)[2]))
    apt = pd.DataFrame(ap_t, columns=["family", "method", "seed", "t", "ap_native", "f1"])
    apt_med = apt.groupby(["family", "method", "t"])[["ap_native", "f1"]].median().reset_index()
    apt_med.to_csv(out / "tables" / "identification_vs_t_median.csv", index=False)
    fig, axes = plt.subplots(2, nf, figsize=(3.25 * nf, 6), sharex=True, sharey=True, squeeze=False)
    for ci, fam in enumerate(FAMILIES):
        for ri, met in enumerate(["ap_native", "f1"]):
            ax = axes[ri, ci]
            for m in CELLS + ["dsp_map"]:
                c = apt_med[(apt_med.family == fam) & (apt_med.method == m)]
                if c.empty:
                    continue
                ax.plot(c.t, c[met], color=COLOR[m], ls=LINESTYLE[m], marker="o", ms=3, label=m)
            ax.set_title(f"{fam}: median {'AP (native_median)' if met == 'ap_native' else 'F1 (p_active>0.5)'}", fontsize=8)
            ax.set_ylim(-0.03, 1.03)
            if ri == 1:
                ax.set_xlabel("iteration t")
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.suptitle("Identification quality over the run (readouts at t=20, 25k, 199), median across complete seeds", y=1.0)
    fig.tight_layout(); fig.savefig(out / "figures" / "identification_vs_t.png"); plt.close(fig)

    # 5b. npz R2 readouts by t (amplitude cells, both prior families) ------------------------------
    r2 = r2_readouts(sorted(run_dirs), run_dirs, manifests)
    r2["complete"] = [k in complete_keys for k in zip(r2.family, r2.method, r2.seed)]
    save_table(r2, out, "r2_readouts_per_run_t")
    r2 = r2[r2.complete]
    r2_med = r2.groupby(["family", "method", "t"]).agg(
        n=("seed", "size"), r2d2_r2_median=("r2d2_r2_median", "median"),
        first_order_r2_median=("first_order_r2_median", "median")).reset_index()
    save_table(order_methods(r2_med).sort_values(["family", "t"], kind="stable"), out, "r2_readouts_vs_t_median")
    r2_final = r2[r2.t == T1].groupby(["method", "family"]).agg(
        n=("seed", "size"), r2d2_r2=("r2d2_r2_median", med_iqr), first_order_r2=("first_order_r2_median", med_iqr)).reset_index()
    r2_final = order_methods(r2_final)

    # 6. diagnostics ---------------------------------------------------------------------------
    def reason_class(r):
        if r.startswith("exception:"):
            return "exception"
        if r == "":
            return "ok"
        parts = [p.strip().split(" ")[0] for p in r.split(";")]
        return "+".join(parts)
    it["reason_class"] = it.reason.map(reason_class)
    diag = it.groupby(["method", "family"]).agg(
        n_iter=("t", "size"), gate_excluded=("gate_excluded", "sum"), exception_rows=("is_exception", "sum"),
        ok=("status", lambda s: int((s == "ok").sum())),
        r_hat_max_median=("r_hat_max", "median"), r_hat_max_q90=("r_hat_max", lambda x: x.quantile(0.9)),
        n_eff_min_median=("n_eff_min", "median"), n_eff_min_q10=("n_eff_min", lambda x: x.quantile(0.1)),
        divergences_mean=("divergences", "mean"), fit_calls_mean=("fit_calls", "mean"),
        nuts_attempts_mean=("nuts_attempts", "mean")).reset_index()
    diag["gate_excluded_rate"] = diag.gate_excluded / diag.n_iter
    diag["method_order"] = diag.method.map({m: i for i, m in enumerate(METHODS)})
    diag = diag.sort_values(["method_order", "family"]).drop(columns="method_order")
    save_table(diag, out, "diagnostics_by_method_family")
    diag_m = it.groupby("method").agg(
        n_iter=("t", "size"), gate_excluded=("gate_excluded", "sum"), exception_rows=("is_exception", "sum"),
        r_hat_max_q10=("r_hat_max", lambda x: x.quantile(0.1)), r_hat_max_median=("r_hat_max", "median"),
        r_hat_max_q90=("r_hat_max", lambda x: x.quantile(0.9)),
        n_eff_min_q10=("n_eff_min", lambda x: x.quantile(0.1)), n_eff_min_median=("n_eff_min", "median"),
        n_eff_min_q90=("n_eff_min", lambda x: x.quantile(0.9)),
        r_hat_max_native_median=("r_hat_max_native", "median"), r_hat_max_ell_median=("r_hat_max_ell", "median"),
        r_hat_max_global_median=("r_hat_max_global", "median"),
        n_eff_min_native_median=("n_eff_min_native", "median"), n_eff_min_ell_median=("n_eff_min_ell", "median"),
        n_eff_min_global_median=("n_eff_min_global", "median"),
        divergences_mean=("divergences", "mean"), divergences_gt5_frac=("divergences", lambda x: float((x > 5).mean())),
        frac_r_hat_below_1_05_median=("frac_r_hat_below_1_05", "median"),
        num_steps_mean_median=("num_steps_mean", "median"),
        frac_at_tree_cap=("num_steps_mean", lambda x: float((x >= 62).mean()) if x.notna().any() else np.nan),
        fit_calls_mean=("fit_calls", "mean"), nuts_attempts_mean=("nuts_attempts", "mean")).reindex(METHODS).reset_index()
    diag_m["gate_excluded_rate"] = diag_m.gate_excluded / diag_m.n_iter
    save_table(diag_m, out, "diagnostics_by_method")
    rc = it[it.is_cell | (it.method == "dsp_map")].groupby(["method", "reason_class"]).size().unstack(fill_value=0)
    rc = rc.reindex([m for m in CELLS + ["dsp_map"] if m in rc.index]).reset_index()
    save_table(rc, out, "gate_reason_breakdown")
    # exclusion rate vs t (bins of 20)
    itcell = it[it.is_cell].copy(); itcell["t_bin"] = (itcell.t // 20) * 20
    exr = itcell.groupby(["method", "t_bin"]).gate_excluded.mean().unstack(0).reset_index()
    exr = exr[["t_bin"] + [m for m in CELLS if m in exr.columns]]
    save_table(exr, out, "gate_exclusion_rate_vs_t")
    # figure: r_hat_max and n_eff_min distributions per cell
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, m in enumerate(CELLS + ["dsp_map"]):
        x = it[it.method == m]
        if x.empty:
            continue
        axes[0].hist(x.r_hat_max.dropna().clip(upper=3), bins=60, range=(1, 3), histtype="step", color=COLOR[m], ls=LINESTYLE[m], label=m, lw=1.4)
        axes[1].hist(np.log10(x.n_eff_min.dropna().clip(lower=1)), bins=60, range=(0, 2.5), histtype="step", color=COLOR[m], ls=LINESTYLE[m], label=m, lw=1.4)
    axes[0].axvline(1.1, color="#0b0b0b", lw=1, ls="--"); axes[0].set_xlabel("r_hat_max (clipped at 3); gate at 1.1"); axes[0].set_ylabel("iterations")
    axes[1].axvline(np.log10(16), color="#0b0b0b", lw=1, ls="--"); axes[1].set_xlabel("log10 n_eff_min; gate at 16")
    for m in CELLS:
        if m not in exr:
            continue
        c = exr[["t_bin", m]].dropna()
        axes[2].plot(c.t_bin + 10, c[m], color=COLOR[m], ls=LINESTYLE[m], marker="o", ms=3, label=m)
    axes[2].set_ylim(-0.03, 1.03); axes[2].set_xlabel("iteration t (bins of 20)"); axes[2].set_ylabel("gate-exclusion rate")
    axes[0].legend(frameon=False, fontsize=6)
    fig.suptitle("NUTS diagnostics over all iterations (all runs); dashed = R2-D2", y=1.02)
    fig.tight_layout(); fig.savefig(out / "figures" / "diagnostics.png", bbox_inches="tight"); plt.close(fig)

    # 7. cost -------------------------------------------------------------------------------------
    it["iter_wall_s"] = it.fit_wall_s + it.acq_wall_s
    cost = it.groupby(["method", "device"]).agg(
        n_iter=("t", "size"), n_runs=("seed", lambda s: s.size and len(set(zip(it.loc[s.index, "family"], s)))),
        fit_wall_s_median=("fit_wall_s", "median"), acq_wall_s_median=("acq_wall_s", "median"),
        iter_wall_s_median=("iter_wall_s", "median"), fit_wall_s_q90=("fit_wall_s", lambda x: x.quantile(0.9)),
        acq_wall_s_q90=("acq_wall_s", lambda x: x.quantile(0.9)),
        hours_fit_plus_acq=("iter_wall_s", lambda x: x.sum() / 3600)).reset_index()
    cost["method_order"] = cost.method.map({m: i for i, m in enumerate(METHODS)})
    cost = cost.sort_values(["method_order", "device"]).drop(columns="method_order")
    save_table(cost, out, "cost_by_method_device")
    tot = it.groupby("method").agg(n_iter=("t", "size"), hours_fit=("fit_wall_s", lambda x: x.sum() / 3600),
                                   hours_acq=("acq_wall_s", lambda x: x.sum() / 3600),
                                   hours_fit_plus_acq=("iter_wall_s", lambda x: x.sum() / 3600)).reindex(METHODS).reset_index()
    per_run = it.groupby(["method", "family", "seed"]).iter_wall_s.sum().div(3600).reset_index(name="run_hours")
    tot = tot.merge(per_run.groupby("method").run_hours.median().reset_index(name="median_run_hours"), on="method", how="left")
    tot = tot.merge(per_run.groupby("method").run_hours.max().reset_index(name="max_run_hours"), on="method", how="left")
    save_table(tot, out, "cost_total_hours_by_method")
    # each R2-D2 cell against its twin, iteration by iteration on one GPU type
    ct = []
    for d in R2_CELLS:
        tw = BY_DIR[d].twin
        for dv in sorted(it.loc[it.method == d, "device"].dropna().unique()):
            x, y = it[(it.method == d) & (it.device == dv)], it[(it.method == tw) & (it.device == dv)]
            row = dict(cell=d, twin=tw, device=dv, n_iter=len(x), n_iter_twin=len(y))
            for c in ("fit_wall_s", "acq_wall_s", "iter_wall_s"):
                row[f"{c}_median"] = float(x[c].median())
                row[f"{c}_median_twin"] = float(y[c].median()) if len(y) else np.nan
                row[f"{c}_ratio"] = row[f"{c}_median"] / row[f"{c}_median_twin"] if len(y) else np.nan
            ct.append(row)
    cost_twin = pd.DataFrame(ct, columns=["cell", "twin", "device", "n_iter", "n_iter_twin"] +
                             [f"{c}_{k}" for c in ("fit_wall_s", "acq_wall_s", "iter_wall_s") for k in ("median", "median_twin", "ratio")])
    save_table(cost_twin, out, "cost_twin_by_device")
    # cost vs t per device (fit and acq), cells only: growth with n
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for m in CELLS:
        for dv, mk in [("NVIDIA H200", "o"), ("NVIDIA H100 80GB HBM3", "s"), ("NVIDIA A100-SXM4-80GB", "^")]:
            x = it[(it.method == m) & (it.device == dv)]
            if len(x) < 50:
                continue
            g = x.groupby((x.t // 10) * 10)[["fit_wall_s", "acq_wall_s"]].median()
            axes[0].plot(g.index + 5, g.fit_wall_s, color=COLOR[m], ls=LINESTYLE[m], marker=mk, ms=2.5, label=f"{m} / {dv.split()[1]}")
            axes[1].plot(g.index + 5, g.acq_wall_s, color=COLOR[m], ls=LINESTYLE[m], marker=mk, ms=2.5, label=f"{m} / {dv.split()[1]}")
    axes[0].set_title("median fit_wall_s vs t (bins of 10)"); axes[1].set_title("median acq_wall_s vs t (bins of 10)")
    for ax in axes:
        ax.set_xlabel("iteration t"); ax.set_ylabel("seconds"); ax.set_yscale("log")
    axes[1].legend(frameon=False, fontsize=5, ncol=2)
    fig.suptitle("Per-iteration cost by cell and GPU (marker: o H200, s H100, ^ A100; dashed = R2-D2); never compare across GPUs", y=1.02)
    fig.tight_layout(); fig.savefig(out / "figures" / "cost_vs_t.png", bbox_inches="tight"); plt.close(fig)

    # anomaly scan --------------------------------------------------------------------------------
    mixed = dev.groupby(["family", "method", "seed"]).device.nunique()
    n_mixed = int((mixed > 1).sum())
    ex_non_dsp = it[it.is_exception & (it.method != "dsp_map")]
    if len(ex_non_dsp):
        notes.append(f"{len(ex_non_dsp)} exception rows OUTSIDE dsp_map: " +
                     ", ".join(f"{r.family}/{r.method}/seed{r.seed} t={r.t} ({r.reason[:50]})" for r in ex_non_dsp.head(10).itertuples()))
    ex_kinds = it[it.is_exception].reason.str.extract(r"exception: (\w+)")[0].value_counts().to_dict()
    if any(k != "NotPSDError" for k in ex_kinds):
        notes.append(f"Exception kinds other than NotPSDError seen: {ex_kinds}")
    infos = []  # expected oddities, reported but not flagged
    n_hashes = int(inv.config_hash.nunique())
    inv_pf = inv[inv.exists].assign(pf=lambda d: d.method.map(lambda m: BY_DIR[m].prior_family or "reference"))
    commits = {pf: sorted(g.commit.unique().tolist()) for pf, g in inv_pf.groupby("pf")}
    infos.append(f"Git commits at creation, by prior family: {commits} (the HC cells and references were launched at "
                 "a51a4b9 and 2866ed2; the R2-D2 cells' commits are checked by `prior_code_ok`).")
    neg = itc[itc.regret < -1e-9]
    if len(neg):
        notes.append(f"{len(neg)} rows with negative regret (best_f above f_star): " +
                     ", ".join(f"{r.family}/{r.method}/seed{r.seed} t={r.t} regret={r.regret:.3g}" for r in neg.head(8).itertuples()))
    # readouts at t=199 for a fit that was gate-excluded
    exc_read = idc.groupby("method").status_at_readout.apply(lambda s: int((s == "excluded").sum())).to_dict()
    # a run whose last coords readout is not at its last iteration
    lastt = it.groupby(["family", "method", "seed"]).t.max()
    last_reason = it.sort_values("t").groupby(["family", "method", "seed"]).reason.last()
    for r in idf.itertuples():
        tl = lastt.loc[(r.family, r.method, r.seed)]
        if r.t_readout != tl:
            msg = f"{r.family}/{r.method}/seed{r.seed}: last coords.csv readout is t={r.t_readout}, last iteration t={tl}"
            if last_reason.loc[(r.family, r.method, r.seed)].startswith("exception:"):
                infos.append(msg + " (the last iteration was an exception row, so no surrogate and no readout; scored at t=%d)" % r.t_readout)
            else:
                notes.append(msg)
    # oracle_S must rank its own S perfectly (metric check); its thresholded p_active is a MAP fit's single 0/1 draw
    orc = idc[idc.method == "oracle_S"]
    if len(orc) and not ((orc.ap_native == 1).all() and (orc.recall_at_S_native == 1).all()):
        notes.append("oracle_S ranking metrics are not perfect; the metric or the oracle readout is off: "
                     f"AP min {orc.ap_native.min():.3g}, recall@|S| min {orc.recall_at_S_native.min():.3g}")
    # degenerate readouts: (nearly) every coordinate declared active
    degen = idc[idc.n_pred_active >= 50]
    for r in degen.itertuples():
        notes.append(f"DEGENERATE readout: {r.family}/{r.method}/seed{r.seed} declares {r.n_pred_active}/100 coordinates active at t={r.t_readout} "
                     f"(AP by native_median {r.ap_native:.2f}, by sobol_hat {r.ap_sobol:.2f}; min native_median on true S {r.native_median_true_min:.3g}, "
                     f"max on inactive {r.native_median_false_max:.3g}; fit status {r.status_at_readout})")
    # NUTS tree-depth saturation
    cap = it[it.is_cell].groupby("method").num_steps_mean.apply(lambda x: float((x >= 62).mean())).to_dict()
    if any(v > 0.5 for v in cap.values()):
        notes.append("NUTS hits max_tree_depth=6 (num_steps_mean >= 62 of the 63 possible) in most fits: fraction per cell " +
                     ", ".join(f"{k} {v:.3f}" for k, v in cap.items()))
    (out / "tables" / "anomalies.md").write_text("\n".join(f"- {n}" for n in notes) + "\n\nExpected oddities:\n" +
                                                 "\n".join(f"- {n}" for n in infos) + "\n")
    # identification quality split by the gate status of the fit that produced the readout
    ig = apt.merge(it[["family", "method", "seed", "t", "gate_excluded"]], on=["family", "method", "seed", "t"], how="left")
    ig = ig[(ig.t >= 100) & ig.method.isin(CELLS)]
    ig_tab = ig.groupby(["method", "gate_excluded"]).agg(n_readouts=("t", "size"), ap_native_mean=("ap_native", "mean"),
                                                         ap_native_median=("ap_native", "median"), f1_mean=("f1", "mean")).reset_index()
    ig_tab["gate_excluded"] = ig_tab.gate_excluded.map({True: "excluded", False: "passed"})
    ig_tab = order_methods(ig_tab)
    save_table(ig_tab, out, "identification_by_gate_status")

    # 8. what changed since previous latest ---------------------------------------------------------
    prev_section = "No previous analysis directory found: this is the first run, nothing to diff against."
    latest = out.parent / "latest"
    prev = None
    if latest.is_symlink() or latest.exists():
        try:
            prev = latest.resolve()
            if prev == out.resolve():
                prev = None
        except OSError:
            prev = None
    if prev is not None and (prev / "tables" / "inventory.csv").exists():
        pinv = pd.read_csv(prev / "tables" / "inventory.csv", dtype={"seed": str})
        pcomp = set(map(tuple, pinv.loc[pinv.complete.fillna(False).astype(bool), ["family", "method", "seed"]].values))
        newly = sorted(complete_keys - pcomp)
        lines = [f"Previous analysis: `{prev.name}`.",
                 f"Newly complete runs since then: {len(newly)}" + (": " + ", ".join(f"{f}/{m}/seed{s}" for f, m, s in newly) if newly else ".")]
        pfr_path = prev / "tables" / "final_regret_t199.csv"
        if pfr_path.exists():
            pfr = pd.read_csv(pfr_path)
            mv = fr.merge(pfr[["family", "method", "median", "rank_in_family"]], on=["family", "method"], suffixes=("", "_prev"))
            moved = mv[(mv.rank_in_family != mv.rank_in_family_prev)]
            lines.append("Final-regret rank changes (family/method: prev rank -> now): " +
                         ("; ".join(f"{r.family}/{r.method}: {int(r.rank_in_family_prev)} -> {int(r.rank_in_family)}" for r in moved.itertuples()) if len(moved) else "none"))
        pid_path = prev / "tables" / "identification_ranking.csv"
        if pid_path.exists():
            pidr = pd.read_csv(pid_path); now = pd.DataFrame(rank_rows)
            m2 = now.merge(pidr, on="family", suffixes=("", "_prev"))
            ch = m2[m2.best_ap_native != m2.best_ap_native_prev]
            lines.append("Best cell by AP(native) changed in: " + ("; ".join(f"{r.family}: {r.best_ap_native_prev} -> {r.best_ap_native}" for r in ch.itertuples()) if len(ch) else "no family"))
        ppt_path = prev / "tables" / "paired_wilcoxon_final_regret.csv"
        if ppt_path.exists():
            ppt = pd.read_csv(ppt_path)
            m3 = pt.merge(ppt[["family", "cell", "reference", "p_value"]], on=["family", "cell", "reference"], suffixes=("", "_prev"))
            flip = m3[((m3.p_value < 0.05) != (m3.p_value_prev < 0.05))]
            lines.append("Wilcoxon p<0.05 status flipped for: " + ("; ".join(f"{r.family} {r.cell} vs {r.reference} (p {r.p_value_prev:.3g} -> {r.p_value:.3g})" for r in flip.itertuples()) if len(flip) else "none"))
        prev_section = "\n".join(lines)
    (out / "tables" / "what_changed.md").write_text(prev_section + "\n")

    # REPORT.md --------------------------------------------------------------------------------------
    R = []
    def h(s, lvl=2): R.append("\n" + "#" * lvl + " " + s + "\n")
    n_complete, n_total = int(inv.complete.sum()), len(inv)
    roots = ", ".join(f"`{r}`" for r in a.runs)
    R.append(f"# sagp study analysis: half-Cauchy and R2-D2 prior families\n\nAs of **{asof}** (data: {roots}; analysis dir `{out}`; "
             f"script `analyze.py`; families {', '.join(FAMILIES)}; seeds {', '.join(SEEDS)}).\n")
    R.append(f"Runs: **{n_complete}/{n_total} complete** (180 rows). Incomplete or missing: " +
             (", ".join(f"{r.family}/{r.method}/seed{r.seed} ({r.rows} rows)" for r in incomplete.itertuples()) if len(incomplete) else "none") + ".\n")
    R.append(f"Headline numbers use complete runs only. Seeds are paired: one (family, seed) is one objective and one shared "
             f"initial design across all {len(METHODS)} methods. \"Family\" alone means the objective family; the half-Cauchy "
             "(HC) and R2-D2 priors are the two prior families.\n")
    R.append((out / "top5.md").read_text() if (out / "top5.md").exists() else "<!-- TOP5 -->\n")

    h("G3 (the pilot gate)")
    if g3r["read"]:
        R.append("**G3: GO to stage 4.** At least one criterion fires; the full replicate (Task 15) may be launched.\n"
                 if g3r["verdict"] == "GO" else
                 "**G3: STOP.** No criterion fires: the full replicate is not run, and the pilot is the appendix result.\n")
    elif g3r["verdict"] == "INCONCLUSIVE":
        R.append(f"**G3: INCONCLUSIVE.** {'; '.join(g3r['reasons'])}. The statistics as computed would read "
                 f"{'GO' if g3r['would_read'] == 'GO' else 'STOP'}; the gate is not read on them.\n")
    else:
        R.append(f"**G3: not read.** {'; '.join(g3r['reasons'])}: the table below is descriptive.\n")
    R.append(f"\nDelta = y(R2-D2) - y(HC twin) at t = 199, y = log10(max(r, 1e-8)), over {g3s['n_pairs']} pairs in "
             f"{g3s['n_units']} (family, seed) units. GO if any of: (a) |median Delta| >= {SESOI} pooled or within any cell; "
             f"(b) the pooled (family, seed)-cluster bootstrap 95 % interval of the mean Delta ({N_BOOT:,} resamples) excludes 0; "
             f"(c) any cell's Wilcoxon signed-rank over its pairs, Holm-adjusted over the 4 cells, below {ALPHA}; "
             f"(d) |median dS| or |median dP| >= {SESOI} over the units. Criteria firing: "
             f"(a) {g3c['a']}, (b) {g3c['b']}, (c) {g3c['c']}, (d) {g3c['d']}. "
             + g3_interval_caveat(g3s["n_units"]) + "\n\n")
    R.append(md_table(g3_tab) + "\n")
    h("What changed since the previous analysis"); R.append(prev_section + "\n")
    h("1. Inventory"); R.append(md_table(inv_summary) + "\n")
    if len(incomplete):
        R.append("\nIncomplete or missing runs:\n\n" + md_table(incomplete[["family", "method", "seed", "rows", "t_max", "n_resumed", "device_created"]]) + "\n")
    S_size = {fam: sorted({len(manifests[k]["objective"]["S"]) for k in manifests if k[0] == fam}) for fam in FAMILIES}
    gammas = {fam: sorted({manifests[k]["objective"].get("gamma") for k in manifests if k[0] == fam}) for fam in FAMILIES}
    any_man = next(iter(manifests.values()))
    R.append(f"\nObjectives: D = 100, noise_sd = {any_man['objective'].get('noise_sd')}; |S| per family {S_size}; gamma per family {gammas}. "
             f"Distinct config_hash values: {n_hashes} (one per run; the hash covers the objective and the method). "
             f"Git commits at creation by prior family: {commits}.\n")
    R.append(f"\nDevices at creation (runs): {inv.device_created.value_counts().to_dict()}. Runs whose iterations span more "
             f"than one GPU type (resumed onto a different partition): {n_mixed}. Resumes per run: {inv.n_resumed.value_counts().sort_index().to_dict()} "
             "(most `resumed` entries are no-op resumes at t=200 recorded by the chain's later Slurm links).\n")
    h("2. Sanity checks"); R.append(md_table(sanity_summary[["check", "n_runs", "n_pass", "n_fail", "failing_runs"]]) + "\n")
    R.append("\n`best_f_ge_cummax_f` is a lower-bound check because best_f also covers the 20-point initial design, which is not a row. "
             "`regret_eq_fstar_minus_bestf` checks the CSV's regret against manifest.objective.f_star to 1e-9. "
             "`log_resumes_eq_manifest` and `log_t_set_eq_csv` verify that log.txt's resume lines match manifest.resumed and that every "
             "iteration in the CSV appears once in the log (device attribution relies on this). `manifest_method_eq_dir`: the manifest's "
             "method is the one its directory names.\n")
    R.append("\n`twin_paired` (D10): each R2-D2 run's row t = 20 `y_mean` and `y_std` are bit-identical to its half-Cauchy twin's, and "
             "`f_star` and S are equal (`tables/sanity_twin_pairing.csv`). `prior_code_ok` (Review Focus 3): every R2-D2 run's creation "
             f"commit and every resume commit resolve, in `{repo}`, to one pair of blobs of {' and '.join(PRIOR_CODE_PATHS)}, the pair "
             "most R2-D2 runs share, and the creation checkout was clean (resume entries record no dirty flag) "
             "(`tables/sanity_prior_code.csv`).\n")
    if a.launch_commit:
        R.append(f"\nLaunch commit `{a.launch_commit}`: blob pair (gp.py/r2d2.py) "
                 f"{'`' + '/'.join(launch_pair) + '`' if launch_pair else '**does not resolve**'}; `prior_code_ok` "
                 "also requires the R2-D2 runs' modal pair to be this one.\n")
    else:
        R.append("\nNo `--launch-commit` given: `prior_code_ok` compares the runs with each other only.\n")
    if len(pc):
        R.append("\nPrior-code blob pairs (gp.py/r2d2.py) over the R2-D2 runs: " +
                 ", ".join(f"`{k or 'unresolved'}` x {v}" for k, v in pc.blob_pair.value_counts().items()) +
                 f"; commits seen: {sorted(set(';'.join(pc.commits).split(';')))}.\n")
    h("3. Optimization performance")
    R.append("![regret curves](figures/regret_curves.png)\n\n![final regret](figures/final_regret_strip.png)\n")
    R.append("\nFinal regret at t=199, median [Q1, Q3] across complete seeds, ranked within family (1 = best); `median_y` is the "
             "median of y = log10(max(r, 1e-8)):\n\n")
    R.append(md_table(fr[["family", "method", "n", "median_iqr", "median_y", "rank_in_family", "mean_rank_within_seed"]]) + "\n")
    h("4. Paired comparisons within each prior family (Wilcoxon signed-rank over seeds, final regret at t=199)")
    R.append("`median_diff` = median over seeds of (cell regret - reference regret); negative means the cell is better. "
             "Two-sided p-values from scipy.stats.wilcoxon (exact for n <= 50 without ties). The smallest attainable two-sided p is "
             "0.002 at 10 seeds and 0.0625 at 5. `p_holm_within_family_kind` is a Holm correction over the 12 cell-vs-reference "
             "tests (or 6 cell-vs-cell tests) of one prior family inside each objective family, as sagp2 did for the half-Cauchy cells. "
             "These tests are on raw regret, as sagp2's; the effects and twin pairs below are on y.\n\n")
    cols = ["family", "prior_family", "cell", "reference", "n_seeds", "median_diff", "n_cell_better", "n_ref_better", "p_value", "p_holm_within_family_kind"]
    R.append(md_table(pt[pt.kind == "cell_vs_ref"][cols]) + "\n")
    R.append("\nCell-vs-cell (same test):\n\n" + md_table(pt[pt.kind == "cell_vs_cell"][cols]) + "\n")
    h("5. Effects: structure S, parameterization P, twin pairs Delta_c, and their prior-family differences")
    R.append("Per (family, seed) on y = log10(max(r_199, 1e-8)): S = 1/2[(y_AA - y_PA) + (y_AL - y_PL)] (additive minus product), "
             "P = 1/2[(y_AA - y_AL) + (y_PA - y_PL)] (amplitude minus lengthscale), within each prior family; Delta_c = y_c(R2-D2) - "
             "y_c(HC twin); dS = S(R2D2) - S(HC), dP likewise. Negative = lower regret for the first-named side. Per family: two-sided "
             "Wilcoxon over seeds, Holm within family across the effects of one kind (main_HC: S_HC, P_HC; main_R2D2: S_R2D2, P_R2D2; "
             f"interaction: dS, dP; twin: the four Delta_c); lo/hi are the seed bootstrap's percentile intervals ({N_BOOT:,} resamples). "
             "Pooled: the (family, seed)-cluster bootstrap of the mean.\n\n")
    ecols = ["effect", "kind", "n_units", "median", "mean", "lo90", "hi90", "lo95", "hi95", "pr2_reading"]
    R.append("Pooled over families:\n\n" + (md_table(eff_pool[ecols]) if len(eff_pool) else "(no complete units)") + "\n")
    fcols = ["family", "effect", "n", "median", "mean", "lo90", "hi90", "p_value", "p_holm_within_family_kind", "pr2_reading"]
    R.append("\nPer family:\n\n" + (md_table(eff_fam[fcols]) if len(eff_fam) else "(no complete units)") + "\n")
    h("6. Preregistered predictions PR1-PR4")
    full = sorted(FAMILIES) == sorted(ALL_FAMILIES) and SEEDS == [f"{s:02d}" for s in range(10)]
    all_r2_complete = all((f, m, s) in complete_keys for f in FAMILIES for m in R2_CELLS for s in SEEDS)
    if full and all_r2_complete:
        R.append("Scope: the full replicate (8 families x 10 seeds, every R2-D2 run complete): this is the confirmatory analysis.\n")
    else:
        R.append("Scope: **not** the confirmatory analysis (that needs all 8 families x 10 seeds with all 320 R2-D2 runs complete; "
                 "no interim analysis is run in stage 4). The readings below are descriptive.\n")
    pr1_tested = pr1[pr1.significant_HC] if len(pr1) else pr1
    R.append(f"\n**PR1** (wherever S(HC) or P(HC) is significant, Holm p < {ALPHA}, the R2-D2 effect has the same sign; sign of the "
             f"median over seeds): tested in {len(pr1_tested)} (family, effect); holds in {int((pr1_tested.holds == 'yes').sum())}, "
             f"fails in {int((pr1_tested.holds == 'no').sum())}.\n\n" + (md_table(pr1) if len(pr1) else "(no data)") + "\n")
    pr2 = eff_pool[eff_pool.kind == "interaction"] if len(eff_pool) else eff_pool
    R.append(f"\n**PR2** (|dS| and |dP| below {SESOI}; pooled 90 % cluster-bootstrap interval inside (-{SESOI}, {SESOI}) = equivalent, "
             "entirely outside = prior family changes the effect, otherwise inconclusive): " +
             ("; ".join(f"{r.effect}: mean {r.mean:.3g}, 90 % [{r.lo90:.3g}, {r.hi90:.3g}] -> {r.pr2_reading}" for r in pr2.itertuples()) if len(pr2) else "no data") +
             ". Per-family readings are in section 5.\n")
    R.append(f"\n**PR3** (Q6's null: |median Delta_c| < {SESOI} for every cell and family): holds in {int(pr3.holds.sum())} of {len(pr3)} "
             f"(cell, family); {'holds' if len(pr3) and pr3.holds.all() else 'does not hold'} overall.\n\n" +
             (md_table(pr3) if len(pr3) else "(no data)") + "\n")
    R.append(f"\n**PR4** (secondary: AP by native_median and by sobol_hat within {PR4_MARGIN} of the twin's; per the prereg's "
             f"clarification, the (family, seed)-cluster bootstrap 90 % interval ({N_BOOT:,} resamples) of the median paired "
             f"difference, R2-D2 minus twin, inside (-{PR4_MARGIN}, {PR4_MARGIN}) = equivalent, entirely outside = differs, "
             "otherwise inconclusive; lo90/hi90 are that interval): " +
             (", ".join(f"{k} {int((pr4.reading == k).sum())}" for k in ("equivalent", "differs", "inconclusive"))
              if len(pr4) else "no data") + f" of {len(pr4)} (cell, metric).\n\n" + (md_table(pr4) if len(pr4) else "(no data)") + "\n")
    h("7. Identification of the active set")
    R.append("Scored at the LAST readout of each run (coords.csv rows are written every iteration; sobol_hat is computed at t = 25, 50, ..., 175 and 199). "
             "`native_median` is the cell's own sparsity parameter (rho_i = 1/ell_i^2 for the lengthscale cells, a_sq_i for the amplitude cells; "
             "dsp_map: its MAP rho), so higher = more active in every cell. AP = average precision of ranking the 100 coordinates by the score; "
             "recall@|S| = fraction of the true S inside the top-|S| coordinates. Thresholded metrics use p_active > 0.5. "
             "oracle_S is included as a metric check (must be perfect); sobol has no model and no readout.\n\n")
    R.append("![identification](figures/identification.png)\n\n![identification vs t](figures/identification_vs_t.png)\n\n")
    R.append("By prior family (each cell under HC and R2-D2, pooled over families, complete runs):\n\n" + md_table(id_pf) + "\n")
    R.append("\nPer method (all families pooled), median and mean over complete runs:\n\n" +
             md_table(by_c[["method", "n_runs", "ap_native_median", "ap_native_mean", "recall_at_S_native_mean", "ap_sobol_mean", "recall_at_S_sobol_mean", "precision_mean", "recall_mean", "f1_mean", "n_pred_active_median"]]) + "\n")
    R.append("\nPer method and family (mean over seeds):\n\n" +
             md_table(by_cf[["method", "family", "n_runs", "ap_native_mean", "recall_at_S_native_mean", "ap_sobol_mean", "recall_at_S_sobol_mean", "precision_mean", "recall_mean", "f1_mean", "n_pred_active_median"]]) + "\n")
    R.append("\nBest cell per family:\n\n" + md_table(pd.DataFrame(rank_rows)) + "\n")
    R.append(f"\nRuns whose t=199 readout came from a gate-excluded fit, per method: {exc_read}.\n")
    h("8. R2 readouts from the npz draws (amplitude cells)")
    R.append("Per fit, medians over its 16 retained draws: `r2d2_r2` = S/(1 + S) with S = sum_i a_sq_i, what the R2-D2 prior's Beta is on "
             "(about R^2/(1 + R^2) on a standardized target, so at most about 1/2); `first_order_r2` = S/(S + sigma^2) for the additive "
             "cells and S/(prod_i(1 + a_sq_i) - 1 + sigma^2) for the product cells (ruling R20), with sigma^2 the raw noise draw clamped at "
             f"{NOISE_FLOOR:g} (the manifest's fixed_noise where the noise is fixed). Read at t = {', '.join(map(str, R2_T))} "
             "(`tables/r2_readouts_vs_t_median.csv`); at t = 199, median [Q1, Q3] over seeds:\n\n" +
             (md_table(r2_final) if len(r2_final) else "(no complete amplitude runs)") + "\n")
    h("9. Diagnostics")
    R.append("![diagnostics](figures/diagnostics.png)\n\nPer method, all iterations of all runs (gate: split-R-hat <= 1.1, ESS >= 16 on every sampled site, <= 5 divergences):\n\n")
    R.append(md_table(diag_m[["method", "n_iter", "gate_excluded_rate", "exception_rows", "r_hat_max_q10", "r_hat_max_median", "r_hat_max_q90", "n_eff_min_q10", "n_eff_min_median", "n_eff_min_q90", "divergences_mean", "divergences_gt5_frac"]]) + "\n")
    R.append("\nSampler effort (num_steps_mean = leapfrog steps per NUTS iteration, max 63 at max_tree_depth = 6; fit_calls > 1 means a retry):\n\n" +
             md_table(diag_m[diag_m.method.isin(CELLS)][["method", "num_steps_mean_median", "frac_at_tree_cap", "fit_calls_mean", "nuts_attempts_mean"]]) + "\n")
    R.append("\nIdentification quality of readouts at t >= 100 split by whether that fit passed the gate (cells):\n\n" +
             md_table(ig_tab) + "\n")
    R.append("\nWhich site fails (medians of the per-site diagnostics):\n\n" +
             md_table(diag_m[diag_m.method.isin(CELLS)][["method", "r_hat_max_native_median", "r_hat_max_ell_median", "r_hat_max_global_median", "n_eff_min_native_median", "n_eff_min_ell_median", "n_eff_min_global_median", "frac_r_hat_below_1_05_median"]]) + "\n")
    R.append("\nGate reason breakdown (iterations):\n\n" + md_table(rc) + "\n")
    R.append("\nPer method and family:\n\n" + md_table(diag[["method", "family", "n_iter", "gate_excluded_rate", "exception_rows", "r_hat_max_median", "n_eff_min_median", "divergences_mean"]]) + "\n")
    R.append("\nGate-exclusion rate vs t (bins of 20), cells:\n\n" + md_table(exr) + "\n")
    h("10. Cost")
    R.append("Wall-times grouped by the GPU that ran the iteration (attributed from the resume segments in log.txt); the reference methods ran on CPU. "
             "**Do not compare across GPU types.** `hours_fit_plus_acq` sums the measured fit and acquisition time only: it excludes the "
             "per-iteration readout (coords.csv), checkpointing, JAX compilation at resume, and Slurm queue time, so it is a lower bound on allocated GPU-hours.\n\n")
    R.append("Each R2-D2 cell against its half-Cauchy twin on the same GPU type (medians over iterations; ratio = R2-D2 / twin):\n\n" +
             (md_table(cost_twin) if len(cost_twin) else "(no R2-D2 iterations)") + "\n\n")
    R.append(md_table(cost[["method", "device", "n_iter", "fit_wall_s_median", "acq_wall_s_median", "iter_wall_s_median", "fit_wall_s_q90", "acq_wall_s_q90", "hours_fit_plus_acq"]]) + "\n")
    R.append("\nTotals per method (all devices, all rows including incomplete runs):\n\n" + md_table(tot) + "\n")
    R.append("\n![cost vs t](figures/cost_vs_t.png)\n")
    h("11. Anomalies and things that look wrong")
    R.append(("\n".join(f"- {n}" for n in notes) if notes else "- Nothing failed beyond the expected incomplete run(s).") + "\n")
    R.append("\nExpected oddities (not flagged):\n\n" + ("\n".join(f"- {n}" for n in infos) if infos else "- none") +
             f"\n- dsp_map: {int((it.reason == 'fit_gpytorch_mll failed; Adam fallback').sum())} iterations used the Adam fallback fitter "
             "(status ok, reason 'fit_gpytorch_mll failed; Adam fallback'); these are neither gate exclusions nor exception rows.\n")
    h("Caveats")
    R.append("- status=\"excluded\" is a label: the draws were still used to choose the next point, so regret curves are valid for every run. "
             "It does mean the posterior readouts (native_median, p_active, sobol_hat) from those iterations are less trustworthy.\n"
             "- Rows whose reason starts with \"exception:\" queried a seeded random point instead; they are counted separately from gate exclusions.\n"
             "- GPU runs (A100/H100/H200) reproduce only in distribution, not bit-for-bit; wall-times are only comparable within one GPU type.\n"
             "- The per-family tests have at most 10 seeds (5 in the pilot, whose smallest two-sided Wilcoxon p is 0.0625).\n"
             "- The regret plots floor the log axis at a small positive value; seeds that reached regret 0 sit on that floor.\n")
    (out / "REPORT.md").write_text("\n".join(R))
    # machine-readable summary for the top-5 writer
    summary = dict(asof=asof, families=FAMILIES, seeds=SEEDS, n_complete=n_complete, n_total=n_total, notes=notes, infos=infos,
                   n_mixed_device_runs=n_mixed, exception_kinds=ex_kinds, readout_from_excluded=exc_read, tree_cap_frac=cap,
                   S_size=S_size, sanity_failures=sanity_failures, g3=dict(statistics=g3s, criteria=g3c, reading=g3r),
                   versions=package_versions())
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"done: {n_complete}/{n_total} complete; notes: {len(notes)}; G3: {g3r['verdict']} (would read {g3r['would_read']})")
    for n in notes:
        print("NOTE:", n)


if __name__ == "__main__":
    main()
