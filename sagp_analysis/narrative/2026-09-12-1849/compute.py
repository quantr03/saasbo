#!/usr/bin/env python
"""Narrative-support computations for the sagp study. Read-only on runs/ and latest/; writes tables/.

Everything here re-derives from coords.csv / iterations.csv / manifest.json (raw runs) or joins tables
that latest/analyze.py already wrote. Definitions match analyze.py: AP = average precision of ranking
the 100 coordinates by a score (NaN ranks last); "detected" = p_active > 0.5.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import SeedSequence, default_rng
from scipy import stats

RUNS = Path("/scratch/work/tranq8/saasbo/runs")
LATEST = Path("/scratch/work/tranq8/sagp_analysis/latest")
OUT = Path(sys.argv[1])
TAB = OUT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
FAMILIES = ["aligned3", "aligned10", "decoupled", "interaction_g0.25"]
CELLS = ["product-lengthscale", "additive-lengthscale", "product-amplitude", "additive-amplitude"]
MODELS = CELLS + ["dsp_map", "oracle_S"]
SEEDS = [f"{s:02d}" for s in range(10)]


def ap(scores, truth):
    scores = np.where(np.isnan(scores), -np.inf, scores)
    order = np.argsort(-scores, kind="stable")
    hits = truth[order].astype(float)
    prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((prec * hits).sum() / hits.sum())


def truth_of(man):
    t = np.zeros(100, bool)
    t[list(man["objective"]["S"])] = True
    return t


def manifest(fam, m, sd):
    return json.loads((RUNS / fam / m / f"seed{sd}" / "manifest.json").read_text())


# ---------------------------------------------------------------- A. decoupled: which coordinates
# synthobj.families.build: S = default_rng(SeedSequence(seed).spawn(3)[0]).permutation(D)[:n_active],
# in RANK order; shares and ells are assigned by rank. The manifest stores S sorted, so the rank
# order is recomputed here and checked against the manifest as a set.
SHARES = (0.21,) * 4 + (0.04,) * 4
ELLS = (1.5, 1.5, 0.08, 0.08, 1.5, 1.5, 0.08, 0.08)
CLASS = ["high-share smooth"] * 2 + ["high-share rough"] * 2 + ["low-share smooth"] * 2 + ["low-share rough"] * 2
rows, checks = [], []
for s in range(10):
    select = SeedSequence(s).spawn(3)[0]
    perm = [int(v) for v in default_rng(select).permutation(100)[:8]]
    for m in MODELS:
        sd = f"{s:02d}"
        man = manifest("decoupled", m, sd)
        S_man = [int(v) for v in man["objective"]["S"]]
        checks.append(dict(seed=sd, method=m, rank_order=" ".join(map(str, perm)), manifest_S=" ".join(map(str, S_man)),
                           set_match=sorted(perm) == sorted(S_man), manifest_is_sorted=S_man == sorted(S_man)))
        c = pd.read_csv(RUNS / "decoupled" / m / f"seed{sd}" / "coords.csv")
        last = c[c.t == c.t.max()].sort_values("i")
        assert len(last) == 100
        nat, pa, sob = last.native_median.values, last.p_active.values, last.sobol_hat.values
        nat_rank = stats.rankdata(-nat, method="min")
        sob_rank = stats.rankdata(-np.nan_to_num(sob, nan=-np.inf), method="min")
        for rank, coord in enumerate(perm):
            rows.append(dict(seed=sd, method=m, t=int(last.t.iloc[0]), rank=rank, coord=coord, share=SHARES[rank],
                             ell=ELLS[rank], cls=CLASS[rank], native_median=nat[coord], p_active=pa[coord],
                             sobol_hat=sob[coord], native_rank=int(nat_rank[coord]), sobol_rank=int(sob_rank[coord]),
                             detected=bool(pa[coord] > 0.5), top8_native=bool(nat_rank[coord] <= 8),
                             top8_sobol=bool(sob_rank[coord] <= 8)))
det = pd.DataFrame(rows)
det.to_csv(TAB / "decoupled_per_coordinate.csv", index=False)
chk = pd.DataFrame(checks)
chk.to_csv(TAB / "decoupled_S_rank_check.csv", index=False)
print("decoupled S check: set_match all", bool(chk.set_match.all()), "| manifest sorted all", bool(chk.manifest_is_sorted.all()))
summ = det.groupby(["method", "cls"]).agg(
    n=("coord", "size"), frac_detected=("detected", "mean"), frac_top8_native=("top8_native", "mean"),
    frac_top8_sobol=("top8_sobol", "mean"), median_native_rank=("native_rank", "median"),
    median_sobol_rank=("sobol_rank", "median"), median_native_median=("native_median", "median"),
    median_p_active=("p_active", "median"), median_sobol_hat=("sobol_hat", "median")).reset_index()
summ["cls"] = pd.Categorical(summ.cls, ["high-share smooth", "high-share rough", "low-share smooth", "low-share rough"])
summ["method"] = pd.Categorical(summ.method, MODELS)
summ = summ.sort_values(["method", "cls"])
summ.to_csv(TAB / "decoupled_breakdown.csv", index=False)
print(summ.to_string(index=False))

# ---------------------------------------------------------------- B. degenerate-readout scan
scan, traj = [], []
for fam in FAMILIES:
    for m in CELLS + ["dsp_map"]:
        for sd in SEEDS:
            d = RUNS / fam / m / f"seed{sd}"
            truth = truth_of(manifest(fam, m, sd))
            c = pd.read_csv(d / "coords.csv", usecols=["t", "i", "native_median", "p_active"])
            per_t = []
            for t, g in c.groupby("t"):
                g = g.sort_values("i")
                nat, pa = g.native_median.values, g.p_active.values
                per_t.append(dict(t=int(t), n_pred_active=int((pa > 0.5).sum()), ap_native=ap(nat, truth),
                                  native_true_min=float(nat[truth].min()), native_false_max=float(nat[~truth].max()),
                                  native_true_median=float(np.median(nat[truth])),
                                  native_false_median=float(np.median(nat[~truth]))))
            pt = pd.DataFrame(per_t)
            bad = pt[pt.n_pred_active >= 50]
            late = pt[pt.t >= 100]
            inv = late[late.ap_native < 0.3]
            scan.append(dict(family=fam, method=m, seed=sd, n_readouts=len(pt), n_readouts_ge50_active=len(bad),
                             first_t_ge50=int(bad.t.min()) if len(bad) else -1, last_t_ge50=int(bad.t.max()) if len(bad) else -1,
                             max_n_pred_active=int(pt.n_pred_active.max()), n_readouts_t_ge_100_ap_lt_0_3=len(inv),
                             min_ap_t_ge_100=float(late.ap_native.min()), ap_t_last=float(pt.ap_native.iloc[-1]),
                             n_pred_active_t_last=int(pt.n_pred_active.iloc[-1])))
            if fam == "aligned3" and m == "additive-lengthscale" and sd in ("00", "02", "08"):
                it = pd.read_csv(d / "iterations.csv", usecols=["t", "status", "r_hat_max", "n_eff_min", "divergences", "regret"])
                pt = pt.merge(it, on="t", how="left")
                pt.insert(0, "seed", sd)
                traj.append(pt)
scan = pd.DataFrame(scan)
scan.to_csv(TAB / "degenerate_scan_all_runs.csv", index=False)
flag = scan[(scan.n_readouts_ge50_active > 0) | (scan.n_readouts_t_ge_100_ap_lt_0_3 > 0)]
flag.to_csv(TAB / "degenerate_scan_flagged.csv", index=False)
print("\nflagged runs (any readout with >=50 active, or AP<0.3 at t>=100):")
print(flag.to_string(index=False))
traj = pd.concat(traj, ignore_index=True)
traj.to_csv(TAB / "degenerate_trajectory_aligned3_AL.csv", index=False)
sub = traj[traj.t.isin([20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 160, 170, 175, 180, 185, 190, 195, 199])]
print("\ntrajectory aligned3/additive-lengthscale seeds 00, 02, 08:")
print(sub.to_string(index=False))

# ---------------------------------------------------------------- C. final regret joined with identification
idr = pd.read_csv(LATEST / "tables" / "identification_per_run.csv", dtype={"seed": str})
fr = pd.read_csv(LATEST / "tables" / "final_regret_per_run.csv", dtype={"seed": str})
j = idr.merge(fr, on=["family", "method", "seed"], how="inner")
j.to_csv(TAB / "regret_vs_identification_per_run.csv", index=False)
rows = []
for (fam, m), g in j.groupby(["family", "method"]):
    for met in ["ap_native", "ap_sobol", "f1", "recall", "n_pred_active"]:
        x, y = g[met].values.astype(float), g.regret.values
        ok = ~np.isnan(x)
        if ok.sum() >= 4 and np.std(x[ok]) > 0:
            r, p = stats.spearmanr(x[ok], y[ok])
        else:
            r, p = np.nan, np.nan
        rows.append(dict(family=fam, method=m, metric=met, n=int(ok.sum()), spearman_vs_regret=r, p=p))
sp = pd.DataFrame(rows)
sp.to_csv(TAB / "regret_vs_identification_spearman.csv", index=False)
print("\nSpearman(identification metric, final regret) per family x method (negative = better identification, lower regret):")
print(sp[sp.metric.isin(["ap_native", "f1"])].pivot_table(index=["family", "method"], columns="metric", values="spearman_vs_regret").round(2).to_string())
wide = fr.pivot_table(index=["family", "seed"], columns="method", values="regret")
wide.to_csv(TAB / "final_regret_wide.csv")

# ---------------------------------------------------------------- D. the native score and p_active ON the true S
rows, offs = [], []
for fam in FAMILIES:
    for m in CELLS:
        for sd in SEEDS:
            man = manifest(fam, m, sd)
            S = [int(v) for v in man["objective"]["S"]]
            c = pd.read_csv(RUNS / fam / m / f"seed{sd}" / "coords.csv")
            last = c[c.t == c.t.max()].sort_values("i")
            nat, pa, sob = last.native_median.values, last.p_active.values, last.sobol_hat.values
            for i in S:
                rows.append(dict(family=fam, method=m, seed=sd, coord=i, native_median=nat[i], p_active=pa[i], sobol_hat=sob[i]))
            off = np.ones(100, bool)
            off[S] = False
            offs.append(dict(family=fam, method=m, seed=sd, native_off_max=float(nat[off].max()), native_off_median=float(np.median(nat[off])),
                             p_active_off_max=float(pa[off].max()), n_off_p_gt_0_5=int((pa[off] > 0.5).sum())))
onS = pd.DataFrame(rows)
onS.to_csv(TAB / "native_on_true_S_per_coordinate.csv", index=False)
pd.DataFrame(offs).to_csv(TAB / "native_off_S_per_run.csv", index=False)
q = lambda p: (lambda x: float(np.quantile(x, p)))
summ = onS.groupby(["method", "family"]).agg(
    n_coords=("coord", "size"), native_q10=("native_median", q(0.1)), native_median=("native_median", "median"),
    native_q90=("native_median", q(0.9)), p_active_q10=("p_active", q(0.1)), p_active_median=("p_active", "median"),
    p_active_q90=("p_active", q(0.9)), frac_p_gt_0_5=("p_active", lambda x: float((x > 0.5).mean())),
    frac_p_gt_0_25=("p_active", lambda x: float((x > 0.25).mean())), frac_p_gt_0=("p_active", lambda x: float((x > 0).mean())),
    sobol_median=("sobol_hat", "median")).reset_index()
summ["method"] = pd.Categorical(summ.method, CELLS)
summ["family"] = pd.Categorical(summ.family, FAMILIES)
summ = summ.sort_values(["method", "family"])
summ.to_csv(TAB / "native_on_true_S_summary.csv", index=False)
print("\nnative score and p_active on the TRUE coordinates at the last readout:")
print(summ.to_string(index=False))

# ---------------------------------------------------------------- E. lock-on times from the median AP/F1 vs t table
v = pd.read_csv(LATEST / "tables" / "identification_vs_t_median.csv")
rows = []
for (fam, m), g in v.groupby(["family", "method"]):
    g = g.sort_values("t")

    def first(col, thr):
        h = g[g[col] >= thr]
        return int(h.t.iloc[0]) if len(h) else -1

    def stays(col, thr):
        h = g[g[col] >= thr]
        if not len(h):
            return False
        return bool((g[g.t >= h.t.iloc[0]][col] >= thr).all())

    rows.append(dict(family=fam, method=m, first_t_ap_ge_0_9=first("ap_native", 0.9), ap_stays_ge_0_9=stays("ap_native", 0.9),
                     first_t_ap_ge_0_95=first("ap_native", 0.95), first_t_f1_ge_0_8=first("f1", 0.8), f1_stays_ge_0_8=stays("f1", 0.8),
                     ap_t199=float(g.ap_native.iloc[-1]), f1_t199=float(g.f1.iloc[-1]), ap_max=float(g.ap_native.max()),
                     t_of_ap_max=int(g.t.iloc[int(np.argmax(g.ap_native.values))])))
lock = pd.DataFrame(rows)
lock.to_csv(TAB / "lockon_times.csv", index=False)
print("\nlock-on times (median-over-seeds AP by native_median, F1 at p_active>0.5; -1 = never):")
print(lock.to_string(index=False))
print("\nDONE")
