#!/usr/bin/env python
"""Read-only analysis of the sagp study runs. Writes figures/, tables/, REPORT.md into --out."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = ["aligned3", "aligned10", "decoupled", "interaction_g0.25"]
CELLS = ["product-lengthscale", "additive-lengthscale", "product-amplitude", "additive-amplitude"]
REFS = ["sobol", "dsp_map", "oracle_S"]
METHODS = CELLS + REFS
SEEDS = [f"{s:02d}" for s in range(10)]
T0, T1, N_ROWS = 20, 199, 180

# fixed categorical slot per method (dataviz reference palette, light mode), never cycled
COLOR = {
    "product-lengthscale": "#2a78d6",   # blue
    "additive-lengthscale": "#eb6834",  # orange
    "product-amplitude": "#1baf7a",     # aqua
    "additive-amplitude": "#eda100",    # yellow
    "sobol": "#e87ba4",                 # magenta
    "dsp_map": "#008300",               # green
    "oracle_S": "#4a3aa7",              # violet
}
LINESTYLE = {m: "-" for m in CELLS} | {"sobol": ":", "dsp_map": "--", "oracle_S": "-."}
ITER_COLS = ["t", "method", "family", "seed", "y", "f", "best_obs", "best_f", "regret", "acq_value",
             "fit_wall_s", "acq_wall_s", "fit_calls", "nuts_attempts", "status", "reason",
             "r_hat_max", "r_hat_median", "frac_r_hat_below_1_05", "n_eff_min", "divergences",
             "num_steps_mean", "r_hat_max_native", "n_eff_min_native", "r_hat_max_ell",
             "n_eff_min_ell", "r_hat_max_global", "n_eff_min_global", "y_mean", "y_std",
             "sobol_computed"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#b8b7b2",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#52514e",
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "lines.linewidth": 1.8,
})


# ----------------------------------------------------------------------------- loading
def load_runs(runs: Path):
    """iterations rows (no x_*), manifests, coords, per-iteration device; one pass over the tree."""
    iters, manifests, coords, log_devices, inventory = [], {}, [], [], []
    for fam in FAMILIES:
        for m in METHODS:
            for s in SEEDS:
                d = runs / fam / m / f"seed{s}"
                key = (fam, m, s)
                if not (d / "iterations.csv").exists():
                    inventory.append(dict(family=fam, method=m, seed=s, rows=0, exists=False))
                    continue
                df = pd.read_csv(d / "iterations.csv", usecols=ITER_COLS)
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
                                      config_hash=man.get("config_hash", "")[:8]))
                if (d / "coords.csv").exists():
                    c = pd.read_csv(d / "coords.csv")
                    c = c[c.t == c.t.max()].assign(family=fam, method=m, seed=s)
                    coords.append(c)
                    # last readout with a sobol_hat (the exact/QMC ones at t = 25k and 199)
                    cs = pd.read_csv(d / "coords.csv")
                    cs = cs[cs.sobol_hat.notna()]
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
    co["which"] = co["which"].fillna("last")
    dev = pd.concat(log_devices, ignore_index=True)
    return it, inv, manifests, co, dev


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
        r["best_f_is_cummax_f"] = bool(np.allclose(g.best_f.values, np.maximum.accumulate(g.f.values),
                                                    rtol=0, atol=1e-9)) or True  # see note below
        # best_f at t counts the initial design too, so cummax over rows alone is a lower bound:
        r["best_f_ge_cummax_f"] = bool((g.best_f.values + 1e-9 >= np.maximum.accumulate(g.f.values)).all())
        r["regret_eq_fstar_minus_bestf"] = bool(np.allclose(g.regret.values, f_star - g.best_f.values,
                                                             rtol=0, atol=1e-9))
        r["regret_nonnegative"] = bool((g.regret.values >= -1e-9).all())
        dv = dev[(dev.family == fam) & (dev.method == m) & (dev.seed == s)]
        r["log_resumes_eq_manifest"] = bool((dv.log_resumes == dv.man_resumes).all())
        r["log_t_set_eq_csv"] = set(dv.t) == set(g.t)
        r["f_star"] = f_star
        r["y_mean_t20"] = float(g.loc[g.t == T0, "y_mean"].iloc[0]) if (g.t == T0).any() else np.nan
        r["y_std_t20"] = float(g.loc[g.t == T0, "y_std"].iloc[0]) if (g.t == T0).any() else np.nan
        r["S"] = tuple(man["objective"]["S"])
        rows.append(r)
    ck = pd.DataFrame(rows).drop(columns=["best_f_is_cummax_f"])
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


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx]); adj[idx] = min(1.0, running)
    return adj


def paired_tests(final):
    """final: family, method, seed, regret at t=199 (complete runs). Wilcoxon signed-rank over seeds."""
    rows = []
    wide = final.pivot_table(index=["family", "seed"], columns="method", values="regret")
    pairs = [(c, r) for c in CELLS for r in REFS] + [(a, b) for i, a in enumerate(CELLS) for b in CELLS[i + 1:]]
    for fam in FAMILIES:
        w = wide.loc[fam] if fam in wide.index.get_level_values(0) else None
        if w is None:
            continue
        for a, b in pairs:
            if a not in w or b not in w:
                continue
            d = (w[a] - w[b]).dropna()
            n = len(d)
            if n < 2:
                p = np.nan; stat = np.nan
            else:
                try:
                    res = stats.wilcoxon(d.values, alternative="two-sided")
                    stat, p = float(res.statistic), float(res.pvalue)
                except ValueError:  # all differences zero
                    stat, p = np.nan, 1.0
            rows.append(dict(family=fam, cell=a, reference=b, kind="cell_vs_ref" if b in REFS else "cell_vs_cell",
                             n_seeds=n, median_diff=float(d.median()), mean_diff=float(d.mean()),
                             n_cell_better=int((d < 0).sum()), n_ref_better=int((d > 0).sum()),
                             wilcoxon_stat=stat, p_value=p))
    df = pd.DataFrame(rows)
    for kind in ["cell_vs_ref", "cell_vs_cell"]:
        for fam in FAMILIES:
            sel = (df.kind == kind) & (df.family == fam) & df.p_value.notna()
            if sel.any():
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


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--asof", default="")
    a = ap.parse_args()
    out = a.out
    asof = a.asof or datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    (out / "figures").mkdir(exist_ok=True); (out / "tables").mkdir(exist_ok=True)
    notes = []  # anything that looks wrong

    it, inv, manifests, co, dev = load_runs(a.runs)
    it = it.merge(dev[["family", "method", "seed", "t", "device"]], on=["family", "method", "seed", "t"], how="left")
    it["is_exception"] = it.reason.str.startswith("exception:")
    it["gate_excluded"] = (it.status == "excluded") & ~it.is_exception
    it["is_cell"] = it.method.isin(CELLS)

    # 1. inventory --------------------------------------------------------------------------
    inv["complete"] = inv["complete"].fillna(False).astype(bool)
    complete_keys = set(map(tuple, inv.loc[inv.complete, ["family", "method", "seed"]].values))
    it["complete"] = [k in complete_keys for k in zip(it.family, it.method, it.seed)]
    save_table(inv.sort_values(["family", "method", "seed"]), out, "inventory")
    incomplete = inv[~inv.complete].sort_values(["family", "method", "seed"])
    inv_summary = inv.groupby("method").agg(runs=("rows", "size"), complete=("complete", "sum"),
                                            rows=("rows", "sum")).reindex(METHODS).reset_index()
    save_table(inv_summary, out, "inventory_summary")

    # 2. sanity ------------------------------------------------------------------------------
    ck, cross = sanity_checks(it, inv, manifests, dev)
    save_table(ck.assign(S=ck.S.astype(str)), out, "sanity_per_run")
    save_table(cross, out, "sanity_cross_method")
    check_cols = ["rows_180", "no_dup_t", "t_contiguous", "regret_nonincreasing", "best_f_nondecreasing",
                  "best_f_ge_cummax_f", "regret_eq_fstar_minus_bestf", "regret_nonnegative",
                  "log_resumes_eq_manifest", "log_t_set_eq_csv"]
    sanity_summary = pd.DataFrame([dict(check=c, n_runs=len(ck), n_pass=int(ck[c].sum()), n_fail=int((~ck[c]).sum()),
                                        failing_runs="; ".join(f"{r.family}/{r.method}/seed{r.seed}" for r in ck[~ck[c]].itertuples()) or "")
                                   for c in check_cols] +
                                  [dict(check=c, n_runs=len(cross), n_pass=int(cross[c].sum()), n_fail=int((~cross[c]).sum()),
                                        failing_runs="; ".join(f"{r.family}/seed{r.seed}" for r in cross[~cross[c]].itertuples()) or "")
                                   for c in ["f_star_identical", "S_identical", "y_mean_t20_identical", "y_std_t20_identical"]])
    save_table(sanity_summary, out, "sanity_summary")
    for r in sanity_summary.itertuples():
        if r.n_fail:
            expected = r.check == "rows_180" and set(r.failing_runs.split("; ")) == set(
                f"{x.family}/{x.method}/seed{x.seed}" for x in incomplete.itertuples())
            if not expected:
                notes.append(f"Sanity check `{r.check}` FAILED for {r.n_fail} run(s): {r.failing_runs}")

    # 3. regret curves + final regret ----------------------------------------------------------
    itc = it[it.complete]
    curves = itc.groupby(["family", "method", "t"]).regret.agg(
        median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75), n="size").reset_index()
    curves.to_csv(out / "tables" / "regret_curves_median_iqr.csv", index=False)
    floor = max(1e-6, float(itc.regret[itc.regret > 0].min()) / 2) if (itc.regret > 0).any() else 1e-6
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, fam in zip(axes.ravel(), FAMILIES):
        for m in METHODS:
            c = curves[(curves.family == fam) & (curves.method == m)]
            if c.empty:
                continue
            ax.plot(c.t, np.maximum(c["median"], floor), color=COLOR[m], ls=LINESTYLE[m], label=f"{m} (n={int(c.n.iloc[-1])})")
            ax.fill_between(c.t, np.maximum(c.q1, floor), np.maximum(c.q3, floor), color=COLOR[m], alpha=0.12, lw=0)
        ax.set_yscale("log"); ax.set_title(fam); ax.set_xlabel("iteration t"); ax.set_ylabel("regret = f* - best_f")
    hd, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Median regret across seeds (band = IQR), complete runs only; log y, floored at {floor:.1e}", y=1.0)
    fig.tight_layout(); fig.savefig(out / "figures" / "regret_curves.png", bbox_inches="tight"); plt.close(fig)

    final = itc[itc.t == T1][["family", "method", "seed", "regret", "best_f"]].copy()
    final.to_csv(out / "tables" / "final_regret_per_run.csv", index=False)
    rows = []
    for fam in FAMILIES:
        for m in METHODS:
            x = final[(final.family == fam) & (final.method == m)].regret.values
            if len(x) == 0:
                continue
            rows.append(dict(family=fam, method=m, n=len(x), median=np.median(x), q1=np.percentile(x, 25),
                             q3=np.percentile(x, 75), mean=x.mean(), min=x.min(), max=x.max(),
                             median_iqr=med_iqr(x)))
    fr = pd.DataFrame(rows)
    fr["rank_in_family"] = fr.groupby("family")["median"].rank(method="min").astype(int)
    fr = fr.sort_values(["family", "rank_in_family"])
    save_table(fr, out, "final_regret_t199")
    # per-seed ranks (a rank-based ranking that is robust to one wild seed)
    final["rank_within_seed"] = final.groupby(["family", "seed"]).regret.rank(method="average")
    mean_rank = final.groupby(["family", "method"]).rank_within_seed.mean().reset_index(name="mean_rank_within_seed")
    fr = fr.merge(mean_rank, on=["family", "method"])
    save_table(fr, out, "final_regret_t199")
    # figure: final regret per family, strip + median
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=False)
    for ax, fam in zip(axes, FAMILIES):
        for j, m in enumerate(METHODS):
            x = final[(final.family == fam) & (final.method == m)].regret.values
            if len(x) == 0:
                continue
            jit = (np.random.default_rng(0).uniform(-0.18, 0.18, len(x)))
            ax.scatter(j + jit, np.maximum(x, floor), s=14, color=COLOR[m], alpha=0.75, lw=0)
            ax.hlines(np.median(x), j - 0.3, j + 0.3, color="#0b0b0b", lw=1.6)
        ax.set_yscale("log"); ax.set_title(fam); ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(["PL", "AL", "PA", "AA", "sobol", "dsp_map", "oracle_S"], fontsize=7)
        ax.set_ylabel("final regret (t=199)")
    fig.suptitle("Final regret per seed (dots) and median (bar); complete runs only, log y. PL/AL = product/additive lengthscale, PA/AA = product/additive amplitude", y=1.02)
    fig.tight_layout(); fig.savefig(out / "figures" / "final_regret_strip.png", bbox_inches="tight"); plt.close(fig)

    # 4. paired tests -------------------------------------------------------------------------
    pt = paired_tests(final)
    save_table(pt, out, "paired_wilcoxon_final_regret")

    # 5. identification ------------------------------------------------------------------------
    idf = identification(co, manifests, it)
    idf["complete"] = [k in complete_keys for k in zip(idf.family, idf.method, idf.seed)]
    save_table(idf, out, "identification_per_run")
    idc = idf[idf.complete]
    metrics = ["ap_native", "recall_at_S_native", "ap_sobol", "recall_at_S_sobol", "precision", "recall", "f1", "n_pred_active"]
    by_cf = idc.groupby(["method", "family"])[metrics].agg(["median", "mean"])
    by_cf.columns = [f"{a}_{b}" for a, b in by_cf.columns]
    by_cf = by_cf.reset_index()
    by_cf["n_runs"] = idc.groupby(["method", "family"]).size().values
    by_cf["method_order"] = by_cf.method.map({m: i for i, m in enumerate(METHODS)})
    by_cf = by_cf.sort_values(["method_order", "family"]).drop(columns="method_order")
    save_table(by_cf, out, "identification_by_cell_family")
    by_c = idc.groupby("method")[metrics].agg(["median", "mean"])
    by_c.columns = [f"{a}_{b}" for a, b in by_c.columns]
    by_c = by_c.reindex([m for m in METHODS if m in by_c.index]).reset_index()
    by_c["n_runs"] = idc.groupby("method").size().reindex(by_c.method).values
    save_table(by_c, out, "identification_by_cell")
    # per-family ranking of the 4 cells on mean AP(native) and mean F1
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
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, met, ttl in zip(axes, ["ap_native", "ap_sobol", "f1"],
                            ["average precision, score = native_median", "average precision, score = sobol_hat",
                             "F1 at p_active > 0.5"]):
        for fi, fam in enumerate(FAMILIES):
            for j, m in enumerate(CELLS + ["dsp_map"]):
                x = idc[(idc.family == fam) & (idc.method == m)][met].dropna().values
                if len(x) == 0:
                    continue
                pos = fi * 6 + j
                jit = np.random.default_rng(1).uniform(-0.2, 0.2, len(x))
                ax.scatter(pos + jit, x, s=12, color=COLOR[m], alpha=0.7, lw=0)
                ax.hlines(np.median(x), pos - 0.35, pos + 0.35, color="#0b0b0b", lw=1.5)
        ax.set_xticks([fi * 6 + 2 for fi in range(4)]); ax.set_xticklabels(FAMILIES, fontsize=8)
        ax.set_title(ttl); ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("score at last readout (1 = perfect)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[m], label=m) for m in CELLS + ["dsp_map"]]
    axes[2].legend(handles=handles, loc="lower right", frameon=False, fontsize=7)
    fig.suptitle("Active-set identification at the last readout (t=199), per seed; bar = median", y=1.02)
    fig.tight_layout(); fig.savefig(out / "figures" / "identification.png", bbox_inches="tight"); plt.close(fig)

    # identification over time (AP native at each readout) for the cells: cheap and informative
    ap_t = []
    for fam in FAMILIES:
        for m in CELLS + ["dsp_map"]:
            for s in SEEDS:
                if (fam, m, s) not in complete_keys:
                    continue
                p = a.runs / fam / m / f"seed{s}" / "coords.csv"
                c = pd.read_csv(p, usecols=["t", "i", "native_median", "p_active"])
                truth = np.zeros(100, bool); truth[list(manifests[(fam, m, s)]["objective"]["S"])] = True
                for t, g in c.groupby("t"):
                    if t % 25 == 0 or t == T1 or t == T0:
                        g = g.sort_values("i")
                        ap_t.append(dict(family=fam, method=m, seed=s, t=t,
                                         ap_native=average_precision(g.native_median.values, truth),
                                         f1=prf(g.p_active.values > 0.5, truth)[2]))
    apt = pd.DataFrame(ap_t)
    apt_med = apt.groupby(["family", "method", "t"])[["ap_native", "f1"]].median().reset_index()
    apt_med.to_csv(out / "tables" / "identification_vs_t_median.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    for ci, fam in enumerate(FAMILIES):
        for ri, met in enumerate(["ap_native", "f1"]):
            ax = axes[ri, ci]
            for m in CELLS + ["dsp_map"]:
                c = apt_med[(apt_med.family == fam) & (apt_med.method == m)]
                if c.empty:
                    continue
                ax.plot(c.t, c[met], color=COLOR[m], ls=LINESTYLE[m], marker="o", ms=3, label=m)
            ax.set_title(f"{fam}: median {'AP (native_median)' if met == 'ap_native' else 'F1 (p_active>0.5)'}")
            ax.set_ylim(-0.03, 1.03)
            if ri == 1:
                ax.set_xlabel("iteration t")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Identification quality over the run (readouts at t=20, 25k, 199), median across complete seeds", y=1.0)
    fig.tight_layout(); fig.savefig(out / "figures" / "identification_vs_t.png"); plt.close(fig)

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
    save_table(exr, out, "gate_exclusion_rate_vs_t")
    # figure: r_hat_max and n_eff_min distributions per cell
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, m in enumerate(CELLS + ["dsp_map"]):
        x = it[it.method == m]
        axes[0].hist(x.r_hat_max.dropna().clip(upper=3), bins=60, range=(1, 3), histtype="step", color=COLOR[m], label=m, lw=1.4)
        axes[1].hist(np.log10(x.n_eff_min.dropna().clip(lower=1)), bins=60, range=(0, 2.5), histtype="step", color=COLOR[m], label=m, lw=1.4)
    axes[0].axvline(1.1, color="#0b0b0b", lw=1, ls="--"); axes[0].set_xlabel("r_hat_max (clipped at 3); gate at 1.1"); axes[0].set_ylabel("iterations")
    axes[1].axvline(np.log10(16), color="#0b0b0b", lw=1, ls="--"); axes[1].set_xlabel("log10 n_eff_min; gate at 16")
    for m in CELLS:
        c = exr[["t_bin", m]].dropna()
        axes[2].plot(c.t_bin + 10, c[m], color=COLOR[m], marker="o", ms=3, label=m)
    axes[2].set_ylim(-0.03, 1.03); axes[2].set_xlabel("iteration t (bins of 20)"); axes[2].set_ylabel("gate-exclusion rate")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("NUTS diagnostics over all iterations (all runs)", y=1.02)
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
    tot = tot.merge(per_run.groupby("method").run_hours.median().reset_index(name="median_run_hours"), on="method")
    tot = tot.merge(per_run.groupby("method").run_hours.max().reset_index(name="max_run_hours"), on="method")
    save_table(tot, out, "cost_total_hours_by_method")
    # cost vs t per device (fit and acq), cells only: growth with n
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for m in CELLS:
        for dv, ls in [("NVIDIA H200", "-"), ("NVIDIA H100 80GB HBM3", "--"), ("NVIDIA A100-SXM4-80GB", ":")]:
            x = it[(it.method == m) & (it.device == dv)]
            if len(x) < 50:
                continue
            g = x.groupby((x.t // 10) * 10)[["fit_wall_s", "acq_wall_s"]].median()
            axes[0].plot(g.index + 5, g.fit_wall_s, color=COLOR[m], ls=ls, label=f"{m} / {dv.split()[1]}")
            axes[1].plot(g.index + 5, g.acq_wall_s, color=COLOR[m], ls=ls, label=f"{m} / {dv.split()[1]}")
    axes[0].set_title("median fit_wall_s vs t (bins of 10)"); axes[1].set_title("median acq_wall_s vs t (bins of 10)")
    for ax in axes:
        ax.set_xlabel("iteration t"); ax.set_ylabel("seconds"); ax.set_yscale("log")
    axes[1].legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle("Per-iteration cost by cell and GPU (solid H200, dashed H100, dotted A100); never compare across GPUs", y=1.02)
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
    commits = inv.commit.unique().tolist()
    if len(commits) > 1:
        notes.append(f"More than one git commit across runs: {commits}")
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
        pcomp = set(map(tuple, pinv.loc[pinv.complete.astype(bool), ["family", "method", "seed"]].values))
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
    R.append(f"# sagp study analysis\n\nAs of **{asof}** (data: `{a.runs}`; analysis dir `{out}`; script `analyze.py`).\n")
    R.append(f"Runs: **{n_complete}/{n_total} complete** (180 rows). Incomplete: " +
             (", ".join(f"{r.family}/{r.method}/seed{r.seed} ({r.rows} rows)" for r in incomplete.itertuples()) if len(incomplete) else "none") + ".\n")
    R.append("Headline numbers use complete runs only. Seeds are paired: one (family, seed) is one objective and one shared "
             "initial design across all 7 methods.\n")
    R.append((out / "top5.md").read_text() if (out / "top5.md").exists() else "<!-- TOP5 -->\n")
    h("What changed since the previous analysis"); R.append(prev_section + "\n")
    h("1. Inventory"); R.append(md_table(inv_summary) + "\n")
    if len(incomplete):
        R.append("\nIncomplete runs:\n\n" + md_table(incomplete[["family", "method", "seed", "rows", "t_max", "n_resumed", "device_created"]]) + "\n")
    S_size = {fam: sorted({len(manifests[k]["objective"]["S"]) for k in manifests if k[0] == fam}) for fam in FAMILIES}
    gammas = {fam: sorted({manifests[k]["objective"].get("gamma") for k in manifests if k[0] == fam}) for fam in FAMILIES}
    R.append(f"\nObjectives: D = 100, noise_sd = {manifests[(FAMILIES[0], METHODS[0], '00')]['objective'].get('noise_sd')}; |S| per family {S_size}; gamma per family {gammas}. "
             f"Distinct config_hash values: {n_hashes} (one per run; the hash covers the objective and the method). Git commits: {commits}.\n")
    R.append(f"\nDevices at creation (runs): {inv.device_created.value_counts().to_dict()}. Runs whose iterations span more "
             f"than one GPU type (resumed onto a different partition): {n_mixed}. Resumes per run: {inv.n_resumed.value_counts().sort_index().to_dict()} "
             "(most `resumed` entries are no-op resumes at t=200 recorded by the chain's later Slurm links).\n")
    h("2. Sanity checks"); R.append(md_table(sanity_summary[["check", "n_runs", "n_pass", "n_fail", "failing_runs"]]) + "\n")
    R.append("\n`best_f_ge_cummax_f` is a lower-bound check because best_f also covers the 20-point initial design, which is not a row. "
             "`regret_eq_fstar_minus_bestf` checks the CSV's regret against manifest.objective.f_star to 1e-9. "
             "`log_resumes_eq_manifest` and `log_t_set_eq_csv` verify that log.txt's resume lines match manifest.resumed and that every "
             "iteration in the CSV appears once in the log (device attribution relies on this).\n")
    h("3. Optimization performance")
    R.append("![regret curves](figures/regret_curves.png)\n\n![final regret](figures/final_regret_strip.png)\n")
    R.append("\nFinal regret at t=199, median [Q1, Q3] across complete seeds, ranked within family (1 = best):\n\n")
    R.append(md_table(fr[["family", "method", "n", "median_iqr", "rank_in_family", "mean_rank_within_seed"]]) + "\n")
    h("4. Paired comparisons (Wilcoxon signed-rank over seeds, final regret at t=199)")
    R.append("`median_diff` = median over seeds of (cell regret - reference regret); negative means the cell is better. "
             "Two-sided p-values from scipy.stats.wilcoxon (exact for n=10 without ties). **10 seeds is small**: the smallest "
             "attainable two-sided p is 0.002, and a single wild seed moves the median. Raw p-values are NOT corrected for "
             "multiple comparisons; `p_holm_within_family_kind` is a Holm correction over the 12 cell-vs-reference tests (or 6 "
             "cell-vs-cell tests) inside each family, shown for reference only.\n\n")
    R.append(md_table(pt[pt.kind == "cell_vs_ref"][["family", "cell", "reference", "n_seeds", "median_diff", "n_cell_better", "n_ref_better", "p_value", "p_holm_within_family_kind"]]) + "\n")
    R.append("\nCell-vs-cell (extra, same test):\n\n" + md_table(pt[pt.kind == "cell_vs_cell"][["family", "cell", "reference", "n_seeds", "median_diff", "n_cell_better", "n_ref_better", "p_value", "p_holm_within_family_kind"]]) + "\n")
    h("5. Identification of the active set (the study's real question)")
    R.append("Scored at the LAST readout of each run (coords.csv rows are written every iteration; sobol_hat is computed at t = 25, 50, ..., 175 and 199). "
             "`native_median` is the cell's own sparsity parameter (rho_i = 1/ell_i^2 for the lengthscale cells, a_sq_i for the amplitude cells; "
             "dsp_map: its MAP rho), so higher = more active in every cell. AP = average precision of ranking the 100 coordinates by the score; "
             "recall@|S| = fraction of the true S inside the top-|S| coordinates. Thresholded metrics use p_active > 0.5. "
             "oracle_S is included as a metric check (must be perfect); sobol has no model and no readout.\n\n")
    R.append("![identification](figures/identification.png)\n\n![identification vs t](figures/identification_vs_t.png)\n\n")
    R.append("Per cell (all families pooled), median and mean over complete runs:\n\n" +
             md_table(by_c[["method", "n_runs", "ap_native_median", "ap_native_mean", "recall_at_S_native_mean", "ap_sobol_mean", "recall_at_S_sobol_mean", "precision_mean", "recall_mean", "f1_mean", "n_pred_active_median"]]) + "\n")
    R.append("\nPer cell and family (mean over seeds):\n\n" +
             md_table(by_cf[["method", "family", "n_runs", "ap_native_mean", "recall_at_S_native_mean", "ap_sobol_mean", "recall_at_S_sobol_mean", "precision_mean", "recall_mean", "f1_mean", "n_pred_active_median"]]) + "\n")
    R.append("\nBest cell per family:\n\n" + md_table(pd.DataFrame(rank_rows)) + "\n")
    R.append(f"\nRuns whose t=199 readout came from a gate-excluded fit, per method: {exc_read}.\n")
    h("6. Diagnostics")
    R.append("![diagnostics](figures/diagnostics.png)\n\nPer method, all iterations of all runs (gate: split-R-hat <= 1.1, ESS >= 16 on every sampled site, <= 5 divergences):\n\n")
    R.append(md_table(diag_m[["method", "n_iter", "gate_excluded_rate", "exception_rows", "r_hat_max_q10", "r_hat_max_median", "r_hat_max_q90", "n_eff_min_q10", "n_eff_min_median", "n_eff_min_q90", "divergences_mean", "divergences_gt5_frac"]]) + "\n")
    R.append("\nSampler effort (num_steps_mean = leapfrog steps per NUTS iteration, max 63 at max_tree_depth = 6; fit_calls > 1 means a retry):\n\n" +
             md_table(diag_m[diag_m.method.isin(CELLS)][["method", "num_steps_mean_median", "frac_at_tree_cap", "fit_calls_mean", "nuts_attempts_mean"]]) + "\n")
    R.append("\nIdentification quality of readouts at t >= 100 split by whether that fit passed the gate (cells; the amplitude cells have no passing fits):\n\n" +
             md_table(ig_tab) + "\n")
    R.append("\nWhich site fails (medians of the per-site diagnostics):\n\n" +
             md_table(diag_m[diag_m.method.isin(CELLS)][["method", "r_hat_max_native_median", "r_hat_max_ell_median", "r_hat_max_global_median", "n_eff_min_native_median", "n_eff_min_ell_median", "n_eff_min_global_median", "frac_r_hat_below_1_05_median"]]) + "\n")
    R.append("\nGate reason breakdown (iterations):\n\n" + md_table(rc) + "\n")
    R.append("\nPer method and family:\n\n" + md_table(diag[["method", "family", "n_iter", "gate_excluded_rate", "exception_rows", "r_hat_max_median", "n_eff_min_median", "divergences_mean"]]) + "\n")
    R.append("\nGate-exclusion rate vs t (bins of 20), cells:\n\n" + md_table(exr) + "\n")
    h("7. Cost")
    R.append("Wall-times grouped by the GPU that ran the iteration (attributed from the resume segments in log.txt); the reference methods ran on CPU. "
             "**Do not compare across GPU types.** `hours_fit_plus_acq` sums the measured fit and acquisition time only: it excludes the "
             "per-iteration readout (coords.csv), checkpointing, JAX compilation at resume, and Slurm queue time, so it is a lower bound on allocated GPU-hours.\n\n")
    R.append(md_table(cost[["method", "device", "n_iter", "fit_wall_s_median", "acq_wall_s_median", "iter_wall_s_median", "fit_wall_s_q90", "acq_wall_s_q90", "hours_fit_plus_acq"]]) + "\n")
    R.append("\nTotals per method (all devices, all rows including the incomplete run):\n\n" + md_table(tot) + "\n")
    R.append("\n![cost vs t](figures/cost_vs_t.png)\n")
    h("8. Anomalies and things that look wrong")
    R.append(("\n".join(f"- {n}" for n in notes) if notes else "- Nothing failed beyond the expected incomplete run(s).") + "\n")
    R.append("\nExpected oddities (not flagged):\n\n" + ("\n".join(f"- {n}" for n in infos) if infos else "- none") +
             f"\n- dsp_map: {int((it.reason == 'fit_gpytorch_mll failed; Adam fallback').sum())} iterations used the Adam fallback fitter "
             "(status ok, reason 'fit_gpytorch_mll failed; Adam fallback'); these are neither gate exclusions nor exception rows.\n")
    h("Caveats")
    R.append("- status=\"excluded\" is a label: the draws were still used to choose the next point, so regret curves are valid for every run. "
             "It does mean the posterior readouts (native_median, p_active, sobol_hat) from those iterations are less trustworthy.\n"
             "- Rows whose reason starts with \"exception:\" queried a seeded random point instead; they are counted separately from gate exclusions.\n"
             "- GPU runs (A100/H100/H200) reproduce only in distribution, not bit-for-bit; wall-times are only comparable within one GPU type.\n"
             "- 10 seeds per (family, method): confidence intervals are wide and no multiplicity correction is applied to the headline p-values.\n"
             "- The regret plots floor the log axis at a small positive value; seeds that reached regret 0 sit on that floor.\n")
    (out / "REPORT.md").write_text("\n".join(R))
    # machine-readable summary for the top-5 writer
    summary = dict(asof=asof, n_complete=n_complete, n_total=n_total, notes=notes, infos=infos, n_mixed_device_runs=n_mixed,
                   exception_kinds=ex_kinds, readout_from_excluded=exc_read, tree_cap_frac=cap, S_size=S_size)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"done: {n_complete}/{n_total} complete; notes: {len(notes)}")
    for n in notes:
        print("NOTE:", n)


if __name__ == "__main__":
    main()
