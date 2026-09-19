"""Second batch of narrative-support computations (read-only on runs/ and latest/)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

RUNS = Path("/scratch/work/tranq8/saasbo/runs"); LATEST = Path("/scratch/work/tranq8/sagp_analysis/latest")
OUT = Path(sys.argv[1]); TAB = OUT / "tables"
RHO_EPS = json.load(open(TAB / "eps_constants.json"))["RHO_EPS"]
FAMILIES = ["aligned3", "aligned10", "decoupled", "interaction_g0.25"]
METHODS = ["product-lengthscale", "additive-lengthscale", "product-amplitude", "additive-amplitude", "sobol", "dsp_map", "oracle_S"]

# ---- (a) what k wins out of 10 paired seeds can bound: exact sign test and Clopper-Pearson 95% CI on the win probability
rows = []
for k in range(5, 11):
    p_sign = float(stats.binomtest(k, 10, 0.5, alternative="two-sided").pvalue)
    lo, hi = stats.binomtest(k, 10).proportion_ci(confidence_level=0.95, method="exact")
    rows.append(dict(wins_of_10=k, sign_test_two_sided_p=p_sign, win_prob_ci95_low=float(lo), win_prob_ci95_high=float(hi)))
sign = pd.DataFrame(rows); sign.to_csv(TAB / "ten_seed_bounds.csv", index=False)
print("k-of-10 bounds:\n", sign.to_string(index=False))

# ---- (b) identical final regret across methods: same queried point?
fr = pd.read_csv(LATEST / "tables" / "final_regret_per_run.csv", dtype={"seed": str})
dups = []
for (fam, sd), g in fr.groupby(["family", "seed"]):
    g = g.sort_values("regret")
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a, b = g.iloc[i], g.iloc[j]
            if abs(a.regret - b.regret) <= 1e-12 * max(1.0, abs(a.regret)):
                dups.append((fam, sd, a.method, b.method, a.regret))
def best_row(fam, m, sd):
    it = pd.read_csv(RUNS / fam / m / f"seed{sd}" / "iterations.csv")
    r = it.loc[it.f.idxmax()]
    x = r[[c for c in it.columns if c.startswith("x_")]].values.astype(float)
    return int(r.t), float(r.f), float(r.best_f), x
rows = []
for fam, sd, m1, m2, reg in dups:
    t1, f1, b1, x1 = best_row(fam, m1, sd); t2, f2, b2, x2 = best_row(fam, m2, sd)
    S = json.loads((RUNS / fam / m1 / f"seed{sd}" / "manifest.json").read_text())["objective"]["S"]
    rows.append(dict(family=fam, seed=sd, method_a=m1, method_b=m2, regret=reg, t_best_a=t1, t_best_b=t2, f_best_a=f1, f_best_b=f2,
                     best_f_final_a=b1, best_f_final_b=b2, max_abs_dx_all=float(np.max(np.abs(x1 - x2))),
                     max_abs_dx_on_S=float(np.max(np.abs(x1[S] - x2[S]))), n_coords_at_bound_a=int(((x1 <= 1e-9) | (x1 >= 1 - 1e-9)).sum()),
                     n_S_coords_at_bound_a=int(((x1[S] <= 1e-9) | (x1[S] >= 1 - 1e-9)).sum()), n_S_coords_at_bound_b=int(((x2[S] <= 1e-9) | (x2[S] >= 1 - 1e-9)).sum())))
dd = pd.DataFrame(rows); dd.to_csv(TAB / "identical_regret_pairs.csv", index=False)
print("\nidentical final regret pairs:\n", dd.to_string(index=False) if len(dd) else "none")
# how often is the incumbent on the boundary in the active coordinates, per method and family (the best-f row of each run)
rows = []
for fam in FAMILIES:
    for m in METHODS:
        for s in range(10):
            sd = f"{s:02d}"
            t, f, b, x = best_row(fam, m, sd)
            S = json.loads((RUNS / fam / m / f"seed{sd}" / "manifest.json").read_text())["objective"]["S"]
            xs = x[S]
            rows.append(dict(family=fam, method=m, seed=sd, t_best=t, n_S=len(S), n_S_at_bound=int(((xs <= 1e-9) | (xs >= 1 - 1e-9)).sum()),
                             n_all_at_bound=int(((x <= 1e-9) | (x >= 1 - 1e-9)).sum())))
bd = pd.DataFrame(rows); bd.to_csv(TAB / "incumbent_boundary_per_run.csv", index=False)
bs = bd.groupby(["family", "method"]).agg(mean_frac_S_at_bound=("n_S_at_bound", lambda x: float(x.sum()) / bd.loc[x.index, "n_S"].sum()),
                                          mean_n_all_at_bound=("n_all_at_bound", "mean"), median_t_best=("t_best", "median")).reset_index()
bs.to_csv(TAB / "incumbent_boundary_summary.csv", index=False)
print("\nincumbent on the box boundary (fraction of active coordinates at 0 or 1, mean over seeds):\n", bs.pivot(index="family", columns="method", values="mean_frac_S_at_bound").round(2).to_string())

# ---- (c) the retained draws in the collapsed readouts (aligned3 / additive-lengthscale)
rows = []
for sd, ts in {"08": [60, 75, 100, 125, 150, 160, 175, 180, 199], "02": [100, 125, 150, 160, 170, 175, 185, 190, 199], "00": [100, 150, 199]}.items():
    d = RUNS / "aligned3" / "additive-lengthscale" / f"seed{sd}"
    S = json.loads((d / "manifest.json").read_text())["objective"]["S"]
    off = np.ones(100, bool); off[S] = False
    for t in ts:
        z = np.load(d / "samples" / f"t{t:03d}.npz", allow_pickle=True)
        rho = z["kernel_inv_length_sq"]  # (16, 100)
        n_active_per_draw = (rho > RHO_EPS).sum(axis=1)
        rows.append(dict(seed=sd, t=t, status=str(z["status"]), n_draws=rho.shape[0], draws_with_ge50_active=int((n_active_per_draw >= 50).sum()),
                         n_active_min=int(n_active_per_draw.min()), n_active_max=int(n_active_per_draw.max()),
                         outputscale_med=float(np.median(z["outputscale"])), outputscale_min=float(z["outputscale"].min()), outputscale_max=float(z["outputscale"].max()),
                         noise_med=float(np.median(z["noise"])), noise_min=float(z["noise"].min()), noise_max=float(z["noise"].max()),
                         tausq_med=float(np.median(z["kernel_tausq"])), tausq_min=float(z["kernel_tausq"].min()), tausq_max=float(z["kernel_tausq"].max()),
                         rho_S_med=float(np.median(rho[:, S])), rho_off_med=float(np.median(rho[:, off])), rho_off_max=float(rho[:, off].max()),
                         ell_S_med=float(np.median(rho[:, S] ** -0.5)), ell_off_med=float(np.median(rho[:, off] ** -0.5)), mean_med=float(np.median(z["mean"]))))
dr = pd.DataFrame(rows); dr.to_csv(TAB / "collapse_draws_aligned3_AL.csv", index=False)
print("\nretained draws, aligned3/additive-lengthscale:\n", dr.to_string(index=False))

# ---- (d) off-S native scores per (method, family)
offs = pd.read_csv(TAB / "native_off_S_per_run.csv")
os_ = offs.groupby(["method", "family"]).agg(native_off_max_median=("native_off_max", "median"), native_off_max_max=("native_off_max", "max"),
                                             native_off_median_median=("native_off_median", "median"), runs_with_false_positive=("n_off_p_gt_0_5", lambda x: int((x > 0).sum())),
                                             false_positives_total=("n_off_p_gt_0_5", "sum")).reset_index()
os_.to_csv(TAB / "native_off_S_summary.csv", index=False)
print("\noff-S native score summary:\n", os_.to_string(index=False))

# ---- (e) near-duplicate design points (RAASP): fraction of BO points at t>=100 within max-norm 1e-2 of an earlier point
rows = []
for fam in FAMILIES:
    for m in ["product-lengthscale", "additive-lengthscale", "product-amplitude", "additive-amplitude", "oracle_S", "dsp_map"]:
        for s in range(10):
            sd = f"{s:02d}"
            it = pd.read_csv(RUNS / fam / m / f"seed{sd}" / "iterations.csv")
            X = it[[c for c in it.columns if c.startswith("x_")]].values.astype(float); t = it.t.values
            late = np.where(t >= 100)[0]; n_dup = 0
            for i in late:
                if np.max(np.abs(X[:i] - X[i]), axis=1).min() < 1e-2:
                    n_dup += 1
            rows.append(dict(family=fam, method=m, seed=sd, n_late=len(late), frac_late_near_duplicate=n_dup / len(late)))
nd = pd.DataFrame(rows); nd.to_csv(TAB / "near_duplicate_queries_per_run.csv", index=False)
nds = nd.groupby(["family", "method"]).frac_late_near_duplicate.mean().unstack().round(2)
nds.to_csv(TAB / "near_duplicate_queries_summary.csv")
print("\nfraction of queries at t>=100 within 1e-2 (max-norm, all 100 coords) of an earlier query, mean over seeds:\n", nds.to_string())
print("\nDONE2")
