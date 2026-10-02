# sagp study analysis: half-Cauchy and R2-D2 prior families

As of **2026-10-02 14:01 EEST** (data: `runs`, `runs_r2d2`; analysis dir `sagp_analysis/r2d2/2026-09-29-0245`; script `analyze.py`; families aligned10, decoupled; seeds 00, 01, 02, 03, 04).

Runs: **110/110 complete** (180 rows). Incomplete or missing: none.

Headline numbers use complete runs only. Seeds are paired: one (family, seed) is one objective and one shared initial design across all 11 methods. "Family" alone means the objective family; the half-Cauchy (HC) and R2-D2 priors are the two prior families.

<!-- TOP5 -->


## G3 (the pilot gate)

**G3: GO to stage 4.** At least one criterion fires; the full replicate (Task 15) may be launched.


Delta = y(R2-D2) - y(HC twin) at t = 199, y = log10(max(r, 1e-8)), over 40 pairs in 10 (family, seed) units. GO if any of: (a) |median Delta| >= 0.3 pooled or within any cell; (b) the pooled (family, seed)-cluster bootstrap 95 % interval of the mean Delta (10,000 resamples) excludes 0; (c) any cell's Wilcoxon signed-rank over its pairs, Holm-adjusted over the 4 cells, below 0.05; (d) |median dS| or |median dP| >= 0.3 over the units. Criteria firing: (a) True, (b) False, (c) True, (d) False. For (b): a percentile bootstrap over 10 clusters tends to under-cover, and the prereg does not fix the interval type.


| criterion | statistic | n | value | rule | fires |
|---|---|---|---|---|---|
| (a) | median Delta, pooled | 40 | 0.138 | abs(value) >= 0.3 | False |
| (a) | median Delta, product-lengthscale_r2d2 | 10 | 0.0122 | abs(value) >= 0.3 | False |
| (a) | median Delta, additive-lengthscale_r2d2 | 10 | 0.138 | abs(value) >= 0.3 | False |
| (a) | median Delta, product-amplitude_r2d2 | 10 | 0.0406 | abs(value) >= 0.3 | False |
| (a) | median Delta, additive-amplitude_r2d2 | 10 | 0.41 | abs(value) >= 0.3 | True |
| (b) | mean Delta, pooled; cluster bootstrap 95 % [-0.0515, 0.518] | 40 | 0.251 | interval excludes 0 | False |
| (c) | Wilcoxon p, product-lengthscale_r2d2 (raw 0.625); Holm over 4 cells | 10 | 1 | value < 0.05 | False |
| (c) | Wilcoxon p, additive-lengthscale_r2d2 (raw 0.557); Holm over 4 cells | 10 | 1 | value < 0.05 | False |
| (c) | Wilcoxon p, product-amplitude_r2d2 (raw 0.625); Holm over 4 cells | 10 | 1 | value < 0.05 | False |
| (c) | Wilcoxon p, additive-amplitude_r2d2 (raw 0.00391); Holm over 4 cells | 10 | 0.0156 | value < 0.05 | True |
| (d) | median dS over units | 10 | 0.145 | abs(value) >= 0.3 | False |
| (d) | median dP over units | 10 | 0.00173 | abs(value) >= 0.3 | False |


## What changed since the previous analysis

No previous analysis directory found: this is the first run, nothing to diff against.


## 1. Inventory

| method | runs | complete | rows |
|---|---|---|---|
| product-lengthscale | 10 | 10 | 1800 |
| additive-lengthscale | 10 | 10 | 1800 |
| product-amplitude | 10 | 10 | 1800 |
| additive-amplitude | 10 | 10 | 1800 |
| product-lengthscale_r2d2 | 10 | 10 | 1800 |
| additive-lengthscale_r2d2 | 10 | 10 | 1800 |
| product-amplitude_r2d2 | 10 | 10 | 1800 |
| additive-amplitude_r2d2 | 10 | 10 | 1800 |
| sobol | 10 | 10 | 1800 |
| dsp_map | 10 | 10 | 1800 |
| oracle_S | 10 | 10 | 1800 |


Objectives: D = 100, noise_sd = 0.1; |S| per family {'aligned10': [10], 'decoupled': [8]}; gamma per family {'aligned10': [0.0], 'decoupled': [0.0]}. Distinct config_hash values: 110 (one per run; the hash covers the objective and the method). Git commits at creation by prior family: {'HC': ['a51a4b9'], 'R2D2': ['97eda92', 'c4d9c38'], 'reference': ['a51a4b9']}.


Devices at creation (runs): {'NVIDIA H200': 62, 'cpu': 30, 'NVIDIA H100 80GB HBM3': 9, 'NVIDIA A100-SXM4-80GB': 9}. Runs whose iterations span more than one GPU type (resumed onto a different partition): 0. Resumes per run: {0: 18, 1: 30, 2: 10, 3: 22, 4: 10, 6: 20} (most `resumed` entries are no-op resumes at t=200 recorded by the chain's later Slurm links).


## 2. Sanity checks

| check | n_runs | n_pass | n_fail | failing_runs |
|---|---|---|---|---|
| rows_180 | 110 | 110 | 0 |  |
| no_dup_t | 110 | 110 | 0 |  |
| t_contiguous | 110 | 110 | 0 |  |
| regret_nonincreasing | 110 | 110 | 0 |  |
| best_f_nondecreasing | 110 | 110 | 0 |  |
| best_f_ge_cummax_f | 110 | 110 | 0 |  |
| regret_eq_fstar_minus_bestf | 110 | 110 | 0 |  |
| regret_nonnegative | 110 | 110 | 0 |  |
| log_resumes_eq_manifest | 110 | 110 | 0 |  |
| log_t_set_eq_csv | 110 | 110 | 0 |  |
| manifest_method_eq_dir | 110 | 110 | 0 |  |
| f_star_identical | 10 | 10 | 0 |  |
| S_identical | 10 | 10 | 0 |  |
| y_mean_t20_identical | 10 | 10 | 0 |  |
| y_std_t20_identical | 10 | 10 | 0 |  |
| twin_paired | 40 | 40 | 0 |  |
| prior_code_ok | 40 | 40 | 0 |  |


`best_f_ge_cummax_f` is a lower-bound check because best_f also covers the 20-point initial design, which is not a row. `regret_eq_fstar_minus_bestf` checks the CSV's regret against manifest.objective.f_star to 1e-9. `log_resumes_eq_manifest` and `log_t_set_eq_csv` verify that log.txt's resume lines match manifest.resumed and that every iteration in the CSV appears once in the log (device attribution relies on this). `manifest_method_eq_dir`: the manifest's method is the one its directory names.


`twin_paired` (D10): each R2-D2 run's row t = 20 `y_mean` and `y_std` are bit-identical to its half-Cauchy twin's, and `f_star` and S are equal (`tables/sanity_twin_pairing.csv`). `prior_code_ok` (Review Focus 3): every R2-D2 run's creation commit and every resume commit resolve, in `/Users/quantr03/Library/CloudStorage/OneDrive-AaltoUniversity/2026 Summer Internship'/saasbo`, to one pair of blobs of sagp/gp.py and sagp/r2d2.py, the pair most R2-D2 runs share, and the creation checkout was clean (resume entries record no dirty flag) (`tables/sanity_prior_code.csv`).


Launch commit `97eda925b`: blob pair (gp.py/r2d2.py) `f3590452a23b3b65c4216c5dd399d33e0ddebbce/45ef19533534f7f4d2ac5d4c84cddebe45d86600`; `prior_code_ok` also requires the R2-D2 runs' modal pair to be this one.


Prior-code blob pairs (gp.py/r2d2.py) over the R2-D2 runs: `f3590452a2/45ef195335` x 40; commits seen: ['97eda925b', 'c4d9c38ac'].


## 3. Optimization performance

![regret curves](figures/regret_curves.png)

![final regret](figures/final_regret_strip.png)


Final regret at t=199, median [Q1, Q3] across complete seeds, ranked within family (1 = best); `median_y` is the median of y = log10(max(r, 1e-8)):


| family | method | n | median_iqr | median_y | rank_in_family | mean_rank_within_seed |
|---|---|---|---|---|---|---|
| aligned10 | additive-amplitude | 5 | 0.0214 [0.0197, 0.041] | -1.67 | 1 | 2 |
| aligned10 | additive-lengthscale | 5 | 0.0364 [0.0265, 0.0547] | -1.44 | 2 | 2.4 |
| aligned10 | additive-amplitude_r2d2 | 5 | 0.0511 [0.0397, 0.0759] | -1.29 | 3 | 4.6 |
| aligned10 | additive-lengthscale_r2d2 | 5 | 0.0554 [0.0298, 0.0628] | -1.26 | 4 | 4.4 |
| aligned10 | product-amplitude | 5 | 0.0694 [0.0149, 0.206] | -1.16 | 5 | 4.2 |
| aligned10 | product-amplitude_r2d2 | 5 | 0.106 [0.0758, 0.213] | -0.975 | 6 | 6.4 |
| aligned10 | product-lengthscale_r2d2 | 5 | 0.394 [0.357, 0.705] | -0.405 | 7 | 8.8 |
| aligned10 | product-lengthscale | 5 | 0.412 [0.0724, 0.556] | -0.385 | 8 | 7.2 |
| aligned10 | oracle_S | 5 | 0.422 [0.346, 0.437] | -0.375 | 9 | 7.2 |
| aligned10 | dsp_map | 5 | 0.531 [0.247, 0.616] | -0.275 | 10 | 7.8 |
| aligned10 | sobol | 5 | 2.3 [2.26, 2.52] | 0.361 | 11 | 11 |
| decoupled | additive-amplitude | 5 | 0.0848 [0.0121, 0.379] | -1.07 | 1 | 1.8 |
| decoupled | product-amplitude_r2d2 | 5 | 0.222 [0.147, 0.396] | -0.653 | 2 | 5 |
| decoupled | additive-lengthscale | 5 | 0.233 [0.111, 0.442] | -0.633 | 3 | 3.6 |
| decoupled | product-amplitude | 5 | 0.334 [0.282, 0.345] | -0.477 | 4 | 3.8 |
| decoupled | additive-amplitude_r2d2 | 5 | 0.419 [0.398, 0.628] | -0.378 | 5 | 6.6 |
| decoupled | product-lengthscale | 5 | 0.489 [0.472, 0.554] | -0.311 | 6 | 8 |
| decoupled | product-lengthscale_r2d2 | 5 | 0.541 [0.343, 1.19] | -0.267 | 7 | 6 |
| decoupled | additive-lengthscale_r2d2 | 5 | 0.582 [0.317, 0.667] | -0.235 | 8 | 6.6 |
| decoupled | dsp_map | 5 | 0.585 [0.34, 0.676] | -0.233 | 9 | 6 |
| decoupled | oracle_S | 5 | 0.762 [0.153, 0.81] | -0.118 | 10 | 7.6 |
| decoupled | sobol | 5 | 2.06 [1.83, 2.4] | 0.314 | 11 | 11 |


## 4. Paired comparisons within each prior family (Wilcoxon signed-rank over seeds, final regret at t=199)

`median_diff` = median over seeds of (cell regret - reference regret); negative means the cell is better. Two-sided p-values from scipy.stats.wilcoxon (exact for n <= 50 without ties). The smallest attainable two-sided p is 0.002 at 10 seeds and 0.0625 at 5. `p_holm_within_family_kind` is a Holm correction over the 12 cell-vs-reference tests (or 6 cell-vs-cell tests) of one prior family inside each objective family, as sagp2 did for the half-Cauchy cells. These tests are on raw regret, as sagp2's; the effects and twin pairs below are on y.


| family | prior_family | cell | reference | n_seeds | median_diff | n_cell_better | n_ref_better | p_value | p_holm_within_family_kind |
|---|---|---|---|---|---|---|---|---|---|
| aligned10 | HC | product-lengthscale | sobol | 5 | -2.27 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | HC | product-lengthscale | dsp_map | 5 | -0.119 | 3 | 2 | 0.438 | 0.875 |
| aligned10 | HC | product-lengthscale | oracle_S | 5 | 0.0661 | 2 | 3 | 1 | 1 |
| aligned10 | HC | additive-lengthscale | sobol | 5 | -2.27 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | HC | additive-lengthscale | dsp_map | 5 | -0.476 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | HC | additive-lengthscale | oracle_S | 5 | -0.291 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | HC | product-amplitude | sobol | 5 | -2.24 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | HC | product-amplitude | dsp_map | 5 | -0.231 | 4 | 1 | 0.125 | 0.75 |
| aligned10 | HC | product-amplitude | oracle_S | 5 | -0.367 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | HC | additive-amplitude | sobol | 5 | -2.28 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | HC | additive-amplitude | dsp_map | 5 | -0.509 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | HC | additive-amplitude | oracle_S | 5 | -0.396 | 5 | 0 | 0.0625 | 0.75 |
| decoupled | HC | product-lengthscale | sobol | 5 | -1.34 | 5 | 0 | 0.0625 | 0.75 |
| decoupled | HC | product-lengthscale | dsp_map | 5 | 0.214 | 2 | 3 | 0.312 | 0.75 |
| decoupled | HC | product-lengthscale | oracle_S | 5 | 0.264 | 2 | 3 | 0.438 | 0.75 |
| decoupled | HC | additive-lengthscale | sobol | 5 | -1.71 | 5 | 0 | 0.0625 | 0.75 |
| decoupled | HC | additive-lengthscale | dsp_map | 5 | -0.107 | 4 | 1 | 0.125 | 0.75 |
| decoupled | HC | additive-lengthscale | oracle_S | 5 | -0.187 | 5 | 0 | 0.0625 | 0.75 |
| decoupled | HC | product-amplitude | sobol | 5 | -1.93 | 5 | 0 | 0.0625 | 0.75 |
| decoupled | HC | product-amplitude | dsp_map | 5 | -0.239 | 4 | 1 | 0.188 | 0.75 |
| decoupled | HC | product-amplitude | oracle_S | 5 | -0.416 | 4 | 1 | 0.188 | 0.75 |
| decoupled | HC | additive-amplitude | sobol | 5 | -1.98 | 5 | 0 | 0.0625 | 0.75 |
| decoupled | HC | additive-amplitude | dsp_map | 5 | -0.328 | 4 | 1 | 0.125 | 0.75 |
| decoupled | HC | additive-amplitude | oracle_S | 5 | -0.383 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | R2D2 | product-lengthscale_r2d2 | sobol | 5 | -1.87 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | R2D2 | product-lengthscale_r2d2 | dsp_map | 5 | 0.0368 | 1 | 4 | 0.625 | 1 |
| aligned10 | R2D2 | product-lengthscale_r2d2 | oracle_S | 5 | 0.0481 | 2 | 3 | 0.625 | 1 |
| aligned10 | R2D2 | additive-lengthscale_r2d2 | sobol | 5 | -2.27 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | R2D2 | additive-lengthscale_r2d2 | dsp_map | 5 | -0.218 | 4 | 1 | 0.188 | 1 |
| aligned10 | R2D2 | additive-lengthscale_r2d2 | oracle_S | 5 | -0.359 | 4 | 1 | 0.438 | 1 |
| aligned10 | R2D2 | product-amplitude_r2d2 | sobol | 5 | -2.15 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | R2D2 | product-amplitude_r2d2 | dsp_map | 5 | -0.425 | 4 | 1 | 0.125 | 0.875 |
| aligned10 | R2D2 | product-amplitude_r2d2 | oracle_S | 5 | -0.24 | 3 | 2 | 0.625 | 1 |
| aligned10 | R2D2 | additive-amplitude_r2d2 | sobol | 5 | -2.24 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | R2D2 | additive-amplitude_r2d2 | dsp_map | 5 | -0.246 | 5 | 0 | 0.0625 | 0.75 |
| aligned10 | R2D2 | additive-amplitude_r2d2 | oracle_S | 5 | -0.322 | 3 | 2 | 0.812 | 1 |
| decoupled | R2D2 | product-lengthscale_r2d2 | sobol | 5 | -1.28 | 5 | 0 | 0.0625 | 0.75 |
| decoupled | R2D2 | product-lengthscale_r2d2 | dsp_map | 5 | -0.134 | 3 | 2 | 0.812 | 1 |
| decoupled | R2D2 | product-lengthscale_r2d2 | oracle_S | 5 | 0.00587 | 2 | 3 | 0.812 | 1 |
| decoupled | R2D2 | additive-lengthscale_r2d2 | sobol | 5 | -1.91 | 5 | 0 | 0.0625 | 0.75 |
| decoupled | R2D2 | additive-lengthscale_r2d2 | dsp_map | 5 | -0.00323 | 3 | 2 | 1 | 1 |
| decoupled | R2D2 | additive-lengthscale_r2d2 | oracle_S | 5 | -0.0945 | 4 | 1 | 0.438 | 1 |
| decoupled | R2D2 | product-amplitude_r2d2 | sobol | 5 | -1.91 | 5 | 0 | 0.0625 | 0.75 |
| decoupled | R2D2 | product-amplitude_r2d2 | dsp_map | 5 | 0.0565 | 2 | 3 | 0.812 | 1 |
| decoupled | R2D2 | product-amplitude_r2d2 | oracle_S | 5 | -0.11 | 4 | 1 | 0.188 | 1 |
| decoupled | R2D2 | additive-amplitude_r2d2 | sobol | 5 | -1.66 | 5 | 0 | 0.0625 | 0.75 |
| decoupled | R2D2 | additive-amplitude_r2d2 | dsp_map | 5 | 0.0433 | 2 | 3 | 0.812 | 1 |
| decoupled | R2D2 | additive-amplitude_r2d2 | oracle_S | 5 | -0.134 | 3 | 2 | 0.625 | 1 |


Cell-vs-cell (same test):

| family | prior_family | cell | reference | n_seeds | median_diff | n_cell_better | n_ref_better | p_value | p_holm_within_family_kind |
|---|---|---|---|---|---|---|---|---|---|
| aligned10 | HC | product-lengthscale | additive-lengthscale | 5 | 0.277 | 0 | 5 | 0.0625 | 0.375 |
| aligned10 | HC | product-lengthscale | product-amplitude | 5 | 0.112 | 1 | 4 | 0.312 | 1 |
| aligned10 | HC | product-lengthscale | additive-amplitude | 5 | 0.39 | 0 | 5 | 0.0625 | 0.375 |
| aligned10 | HC | additive-lengthscale | product-amplitude | 5 | -0.0329 | 4 | 1 | 0.625 | 1 |
| aligned10 | HC | additive-lengthscale | additive-amplitude | 5 | 0.0123 | 2 | 3 | 0.625 | 1 |
| aligned10 | HC | product-amplitude | additive-amplitude | 5 | 0.0283 | 2 | 3 | 0.312 | 1 |
| decoupled | HC | product-lengthscale | additive-lengthscale | 5 | 0.321 | 0 | 5 | 0.0625 | 0.375 |
| decoupled | HC | product-lengthscale | product-amplitude | 5 | 0.271 | 0 | 5 | 0.0625 | 0.375 |
| decoupled | HC | product-lengthscale | additive-amplitude | 5 | 0.479 | 0 | 5 | 0.0625 | 0.375 |
| decoupled | HC | additive-lengthscale | product-amplitude | 5 | -0.026 | 3 | 2 | 1 | 1 |
| decoupled | HC | additive-lengthscale | additive-amplitude | 5 | 0.101 | 0 | 5 | 0.0625 | 0.375 |
| decoupled | HC | product-amplitude | additive-amplitude | 5 | 0.0483 | 1 | 4 | 0.188 | 0.375 |
| aligned10 | R2D2 | product-lengthscale_r2d2 | additive-lengthscale_r2d2 | 5 | 0.327 | 1 | 4 | 0.188 | 0.75 |
| aligned10 | R2D2 | product-lengthscale_r2d2 | product-amplitude_r2d2 | 5 | 0.288 | 0 | 5 | 0.0625 | 0.375 |
| aligned10 | R2D2 | product-lengthscale_r2d2 | additive-amplitude_r2d2 | 5 | 0.306 | 0 | 5 | 0.0625 | 0.375 |
| aligned10 | R2D2 | additive-lengthscale_r2d2 | product-amplitude_r2d2 | 5 | -0.0767 | 4 | 1 | 0.438 | 1 |
| aligned10 | R2D2 | additive-lengthscale_r2d2 | additive-amplitude_r2d2 | 5 | 0.00515 | 2 | 3 | 1 | 1 |
| aligned10 | R2D2 | product-amplitude_r2d2 | additive-amplitude_r2d2 | 5 | 0.0361 | 2 | 3 | 0.812 | 1 |
| decoupled | R2D2 | product-lengthscale_r2d2 | additive-lengthscale_r2d2 | 5 | -0.111 | 3 | 2 | 0.812 | 1 |
| decoupled | R2D2 | product-lengthscale_r2d2 | product-amplitude_r2d2 | 5 | 0.319 | 2 | 3 | 0.312 | 1 |
| decoupled | R2D2 | product-lengthscale_r2d2 | additive-amplitude_r2d2 | 5 | 0.285 | 2 | 3 | 0.312 | 1 |
| decoupled | R2D2 | additive-lengthscale_r2d2 | product-amplitude_r2d2 | 5 | 0.0154 | 1 | 4 | 0.312 | 1 |
| decoupled | R2D2 | additive-lengthscale_r2d2 | additive-amplitude_r2d2 | 5 | -0.102 | 3 | 2 | 0.812 | 1 |
| decoupled | R2D2 | product-amplitude_r2d2 | additive-amplitude_r2d2 | 5 | -0.0338 | 4 | 1 | 0.188 | 1 |


## 5. Effects: structure S, parameterization P, twin pairs Delta_c, and their prior-family differences

Per (family, seed) on y = log10(max(r_199, 1e-8)): S = 1/2[(y_AA - y_PA) + (y_AL - y_PL)] (additive minus product), P = 1/2[(y_AA - y_AL) + (y_PA - y_PL)] (amplitude minus lengthscale), within each prior family; Delta_c = y_c(R2-D2) - y_c(HC twin); dS = S(R2D2) - S(HC), dP likewise. Negative = lower regret for the first-named side. Per family: two-sided Wilcoxon over seeds, Holm within family across the effects of one kind (main_HC: S_HC, P_HC; main_R2D2: S_R2D2, P_R2D2; interaction: dS, dP; twin: the four Delta_c); lo/hi are the seed bootstrap's percentile intervals (10,000 resamples). Pooled: the (family, seed)-cluster bootstrap of the mean.


Pooled over families:

| effect | kind | n_units | median | mean | lo90 | hi90 | lo95 | hi95 | pr2_reading |
|---|---|---|---|---|---|---|---|---|---|
| S_HC | main_HC | 10 | -0.501 | -0.514 | -0.723 | -0.301 | -0.76 | -0.265 |  |
| P_HC | main_HC | 10 | -0.367 | -0.417 | -0.668 | -0.187 | -0.726 | -0.151 |  |
| S_R2D2 | main_R2D2 | 10 | -0.0916 | -0.0994 | -0.411 | 0.255 | -0.46 | 0.337 |  |
| P_R2D2 | main_R2D2 | 10 | -0.181 | -0.294 | -0.629 | 0.00709 | -0.694 | 0.0511 |  |
| dS | interaction | 10 | 0.145 | 0.415 | 0.0712 | 0.763 | 0.0123 | 0.824 | inconclusive |
| dP | interaction | 10 | 0.00173 | 0.123 | -0.284 | 0.58 | -0.348 | 0.681 | inconclusive |
| Delta_product-lengthscale_r2d2 | twin | 10 | 0.0122 | 0.138 | -0.0595 | 0.359 | -0.0897 | 0.401 |  |
| Delta_additive-lengthscale_r2d2 | twin | 10 | 0.138 | 0.242 | -0.0913 | 0.651 | -0.137 | 0.735 |  |
| Delta_product-amplitude_r2d2 | twin | 10 | 0.0406 | -0.05 | -0.68 | 0.52 | -0.86 | 0.62 |  |
| Delta_additive-amplitude_r2d2 | twin | 10 | 0.41 | 0.676 | 0.358 | 1.02 | 0.308 | 1.09 |  |


Per family:

| family | effect | n | median | mean | lo90 | hi90 | p_value | p_holm_within_family_kind | pr2_reading |
|---|---|---|---|---|---|---|---|---|---|
| aligned10 | S_HC | 5 | -0.609 | -0.541 | -0.817 | -0.241 | 0.125 | 0.25 |  |
| aligned10 | P_HC | 5 | -0.273 | -0.377 | -0.851 | 0.027 | 0.438 | 0.438 |  |
| aligned10 | S_R2D2 | 5 | -0.613 | -0.511 | -0.763 | -0.252 | 0.125 | 0.25 |  |
| aligned10 | P_R2D2 | 5 | -0.121 | -0.185 | -0.563 | 0.167 | 0.625 | 0.625 |  |
| aligned10 | dS | 5 | 0.00626 | 0.0299 | -0.326 | 0.412 | 1 | 1 | inconclusive |
| aligned10 | dP | 5 | -0.137 | 0.192 | -0.354 | 1.03 | 0.625 | 1 | inconclusive |
| aligned10 | Delta_product-lengthscale_r2d2 | 5 | 0.117 | 0.314 | 0.029 | 0.633 | 0.312 | 0.938 |  |
| aligned10 | Delta_additive-lengthscale_r2d2 | 5 | 0.051 | 0.285 | -0.333 | 1.08 | 1 | 1 |  |
| aligned10 | Delta_product-amplitude_r2d2 | 5 | 0.0388 | 0.447 | -0.141 | 1.11 | 0.312 | 0.938 |  |
| aligned10 | Delta_additive-amplitude_r2d2 | 5 | 0.262 | 0.536 | 0.109 | 1.1 | 0.125 | 0.5 |  |
| decoupled | S_HC | 5 | -0.393 | -0.487 | -0.8 | -0.192 | 0.125 | 0.125 |  |
| decoupled | P_HC | 5 | -0.499 | -0.457 | -0.633 | -0.279 | 0.0625 | 0.125 |  |
| decoupled | S_R2D2 | 5 | 0.106 | 0.313 | -0.0355 | 0.854 | 0.438 | 0.625 |  |
| decoupled | P_R2D2 | 5 | -0.241 | -0.403 | -0.954 | 0.0513 | 0.312 | 0.625 |  |
| decoupled | dS | 5 | 0.979 | 0.8 | 0.346 | 1.21 | 0.0625 | 0.125 | prior family changes the effect |
| decoupled | dP | 5 | 0.193 | 0.0539 | -0.477 | 0.52 | 0.625 | 0.625 | inconclusive |
| decoupled | Delta_product-lengthscale_r2d2 | 5 | -0.122 | -0.0385 | -0.249 | 0.175 | 0.812 | 1 |  |
| decoupled | Delta_additive-lengthscale_r2d2 | 5 | 0.144 | 0.199 | -0.0122 | 0.447 | 0.438 | 1 |  |
| decoupled | Delta_product-amplitude_r2d2 | 5 | 0.0424 | -0.547 | -1.73 | 0.157 | 1 | 1 |  |
| decoupled | Delta_additive-amplitude_r2d2 | 5 | 0.672 | 0.815 | 0.416 | 1.25 | 0.0625 | 0.25 |  |


## 6. Preregistered predictions PR1-PR4

Scope: **not** the confirmatory analysis (that needs all 8 families x 10 seeds with all 320 R2-D2 runs complete; no interim analysis is run in stage 4). The readings below are descriptive.


**PR1** (wherever S(HC) or P(HC) is significant, Holm p < 0.05, the R2-D2 effect has the same sign; sign of the median over seeds): tested in 0 (family, effect); holds in 0, fails in 0.

| family | effect | n_HC | median_HC | p_holm_HC | significant_HC | n_R2D2 | median_R2D2 | holds |
|---|---|---|---|---|---|---|---|---|
| aligned10 | S | 5 | -0.609 | 0.25 | False | 5 | -0.613 | not tested |
| aligned10 | P | 5 | -0.273 | 0.438 | False | 5 | -0.121 | not tested |
| decoupled | S | 5 | -0.393 | 0.125 | False | 5 | 0.106 | not tested |
| decoupled | P | 5 | -0.499 | 0.125 | False | 5 | -0.241 | not tested |


**PR2** (|dS| and |dP| below 0.3; pooled 90 % cluster-bootstrap interval inside (-0.3, 0.3) = equivalent, entirely outside = prior family changes the effect, otherwise inconclusive): dS: mean 0.415, 90 % [0.0712, 0.763] -> inconclusive; dP: mean 0.123, 90 % [-0.284, 0.58] -> inconclusive. Per-family readings are in section 5.


**PR3** (Q6's null: |median Delta_c| < 0.3 for every cell and family): holds in 7 of 8 (cell, family); does not hold overall.

| family | effect | n | median | mean | lo90 | hi90 | p_value | p_holm_within_family_kind | holds |
|---|---|---|---|---|---|---|---|---|---|
| aligned10 | Delta_product-lengthscale_r2d2 | 5 | 0.117 | 0.314 | 0.029 | 0.633 | 0.312 | 0.938 | True |
| aligned10 | Delta_additive-lengthscale_r2d2 | 5 | 0.051 | 0.285 | -0.333 | 1.08 | 1 | 1 | True |
| aligned10 | Delta_product-amplitude_r2d2 | 5 | 0.0388 | 0.447 | -0.141 | 1.11 | 0.312 | 0.938 | True |
| aligned10 | Delta_additive-amplitude_r2d2 | 5 | 0.262 | 0.536 | 0.109 | 1.1 | 0.125 | 0.5 | True |
| decoupled | Delta_product-lengthscale_r2d2 | 5 | -0.122 | -0.0385 | -0.249 | 0.175 | 0.812 | 1 | True |
| decoupled | Delta_additive-lengthscale_r2d2 | 5 | 0.144 | 0.199 | -0.0122 | 0.447 | 0.438 | 1 | True |
| decoupled | Delta_product-amplitude_r2d2 | 5 | 0.0424 | -0.547 | -1.73 | 0.157 | 1 | 1 | True |
| decoupled | Delta_additive-amplitude_r2d2 | 5 | 0.672 | 0.815 | 0.416 | 1.25 | 0.0625 | 0.25 | False |


**PR4** (secondary: AP by native_median and by sobol_hat within 0.05 of the twin's; per the prereg's clarification, the (family, seed)-cluster bootstrap 90 % interval (10,000 resamples) of the median paired difference, R2-D2 minus twin, inside (-0.05, 0.05) = equivalent, entirely outside = differs, otherwise inconclusive; lo90/hi90 are that interval): equivalent 0, differs 3, inconclusive 5 of 8 (cell, metric).

| cell | twin | metric | n_pairs | mean_diff | median_diff | lo90 | hi90 | reading |
|---|---|---|---|---|---|---|---|---|
| product-lengthscale_r2d2 | product-lengthscale | ap_native | 10 | -0.17 | -0.131 | -0.237 | -0.0354 | inconclusive |
| product-lengthscale_r2d2 | product-lengthscale | ap_sobol | 10 | -0.226 | -0.199 | -0.327 | -0.0867 | differs |
| additive-lengthscale_r2d2 | additive-lengthscale | ap_native | 10 | -0.23 | -0.229 | -0.31 | -0.162 | differs |
| additive-lengthscale_r2d2 | additive-lengthscale | ap_sobol | 10 | -0.208 | -0.154 | -0.315 | -0.116 | differs |
| product-amplitude_r2d2 | product-amplitude | ap_native | 10 | -0.0895 | -0.0938 | -0.178 | -0.0406 | inconclusive |
| product-amplitude_r2d2 | product-amplitude | ap_sobol | 10 | -0.131 | -0.132 | -0.24 | -0.0413 | inconclusive |
| additive-amplitude_r2d2 | additive-amplitude | ap_native | 10 | -0.0229 | -0.00481 | -0.0579 | 0.0191 | inconclusive |
| additive-amplitude_r2d2 | additive-amplitude | ap_sobol | 10 | -0.0537 | -0.0433 | -0.118 | 0.00882 | inconclusive |


## 7. Identification of the active set

Scored at the LAST readout of each run (coords.csv rows are written every iteration; sobol_hat is computed at t = 25, 50, ..., 175 and 199). `native_median` is the cell's own sparsity parameter (rho_i = 1/ell_i^2 for the lengthscale cells, a_sq_i for the amplitude cells; dsp_map: its MAP rho), so higher = more active in every cell. AP = average precision of ranking the 100 coordinates by the score; recall@|S| = fraction of the true S inside the top-|S| coordinates. Thresholded metrics use p_active > 0.5. oracle_S is included as a metric check (must be perfect); sobol has no model and no readout.


![identification](figures/identification.png)

![identification vs t](figures/identification_vs_t.png)


By prior family (each cell under HC and R2-D2, pooled over families, complete runs):

| structure | parameterization | prior_family | n_runs | ap_native_mean | ap_native_median | ap_sobol_mean | ap_sobol_median | f1_mean | n_pred_active_median |
|---|---|---|---|---|---|---|---|---|---|
| additive | amplitude | HC | 10 | 0.898 | 0.892 | 0.913 | 0.921 | 0.637 | 5 |
| additive | amplitude | R2D2 | 10 | 0.875 | 0.895 | 0.859 | 0.89 | 0.384 | 0 |
| additive | lengthscale | HC | 10 | 0.973 | 1 | 0.982 | 0.993 | 0.903 | 7 |
| additive | lengthscale | R2D2 | 10 | 0.743 | 0.764 | 0.773 | 0.832 | 0.436 | 30 |
| product | amplitude | HC | 10 | 0.909 | 0.918 | 0.938 | 0.984 | 0.49 | 3 |
| product | amplitude | R2D2 | 10 | 0.819 | 0.84 | 0.806 | 0.832 | 0.364 | 0 |
| product | lengthscale | HC | 10 | 0.899 | 0.947 | 0.883 | 0.952 | 0.568 | 4 |
| product | lengthscale | R2D2 | 10 | 0.729 | 0.82 | 0.658 | 0.673 | 0.448 | 3.5 |


Per method (all families pooled), median and mean over complete runs:

| method | n_runs | ap_native_median | ap_native_mean | recall_at_S_native_mean | ap_sobol_mean | recall_at_S_sobol_mean | precision_mean | recall_mean | f1_mean | n_pred_active_median |
|---|---|---|---|---|---|---|---|---|---|---|
| product-lengthscale | 10 | 0.947 | 0.899 | 0.865 | 0.883 | 0.83 | 0.983 | 0.425 | 0.568 | 4 |
| additive-lengthscale | 10 | 1 | 0.973 | 0.943 | 0.982 | 0.94 | 1 | 0.832 | 0.903 | 7 |
| product-amplitude | 10 | 0.918 | 0.909 | 0.845 | 0.938 | 0.88 | 1 | 0.33 | 0.49 | 3 |
| additive-amplitude | 10 | 0.892 | 0.898 | 0.878 | 0.913 | 0.887 | 0.98 | 0.483 | 0.637 | 5 |
| product-lengthscale_r2d2 | 10 | 0.82 | 0.729 | 0.657 | 0.658 | 0.605 | 0.716 | 0.318 | 0.448 | 3.5 |
| additive-lengthscale_r2d2 | 10 | 0.764 | 0.743 | 0.682 | 0.773 | 0.693 | 0.324 | 0.948 | 0.436 | 30 |
| product-amplitude_r2d2 | 10 | 0.84 | 0.819 | 0.722 | 0.806 | 0.713 | 1 | 0.0475 | 0.364 | 0 |
| additive-amplitude_r2d2 | 10 | 0.895 | 0.875 | 0.78 | 0.859 | 0.782 | 1 | 0.05 | 0.384 | 0 |
| dsp_map | 10 | 0.568 | 0.539 | 0.483 | 0.415 | 0.438 | 0.327 | 0.667 | 0.423 | 15 |
| oracle_S | 10 | 1 | 1 | 1 | 0.956 | 0.953 | 1 | 0.48 | 0.592 | 3.5 |


Per method and family (mean over seeds):

| method | family | n_runs | ap_native_mean | recall_at_S_native_mean | ap_sobol_mean | recall_at_S_sobol_mean | precision_mean | recall_mean | f1_mean | n_pred_active_median |
|---|---|---|---|---|---|---|---|---|---|---|
| product-lengthscale | aligned10 | 5 | 0.914 | 0.88 | 0.913 | 0.86 | 0.967 | 0.4 | 0.542 | 5 |
| product-lengthscale | decoupled | 5 | 0.883 | 0.85 | 0.853 | 0.8 | 1 | 0.45 | 0.594 | 3 |
| additive-lengthscale | aligned10 | 5 | 0.98 | 0.96 | 0.989 | 0.98 | 1 | 0.94 | 0.967 | 10 |
| additive-lengthscale | decoupled | 5 | 0.967 | 0.925 | 0.974 | 0.9 | 1 | 0.725 | 0.84 | 6 |
| product-amplitude | aligned10 | 5 | 0.908 | 0.84 | 0.935 | 0.86 | 1 | 0.26 | 0.41 | 3 |
| product-amplitude | decoupled | 5 | 0.91 | 0.85 | 0.94 | 0.9 | 1 | 0.4 | 0.57 | 3 |
| additive-amplitude | aligned10 | 5 | 0.89 | 0.88 | 0.9 | 0.9 | 0.96 | 0.44 | 0.59 | 5 |
| additive-amplitude | decoupled | 5 | 0.906 | 0.875 | 0.926 | 0.875 | 1 | 0.525 | 0.683 | 4 |
| product-lengthscale_r2d2 | aligned10 | 5 | 0.876 | 0.84 | 0.756 | 0.76 | 0.927 | 0.36 | 0.563 | 5 |
| product-lengthscale_r2d2 | decoupled | 5 | 0.582 | 0.475 | 0.559 | 0.45 | 0.436 | 0.275 | 0.294 | 2 |
| additive-lengthscale_r2d2 | aligned10 | 5 | 0.836 | 0.74 | 0.834 | 0.76 | 0.442 | 0.92 | 0.561 | 26 |
| additive-lengthscale_r2d2 | decoupled | 5 | 0.65 | 0.625 | 0.713 | 0.625 | 0.206 | 0.975 | 0.31 | 64 |
| product-amplitude_r2d2 | aligned10 | 5 | 0.816 | 0.72 | 0.821 | 0.7 | 1 | 0.02 | 0.182 | 0 |
| product-amplitude_r2d2 | decoupled | 5 | 0.823 | 0.725 | 0.791 | 0.725 | 1 | 0.075 | 0.545 | 0 |
| additive-amplitude_r2d2 | aligned10 | 5 | 0.931 | 0.86 | 0.914 | 0.84 | nan | 0 | nan | 0 |
| additive-amplitude_r2d2 | decoupled | 5 | 0.819 | 0.7 | 0.804 | 0.725 | 1 | 0.1 | 0.384 | 0 |
| dsp_map | aligned10 | 5 | 0.469 | 0.44 | 0.39 | 0.4 | 0.353 | 0.56 | 0.423 | 14 |
| dsp_map | decoupled | 5 | 0.61 | 0.525 | 0.44 | 0.475 | 0.302 | 0.775 | 0.423 | 26 |
| oracle_S | aligned10 | 5 | 1 | 1 | 0.982 | 0.98 | 1 | 0.46 | 0.567 | 4 |
| oracle_S | decoupled | 5 | 1 | 1 | 0.931 | 0.925 | 1 | 0.5 | 0.616 | 3 |


Best cell per family:

| family | best_ap_native | best_ap_sobol | best_f1 | order_by_ap_native |
|---|---|---|---|---|
| aligned10 | additive-lengthscale | additive-lengthscale | additive-lengthscale | additive-lengthscale > additive-amplitude_r2d2 > product-lengthscale > product-amplitude > additive-amplitude > product-lengthscale_r2d2 > additive-lengthscale_r2d2 > product-amplitude_r2d2 |
| decoupled | additive-lengthscale | additive-lengthscale | additive-lengthscale | additive-lengthscale > product-amplitude > additive-amplitude > product-lengthscale > product-amplitude_r2d2 > additive-amplitude_r2d2 > additive-lengthscale_r2d2 > product-lengthscale_r2d2 |


Runs whose t=199 readout came from a gate-excluded fit, per method: {'additive-amplitude': 10, 'additive-amplitude_r2d2': 10, 'additive-lengthscale': 5, 'additive-lengthscale_r2d2': 0, 'dsp_map': 0, 'oracle_S': 0, 'product-amplitude': 10, 'product-amplitude_r2d2': 10, 'product-lengthscale': 4, 'product-lengthscale_r2d2': 0}.


## 8. R2 readouts from the npz draws (amplitude cells)

Per fit, medians over its 16 retained draws: `r2d2_r2` = S/(1 + S) with S = sum_i a_sq_i, what the R2-D2 prior's Beta is on (about R^2/(1 + R^2) on a standardized target, so at most about 1/2); `first_order_r2` = S/(S + sigma^2) for the additive cells and S/(prod_i(1 + a_sq_i) - 1 + sigma^2) for the product cells (ruling R20), with sigma^2 the raw noise draw clamped at 0.0001 (the manifest's fixed_noise where the noise is fixed). Read at t = 20, 25, 50, 75, 100, 125, 150, 175, 199 (`tables/r2_readouts_vs_t_median.csv`); at t = 199, median [Q1, Q3] over seeds:

| method | family | n | r2d2_r2 | first_order_r2 |
|---|---|---|---|---|
| product-amplitude | aligned10 | 5 | 0.162 [0.153, 0.179] | 0.913 [0.904, 0.915] |
| product-amplitude | decoupled | 5 | 0.203 [0.194, 0.204] | 0.897 [0.889, 0.903] |
| additive-amplitude | aligned10 | 5 | 0.29 [0.264, 0.291] | 0.992 [0.988, 0.992] |
| additive-amplitude | decoupled | 5 | 0.293 [0.262, 0.325] | 0.989 [0.987, 0.99] |
| product-amplitude_r2d2 | aligned10 | 5 | 0.121 [0.111, 0.126] | 0.916 [0.914, 0.918] |
| product-amplitude_r2d2 | decoupled | 5 | 0.138 [0.128, 0.176] | 0.9 [0.883, 0.904] |
| additive-amplitude_r2d2 | aligned10 | 5 | 0.137 [0.135, 0.156] | 0.98 [0.972, 0.981] |
| additive-amplitude_r2d2 | decoupled | 5 | 0.158 [0.122, 0.16] | 0.973 [0.966, 0.979] |


## 9. Diagnostics

![diagnostics](figures/diagnostics.png)

Per method, all iterations of all runs (gate: split-R-hat <= 1.1, ESS >= 16 on every sampled site, <= 5 divergences):


| method | n_iter | gate_excluded_rate | exception_rows | r_hat_max_q10 | r_hat_max_median | r_hat_max_q90 | n_eff_min_q10 | n_eff_min_median | n_eff_min_q90 | divergences_mean | divergences_gt5_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| product-lengthscale | 1800 | 0.674 | 0 | 1.05 | 1.13 | 1.38 | 6.51 | 16.6 | 41.5 | 0.00389 | 0 |
| additive-lengthscale | 1800 | 0.655 | 0 | 1.05 | 1.12 | 1.43 | 5.5 | 16.5 | 41.8 | 0.0194 | 0.000556 |
| product-amplitude | 1800 | 1 | 0 | 1.36 | 1.5 | 1.7 | 4.78 | 6.04 | 7.55 | 0.00167 | 0 |
| additive-amplitude | 1800 | 1 | 0 | 1.38 | 1.52 | 1.74 | 4.55 | 5.79 | 7.33 | 0.00111 | 0 |
| product-lengthscale_r2d2 | 1800 | 0.00778 | 0 | 1.02 | 1.03 | 1.06 | 59.7 | 92.7 | 123 | 0.00722 | 0 |
| additive-lengthscale_r2d2 | 1800 | 0.0322 | 0 | 1.02 | 1.04 | 1.06 | 43.6 | 82.2 | 114 | 0.384 | 0.0122 |
| product-amplitude_r2d2 | 1800 | 1 | 0 | 1.37 | 1.5 | 1.7 | 4.78 | 5.97 | 7.53 | 0.000556 | 0 |
| additive-amplitude_r2d2 | 1800 | 1 | 0 | 1.4 | 1.54 | 1.76 | 4.53 | 5.68 | 7.17 | 0 | 0 |
| sobol | 1800 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | 0 |
| dsp_map | 1800 | 0 | 2 | nan | nan | nan | nan | nan | nan | nan | 0 |
| oracle_S | 1800 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | 0 |


Sampler effort (num_steps_mean = leapfrog steps per NUTS iteration, max 63 at max_tree_depth = 6; fit_calls > 1 means a retry):

| method | num_steps_mean_median | frac_at_tree_cap | fit_calls_mean | nuts_attempts_mean |
|---|---|---|---|---|
| product-lengthscale | 63 | 0.993 | 1 | 1 |
| additive-lengthscale | 63 | 0.99 | 1 | 1 |
| product-amplitude | 63 | 1 | 1 | 1 |
| additive-amplitude | 63 | 1 | 1 | 1 |
| product-lengthscale_r2d2 | 60.6 | 0.156 | 1 | 1 |
| additive-lengthscale_r2d2 | 60.9 | 0.2 | 1 | 1 |
| product-amplitude_r2d2 | 63 | 1 | 1 | 1 |
| additive-amplitude_r2d2 | 63 | 1 | 1 | 1 |


Identification quality of readouts at t >= 100 split by whether that fit passed the gate (cells):

| method | gate_excluded | n_readouts | ap_native_mean | ap_native_median | f1_mean |
|---|---|---|---|---|---|
| product-lengthscale | passed | 14 | 0.873 | 0.933 | 0.508 |
| product-lengthscale | excluded | 36 | 0.742 | 0.758 | 0.562 |
| additive-lengthscale | passed | 17 | 0.973 | 1 | 0.912 |
| additive-lengthscale | excluded | 33 | 0.845 | 0.887 | 0.736 |
| product-amplitude | excluded | 50 | 0.748 | 0.786 | 0.395 |
| additive-amplitude | excluded | 50 | 0.796 | 0.839 | 0.541 |
| product-lengthscale_r2d2 | passed | 50 | 0.613 | 0.648 | 0.374 |
| additive-lengthscale_r2d2 | passed | 50 | 0.603 | 0.605 | 0.364 |
| product-amplitude_r2d2 | excluded | 50 | 0.712 | 0.757 | 0.263 |
| additive-amplitude_r2d2 | excluded | 50 | 0.735 | 0.737 | 0.365 |


Which site fails (medians of the per-site diagnostics):

| method | r_hat_max_native_median | r_hat_max_ell_median | r_hat_max_global_median | n_eff_min_native_median | n_eff_min_ell_median | n_eff_min_global_median | frac_r_hat_below_1_05_median |
|---|---|---|---|---|---|---|---|
| product-lengthscale | 1.13 | nan | 1.02 | 16.6 | nan | 89.3 | 0.942 |
| additive-lengthscale | 1.12 | nan | 1.02 | 16.5 | nan | 74.1 | 0.942 |
| product-amplitude | 1.42 | 1.44 | 1.05 | 6.82 | 6.62 | 36.7 | 0.631 |
| additive-amplitude | 1.44 | 1.45 | 1.04 | 6.48 | 6.52 | 37.4 | 0.626 |
| product-lengthscale_r2d2 | 1.03 | nan | 1 | 93.2 | nan | 177 | 1 |
| additive-lengthscale_r2d2 | 1.03 | nan | 1 | 85.3 | nan | 161 | 1 |
| product-amplitude_r2d2 | 1.3 | 1.5 | 1.03 | 9.18 | 6.02 | 54.4 | 0.655 |
| additive-amplitude_r2d2 | 1.31 | 1.53 | 1.02 | 8.97 | 5.71 | 87.4 | 0.645 |


Gate reason breakdown (iterations):

| method | divergences | exception | fit_gpytorch_mll+Adam | n_eff_min | ok | r_hat_max | r_hat_max+divergences | r_hat_max+n_eff_min | r_hat_max+n_eff_min+divergences |
|---|---|---|---|---|---|---|---|---|---|
| product-lengthscale | 0 | 0 | 0 | 87 | 587 | 347 | 0 | 779 | 0 |
| additive-lengthscale | 0 | 0 | 0 | 68 | 621 | 316 | 0 | 794 | 1 |
| product-amplitude | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1800 | 0 |
| additive-amplitude | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1800 | 0 |
| product-lengthscale_r2d2 | 0 | 0 | 0 | 0 | 1786 | 11 | 0 | 3 | 0 |
| additive-lengthscale_r2d2 | 10 | 0 | 0 | 3 | 1742 | 24 | 4 | 9 | 8 |
| product-amplitude_r2d2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1800 | 0 |
| additive-amplitude_r2d2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1800 | 0 |
| dsp_map | 0 | 2 | 31 | 0 | 1767 | 0 | 0 | 0 | 0 |


Per method and family:

| method | family | n_iter | gate_excluded_rate | exception_rows | r_hat_max_median | n_eff_min_median | divergences_mean |
|---|---|---|---|---|---|---|---|
| product-lengthscale | aligned10 | 900 | 0.742 | 0 | 1.14 | 15.1 | 0.00333 |
| product-lengthscale | decoupled | 900 | 0.606 | 0 | 1.12 | 18.8 | 0.00444 |
| additive-lengthscale | aligned10 | 900 | 0.616 | 0 | 1.12 | 18.6 | 0.0278 |
| additive-lengthscale | decoupled | 900 | 0.694 | 0 | 1.14 | 14 | 0.0111 |
| product-amplitude | aligned10 | 900 | 1 | 0 | 1.49 | 6.06 | 0 |
| product-amplitude | decoupled | 900 | 1 | 0 | 1.5 | 6.03 | 0.00333 |
| additive-amplitude | aligned10 | 900 | 1 | 0 | 1.52 | 5.74 | 0 |
| additive-amplitude | decoupled | 900 | 1 | 0 | 1.51 | 5.86 | 0.00222 |
| product-lengthscale_r2d2 | aligned10 | 900 | 0.00333 | 0 | 1.03 | 91.5 | 0.00889 |
| product-lengthscale_r2d2 | decoupled | 900 | 0.0122 | 0 | 1.03 | 93.5 | 0.00556 |
| additive-lengthscale_r2d2 | aligned10 | 900 | 0.0222 | 0 | 1.04 | 82.3 | 0.22 |
| additive-lengthscale_r2d2 | decoupled | 900 | 0.0422 | 0 | 1.04 | 82.2 | 0.548 |
| product-amplitude_r2d2 | aligned10 | 900 | 1 | 0 | 1.5 | 5.94 | 0 |
| product-amplitude_r2d2 | decoupled | 900 | 1 | 0 | 1.5 | 6.02 | 0.00111 |
| additive-amplitude_r2d2 | aligned10 | 900 | 1 | 0 | 1.55 | 5.59 | 0 |
| additive-amplitude_r2d2 | decoupled | 900 | 1 | 0 | 1.53 | 5.79 | 0 |
| sobol | aligned10 | 900 | 0 | 0 | nan | nan | nan |
| sobol | decoupled | 900 | 0 | 0 | nan | nan | nan |
| dsp_map | aligned10 | 900 | 0 | 2 | nan | nan | nan |
| dsp_map | decoupled | 900 | 0 | 0 | nan | nan | nan |
| oracle_S | aligned10 | 900 | 0 | 0 | nan | nan | nan |
| oracle_S | decoupled | 900 | 0 | 0 | nan | nan | nan |


Gate-exclusion rate vs t (bins of 20), cells:

| t_bin | product-lengthscale | additive-lengthscale | product-amplitude | additive-amplitude | product-lengthscale_r2d2 | additive-lengthscale_r2d2 | product-amplitude_r2d2 | additive-amplitude_r2d2 |
|---|---|---|---|---|---|---|---|---|
| 20 | 0.335 | 0.445 | 1 | 1 | 0.03 | 0.095 | 1 | 1 |
| 40 | 0.625 | 0.795 | 1 | 1 | 0.01 | 0.105 | 1 | 1 |
| 60 | 0.835 | 0.87 | 1 | 1 | 0.005 | 0.035 | 1 | 1 |
| 80 | 0.885 | 0.815 | 1 | 1 | 0 | 0.03 | 1 | 1 |
| 100 | 0.75 | 0.755 | 1 | 1 | 0.005 | 0.005 | 1 | 1 |
| 120 | 0.74 | 0.65 | 1 | 1 | 0.015 | 0 | 1 | 1 |
| 140 | 0.675 | 0.625 | 1 | 1 | 0.005 | 0.01 | 1 | 1 |
| 160 | 0.64 | 0.445 | 1 | 1 | 0 | 0.005 | 1 | 1 |
| 180 | 0.58 | 0.495 | 1 | 1 | 0 | 0.005 | 1 | 1 |


## 10. Cost

Wall-times grouped by the GPU that ran the iteration (attributed from the resume segments in log.txt); the reference methods ran on CPU. **Do not compare across GPU types.** `hours_fit_plus_acq` sums the measured fit and acquisition time only: it excludes the per-iteration readout (coords.csv), checkpointing, JAX compilation at resume, and Slurm queue time, so it is a lower bound on allocated GPU-hours.


Each R2-D2 cell against its half-Cauchy twin on the same GPU type (medians over iterations; ratio = R2-D2 / twin):

| cell | twin | device | n_iter | n_iter_twin | fit_wall_s_median | fit_wall_s_median_twin | fit_wall_s_ratio | acq_wall_s_median | acq_wall_s_median_twin | acq_wall_s_ratio | iter_wall_s_median | iter_wall_s_median_twin | iter_wall_s_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| product-lengthscale_r2d2 | product-lengthscale | NVIDIA H200 | 1800 | 0 | 43.5 | nan | nan | 6.62 | nan | nan | 50.3 | nan | nan |
| additive-lengthscale_r2d2 | additive-lengthscale | NVIDIA H200 | 1800 | 1440 | 48.1 | 39.2 | 1.23 | 193 | 164 | 1.18 | 241 | 206 | 1.17 |
| product-amplitude_r2d2 | product-amplitude | NVIDIA H200 | 1800 | 1800 | 44.5 | 43.3 | 1.03 | 248 | 238 | 1.04 | 296 | 282 | 1.05 |
| additive-amplitude_r2d2 | additive-amplitude | NVIDIA H200 | 1800 | 720 | 44 | 42.8 | 1.03 | 267 | 290 | 0.921 | 314 | 334 | 0.939 |


| method | device | n_iter | fit_wall_s_median | acq_wall_s_median | iter_wall_s_median | fit_wall_s_q90 | acq_wall_s_q90 | hours_fit_plus_acq |
|---|---|---|---|---|---|---|---|---|
| product-lengthscale | NVIDIA A100-SXM4-80GB | 900 | 47.8 | 10.6 | 59.6 | 57.2 | 18 | 14.6 |
| product-lengthscale | NVIDIA H100 80GB HBM3 | 900 | 39.3 | 8.22 | 49.2 | 47.9 | 14.5 | 12.1 |
| additive-lengthscale | NVIDIA A100-SXM4-80GB | 360 | 51.4 | 240 | 293 | 64.7 | 296 | 29.5 |
| additive-lengthscale | NVIDIA H200 | 1440 | 39.2 | 164 | 206 | 50.6 | 220 | 83.1 |
| product-amplitude | NVIDIA H200 | 1800 | 43.3 | 238 | 282 | 55.7 | 313 | 138 |
| additive-amplitude | NVIDIA A100-SXM4-80GB | 360 | 57.9 | 457 | 517 | 73.7 | 506 | 47.9 |
| additive-amplitude | NVIDIA H100 80GB HBM3 | 720 | 45 | 249 | 296 | 56.9 | 320 | 58.6 |
| additive-amplitude | NVIDIA H200 | 720 | 42.8 | 290 | 334 | 54.3 | 319 | 62.6 |
| product-lengthscale_r2d2 | NVIDIA H200 | 1800 | 43.5 | 6.62 | 50.3 | 56.4 | 10.6 | 25.4 |
| additive-lengthscale_r2d2 | NVIDIA H200 | 1800 | 48.1 | 193 | 241 | 61.8 | 272 | 123 |
| product-amplitude_r2d2 | NVIDIA H200 | 1800 | 44.5 | 248 | 296 | 56.7 | 323 | 144 |
| additive-amplitude_r2d2 | NVIDIA H200 | 1800 | 44 | 267 | 314 | 55.5 | 314 | 148 |
| sobol | cpu | 1800 | nan | 3.37e-06 | nan | nan | 4.97e-06 | 0 |
| dsp_map | cpu | 1800 | 4.94 | 9 | 14.8 | 16.9 | 12.5 | 9.09 |
| oracle_S | cpu | 1800 | 1.34 | 2.74 | 4.15 | 2.86 | 3.41 | 2.27 |


Totals per method (all devices, all rows including incomplete runs):

| method | n_iter | hours_fit | hours_acq | hours_fit_plus_acq | median_run_hours | max_run_hours |
|---|---|---|---|---|---|---|
| product-lengthscale | 1800 | 21.6 | 5.1 | 26.7 | 2.7 | 3.04 |
| additive-lengthscale | 1800 | 21.1 | 91.5 | 113 | 10.8 | 14.8 |
| product-amplitude | 1800 | 21.7 | 117 | 138 | 14.3 | 16.4 |
| additive-amplitude | 1800 | 23.2 | 146 | 169 | 16.1 | 24 |
| product-lengthscale_r2d2 | 1800 | 21.9 | 3.53 | 25.4 | 2.65 | 2.69 |
| additive-lengthscale_r2d2 | 1800 | 24.1 | 98.6 | 123 | 12.9 | 14.7 |
| product-amplitude_r2d2 | 1800 | 22.2 | 122 | 144 | 14.2 | 16.1 |
| additive-amplitude_r2d2 | 1800 | 21.9 | 126 | 148 | 15.1 | 16.5 |
| sobol | 1800 | 0 | 1.91e-06 | 0 | 0 | 0 |
| dsp_map | 1800 | 4.76 | 4.38 | 9.09 | 0.922 | 1.16 |
| oracle_S | 1800 | 0.858 | 1.41 | 2.27 | 0.219 | 0.328 |


![cost vs t](figures/cost_vs_t.png)


## 11. Anomalies and things that look wrong

- DEGENERATE readout: aligned10/additive-lengthscale_r2d2/seed01 declares 52/100 coordinates active at t=199 (AP by native_median 0.83, by sobol_hat 0.86; min native_median on true S 0.481, max on inactive 2.53; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed00 declares 64/100 coordinates active at t=199 (AP by native_median 0.63, by sobol_hat 0.74; min native_median on true S 0.558, max on inactive 3.41; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed02 declares 99/100 coordinates active at t=199 (AP by native_median 0.61, by sobol_hat 0.57; min native_median on true S 3.67, max on inactive 18; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed03 declares 97/100 coordinates active at t=199 (AP by native_median 0.51, by sobol_hat 0.83; min native_median on true S 2.5, max on inactive 17.9; fit status ok)
- NUTS hits max_tree_depth=6 (num_steps_mean >= 62 of the 63 possible) in most fits: fraction per cell additive-amplitude 1.000, additive-amplitude_r2d2 1.000, additive-lengthscale 0.990, additive-lengthscale_r2d2 0.200, product-amplitude 1.000, product-amplitude_r2d2 1.000, product-lengthscale 0.993, product-lengthscale_r2d2 0.156


Expected oddities (not flagged):

- Git commits at creation, by prior family: {'HC': ['a51a4b9'], 'R2D2': ['97eda92', 'c4d9c38'], 'reference': ['a51a4b9']} (the HC cells and references were launched at a51a4b9 and 2866ed2; the R2-D2 cells' commits are checked by `prior_code_ok`).
- dsp_map: 31 iterations used the Adam fallback fitter (status ok, reason 'fit_gpytorch_mll failed; Adam fallback'); these are neither gate exclusions nor exception rows.


## Caveats

- status="excluded" is a label: the draws were still used to choose the next point, so regret curves are valid for every run. It does mean the posterior readouts (native_median, p_active, sobol_hat) from those iterations are less trustworthy.
- Rows whose reason starts with "exception:" queried a seeded random point instead; they are counted separately from gate exclusions.
- GPU runs (A100/H100/H200) reproduce only in distribution, not bit-for-bit; wall-times are only comparable within one GPU type.
- The per-family tests have at most 10 seeds (5 in the pilot, whose smallest two-sided Wilcoxon p is 0.0625).
- The regret plots floor the log axis at a small positive value; seeds that reached regret 0 sit on that floor.
