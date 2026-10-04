# sagp study analysis: half-Cauchy and R2-D2 prior families

As of **2026-10-04 14:25 EEST** (data: `runs`, `runs_r2d2`; analysis dir `sagp_analysis/r2d2/2026-10-02-1420`; script `analyze.py`; families aligned3, aligned10, anti_aligned, decoupled, interaction_g0.00, interaction_g0.25, interaction_g0.50, interaction_g0.75; seeds 00, 01, 02, 03, 04, 05, 06, 07, 08, 09).

Runs: **880/880 complete** (180 rows). Incomplete or missing: none.

Headline numbers use complete runs only. Seeds are paired: one (family, seed) is one objective and one shared initial design across all 11 methods. "Family" alone means the objective family; the half-Cauchy (HC) and R2-D2 priors are the two prior families.

<!-- TOP5 -->


## G3 (the pilot gate)

**G3: not read.** G3 is the pilot gate (families aligned10, decoupled, seeds 0-4); this run's scope is families aligned3, aligned10, anti_aligned, decoupled, interaction_g0.00, interaction_g0.25, interaction_g0.50, interaction_g0.75, seeds 00, 01, 02, 03, 04, 05, 06, 07, 08, 09: the table below is descriptive.


Delta = y(R2-D2) - y(HC twin) at t = 199, y = log10(max(r, 1e-8)), over 320 pairs in 80 (family, seed) units. GO if any of: (a) |median Delta| >= 0.3 pooled or within any cell; (b) the pooled (family, seed)-cluster bootstrap 95 % interval of the mean Delta (10,000 resamples) excludes 0; (c) any cell's Wilcoxon signed-rank over its pairs, Holm-adjusted over the 4 cells, below 0.05; (d) |median dS| or |median dP| >= 0.3 over the units. Criteria firing: (a) True, (b) True, (c) True, (d) False. For (b): a percentile bootstrap over 80 clusters tends to under-cover, and the prereg does not fix the interval type.


| criterion | statistic | n | value | rule | fires |
|---|---|---|---|---|---|
| (a) | median Delta, pooled | 320 | 0.457 | abs(value) >= 0.3 | True |
| (a) | median Delta, product-lengthscale_r2d2 | 80 | 0.492 | abs(value) >= 0.3 | True |
| (a) | median Delta, additive-lengthscale_r2d2 | 80 | 0.508 | abs(value) >= 0.3 | True |
| (a) | median Delta, product-amplitude_r2d2 | 80 | 0.502 | abs(value) >= 0.3 | True |
| (a) | median Delta, additive-amplitude_r2d2 | 80 | 0.364 | abs(value) >= 0.3 | True |
| (b) | mean Delta, pooled; cluster bootstrap 95 % [0.53, 0.877] | 320 | 0.707 | interval excludes 0 | True |
| (c) | Wilcoxon p, product-lengthscale_r2d2 (raw 1.05e-07); Holm over 4 cells | 80 | 2.09e-07 | value < 0.05 | True |
| (c) | Wilcoxon p, additive-lengthscale_r2d2 (raw 4.78e-10); Holm over 4 cells | 80 | 1.91e-09 | value < 0.05 | True |
| (c) | Wilcoxon p, product-amplitude_r2d2 (raw 4.68e-08); Holm over 4 cells | 80 | 1.4e-07 | value < 0.05 | True |
| (c) | Wilcoxon p, additive-amplitude_r2d2 (raw 1.48e-05); Holm over 4 cells | 80 | 1.48e-05 | value < 0.05 | True |
| (d) | median dS over units | 80 | 0.00339 | abs(value) >= 0.3 | False |
| (d) | median dP over units | 80 | -0.103 | abs(value) >= 0.3 | False |


## What changed since the previous analysis

No previous analysis directory found: this is the first run, nothing to diff against.


## 1. Inventory

| method | runs | complete | rows |
|---|---|---|---|
| product-lengthscale | 80 | 80 | 14400 |
| additive-lengthscale | 80 | 80 | 14400 |
| product-amplitude | 80 | 80 | 14400 |
| additive-amplitude | 80 | 80 | 14400 |
| product-lengthscale_r2d2 | 80 | 80 | 14400 |
| additive-lengthscale_r2d2 | 80 | 80 | 14400 |
| product-amplitude_r2d2 | 80 | 80 | 14400 |
| additive-amplitude_r2d2 | 80 | 80 | 14400 |
| sobol | 80 | 80 | 14400 |
| dsp_map | 80 | 80 | 14400 |
| oracle_S | 80 | 80 | 14400 |


Objectives: D = 100, noise_sd = 0.1; |S| per family {'aligned3': [3], 'aligned10': [10], 'anti_aligned': [6], 'decoupled': [8], 'interaction_g0.00': [5], 'interaction_g0.25': [5], 'interaction_g0.50': [5], 'interaction_g0.75': [5]}; gamma per family {'aligned3': [0.0], 'aligned10': [0.0], 'anti_aligned': [0.0], 'decoupled': [0.0], 'interaction_g0.00': [0.0], 'interaction_g0.25': [0.25000000000000006], 'interaction_g0.50': [0.5], 'interaction_g0.75': [0.7499999999999999]}. Distinct config_hash values: 880 (one per run; the hash covers the objective and the method). Git commits at creation by prior family: {'HC': ['2866ed2', 'a51a4b9'], 'R2D2': ['5a485f3', '97eda92', 'c4d9c38'], 'reference': ['2866ed2', 'a51a4b9']}.


Devices at creation (runs): {'NVIDIA H200': 532, 'cpu': 240, 'NVIDIA A100-SXM4-80GB': 67, 'NVIDIA H100 80GB HBM3': 41}. Runs whose iterations span more than one GPU type (resumed onto a different partition): 42. Resumes per run: {0: 144, 1: 274, 2: 33, 3: 309, 4: 40, 6: 80} (most `resumed` entries are no-op resumes at t=200 recorded by the chain's later Slurm links).


## 2. Sanity checks

| check | n_runs | n_pass | n_fail | failing_runs |
|---|---|---|---|---|
| rows_180 | 880 | 880 | 0 |  |
| no_dup_t | 880 | 880 | 0 |  |
| t_contiguous | 880 | 880 | 0 |  |
| regret_nonincreasing | 880 | 880 | 0 |  |
| best_f_nondecreasing | 880 | 880 | 0 |  |
| best_f_ge_cummax_f | 880 | 880 | 0 |  |
| regret_eq_fstar_minus_bestf | 880 | 880 | 0 |  |
| regret_nonnegative | 880 | 880 | 0 |  |
| log_resumes_eq_manifest | 880 | 880 | 0 |  |
| log_t_set_eq_csv | 880 | 880 | 0 |  |
| manifest_method_eq_dir | 880 | 880 | 0 |  |
| f_star_identical | 80 | 80 | 0 |  |
| S_identical | 80 | 80 | 0 |  |
| y_mean_t20_identical | 80 | 80 | 0 |  |
| y_std_t20_identical | 80 | 80 | 0 |  |
| twin_paired | 320 | 320 | 0 |  |
| prior_code_ok | 320 | 320 | 0 |  |


`best_f_ge_cummax_f` is a lower-bound check because best_f also covers the 20-point initial design, which is not a row. `regret_eq_fstar_minus_bestf` checks the CSV's regret against manifest.objective.f_star to 1e-9. `log_resumes_eq_manifest` and `log_t_set_eq_csv` verify that log.txt's resume lines match manifest.resumed and that every iteration in the CSV appears once in the log (device attribution relies on this). `manifest_method_eq_dir`: the manifest's method is the one its directory names.


`twin_paired` (D10): each R2-D2 run's row t = 20 `y_mean` and `y_std` are bit-identical to its half-Cauchy twin's, and `f_star` and S are equal (`tables/sanity_twin_pairing.csv`). `prior_code_ok` (Review Focus 3): every R2-D2 run's creation commit and every resume commit resolve, in `/Users/quantr03/Library/CloudStorage/OneDrive-AaltoUniversity/2026 Summer Internship'/saasbo`, to one pair of blobs of sagp/gp.py and sagp/r2d2.py, the pair most R2-D2 runs share, and the creation checkout was clean (resume entries record no dirty flag) (`tables/sanity_prior_code.csv`).


Launch commit `97eda925b`: blob pair (gp.py/r2d2.py) `f3590452a23b3b65c4216c5dd399d33e0ddebbce/45ef19533534f7f4d2ac5d4c84cddebe45d86600`; `prior_code_ok` also requires the R2-D2 runs' modal pair to be this one.


Prior-code blob pairs (gp.py/r2d2.py) over the R2-D2 runs: `f3590452a2/45ef195335` x 320; commits seen: ['5a485f33b', '97eda925b', 'c4d9c38ac'].


## 3. Optimization performance

![regret curves](figures/regret_curves.png)

![final regret](figures/final_regret_strip.png)


Final regret at t=199, median [Q1, Q3] across complete seeds, ranked within family (1 = best); `median_y` is the median of y = log10(max(r, 1e-8)):


| family | method | n | median_iqr | median_y | rank_in_family | mean_rank_within_seed |
|---|---|---|---|---|---|---|
| aligned10 | additive-amplitude | 10 | 0.0273 [0.0201, 0.0391] | -1.57 | 1 | 2.3 |
| aligned10 | additive-lengthscale | 10 | 0.0323 [0.0184, 0.0502] | -1.49 | 2 | 2.8 |
| aligned10 | additive-amplitude_r2d2 | 10 | 0.0456 [0.038, 0.0537] | -1.34 | 3 | 3.9 |
| aligned10 | product-amplitude | 10 | 0.0653 [0.0181, 0.0897] | -1.19 | 4 | 4.1 |
| aligned10 | product-amplitude_r2d2 | 10 | 0.0702 [0.0529, 0.0983] | -1.16 | 5 | 5.1 |
| aligned10 | additive-lengthscale_r2d2 | 10 | 0.107 [0.0573, 0.231] | -0.982 | 6 | 5.9 |
| aligned10 | dsp_map | 10 | 0.282 [0.195, 0.509] | -0.554 | 7 | 7.5 |
| aligned10 | product-lengthscale | 10 | 0.289 [0.0783, 0.52] | -0.554 | 8 | 6.9 |
| aligned10 | oracle_S | 10 | 0.429 [0.264, 0.583] | -0.368 | 9 | 7.7 |
| aligned10 | product-lengthscale_r2d2 | 10 | 0.555 [0.366, 0.773] | -0.256 | 10 | 8.8 |
| aligned10 | sobol | 10 | 2.28 [2.16, 2.53] | 0.357 | 11 | 11 |
| aligned3 | product-amplitude | 10 | 0.000115 [2.22e-05, 0.000616] | -3.94 | 1 | 3.3 |
| aligned3 | additive-lengthscale | 10 | 0.000242 [1.09e-05, 0.00116] | -3.7 | 2 | 3.5 |
| aligned3 | additive-amplitude | 10 | 0.000291 [3.03e-05, 0.00103] | -3.67 | 3 | 3.9 |
| aligned3 | dsp_map | 10 | 0.000411 [2.11e-05, 0.0084] | -3.46 | 4 | 5.6 |
| aligned3 | additive-amplitude_r2d2 | 10 | 0.00121 [0.000685, 0.00637] | -2.92 | 5 | 5.8 |
| aligned3 | product-lengthscale | 10 | 0.00124 [0.00021, 0.0044] | -2.91 | 6 | 5.3 |
| aligned3 | product-amplitude_r2d2 | 10 | 0.00307 [0.000606, 0.00685] | -2.51 | 7 | 6.3 |
| aligned3 | oracle_S | 10 | 0.0033 [0.00133, 0.00515] | -2.49 | 8 | 6.9 |
| aligned3 | additive-lengthscale_r2d2 | 10 | 0.00364 [0.00163, 0.0158] | -2.45 | 9 | 7.05 |
| aligned3 | product-lengthscale_r2d2 | 10 | 0.0382 [0.0113, 0.0906] | -1.48 | 10 | 7.55 |
| aligned3 | sobol | 10 | 0.359 [0.167, 0.43] | -0.446 | 11 | 10.8 |
| anti_aligned | additive-amplitude | 10 | 0.0563 [0.0172, 0.0849] | -1.26 | 1 | 2.3 |
| anti_aligned | product-amplitude | 10 | 0.111 [0.0756, 0.158] | -0.958 | 2 | 3 |
| anti_aligned | additive-lengthscale | 10 | 0.115 [0.0422, 0.225] | -0.943 | 3 | 3.2 |
| anti_aligned | additive-amplitude_r2d2 | 10 | 0.129 [0.0797, 0.168] | -0.889 | 4 | 4.4 |
| anti_aligned | product-amplitude_r2d2 | 10 | 0.197 [0.085, 0.284] | -0.712 | 5 | 5.6 |
| anti_aligned | dsp_map | 10 | 0.255 [0.168, 0.441] | -0.597 | 6 | 6.3 |
| anti_aligned | oracle_S | 10 | 0.269 [0.2, 0.405] | -0.572 | 7 | 6.5 |
| anti_aligned | product-lengthscale | 10 | 0.302 [0.224, 0.495] | -0.525 | 8 | 7.2 |
| anti_aligned | additive-lengthscale_r2d2 | 10 | 0.354 [0.291, 0.467] | -0.451 | 9 | 7.9 |
| anti_aligned | product-lengthscale_r2d2 | 10 | 0.379 [0.304, 0.492] | -0.422 | 10 | 8.6 |
| anti_aligned | sobol | 10 | 1.44 [1.02, 1.56] | 0.159 | 11 | 11 |
| decoupled | additive-amplitude | 10 | 0.0581 [0.0153, 0.2] | -1.29 | 1 | 2.4 |
| decoupled | product-amplitude | 10 | 0.126 [0.0615, 0.321] | -0.9 | 2 | 3.2 |
| decoupled | additive-lengthscale | 10 | 0.157 [0.0892, 0.367] | -0.823 | 3 | 3.9 |
| decoupled | product-amplitude_r2d2 | 10 | 0.191 [0.0746, 0.411] | -0.725 | 4 | 5.2 |
| decoupled | dsp_map | 10 | 0.287 [0.097, 0.653] | -0.55 | 5 | 5.6 |
| decoupled | additive-lengthscale_r2d2 | 10 | 0.334 [0.222, 0.564] | -0.477 | 6 | 6.1 |
| decoupled | additive-amplitude_r2d2 | 10 | 0.409 [0.287, 0.697] | -0.389 | 7 | 6.4 |
| decoupled | oracle_S | 10 | 0.443 [0.163, 0.798] | -0.365 | 8 | 7.6 |
| decoupled | product-lengthscale | 10 | 0.465 [0.191, 0.537] | -0.332 | 9 | 7.2 |
| decoupled | product-lengthscale_r2d2 | 10 | 0.561 [0.47, 0.883] | -0.251 | 10 | 7.4 |
| decoupled | sobol | 10 | 1.81 [1.56, 2.11] | 0.257 | 11 | 11 |
| interaction_g0.00 | additive-lengthscale | 10 | 0.00339 [0.00202, 0.0104] | -2.48 | 1 | 2.3 |
| interaction_g0.00 | product-amplitude | 10 | 0.00441 [0.0021, 0.00958] | -2.36 | 2 | 2.3 |
| interaction_g0.00 | additive-amplitude | 10 | 0.00472 [0.00161, 0.012] | -2.35 | 3 | 2.9 |
| interaction_g0.00 | product-lengthscale | 10 | 0.0116 [0.0061, 0.0302] | -1.94 | 4 | 5.15 |
| interaction_g0.00 | product-amplitude_r2d2 | 10 | 0.0241 [0.00891, 0.0388] | -1.62 | 5 | 5.25 |
| interaction_g0.00 | oracle_S | 10 | 0.0322 [0.0181, 0.0637] | -1.49 | 6 | 7.05 |
| interaction_g0.00 | additive-amplitude_r2d2 | 10 | 0.0345 [0.0091, 0.0896] | -1.48 | 7 | 7.05 |
| interaction_g0.00 | additive-lengthscale_r2d2 | 10 | 0.0361 [0.0218, 0.0782] | -1.46 | 8 | 7.45 |
| interaction_g0.00 | dsp_map | 10 | 0.0891 [0.0455, 0.26] | -1.05 | 9 | 7.1 |
| interaction_g0.00 | product-lengthscale_r2d2 | 10 | 0.1 [0.0503, 0.248] | -0.998 | 10 | 8.55 |
| interaction_g0.00 | sobol | 10 | 1.05 [0.607, 1.83] | 0.0214 | 11 | 10.9 |
| interaction_g0.25 | product-lengthscale | 10 | 0.000699 [0.000227, 0.00994] | -3.16 | 1 | 2.3 |
| interaction_g0.25 | product-amplitude | 10 | 0.00125 [0.00108, 0.0214] | -2.9 | 2 | 2.7 |
| interaction_g0.25 | oracle_S | 10 | 0.0119 [0.00445, 0.0475] | -1.93 | 3 | 4.1 |
| interaction_g0.25 | additive-amplitude | 10 | 0.0173 [0.00207, 0.164] | -1.78 | 4 | 4.7 |
| interaction_g0.25 | additive-lengthscale | 10 | 0.0214 [0.00336, 0.205] | -1.68 | 5 | 4.3 |
| interaction_g0.25 | additive-amplitude_r2d2 | 10 | 0.0904 [0.0187, 0.51] | -1.11 | 6 | 6.1 |
| interaction_g0.25 | product-amplitude_r2d2 | 10 | 0.148 [0.0602, 0.698] | -0.835 | 7 | 7.8 |
| interaction_g0.25 | additive-lengthscale_r2d2 | 10 | 0.206 [0.0817, 0.455] | -0.694 | 8 | 7.25 |
| interaction_g0.25 | product-lengthscale_r2d2 | 10 | 0.363 [0.189, 0.908] | -0.463 | 9 | 8.75 |
| interaction_g0.25 | dsp_map | 10 | 0.415 [0.0589, 1.31] | -0.516 | 10 | 7.7 |
| interaction_g0.25 | sobol | 10 | 1.33 [1.09, 1.62] | 0.125 | 11 | 10.3 |
| interaction_g0.50 | product-amplitude | 10 | 0.0305 [0.0013, 0.283] | -1.56 | 1 | 3.55 |
| interaction_g0.50 | product-lengthscale | 10 | 0.0356 [0.0122, 0.145] | -1.51 | 2 | 3.45 |
| interaction_g0.50 | oracle_S | 10 | 0.0413 [0.0189, 0.159] | -1.4 | 3 | 3.45 |
| interaction_g0.50 | additive-lengthscale | 10 | 0.265 [0.028, 0.616] | -0.765 | 4 | 4.8 |
| interaction_g0.50 | additive-lengthscale_r2d2 | 10 | 0.508 [0.353, 1.09] | -0.294 | 5 | 7.15 |
| interaction_g0.50 | additive-amplitude_r2d2 | 10 | 0.51 [0.0928, 0.791] | -0.301 | 6 | 6 |
| interaction_g0.50 | product-amplitude_r2d2 | 10 | 0.645 [0.444, 0.785] | -0.191 | 7 | 6.7 |
| interaction_g0.50 | additive-amplitude | 10 | 0.65 [0.132, 0.875] | -0.188 | 8 | 5.8 |
| interaction_g0.50 | product-lengthscale_r2d2 | 10 | 0.879 [0.641, 1.17] | -0.0577 | 9 | 6.9 |
| interaction_g0.50 | dsp_map | 10 | 0.949 [0.634, 1.15] | -0.0307 | 10 | 8 |
| interaction_g0.50 | sobol | 10 | 1.57 [1.14, 2.05] | 0.193 | 11 | 10.2 |
| interaction_g0.75 | product-amplitude | 10 | 0.00828 [0.00234, 0.0421] | -2.08 | 1 | 2.5 |
| interaction_g0.75 | oracle_S | 10 | 0.0164 [0.00958, 0.0393] | -1.8 | 2 | 4.05 |
| interaction_g0.75 | product-lengthscale | 10 | 0.119 [0.0221, 0.191] | -0.923 | 3 | 5.15 |
| interaction_g0.75 | additive-lengthscale | 10 | 0.381 [0.0341, 0.825] | -0.591 | 4 | 5.35 |
| interaction_g0.75 | additive-amplitude_r2d2 | 10 | 0.514 [0.00378, 1.77] | -0.355 | 5 | 5.7 |
| interaction_g0.75 | additive-amplitude | 10 | 0.522 [0.04, 1.69] | -0.395 | 6 | 5.3 |
| interaction_g0.75 | product-amplitude_r2d2 | 10 | 0.595 [0.0903, 1.08] | -0.254 | 7 | 6.15 |
| interaction_g0.75 | product-lengthscale_r2d2 | 10 | 0.933 [0.135, 1.43] | -0.0464 | 8 | 6.6 |
| interaction_g0.75 | dsp_map | 10 | 1.06 [0.65, 1.74] | 0.0105 | 9 | 6.7 |
| interaction_g0.75 | additive-lengthscale_r2d2 | 10 | 1.1 [0.876, 1.88] | 0.0373 | 10 | 9 |
| interaction_g0.75 | sobol | 10 | 2.06 [1.33, 2.61] | 0.313 | 11 | 9.5 |


## 4. Paired comparisons within each prior family (Wilcoxon signed-rank over seeds, final regret at t=199)

`median_diff` = median over seeds of (cell regret - reference regret); negative means the cell is better. Two-sided p-values from scipy.stats.wilcoxon (exact for n <= 50 without ties). The smallest attainable two-sided p is 0.002 at 10 seeds and 0.0625 at 5. `p_holm_within_family_kind` is a Holm correction over the 12 cell-vs-reference tests (or 6 cell-vs-cell tests) of one prior family inside each objective family, as sagp2 did for the half-Cauchy cells. These tests are on raw regret, as sagp2's; the effects and twin pairs below are on y.


| family | prior_family | cell | reference | n_seeds | median_diff | n_cell_better | n_ref_better | p_value | p_holm_within_family_kind |
|---|---|---|---|---|---|---|---|---|---|
| aligned3 | HC | product-lengthscale | sobol | 10 | -0.358 | 10 | 0 | 0.00195 | 0.0234 |
| aligned3 | HC | product-lengthscale | dsp_map | 10 | -0.00024 | 5 | 5 | 1 | 1 |
| aligned3 | HC | product-lengthscale | oracle_S | 10 | -8.36e-05 | 7 | 1 | 0.109 | 0.438 |
| aligned3 | HC | additive-lengthscale | sobol | 10 | -0.358 | 10 | 0 | 0.00195 | 0.0234 |
| aligned3 | HC | additive-lengthscale | dsp_map | 10 | -0.00031 | 6 | 4 | 0.232 | 0.48 |
| aligned3 | HC | additive-lengthscale | oracle_S | 10 | -0.00195 | 8 | 2 | 0.0645 | 0.387 |
| aligned3 | HC | product-amplitude | sobol | 10 | -0.359 | 10 | 0 | 0.00195 | 0.0234 |
| aligned3 | HC | product-amplitude | dsp_map | 10 | -0.000257 | 8 | 2 | 0.084 | 0.42 |
| aligned3 | HC | product-amplitude | oracle_S | 10 | -0.00293 | 9 | 1 | 0.0488 | 0.342 |
| aligned3 | HC | additive-amplitude | sobol | 10 | -0.358 | 10 | 0 | 0.00195 | 0.0234 |
| aligned3 | HC | additive-amplitude | dsp_map | 10 | -0.000103 | 6 | 4 | 0.16 | 0.48 |
| aligned3 | HC | additive-amplitude | oracle_S | 10 | -0.00281 | 9 | 1 | 0.0137 | 0.109 |
| aligned10 | HC | product-lengthscale | sobol | 10 | -1.89 | 10 | 0 | 0.00195 | 0.0234 |
| aligned10 | HC | product-lengthscale | dsp_map | 10 | -0.0541 | 7 | 3 | 0.375 | 0.551 |
| aligned10 | HC | product-lengthscale | oracle_S | 10 | -0.122 | 6 | 4 | 0.275 | 0.551 |
| aligned10 | HC | additive-lengthscale | sobol | 10 | -2.26 | 10 | 0 | 0.00195 | 0.0234 |
| aligned10 | HC | additive-lengthscale | dsp_map | 10 | -0.26 | 9 | 1 | 0.0273 | 0.082 |
| aligned10 | HC | additive-lengthscale | oracle_S | 10 | -0.248 | 9 | 1 | 0.00391 | 0.0234 |
| aligned10 | HC | product-amplitude | sobol | 10 | -2.17 | 10 | 0 | 0.00195 | 0.0234 |
| aligned10 | HC | product-amplitude | dsp_map | 10 | -0.213 | 8 | 2 | 0.00977 | 0.0391 |
| aligned10 | HC | product-amplitude | oracle_S | 10 | -0.39 | 9 | 1 | 0.00391 | 0.0234 |
| aligned10 | HC | additive-amplitude | sobol | 10 | -2.26 | 10 | 0 | 0.00195 | 0.0234 |
| aligned10 | HC | additive-amplitude | dsp_map | 10 | -0.229 | 10 | 0 | 0.00195 | 0.0234 |
| aligned10 | HC | additive-amplitude | oracle_S | 10 | -0.399 | 10 | 0 | 0.00195 | 0.0234 |
| anti_aligned | HC | product-lengthscale | sobol | 10 | -1.01 | 10 | 0 | 0.00195 | 0.0234 |
| anti_aligned | HC | product-lengthscale | dsp_map | 10 | 0.102 | 4 | 6 | 0.695 | 0.695 |
| anti_aligned | HC | product-lengthscale | oracle_S | 10 | 0.0356 | 3 | 7 | 0.193 | 0.422 |
| anti_aligned | HC | additive-lengthscale | sobol | 10 | -1.26 | 10 | 0 | 0.00195 | 0.0234 |
| anti_aligned | HC | additive-lengthscale | dsp_map | 10 | -0.2 | 6 | 4 | 0.131 | 0.422 |
| anti_aligned | HC | additive-lengthscale | oracle_S | 10 | -0.168 | 8 | 2 | 0.105 | 0.422 |
| anti_aligned | HC | product-amplitude | sobol | 10 | -1.36 | 10 | 0 | 0.00195 | 0.0234 |
| anti_aligned | HC | product-amplitude | dsp_map | 10 | -0.143 | 9 | 1 | 0.00586 | 0.0352 |
| anti_aligned | HC | product-amplitude | oracle_S | 10 | -0.165 | 9 | 1 | 0.00586 | 0.0352 |
| anti_aligned | HC | additive-amplitude | sobol | 10 | -1.39 | 10 | 0 | 0.00195 | 0.0234 |
| anti_aligned | HC | additive-amplitude | dsp_map | 10 | -0.203 | 10 | 0 | 0.00195 | 0.0234 |
| anti_aligned | HC | additive-amplitude | oracle_S | 10 | -0.16 | 10 | 0 | 0.00195 | 0.0234 |
| decoupled | HC | product-lengthscale | sobol | 10 | -1.36 | 10 | 0 | 0.00195 | 0.0234 |
| decoupled | HC | product-lengthscale | dsp_map | 10 | 0.152 | 4 | 6 | 0.375 | 0.967 |
| decoupled | HC | product-lengthscale | oracle_S | 10 | -0.0519 | 6 | 4 | 0.922 | 0.967 |
| decoupled | HC | additive-lengthscale | sobol | 10 | -1.49 | 10 | 0 | 0.00195 | 0.0234 |
| decoupled | HC | additive-lengthscale | dsp_map | 10 | -0.0918 | 6 | 4 | 0.322 | 0.967 |
| decoupled | HC | additive-lengthscale | oracle_S | 10 | -0.179 | 9 | 1 | 0.00977 | 0.0684 |
| decoupled | HC | product-amplitude | sobol | 10 | -1.49 | 10 | 0 | 0.00195 | 0.0234 |
| decoupled | HC | product-amplitude | dsp_map | 10 | -0.144 | 8 | 2 | 0.0371 | 0.186 |
| decoupled | HC | product-amplitude | oracle_S | 10 | -0.344 | 9 | 1 | 0.0137 | 0.082 |
| decoupled | HC | additive-amplitude | sobol | 10 | -1.69 | 10 | 0 | 0.00195 | 0.0234 |
| decoupled | HC | additive-amplitude | dsp_map | 10 | -0.172 | 7 | 3 | 0.0488 | 0.195 |
| decoupled | HC | additive-amplitude | oracle_S | 10 | -0.277 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.00 | HC | product-lengthscale | sobol | 10 | -1.01 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.00 | HC | product-lengthscale | dsp_map | 10 | -0.0721 | 7 | 3 | 0.0371 | 0.0742 |
| interaction_g0.00 | HC | product-lengthscale | oracle_S | 10 | -0.00303 | 6 | 3 | 0.203 | 0.203 |
| interaction_g0.00 | HC | additive-lengthscale | sobol | 10 | -1.05 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.00 | HC | additive-lengthscale | dsp_map | 10 | -0.0868 | 8 | 2 | 0.00977 | 0.0391 |
| interaction_g0.00 | HC | additive-lengthscale | oracle_S | 10 | -0.0195 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.00 | HC | product-amplitude | sobol | 10 | -1.04 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.00 | HC | product-amplitude | dsp_map | 10 | -0.0825 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.00 | HC | product-amplitude | oracle_S | 10 | -0.0308 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.00 | HC | additive-amplitude | sobol | 10 | -1.04 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.00 | HC | additive-amplitude | dsp_map | 10 | -0.0792 | 8 | 2 | 0.00977 | 0.0391 |
| interaction_g0.00 | HC | additive-amplitude | oracle_S | 10 | -0.026 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.25 | HC | product-lengthscale | sobol | 10 | -1.3 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.25 | HC | product-lengthscale | dsp_map | 10 | -0.383 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.25 | HC | product-lengthscale | oracle_S | 10 | -0.00218 | 8 | 2 | 0.0488 | 0.293 |
| interaction_g0.25 | HC | additive-lengthscale | sobol | 10 | -1.25 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.25 | HC | additive-lengthscale | dsp_map | 10 | -0.078 | 9 | 1 | 0.0488 | 0.293 |
| interaction_g0.25 | HC | additive-lengthscale | oracle_S | 10 | 0.00165 | 4 | 6 | 0.557 | 1 |
| interaction_g0.25 | HC | product-amplitude | sobol | 10 | -1.24 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.25 | HC | product-amplitude | dsp_map | 10 | -0.0982 | 9 | 1 | 0.0371 | 0.26 |
| interaction_g0.25 | HC | product-amplitude | oracle_S | 10 | -0.00336 | 7 | 3 | 0.322 | 0.967 |
| interaction_g0.25 | HC | additive-amplitude | sobol | 10 | -1.24 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.25 | HC | additive-amplitude | dsp_map | 10 | -0.0878 | 8 | 2 | 0.084 | 0.336 |
| interaction_g0.25 | HC | additive-amplitude | oracle_S | 10 | 0.00571 | 4 | 6 | 0.557 | 1 |
| interaction_g0.50 | HC | product-lengthscale | sobol | 10 | -1.44 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.50 | HC | product-lengthscale | dsp_map | 10 | -0.878 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.50 | HC | product-lengthscale | oracle_S | 10 | 0 | 4 | 3 | 0.812 | 1 |
| interaction_g0.50 | HC | additive-lengthscale | sobol | 10 | -1.06 | 9 | 1 | 0.00391 | 0.0391 |
| interaction_g0.50 | HC | additive-lengthscale | dsp_map | 10 | -0.196 | 8 | 2 | 0.193 | 0.967 |
| interaction_g0.50 | HC | additive-lengthscale | oracle_S | 10 | 0.205 | 5 | 5 | 0.232 | 0.967 |
| interaction_g0.50 | HC | product-amplitude | sobol | 10 | -1.13 | 9 | 1 | 0.00391 | 0.0391 |
| interaction_g0.50 | HC | product-amplitude | dsp_map | 10 | -0.528 | 9 | 1 | 0.0645 | 0.451 |
| interaction_g0.50 | HC | product-amplitude | oracle_S | 10 | 4.03e-06 | 4 | 5 | 0.652 | 1 |
| interaction_g0.50 | HC | additive-amplitude | sobol | 10 | -1.11 | 9 | 1 | 0.00391 | 0.0391 |
| interaction_g0.50 | HC | additive-amplitude | dsp_map | 10 | -0.0968 | 7 | 3 | 0.322 | 0.967 |
| interaction_g0.50 | HC | additive-amplitude | oracle_S | 10 | 0.631 | 3 | 7 | 0.0645 | 0.451 |
| interaction_g0.75 | HC | product-lengthscale | sobol | 10 | -1.92 | 9 | 1 | 0.0195 | 0.137 |
| interaction_g0.75 | HC | product-lengthscale | dsp_map | 10 | -0.623 | 8 | 2 | 0.00977 | 0.107 |
| interaction_g0.75 | HC | product-lengthscale | oracle_S | 10 | 0.0574 | 2 | 5 | 0.0781 | 0.371 |
| interaction_g0.75 | HC | additive-lengthscale | sobol | 10 | -1.16 | 9 | 1 | 0.0137 | 0.109 |
| interaction_g0.75 | HC | additive-lengthscale | dsp_map | 10 | -0.365 | 6 | 4 | 0.275 | 0.826 |
| interaction_g0.75 | HC | additive-lengthscale | oracle_S | 10 | 0.352 | 3 | 6 | 0.0742 | 0.371 |
| interaction_g0.75 | HC | product-amplitude | sobol | 10 | -2.06 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.75 | HC | product-amplitude | dsp_map | 10 | -0.919 | 8 | 2 | 0.00977 | 0.107 |
| interaction_g0.75 | HC | product-amplitude | oracle_S | 10 | -0.00751 | 8 | 2 | 0.322 | 0.826 |
| interaction_g0.75 | HC | additive-amplitude | sobol | 10 | -0.997 | 9 | 1 | 0.00977 | 0.107 |
| interaction_g0.75 | HC | additive-amplitude | dsp_map | 10 | -0.358 | 6 | 4 | 0.322 | 0.826 |
| interaction_g0.75 | HC | additive-amplitude | oracle_S | 10 | 0.507 | 3 | 7 | 0.0371 | 0.223 |
| aligned3 | R2D2 | product-lengthscale_r2d2 | sobol | 10 | -0.311 | 9 | 1 | 0.0371 | 0.334 |
| aligned3 | R2D2 | product-lengthscale_r2d2 | dsp_map | 10 | 0.0362 | 4 | 6 | 0.131 | 0.916 |
| aligned3 | R2D2 | product-lengthscale_r2d2 | oracle_S | 10 | 0.0125 | 2 | 6 | 0.0391 | 0.334 |
| aligned3 | R2D2 | additive-lengthscale_r2d2 | sobol | 10 | -0.335 | 9 | 1 | 0.00391 | 0.0391 |
| aligned3 | R2D2 | additive-lengthscale_r2d2 | dsp_map | 10 | 0.00208 | 5 | 5 | 0.492 | 1 |
| aligned3 | R2D2 | additive-lengthscale_r2d2 | oracle_S | 10 | -3.37e-05 | 5 | 4 | 0.91 | 1 |
| aligned3 | R2D2 | product-amplitude_r2d2 | sobol | 10 | -0.356 | 10 | 0 | 0.00195 | 0.0234 |
| aligned3 | R2D2 | product-amplitude_r2d2 | dsp_map | 10 | 0.000197 | 4 | 6 | 0.922 | 1 |
| aligned3 | R2D2 | product-amplitude_r2d2 | oracle_S | 10 | 0.000485 | 5 | 5 | 1 | 1 |
| aligned3 | R2D2 | additive-amplitude_r2d2 | sobol | 10 | -0.359 | 10 | 0 | 0.00195 | 0.0234 |
| aligned3 | R2D2 | additive-amplitude_r2d2 | dsp_map | 10 | 0.00039 | 4 | 6 | 0.557 | 1 |
| aligned3 | R2D2 | additive-amplitude_r2d2 | oracle_S | 10 | -0.000702 | 5 | 4 | 0.734 | 1 |
| aligned10 | R2D2 | product-lengthscale_r2d2 | sobol | 10 | -1.73 | 10 | 0 | 0.00195 | 0.0234 |
| aligned10 | R2D2 | product-lengthscale_r2d2 | dsp_map | 10 | 0.099 | 2 | 8 | 0.0488 | 0.293 |
| aligned10 | R2D2 | product-lengthscale_r2d2 | oracle_S | 10 | 0.17 | 4 | 6 | 0.375 | 0.75 |
| aligned10 | R2D2 | additive-lengthscale_r2d2 | sobol | 10 | -2.25 | 10 | 0 | 0.00195 | 0.0234 |
| aligned10 | R2D2 | additive-lengthscale_r2d2 | dsp_map | 10 | -0.105 | 6 | 4 | 0.375 | 0.75 |
| aligned10 | R2D2 | additive-lengthscale_r2d2 | oracle_S | 10 | -0.335 | 7 | 3 | 0.16 | 0.523 |
| aligned10 | R2D2 | product-amplitude_r2d2 | sobol | 10 | -2.12 | 10 | 0 | 0.00195 | 0.0234 |
| aligned10 | R2D2 | product-amplitude_r2d2 | dsp_map | 10 | -0.202 | 9 | 1 | 0.00391 | 0.0273 |
| aligned10 | R2D2 | product-amplitude_r2d2 | oracle_S | 10 | -0.3 | 8 | 2 | 0.0488 | 0.293 |
| aligned10 | R2D2 | additive-amplitude_r2d2 | sobol | 10 | -2.21 | 10 | 0 | 0.00195 | 0.0234 |
| aligned10 | R2D2 | additive-amplitude_r2d2 | dsp_map | 10 | -0.221 | 10 | 0 | 0.00195 | 0.0234 |
| aligned10 | R2D2 | additive-amplitude_r2d2 | oracle_S | 10 | -0.359 | 8 | 2 | 0.131 | 0.523 |
| anti_aligned | R2D2 | product-lengthscale_r2d2 | sobol | 10 | -1.07 | 10 | 0 | 0.00195 | 0.0234 |
| anti_aligned | R2D2 | product-lengthscale_r2d2 | dsp_map | 10 | 0.0817 | 4 | 6 | 0.322 | 0.523 |
| anti_aligned | R2D2 | product-lengthscale_r2d2 | oracle_S | 10 | 0.133 | 2 | 8 | 0.131 | 0.523 |
| anti_aligned | R2D2 | additive-lengthscale_r2d2 | sobol | 10 | -0.812 | 10 | 0 | 0.00195 | 0.0234 |
| anti_aligned | R2D2 | additive-lengthscale_r2d2 | dsp_map | 10 | 0.187 | 2 | 8 | 0.131 | 0.523 |
| anti_aligned | R2D2 | additive-lengthscale_r2d2 | oracle_S | 10 | 0.0815 | 3 | 7 | 0.193 | 0.523 |
| anti_aligned | R2D2 | product-amplitude_r2d2 | sobol | 10 | -1.31 | 10 | 0 | 0.00195 | 0.0234 |
| anti_aligned | R2D2 | product-amplitude_r2d2 | dsp_map | 10 | -0.129 | 6 | 4 | 0.084 | 0.42 |
| anti_aligned | R2D2 | product-amplitude_r2d2 | oracle_S | 10 | -0.152 | 7 | 3 | 0.0645 | 0.387 |
| anti_aligned | R2D2 | additive-amplitude_r2d2 | sobol | 10 | -1.35 | 10 | 0 | 0.00195 | 0.0234 |
| anti_aligned | R2D2 | additive-amplitude_r2d2 | dsp_map | 10 | -0.121 | 8 | 2 | 0.00977 | 0.0781 |
| anti_aligned | R2D2 | additive-amplitude_r2d2 | oracle_S | 10 | -0.152 | 7 | 3 | 0.0273 | 0.191 |
| decoupled | R2D2 | product-lengthscale_r2d2 | sobol | 10 | -1.18 | 10 | 0 | 0.00195 | 0.0234 |
| decoupled | R2D2 | product-lengthscale_r2d2 | dsp_map | 10 | 0.232 | 4 | 6 | 0.193 | 1 |
| decoupled | R2D2 | product-lengthscale_r2d2 | oracle_S | 10 | 0.121 | 4 | 6 | 0.695 | 1 |
| decoupled | R2D2 | additive-lengthscale_r2d2 | sobol | 10 | -1.25 | 10 | 0 | 0.00195 | 0.0234 |
| decoupled | R2D2 | additive-lengthscale_r2d2 | dsp_map | 10 | 0.0396 | 5 | 5 | 0.922 | 1 |
| decoupled | R2D2 | additive-lengthscale_r2d2 | oracle_S | 10 | -0.196 | 7 | 3 | 0.232 | 1 |
| decoupled | R2D2 | product-amplitude_r2d2 | sobol | 10 | -1.55 | 10 | 0 | 0.00195 | 0.0234 |
| decoupled | R2D2 | product-amplitude_r2d2 | dsp_map | 10 | 0.0112 | 5 | 5 | 0.695 | 1 |
| decoupled | R2D2 | product-amplitude_r2d2 | oracle_S | 10 | -0.135 | 7 | 3 | 0.131 | 1 |
| decoupled | R2D2 | additive-amplitude_r2d2 | sobol | 10 | -1.39 | 10 | 0 | 0.00195 | 0.0234 |
| decoupled | R2D2 | additive-amplitude_r2d2 | dsp_map | 10 | -0.00828 | 5 | 5 | 0.922 | 1 |
| decoupled | R2D2 | additive-amplitude_r2d2 | oracle_S | 10 | -0.135 | 6 | 4 | 0.557 | 1 |
| interaction_g0.00 | R2D2 | product-lengthscale_r2d2 | sobol | 10 | -0.83 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.00 | R2D2 | product-lengthscale_r2d2 | dsp_map | 10 | 0.0288 | 4 | 6 | 0.846 | 1 |
| interaction_g0.00 | R2D2 | product-lengthscale_r2d2 | oracle_S | 10 | 0.0703 | 1 | 8 | 0.0391 | 0.312 |
| interaction_g0.00 | R2D2 | additive-lengthscale_r2d2 | sobol | 10 | -1.02 | 9 | 1 | 0.00391 | 0.0352 |
| interaction_g0.00 | R2D2 | additive-lengthscale_r2d2 | dsp_map | 10 | -0.0359 | 6 | 4 | 0.557 | 1 |
| interaction_g0.00 | R2D2 | additive-lengthscale_r2d2 | oracle_S | 10 | -0.000769 | 5 | 4 | 1 | 1 |
| interaction_g0.00 | R2D2 | product-amplitude_r2d2 | sobol | 10 | -1.03 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.00 | R2D2 | product-amplitude_r2d2 | dsp_map | 10 | -0.0734 | 7 | 3 | 0.0645 | 0.451 |
| interaction_g0.00 | R2D2 | product-amplitude_r2d2 | oracle_S | 10 | -0.00522 | 7 | 2 | 0.426 | 1 |
| interaction_g0.00 | R2D2 | additive-amplitude_r2d2 | sobol | 10 | -1.04 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.00 | R2D2 | additive-amplitude_r2d2 | dsp_map | 10 | 0.000917 | 5 | 5 | 0.625 | 1 |
| interaction_g0.00 | R2D2 | additive-amplitude_r2d2 | oracle_S | 10 | -0.000561 | 5 | 4 | 1 | 1 |
| interaction_g0.25 | R2D2 | product-lengthscale_r2d2 | sobol | 10 | -0.753 | 8 | 2 | 0.084 | 0.504 |
| interaction_g0.25 | R2D2 | product-lengthscale_r2d2 | dsp_map | 10 | 0.0366 | 4 | 6 | 0.492 | 1 |
| interaction_g0.25 | R2D2 | product-lengthscale_r2d2 | oracle_S | 10 | 0.33 | 1 | 8 | 0.00781 | 0.0781 |
| interaction_g0.25 | R2D2 | additive-lengthscale_r2d2 | sobol | 10 | -1.04 | 9 | 1 | 0.00391 | 0.0469 |
| interaction_g0.25 | R2D2 | additive-lengthscale_r2d2 | dsp_map | 10 | -0.0371 | 6 | 4 | 0.492 | 1 |
| interaction_g0.25 | R2D2 | additive-lengthscale_r2d2 | oracle_S | 10 | 0.12 | 1 | 8 | 0.00781 | 0.0781 |
| interaction_g0.25 | R2D2 | product-amplitude_r2d2 | sobol | 10 | -0.968 | 9 | 1 | 0.0195 | 0.156 |
| interaction_g0.25 | R2D2 | product-amplitude_r2d2 | dsp_map | 10 | 0.0317 | 3 | 7 | 0.432 | 1 |
| interaction_g0.25 | R2D2 | product-amplitude_r2d2 | oracle_S | 10 | 0.102 | 1 | 9 | 0.0273 | 0.191 |
| interaction_g0.25 | R2D2 | additive-amplitude_r2d2 | sobol | 10 | -1.07 | 9 | 1 | 0.00391 | 0.0469 |
| interaction_g0.25 | R2D2 | additive-amplitude_r2d2 | dsp_map | 10 | -0.0174 | 6 | 4 | 0.432 | 1 |
| interaction_g0.25 | R2D2 | additive-amplitude_r2d2 | oracle_S | 10 | 0.0427 | 4 | 6 | 0.16 | 0.801 |
| interaction_g0.50 | R2D2 | product-lengthscale_r2d2 | sobol | 10 | -0.776 | 10 | 0 | 0.00195 | 0.0234 |
| interaction_g0.50 | R2D2 | product-lengthscale_r2d2 | dsp_map | 10 | -0.0277 | 6 | 4 | 0.625 | 1 |
| interaction_g0.50 | R2D2 | product-lengthscale_r2d2 | oracle_S | 10 | 0.856 | 1 | 9 | 0.00391 | 0.043 |
| interaction_g0.50 | R2D2 | additive-lengthscale_r2d2 | sobol | 10 | -0.724 | 8 | 2 | 0.00977 | 0.0684 |
| interaction_g0.50 | R2D2 | additive-lengthscale_r2d2 | dsp_map | 10 | -0.00364 | 5 | 5 | 0.922 | 1 |
| interaction_g0.50 | R2D2 | additive-lengthscale_r2d2 | oracle_S | 10 | 0.358 | 1 | 8 | 0.00781 | 0.0625 |
| interaction_g0.50 | R2D2 | product-amplitude_r2d2 | sobol | 10 | -0.862 | 9 | 1 | 0.00391 | 0.043 |
| interaction_g0.50 | R2D2 | product-amplitude_r2d2 | dsp_map | 10 | -0.177 | 7 | 3 | 0.275 | 0.826 |
| interaction_g0.50 | R2D2 | product-amplitude_r2d2 | oracle_S | 10 | 0.622 | 2 | 8 | 0.00977 | 0.0684 |
| interaction_g0.50 | R2D2 | additive-amplitude_r2d2 | sobol | 10 | -1.14 | 9 | 1 | 0.00391 | 0.043 |
| interaction_g0.50 | R2D2 | additive-amplitude_r2d2 | dsp_map | 10 | -0.436 | 7 | 3 | 0.16 | 0.641 |
| interaction_g0.50 | R2D2 | additive-amplitude_r2d2 | oracle_S | 10 | 0.474 | 2 | 8 | 0.0371 | 0.186 |
| interaction_g0.75 | R2D2 | product-lengthscale_r2d2 | sobol | 10 | -0.98 | 8 | 2 | 0.00977 | 0.107 |
| interaction_g0.75 | R2D2 | product-lengthscale_r2d2 | dsp_map | 10 | 0.021 | 4 | 6 | 1 | 1 |
| interaction_g0.75 | R2D2 | product-lengthscale_r2d2 | oracle_S | 10 | 0.925 | 2 | 8 | 0.0137 | 0.137 |
| interaction_g0.75 | R2D2 | additive-lengthscale_r2d2 | sobol | 10 | -0.737 | 7 | 3 | 0.275 | 1 |
| interaction_g0.75 | R2D2 | additive-lengthscale_r2d2 | dsp_map | 10 | 0.26 | 4 | 6 | 0.432 | 1 |
| interaction_g0.75 | R2D2 | additive-lengthscale_r2d2 | oracle_S | 10 | 1.09 | 1 | 8 | 0.00781 | 0.0938 |
| interaction_g0.75 | R2D2 | product-amplitude_r2d2 | sobol | 10 | -0.989 | 8 | 2 | 0.0273 | 0.246 |
| interaction_g0.75 | R2D2 | product-amplitude_r2d2 | dsp_map | 10 | -0.097 | 6 | 4 | 0.557 | 1 |
| interaction_g0.75 | R2D2 | product-amplitude_r2d2 | oracle_S | 10 | 0.579 | 3 | 7 | 0.0273 | 0.246 |
| interaction_g0.75 | R2D2 | additive-amplitude_r2d2 | sobol | 10 | -0.926 | 7 | 3 | 0.105 | 0.633 |
| interaction_g0.75 | R2D2 | additive-amplitude_r2d2 | dsp_map | 10 | -0.229 | 5 | 5 | 0.492 | 1 |
| interaction_g0.75 | R2D2 | additive-amplitude_r2d2 | oracle_S | 10 | 0.491 | 4 | 6 | 0.084 | 0.588 |


Cell-vs-cell (same test):

| family | prior_family | cell | reference | n_seeds | median_diff | n_cell_better | n_ref_better | p_value | p_holm_within_family_kind |
|---|---|---|---|---|---|---|---|---|---|
| aligned3 | HC | product-lengthscale | additive-lengthscale | 10 | 6.69e-05 | 3 | 7 | 0.275 | 1 |
| aligned3 | HC | product-lengthscale | product-amplitude | 10 | 0.000711 | 3 | 7 | 0.16 | 0.961 |
| aligned3 | HC | product-lengthscale | additive-amplitude | 10 | 3.32e-05 | 4 | 6 | 0.275 | 1 |
| aligned3 | HC | additive-lengthscale | product-amplitude | 10 | -4.79e-05 | 5 | 5 | 0.922 | 1 |
| aligned3 | HC | additive-lengthscale | additive-amplitude | 10 | -2.91e-06 | 5 | 5 | 0.922 | 1 |
| aligned3 | HC | product-amplitude | additive-amplitude | 10 | -4.58e-05 | 6 | 4 | 1 | 1 |
| aligned10 | HC | product-lengthscale | additive-lengthscale | 10 | 0.17 | 1 | 9 | 0.0273 | 0.137 |
| aligned10 | HC | product-lengthscale | product-amplitude | 10 | 0.15 | 2 | 8 | 0.0371 | 0.148 |
| aligned10 | HC | product-lengthscale | additive-amplitude | 10 | 0.261 | 0 | 10 | 0.00195 | 0.0117 |
| aligned10 | HC | additive-lengthscale | product-amplitude | 10 | -0.0303 | 7 | 3 | 0.557 | 0.984 |
| aligned10 | HC | additive-lengthscale | additive-amplitude | 10 | 0.00616 | 4 | 6 | 0.492 | 0.984 |
| aligned10 | HC | product-amplitude | additive-amplitude | 10 | 0.0132 | 5 | 5 | 0.232 | 0.697 |
| anti_aligned | HC | product-lengthscale | additive-lengthscale | 10 | 0.217 | 1 | 9 | 0.00391 | 0.0195 |
| anti_aligned | HC | product-lengthscale | product-amplitude | 10 | 0.214 | 0 | 10 | 0.00195 | 0.0117 |
| anti_aligned | HC | product-lengthscale | additive-amplitude | 10 | 0.233 | 1 | 9 | 0.00391 | 0.0195 |
| anti_aligned | HC | additive-lengthscale | product-amplitude | 10 | -0.0288 | 6 | 4 | 1 | 1 |
| anti_aligned | HC | additive-lengthscale | additive-amplitude | 10 | 0.00283 | 4 | 6 | 0.432 | 1 |
| anti_aligned | HC | product-amplitude | additive-amplitude | 10 | 0.00165 | 4 | 6 | 0.375 | 1 |
| decoupled | HC | product-lengthscale | additive-lengthscale | 10 | 0.283 | 0 | 10 | 0.00195 | 0.0117 |
| decoupled | HC | product-lengthscale | product-amplitude | 10 | 0.213 | 1 | 9 | 0.00391 | 0.0156 |
| decoupled | HC | product-lengthscale | additive-amplitude | 10 | 0.278 | 0 | 10 | 0.00195 | 0.0117 |
| decoupled | HC | additive-lengthscale | product-amplitude | 10 | 0.013 | 4 | 6 | 0.492 | 0.863 |
| decoupled | HC | additive-lengthscale | additive-amplitude | 10 | 0.0576 | 1 | 9 | 0.0137 | 0.041 |
| decoupled | HC | product-amplitude | additive-amplitude | 10 | 0.00659 | 4 | 6 | 0.432 | 0.863 |
| interaction_g0.00 | HC | product-lengthscale | additive-lengthscale | 10 | 0.00887 | 2 | 8 | 0.0273 | 0.137 |
| interaction_g0.00 | HC | product-lengthscale | product-amplitude | 10 | 0.00744 | 2 | 8 | 0.00977 | 0.0586 |
| interaction_g0.00 | HC | product-lengthscale | additive-amplitude | 10 | 0.00721 | 2 | 8 | 0.0645 | 0.258 |
| interaction_g0.00 | HC | additive-lengthscale | product-amplitude | 10 | 0.000207 | 5 | 5 | 0.922 | 1 |
| interaction_g0.00 | HC | additive-lengthscale | additive-amplitude | 10 | -0.000391 | 7 | 3 | 0.557 | 1 |
| interaction_g0.00 | HC | product-amplitude | additive-amplitude | 10 | -0.00133 | 6 | 4 | 0.625 | 1 |
| interaction_g0.25 | HC | product-lengthscale | additive-lengthscale | 10 | -0.0165 | 8 | 2 | 0.0645 | 0.195 |
| interaction_g0.25 | HC | product-lengthscale | product-amplitude | 10 | -0.000558 | 6 | 4 | 0.322 | 0.645 |
| interaction_g0.25 | HC | product-lengthscale | additive-amplitude | 10 | -0.00789 | 8 | 2 | 0.0488 | 0.195 |
| interaction_g0.25 | HC | additive-lengthscale | product-amplitude | 10 | 0.00916 | 2 | 8 | 0.00977 | 0.0586 |
| interaction_g0.25 | HC | additive-lengthscale | additive-amplitude | 10 | -0.000604 | 6 | 4 | 1 | 1 |
| interaction_g0.25 | HC | product-amplitude | additive-amplitude | 10 | -0.0105 | 8 | 2 | 0.0137 | 0.0684 |
| interaction_g0.50 | HC | product-lengthscale | additive-lengthscale | 10 | -0.0246 | 5 | 5 | 0.322 | 0.967 |
| interaction_g0.50 | HC | product-lengthscale | product-amplitude | 10 | 0.00878 | 4 | 5 | 0.734 | 0.967 |
| interaction_g0.50 | HC | product-lengthscale | additive-amplitude | 10 | -0.343 | 8 | 2 | 0.0371 | 0.223 |
| interaction_g0.50 | HC | additive-lengthscale | product-amplitude | 10 | 0.0301 | 3 | 7 | 0.131 | 0.654 |
| interaction_g0.50 | HC | additive-lengthscale | additive-amplitude | 10 | -0.0235 | 6 | 4 | 0.375 | 0.967 |
| interaction_g0.50 | HC | product-amplitude | additive-amplitude | 10 | -0.116 | 7 | 3 | 0.131 | 0.654 |
| interaction_g0.75 | HC | product-lengthscale | additive-lengthscale | 10 | -0.23 | 5 | 4 | 0.359 | 1 |
| interaction_g0.75 | HC | product-lengthscale | product-amplitude | 10 | 0.103 | 1 | 9 | 0.0488 | 0.195 |
| interaction_g0.75 | HC | product-lengthscale | additive-amplitude | 10 | -0.418 | 5 | 5 | 0.432 | 1 |
| interaction_g0.75 | HC | additive-lengthscale | product-amplitude | 10 | 0.335 | 1 | 9 | 0.00977 | 0.0586 |
| interaction_g0.75 | HC | additive-lengthscale | additive-amplitude | 10 | -0.00798 | 6 | 4 | 0.432 | 1 |
| interaction_g0.75 | HC | product-amplitude | additive-amplitude | 10 | -0.386 | 8 | 2 | 0.0273 | 0.137 |
| aligned3 | R2D2 | product-lengthscale_r2d2 | additive-lengthscale_r2d2 | 10 | 0.00296 | 3 | 5 | 0.25 | 0.75 |
| aligned3 | R2D2 | product-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.0366 | 3 | 7 | 0.0488 | 0.293 |
| aligned3 | R2D2 | product-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.0304 | 3 | 6 | 0.164 | 0.656 |
| aligned3 | R2D2 | additive-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.000491 | 4 | 6 | 0.432 | 0.863 |
| aligned3 | R2D2 | additive-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.000994 | 3 | 6 | 0.0977 | 0.488 |
| aligned3 | R2D2 | product-amplitude_r2d2 | additive-amplitude_r2d2 | 10 | 0.00101 | 3 | 7 | 0.77 | 0.863 |
| aligned10 | R2D2 | product-lengthscale_r2d2 | additive-lengthscale_r2d2 | 10 | 0.382 | 2 | 8 | 0.0645 | 0.258 |
| aligned10 | R2D2 | product-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.496 | 0 | 10 | 0.00195 | 0.0117 |
| aligned10 | R2D2 | product-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.423 | 0 | 10 | 0.00195 | 0.0117 |
| aligned10 | R2D2 | additive-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.048 | 4 | 6 | 0.557 | 0.557 |
| aligned10 | R2D2 | additive-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.061 | 2 | 8 | 0.16 | 0.48 |
| aligned10 | R2D2 | product-amplitude_r2d2 | additive-amplitude_r2d2 | 10 | 0.0156 | 3 | 7 | 0.193 | 0.48 |
| anti_aligned | R2D2 | product-lengthscale_r2d2 | additive-lengthscale_r2d2 | 10 | 0.0121 | 4 | 6 | 1 | 1 |
| anti_aligned | R2D2 | product-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.242 | 2 | 8 | 0.00977 | 0.0391 |
| anti_aligned | R2D2 | product-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.241 | 0 | 10 | 0.00195 | 0.0117 |
| anti_aligned | R2D2 | additive-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.179 | 3 | 7 | 0.0488 | 0.146 |
| anti_aligned | R2D2 | additive-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.248 | 1 | 9 | 0.00586 | 0.0293 |
| anti_aligned | R2D2 | product-amplitude_r2d2 | additive-amplitude_r2d2 | 10 | 0.0191 | 3 | 7 | 0.16 | 0.32 |
| decoupled | R2D2 | product-lengthscale_r2d2 | additive-lengthscale_r2d2 | 10 | 0.254 | 3 | 7 | 0.0645 | 0.322 |
| decoupled | R2D2 | product-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.361 | 2 | 8 | 0.0371 | 0.223 |
| decoupled | R2D2 | product-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.264 | 3 | 7 | 0.105 | 0.422 |
| decoupled | R2D2 | additive-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.0791 | 3 | 7 | 0.232 | 0.465 |
| decoupled | R2D2 | additive-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | -0.0655 | 6 | 4 | 0.77 | 0.77 |
| decoupled | R2D2 | product-amplitude_r2d2 | additive-amplitude_r2d2 | 10 | -0.0288 | 7 | 3 | 0.131 | 0.422 |
| interaction_g0.00 | R2D2 | product-lengthscale_r2d2 | additive-lengthscale_r2d2 | 10 | 0.0465 | 2 | 7 | 0.301 | 1 |
| interaction_g0.00 | R2D2 | product-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.065 | 1 | 8 | 0.0273 | 0.137 |
| interaction_g0.00 | R2D2 | product-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.0275 | 4 | 5 | 0.359 | 1 |
| interaction_g0.00 | R2D2 | additive-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.0209 | 1 | 8 | 0.0195 | 0.117 |
| interaction_g0.00 | R2D2 | additive-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.0201 | 3 | 6 | 0.57 | 1 |
| interaction_g0.00 | R2D2 | product-amplitude_r2d2 | additive-amplitude_r2d2 | 10 | -0.00187 | 6 | 3 | 0.426 | 1 |
| interaction_g0.25 | R2D2 | product-lengthscale_r2d2 | additive-lengthscale_r2d2 | 10 | 0.118 | 1 | 7 | 0.0547 | 0.219 |
| interaction_g0.25 | R2D2 | product-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.136 | 2 | 8 | 0.0195 | 0.0977 |
| interaction_g0.25 | R2D2 | product-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.244 | 1 | 9 | 0.0137 | 0.082 |
| interaction_g0.25 | R2D2 | additive-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.00333 | 4 | 6 | 0.846 | 0.863 |
| interaction_g0.25 | R2D2 | additive-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.0403 | 4 | 6 | 0.432 | 0.863 |
| interaction_g0.25 | R2D2 | product-amplitude_r2d2 | additive-amplitude_r2d2 | 10 | 0.0392 | 2 | 8 | 0.084 | 0.252 |
| interaction_g0.50 | R2D2 | product-lengthscale_r2d2 | additive-lengthscale_r2d2 | 10 | -0.00634 | 5 | 5 | 1 | 1 |
| interaction_g0.50 | R2D2 | product-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.0813 | 5 | 5 | 0.625 | 1 |
| interaction_g0.50 | R2D2 | product-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.229 | 4 | 6 | 0.232 | 1 |
| interaction_g0.50 | R2D2 | additive-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.00546 | 4 | 6 | 0.922 | 1 |
| interaction_g0.50 | R2D2 | additive-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.27 | 4 | 6 | 0.432 | 1 |
| interaction_g0.50 | R2D2 | product-amplitude_r2d2 | additive-amplitude_r2d2 | 10 | 0.00464 | 4 | 6 | 0.432 | 1 |
| interaction_g0.75 | R2D2 | product-lengthscale_r2d2 | additive-lengthscale_r2d2 | 10 | -0.54 | 7 | 3 | 0.16 | 0.641 |
| interaction_g0.75 | R2D2 | product-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.0437 | 4 | 6 | 0.77 | 1 |
| interaction_g0.75 | R2D2 | product-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | -0.00738 | 5 | 5 | 0.77 | 1 |
| interaction_g0.75 | R2D2 | additive-lengthscale_r2d2 | product-amplitude_r2d2 | 10 | 0.262 | 1 | 8 | 0.0742 | 0.445 |
| interaction_g0.75 | R2D2 | additive-lengthscale_r2d2 | additive-amplitude_r2d2 | 10 | 0.81 | 1 | 9 | 0.084 | 0.445 |
| interaction_g0.75 | R2D2 | product-amplitude_r2d2 | additive-amplitude_r2d2 | 10 | 0.0775 | 3 | 7 | 0.695 | 1 |


## 5. Effects: structure S, parameterization P, twin pairs Delta_c, and their prior-family differences

Per (family, seed) on y = log10(max(r_199, 1e-8)): S = 1/2[(y_AA - y_PA) + (y_AL - y_PL)] (additive minus product), P = 1/2[(y_AA - y_AL) + (y_PA - y_PL)] (amplitude minus lengthscale), within each prior family; Delta_c = y_c(R2-D2) - y_c(HC twin); dS = S(R2D2) - S(HC), dP likewise. Negative = lower regret for the first-named side. Per family: two-sided Wilcoxon over seeds, Holm within family across the effects of one kind (main_HC: S_HC, P_HC; main_R2D2: S_R2D2, P_R2D2; interaction: dS, dP; twin: the four Delta_c); lo/hi are the seed bootstrap's percentile intervals (10,000 resamples). Pooled: the (family, seed)-cluster bootstrap of the mean.


Pooled over families:

| effect | kind | n_units | median | mean | lo90 | hi90 | lo95 | hi95 | pr2_reading |
|---|---|---|---|---|---|---|---|---|---|
| S_HC | main_HC | 80 | -0.00917 | 0.142 | -0.0666 | 0.356 | -0.102 | 0.398 |  |
| P_HC | main_HC | 80 | -0.339 | -0.397 | -0.55 | -0.244 | -0.585 | -0.214 |  |
| S_R2D2 | main_R2D2 | 80 | -0.121 | -0.218 | -0.399 | -0.0394 | -0.438 | -0.0015 |  |
| P_R2D2 | main_R2D2 | 80 | -0.369 | -0.482 | -0.676 | -0.293 | -0.715 | -0.261 |  |
| dS | interaction | 80 | 0.00339 | -0.36 | -0.666 | -0.0762 | -0.736 | -0.0236 | inconclusive |
| dP | interaction | 80 | -0.103 | -0.0843 | -0.311 | 0.138 | -0.354 | 0.181 | inconclusive |
| Delta_product-lengthscale_r2d2 | twin | 80 | 0.492 | 0.803 | 0.59 | 1.02 | 0.551 | 1.07 |  |
| Delta_additive-lengthscale_r2d2 | twin | 80 | 0.508 | 0.696 | 0.503 | 0.889 | 0.47 | 0.929 |  |
| Delta_product-amplitude_r2d2 | twin | 80 | 0.502 | 0.973 | 0.686 | 1.27 | 0.631 | 1.33 |  |
| Delta_additive-amplitude_r2d2 | twin | 80 | 0.364 | 0.358 | 0.0222 | 0.674 | -0.052 | 0.723 |  |


Per family:

| family | effect | n | median | mean | lo90 | hi90 | p_value | p_holm_within_family_kind | pr2_reading |
|---|---|---|---|---|---|---|---|---|---|
| aligned3 | S_HC | 10 | -0.293 | -0.397 | -1.09 | 0.176 | 0.695 | 0.695 |  |
| aligned3 | P_HC | 10 | -0.642 | -0.549 | -1.04 | -0.0627 | 0.105 | 0.211 |  |
| aligned3 | S_R2D2 | 10 | -0.331 | -0.172 | -1.05 | 0.68 | 0.846 | 0.846 |  |
| aligned3 | P_R2D2 | 10 | -0.821 | -0.797 | -1.65 | 0.0296 | 0.193 | 0.387 |  |
| aligned3 | dS | 10 | 0.0596 | 0.225 | -0.448 | 0.918 | 0.77 | 1 | inconclusive |
| aligned3 | dP | 10 | -0.212 | -0.248 | -0.782 | 0.274 | 0.695 | 1 | inconclusive |
| aligned3 | Delta_product-lengthscale_r2d2 | 10 | 1.22 | 0.991 | 0.378 | 1.62 | 0.0391 | 0.117 |  |
| aligned3 | Delta_additive-lengthscale_r2d2 | 10 | 1.06 | 1.32 | 0.592 | 2.12 | 0.0137 | 0.0547 |  |
| aligned3 | Delta_product-amplitude_r2d2 | 10 | 1.11 | 0.847 | 0.0664 | 1.55 | 0.105 | 0.211 |  |
| aligned3 | Delta_additive-amplitude_r2d2 | 10 | 0.84 | 0.968 | 0.208 | 1.74 | 0.105 | 0.211 |  |
| aligned10 | S_HC | 10 | -0.427 | -0.453 | -0.645 | -0.255 | 0.00977 | 0.0195 |  |
| aligned10 | P_HC | 10 | -0.186 | -0.399 | -0.684 | -0.135 | 0.0371 | 0.0371 |  |
| aligned10 | S_R2D2 | 10 | -0.384 | -0.391 | -0.564 | -0.211 | 0.00977 | 0.0195 |  |
| aligned10 | P_R2D2 | 10 | -0.677 | -0.525 | -0.776 | -0.245 | 0.0195 | 0.0195 |  |
| aligned10 | dS | 10 | 0.0111 | 0.0619 | -0.124 | 0.261 | 0.492 | 0.492 | equivalent |
| aligned10 | dP | 10 | -0.308 | -0.126 | -0.476 | 0.324 | 0.16 | 0.32 | inconclusive |
| aligned10 | Delta_product-lengthscale_r2d2 | 10 | 0.266 | 0.372 | 0.151 | 0.607 | 0.084 | 0.252 |  |
| aligned10 | Delta_additive-lengthscale_r2d2 | 10 | 0.4 | 0.488 | 0.116 | 0.885 | 0.084 | 0.252 |  |
| aligned10 | Delta_product-amplitude_r2d2 | 10 | 0.0273 | 0.301 | -0.0294 | 0.687 | 0.432 | 0.432 |  |
| aligned10 | Delta_additive-amplitude_r2d2 | 10 | 0.147 | 0.308 | 0.0502 | 0.632 | 0.0488 | 0.195 |  |
| anti_aligned | S_HC | 10 | -0.391 | -0.39 | -0.598 | -0.189 | 0.0137 | 0.0273 |  |
| anti_aligned | P_HC | 10 | -0.195 | -0.466 | -0.786 | -0.187 | 0.0137 | 0.0273 |  |
| anti_aligned | S_R2D2 | 10 | -0.0413 | -0.0453 | -0.194 | 0.0948 | 0.695 | 0.695 |  |
| anti_aligned | P_R2D2 | 10 | -0.417 | -0.465 | -0.665 | -0.275 | 0.00195 | 0.00391 |  |
| anti_aligned | dS | 10 | 0.306 | 0.345 | 0.0692 | 0.604 | 0.084 | 0.168 | inconclusive |
| anti_aligned | dP | 10 | 0.0234 | 0.00119 | -0.257 | 0.263 | 1 | 1 | equivalent |
| anti_aligned | Delta_product-lengthscale_r2d2 | 10 | 0.0241 | 0.0713 | -0.0219 | 0.165 | 0.131 | 0.131 |  |
| anti_aligned | Delta_additive-lengthscale_r2d2 | 10 | 0.375 | 0.611 | 0.313 | 0.937 | 0.00586 | 0.0234 |  |
| anti_aligned | Delta_product-amplitude_r2d2 | 10 | 0.142 | 0.268 | 0.0686 | 0.541 | 0.0488 | 0.0977 |  |
| anti_aligned | Delta_additive-amplitude_r2d2 | 10 | 0.324 | 0.417 | 0.211 | 0.648 | 0.00977 | 0.0293 |  |
| decoupled | S_HC | 10 | -0.247 | -0.323 | -0.523 | -0.145 | 0.0195 | 0.0195 |  |
| decoupled | P_HC | 10 | -0.459 | -0.488 | -0.614 | -0.36 | 0.00195 | 0.00391 |  |
| decoupled | S_R2D2 | 10 | 0.0436 | 0.0867 | -0.137 | 0.372 | 1 | 1 |  |
| decoupled | P_R2D2 | 10 | -0.277 | -0.37 | -0.711 | -0.0525 | 0.131 | 0.262 |  |
| decoupled | dS | 10 | 0.242 | 0.41 | 0.0926 | 0.739 | 0.105 | 0.211 | inconclusive |
| decoupled | dP | 10 | 0.125 | 0.118 | -0.236 | 0.472 | 0.557 | 0.557 | inconclusive |
| decoupled | Delta_product-lengthscale_r2d2 | 10 | 0.0197 | 0.242 | -0.00369 | 0.497 | 0.432 | 0.645 |  |
| decoupled | Delta_additive-lengthscale_r2d2 | 10 | 0.161 | 0.299 | -0.0256 | 0.604 | 0.193 | 0.58 |  |
| decoupled | Delta_product-amplitude_r2d2 | 10 | 0.138 | 0.00696 | -0.617 | 0.498 | 0.322 | 0.645 |  |
| decoupled | Delta_additive-amplitude_r2d2 | 10 | 1.02 | 0.769 | 0.359 | 1.11 | 0.0195 | 0.0781 |  |
| interaction_g0.00 | S_HC | 10 | -0.178 | -0.285 | -0.615 | 0.0201 | 0.322 | 0.322 |  |
| interaction_g0.00 | P_HC | 10 | -0.322 | -0.251 | -0.474 | -0.0185 | 0.131 | 0.262 |  |
| interaction_g0.00 | S_R2D2 | 10 | -0.016 | -0.101 | -0.327 | 0.105 | 0.652 | 0.652 |  |
| interaction_g0.00 | P_R2D2 | 10 | -0.3 | -0.394 | -0.611 | -0.182 | 0.0273 | 0.0547 |  |
| interaction_g0.00 | dS | 10 | 0.0752 | 0.183 | -0.148 | 0.496 | 0.16 | 0.32 | inconclusive |
| interaction_g0.00 | dP | 10 | -0.0283 | -0.143 | -0.516 | 0.197 | 0.922 | 0.922 | inconclusive |
| interaction_g0.00 | Delta_product-lengthscale_r2d2 | 10 | 0.693 | 0.893 | 0.516 | 1.29 | 0.00391 | 0.00781 |  |
| interaction_g0.00 | Delta_additive-lengthscale_r2d2 | 10 | 1.22 | 1.16 | 0.819 | 1.5 | 0.00195 | 0.00781 |  |
| interaction_g0.00 | Delta_product-amplitude_r2d2 | 10 | 0.57 | 0.833 | 0.359 | 1.34 | 0.0195 | 0.0195 |  |
| interaction_g0.00 | Delta_additive-amplitude_r2d2 | 10 | 0.807 | 0.934 | 0.557 | 1.38 | 0.00195 | 0.00781 |  |
| interaction_g0.25 | S_HC | 10 | 0.765 | 1.06 | 0.468 | 1.87 | 0.00977 | 0.0195 |  |
| interaction_g0.25 | P_HC | 10 | -0.204 | 0.17 | -0.234 | 0.59 | 0.557 | 0.557 |  |
| interaction_g0.25 | S_R2D2 | 10 | -0.398 | -0.435 | -1.18 | 0.285 | 0.0645 | 0.129 |  |
| interaction_g0.25 | P_R2D2 | 10 | -0.296 | -0.391 | -1.18 | 0.319 | 0.375 | 0.375 |  |
| interaction_g0.25 | dS | 10 | -1.16 | -1.49 | -3.04 | -0.301 | 0.0645 | 0.129 | prior family changes the effect |
| interaction_g0.25 | dP | 10 | -0.266 | -0.56 | -1.37 | 0.203 | 0.322 | 0.322 | inconclusive |
| interaction_g0.25 | Delta_product-lengthscale_r2d2 | 10 | 2.35 | 2.13 | 1.24 | 2.93 | 0.00586 | 0.0176 |  |
| interaction_g0.25 | Delta_additive-lengthscale_r2d2 | 10 | 0.432 | 1.02 | 0.441 | 1.67 | 0.00977 | 0.0195 |  |
| interaction_g0.25 | Delta_product-amplitude_r2d2 | 10 | 1.83 | 1.95 | 1.29 | 2.71 | 0.00391 | 0.0156 |  |
| interaction_g0.25 | Delta_additive-amplitude_r2d2 | 10 | 0.576 | 0.0732 | -1.57 | 1.31 | 0.193 | 0.193 |  |
| interaction_g0.50 | S_HC | 10 | 0.939 | 0.974 | 0.3 | 1.61 | 0.0488 | 0.0977 |  |
| interaction_g0.50 | P_HC | 10 | -0.238 | -0.417 | -1.07 | 0.224 | 0.375 | 0.375 |  |
| interaction_g0.50 | S_R2D2 | 10 | -0.25 | -0.447 | -0.929 | 0.00215 | 0.322 | 0.645 |  |
| interaction_g0.50 | P_R2D2 | 10 | -0.0536 | -0.103 | -0.621 | 0.476 | 0.695 | 0.695 |  |
| interaction_g0.50 | dS | 10 | -0.974 | -1.42 | -2.41 | -0.51 | 0.0371 | 0.0742 | prior family changes the effect |
| interaction_g0.50 | dP | 10 | -0.313 | 0.314 | -0.535 | 1.28 | 1 | 1 | inconclusive |
| interaction_g0.50 | Delta_product-lengthscale_r2d2 | 10 | 1.46 | 1.3 | 0.887 | 1.71 | 0.00977 | 0.0391 |  |
| interaction_g0.50 | Delta_additive-lengthscale_r2d2 | 10 | 0.106 | 0.0807 | -0.704 | 0.696 | 0.232 | 0.465 |  |
| interaction_g0.50 | Delta_product-amplitude_r2d2 | 10 | 0.666 | 1.82 | 0.717 | 3.01 | 0.0273 | 0.082 |  |
| interaction_g0.50 | Delta_additive-amplitude_r2d2 | 10 | 0.00147 | 0.198 | -0.631 | 1.17 | 0.695 | 0.695 |  |
| interaction_g0.75 | S_HC | 10 | 1.12 | 0.953 | 0.376 | 1.49 | 0.0371 | 0.0742 |  |
| interaction_g0.75 | P_HC | 10 | -0.439 | -0.777 | -1.4 | -0.229 | 0.0488 | 0.0742 |  |
| interaction_g0.75 | S_R2D2 | 10 | 0.0851 | -0.24 | -0.902 | 0.313 | 0.922 | 0.922 |  |
| interaction_g0.75 | P_R2D2 | 10 | -0.468 | -0.808 | -1.58 | -0.184 | 0.084 | 0.168 |  |
| interaction_g0.75 | dS | 10 | -1.41 | -1.19 | -2.09 | -0.29 | 0.084 | 0.168 | inconclusive |
| interaction_g0.75 | dP | 10 | 0.294 | -0.0309 | -1.07 | 0.877 | 0.846 | 0.846 | inconclusive |
| interaction_g0.75 | Delta_product-lengthscale_r2d2 | 10 | 0.542 | 0.42 | -0.327 | 1.15 | 0.432 | 0.863 |  |
| interaction_g0.75 | Delta_additive-lengthscale_r2d2 | 10 | 0.249 | 0.596 | 0.3 | 0.906 | 0.00391 | 0.0156 |  |
| interaction_g0.75 | Delta_product-amplitude_r2d2 | 10 | 1.27 | 1.76 | 0.636 | 2.96 | 0.0371 | 0.111 |  |
| interaction_g0.75 | Delta_additive-amplitude_r2d2 | 10 | 0.0207 | -0.804 | -2.4 | 0.449 | 1 | 1 |  |


## 6. Preregistered predictions PR1-PR4

Scope: the full replicate (8 families x 10 seeds, every R2-D2 run complete): this is the confirmatory analysis.


**PR1** (wherever S(HC) or P(HC) is significant, Holm p < 0.05, the R2-D2 effect has the same sign; sign of the median over seeds): tested in 7 (family, effect); holds in 5, fails in 2.

| family | effect | n_HC | median_HC | p_holm_HC | significant_HC | n_R2D2 | median_R2D2 | holds |
|---|---|---|---|---|---|---|---|---|
| aligned3 | S | 10 | -0.293 | 0.695 | False | 10 | -0.331 | not tested |
| aligned3 | P | 10 | -0.642 | 0.211 | False | 10 | -0.821 | not tested |
| aligned10 | S | 10 | -0.427 | 0.0195 | True | 10 | -0.384 | yes |
| aligned10 | P | 10 | -0.186 | 0.0371 | True | 10 | -0.677 | yes |
| anti_aligned | S | 10 | -0.391 | 0.0273 | True | 10 | -0.0413 | yes |
| anti_aligned | P | 10 | -0.195 | 0.0273 | True | 10 | -0.417 | yes |
| decoupled | S | 10 | -0.247 | 0.0195 | True | 10 | 0.0436 | no |
| decoupled | P | 10 | -0.459 | 0.00391 | True | 10 | -0.277 | yes |
| interaction_g0.00 | S | 10 | -0.178 | 0.322 | False | 10 | -0.016 | not tested |
| interaction_g0.00 | P | 10 | -0.322 | 0.262 | False | 10 | -0.3 | not tested |
| interaction_g0.25 | S | 10 | 0.765 | 0.0195 | True | 10 | -0.398 | no |
| interaction_g0.25 | P | 10 | -0.204 | 0.557 | False | 10 | -0.296 | not tested |
| interaction_g0.50 | S | 10 | 0.939 | 0.0977 | False | 10 | -0.25 | not tested |
| interaction_g0.50 | P | 10 | -0.238 | 0.375 | False | 10 | -0.0536 | not tested |
| interaction_g0.75 | S | 10 | 1.12 | 0.0742 | False | 10 | 0.0851 | not tested |
| interaction_g0.75 | P | 10 | -0.439 | 0.0742 | False | 10 | -0.468 | not tested |


**PR2** (|dS| and |dP| below 0.3; pooled 90 % cluster-bootstrap interval inside (-0.3, 0.3) = equivalent, entirely outside = prior family changes the effect, otherwise inconclusive): dS: mean -0.36, 90 % [-0.666, -0.0762] -> inconclusive; dP: mean -0.0843, 90 % [-0.311, 0.138] -> inconclusive. Per-family readings are in section 5.


**PR3** (Q6's null: |median Delta_c| < 0.3 for every cell and family): holds in 12 of 32 (cell, family); does not hold overall.

| family | effect | n | median | mean | lo90 | hi90 | p_value | p_holm_within_family_kind | holds |
|---|---|---|---|---|---|---|---|---|---|
| aligned3 | Delta_product-lengthscale_r2d2 | 10 | 1.22 | 0.991 | 0.378 | 1.62 | 0.0391 | 0.117 | False |
| aligned3 | Delta_additive-lengthscale_r2d2 | 10 | 1.06 | 1.32 | 0.592 | 2.12 | 0.0137 | 0.0547 | False |
| aligned3 | Delta_product-amplitude_r2d2 | 10 | 1.11 | 0.847 | 0.0664 | 1.55 | 0.105 | 0.211 | False |
| aligned3 | Delta_additive-amplitude_r2d2 | 10 | 0.84 | 0.968 | 0.208 | 1.74 | 0.105 | 0.211 | False |
| aligned10 | Delta_product-lengthscale_r2d2 | 10 | 0.266 | 0.372 | 0.151 | 0.607 | 0.084 | 0.252 | True |
| aligned10 | Delta_additive-lengthscale_r2d2 | 10 | 0.4 | 0.488 | 0.116 | 0.885 | 0.084 | 0.252 | False |
| aligned10 | Delta_product-amplitude_r2d2 | 10 | 0.0273 | 0.301 | -0.0294 | 0.687 | 0.432 | 0.432 | True |
| aligned10 | Delta_additive-amplitude_r2d2 | 10 | 0.147 | 0.308 | 0.0502 | 0.632 | 0.0488 | 0.195 | True |
| anti_aligned | Delta_product-lengthscale_r2d2 | 10 | 0.0241 | 0.0713 | -0.0219 | 0.165 | 0.131 | 0.131 | True |
| anti_aligned | Delta_additive-lengthscale_r2d2 | 10 | 0.375 | 0.611 | 0.313 | 0.937 | 0.00586 | 0.0234 | False |
| anti_aligned | Delta_product-amplitude_r2d2 | 10 | 0.142 | 0.268 | 0.0686 | 0.541 | 0.0488 | 0.0977 | True |
| anti_aligned | Delta_additive-amplitude_r2d2 | 10 | 0.324 | 0.417 | 0.211 | 0.648 | 0.00977 | 0.0293 | False |
| decoupled | Delta_product-lengthscale_r2d2 | 10 | 0.0197 | 0.242 | -0.00369 | 0.497 | 0.432 | 0.645 | True |
| decoupled | Delta_additive-lengthscale_r2d2 | 10 | 0.161 | 0.299 | -0.0256 | 0.604 | 0.193 | 0.58 | True |
| decoupled | Delta_product-amplitude_r2d2 | 10 | 0.138 | 0.00696 | -0.617 | 0.498 | 0.322 | 0.645 | True |
| decoupled | Delta_additive-amplitude_r2d2 | 10 | 1.02 | 0.769 | 0.359 | 1.11 | 0.0195 | 0.0781 | False |
| interaction_g0.00 | Delta_product-lengthscale_r2d2 | 10 | 0.693 | 0.893 | 0.516 | 1.29 | 0.00391 | 0.00781 | False |
| interaction_g0.00 | Delta_additive-lengthscale_r2d2 | 10 | 1.22 | 1.16 | 0.819 | 1.5 | 0.00195 | 0.00781 | False |
| interaction_g0.00 | Delta_product-amplitude_r2d2 | 10 | 0.57 | 0.833 | 0.359 | 1.34 | 0.0195 | 0.0195 | False |
| interaction_g0.00 | Delta_additive-amplitude_r2d2 | 10 | 0.807 | 0.934 | 0.557 | 1.38 | 0.00195 | 0.00781 | False |
| interaction_g0.25 | Delta_product-lengthscale_r2d2 | 10 | 2.35 | 2.13 | 1.24 | 2.93 | 0.00586 | 0.0176 | False |
| interaction_g0.25 | Delta_additive-lengthscale_r2d2 | 10 | 0.432 | 1.02 | 0.441 | 1.67 | 0.00977 | 0.0195 | False |
| interaction_g0.25 | Delta_product-amplitude_r2d2 | 10 | 1.83 | 1.95 | 1.29 | 2.71 | 0.00391 | 0.0156 | False |
| interaction_g0.25 | Delta_additive-amplitude_r2d2 | 10 | 0.576 | 0.0732 | -1.57 | 1.31 | 0.193 | 0.193 | False |
| interaction_g0.50 | Delta_product-lengthscale_r2d2 | 10 | 1.46 | 1.3 | 0.887 | 1.71 | 0.00977 | 0.0391 | False |
| interaction_g0.50 | Delta_additive-lengthscale_r2d2 | 10 | 0.106 | 0.0807 | -0.704 | 0.696 | 0.232 | 0.465 | True |
| interaction_g0.50 | Delta_product-amplitude_r2d2 | 10 | 0.666 | 1.82 | 0.717 | 3.01 | 0.0273 | 0.082 | False |
| interaction_g0.50 | Delta_additive-amplitude_r2d2 | 10 | 0.00147 | 0.198 | -0.631 | 1.17 | 0.695 | 0.695 | True |
| interaction_g0.75 | Delta_product-lengthscale_r2d2 | 10 | 0.542 | 0.42 | -0.327 | 1.15 | 0.432 | 0.863 | False |
| interaction_g0.75 | Delta_additive-lengthscale_r2d2 | 10 | 0.249 | 0.596 | 0.3 | 0.906 | 0.00391 | 0.0156 | True |
| interaction_g0.75 | Delta_product-amplitude_r2d2 | 10 | 1.27 | 1.76 | 0.636 | 2.96 | 0.0371 | 0.111 | False |
| interaction_g0.75 | Delta_additive-amplitude_r2d2 | 10 | 0.0207 | -0.804 | -2.4 | 0.449 | 1 | 1 | True |


**PR4** (secondary: AP by native_median and by sobol_hat within 0.05 of the twin's; per the prereg's clarification, the (family, seed)-cluster bootstrap 90 % interval (10,000 resamples) of the median paired difference, R2-D2 minus twin, inside (-0.05, 0.05) = equivalent, entirely outside = differs, otherwise inconclusive; lo90/hi90 are that interval): equivalent 0, differs 4, inconclusive 4 of 8 (cell, metric).

| cell | twin | metric | n_pairs | mean_diff | median_diff | lo90 | hi90 | reading |
|---|---|---|---|---|---|---|---|---|
| product-lengthscale_r2d2 | product-lengthscale | ap_native | 80 | -0.187 | -0.15 | -0.195 | -0.121 | differs |
| product-lengthscale_r2d2 | product-lengthscale | ap_sobol | 80 | -0.241 | -0.223 | -0.273 | -0.155 | differs |
| additive-lengthscale_r2d2 | additive-lengthscale | ap_native | 80 | -0.182 | -0.173 | -0.225 | -0.128 | differs |
| additive-lengthscale_r2d2 | additive-lengthscale | ap_sobol | 80 | -0.234 | -0.188 | -0.248 | -0.135 | differs |
| product-amplitude_r2d2 | product-amplitude | ap_native | 80 | -0.0718 | -0.0659 | -0.0819 | -0.0333 | inconclusive |
| product-amplitude_r2d2 | product-amplitude | ap_sobol | 80 | -0.0908 | -0.0707 | -0.0898 | -0.0362 | inconclusive |
| additive-amplitude_r2d2 | additive-amplitude | ap_native | 80 | -0.0789 | -0.0284 | -0.0563 | -0.00207 | inconclusive |
| additive-amplitude_r2d2 | additive-amplitude | ap_sobol | 80 | -0.0828 | -0.0581 | -0.0833 | -0.0238 | inconclusive |


## 7. Identification of the active set

Scored at the LAST readout of each run (coords.csv rows are written every iteration; sobol_hat is computed at t = 25, 50, ..., 175 and 199). `native_median` is the cell's own sparsity parameter (rho_i = 1/ell_i^2 for the lengthscale cells, a_sq_i for the amplitude cells; dsp_map: its MAP rho), so higher = more active in every cell. AP = average precision of ranking the 100 coordinates by the score; recall@|S| = fraction of the true S inside the top-|S| coordinates. Thresholded metrics use p_active > 0.5. oracle_S is included as a metric check (must be perfect); sobol has no model and no readout.


![identification](figures/identification.png)

![identification vs t](figures/identification_vs_t.png)


By prior family (each cell under HC and R2-D2, pooled over families, complete runs):

| structure | parameterization | prior_family | n_runs | ap_native_mean | ap_native_median | ap_sobol_mean | ap_sobol_median | f1_mean | n_pred_active_median |
|---|---|---|---|---|---|---|---|---|---|
| additive | amplitude | HC | 80 | 0.911 | 1 | 0.941 | 1 | 0.84 | 5 |
| additive | amplitude | R2D2 | 80 | 0.832 | 0.89 | 0.859 | 0.901 | 0.562 | 2 |
| additive | lengthscale | HC | 80 | 0.871 | 1 | 0.943 | 1 | 0.867 | 5 |
| additive | lengthscale | R2D2 | 80 | 0.688 | 0.733 | 0.709 | 0.74 | 0.401 | 14 |
| product | amplitude | HC | 80 | 0.972 | 1 | 0.978 | 1 | 0.795 | 4 |
| product | amplitude | R2D2 | 80 | 0.9 | 0.911 | 0.887 | 0.921 | 0.56 | 1 |
| product | lengthscale | HC | 80 | 0.955 | 1 | 0.934 | 1 | 0.828 | 4 |
| product | lengthscale | R2D2 | 80 | 0.769 | 0.81 | 0.692 | 0.713 | 0.596 | 4 |


Per method (all families pooled), median and mean over complete runs:

| method | n_runs | ap_native_median | ap_native_mean | recall_at_S_native_mean | ap_sobol_mean | recall_at_S_sobol_mean | precision_mean | recall_mean | f1_mean | n_pred_active_median |
|---|---|---|---|---|---|---|---|---|---|---|
| product-lengthscale | 80 | 1 | 0.955 | 0.937 | 0.934 | 0.909 | 0.988 | 0.748 | 0.828 | 4 |
| additive-lengthscale | 80 | 1 | 0.871 | 0.873 | 0.943 | 0.91 | 0.919 | 0.869 | 0.867 | 5 |
| product-amplitude | 80 | 1 | 0.972 | 0.957 | 0.978 | 0.964 | 0.995 | 0.707 | 0.795 | 4 |
| additive-amplitude | 80 | 1 | 0.911 | 0.908 | 0.941 | 0.92 | 0.941 | 0.789 | 0.84 | 5 |
| product-lengthscale_r2d2 | 80 | 0.81 | 0.769 | 0.692 | 0.692 | 0.623 | 0.739 | 0.484 | 0.596 | 4 |
| additive-lengthscale_r2d2 | 80 | 0.733 | 0.688 | 0.602 | 0.709 | 0.627 | 0.291 | 0.779 | 0.401 | 14 |
| product-amplitude_r2d2 | 80 | 0.911 | 0.9 | 0.827 | 0.887 | 0.809 | 0.987 | 0.268 | 0.56 | 1 |
| additive-amplitude_r2d2 | 80 | 0.89 | 0.832 | 0.774 | 0.859 | 0.786 | 0.878 | 0.33 | 0.562 | 2 |
| dsp_map | 80 | 0.674 | 0.658 | 0.622 | 0.573 | 0.578 | 0.347 | 0.771 | 0.457 | 12.5 |
| oracle_S | 80 | 1 | 1 | 1 | 0.977 | 0.975 | 1 | 0.742 | 0.833 | 4 |


Per method and family (mean over seeds):

| method | family | n_runs | ap_native_mean | recall_at_S_native_mean | ap_sobol_mean | recall_at_S_sobol_mean | precision_mean | recall_mean | f1_mean | n_pred_active_median |
|---|---|---|---|---|---|---|---|---|---|---|
| product-lengthscale | aligned10 | 10 | 0.913 | 0.87 | 0.9 | 0.86 | 0.983 | 0.53 | 0.651 | 6 |
| product-lengthscale | aligned3 | 10 | 1 | 1 | 1 | 1 | 1 | 0.933 | 0.96 | 3 |
| product-lengthscale | anti_aligned | 10 | 0.889 | 0.833 | 0.87 | 0.8 | 1 | 0.617 | 0.744 | 4 |
| product-lengthscale | decoupled | 10 | 0.895 | 0.85 | 0.859 | 0.812 | 1 | 0.562 | 0.691 | 4 |
| product-lengthscale | interaction_g0.00 | 10 | 0.945 | 0.94 | 0.973 | 0.94 | 0.917 | 0.68 | 0.765 | 3 |
| product-lengthscale | interaction_g0.25 | 10 | 1 | 1 | 0.981 | 0.98 | 1 | 0.92 | 0.956 | 5 |
| product-lengthscale | interaction_g0.50 | 10 | 1 | 1 | 0.924 | 0.92 | 1 | 0.88 | 0.933 | 4 |
| product-lengthscale | interaction_g0.75 | 10 | 1 | 1 | 0.962 | 0.96 | 1 | 0.86 | 0.922 | 4 |
| additive-lengthscale | aligned10 | 10 | 0.979 | 0.94 | 0.983 | 0.96 | 1 | 0.9 | 0.941 | 10 |
| additive-lengthscale | aligned3 | 10 | 0.806 | 0.8 | 0.992 | 0.967 | 0.806 | 1 | 0.812 | 3 |
| additive-lengthscale | anti_aligned | 10 | 0.954 | 0.95 | 0.981 | 0.95 | 1 | 0.783 | 0.854 | 5 |
| additive-lengthscale | decoupled | 10 | 0.971 | 0.938 | 0.982 | 0.925 | 1 | 0.787 | 0.876 | 6 |
| additive-lengthscale | interaction_g0.00 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 5 |
| additive-lengthscale | interaction_g0.25 | 10 | 0.811 | 0.82 | 0.886 | 0.86 | 0.867 | 0.9 | 0.875 | 5.5 |
| additive-lengthscale | interaction_g0.50 | 10 | 0.744 | 0.78 | 0.913 | 0.84 | 0.839 | 0.86 | 0.835 | 5.5 |
| additive-lengthscale | interaction_g0.75 | 10 | 0.701 | 0.76 | 0.808 | 0.78 | 0.837 | 0.72 | 0.742 | 5 |
| product-amplitude | aligned10 | 10 | 0.925 | 0.86 | 0.954 | 0.88 | 1 | 0.25 | 0.393 | 2.5 |
| product-amplitude | aligned3 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 3 |
| product-amplitude | anti_aligned | 10 | 0.951 | 0.933 | 0.976 | 0.95 | 1 | 0.717 | 0.83 | 4 |
| product-amplitude | decoupled | 10 | 0.934 | 0.9 | 0.948 | 0.925 | 1 | 0.388 | 0.555 | 3 |
| product-amplitude | interaction_g0.00 | 10 | 1 | 1 | 1 | 1 | 1 | 0.88 | 0.931 | 4.5 |
| product-amplitude | interaction_g0.25 | 10 | 1 | 1 | 1 | 1 | 1 | 0.84 | 0.908 | 4 |
| product-amplitude | interaction_g0.50 | 10 | 1 | 1 | 1 | 1 | 1 | 0.8 | 0.886 | 4 |
| product-amplitude | interaction_g0.75 | 10 | 0.963 | 0.96 | 0.946 | 0.96 | 0.96 | 0.78 | 0.86 | 4 |
| additive-amplitude | aligned10 | 10 | 0.926 | 0.91 | 0.936 | 0.92 | 0.98 | 0.51 | 0.656 | 5 |
| additive-amplitude | aligned3 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 3 |
| additive-amplitude | anti_aligned | 10 | 0.979 | 0.967 | 0.975 | 0.967 | 1 | 0.783 | 0.876 | 5 |
| additive-amplitude | decoupled | 10 | 0.946 | 0.925 | 0.955 | 0.912 | 1 | 0.537 | 0.692 | 4 |
| additive-amplitude | interaction_g0.00 | 10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 5 |
| additive-amplitude | interaction_g0.25 | 10 | 0.892 | 0.86 | 0.94 | 0.9 | 0.91 | 0.92 | 0.911 | 5 |
| additive-amplitude | interaction_g0.50 | 10 | 0.793 | 0.8 | 0.843 | 0.8 | 0.822 | 0.8 | 0.806 | 5 |
| additive-amplitude | interaction_g0.75 | 10 | 0.753 | 0.8 | 0.884 | 0.86 | 0.813 | 0.76 | 0.781 | 5 |
| product-lengthscale_r2d2 | aligned10 | 10 | 0.806 | 0.77 | 0.745 | 0.71 | 0.911 | 0.29 | 0.519 | 2.5 |
| product-lengthscale_r2d2 | aligned3 | 10 | 0.9 | 0.8 | 0.901 | 0.833 | 0.761 | 0.667 | 0.723 | 3 |
| product-lengthscale_r2d2 | anti_aligned | 10 | 0.683 | 0.617 | 0.585 | 0.567 | 0.74 | 0.367 | 0.572 | 3 |
| product-lengthscale_r2d2 | decoupled | 10 | 0.592 | 0.512 | 0.54 | 0.438 | 0.472 | 0.325 | 0.38 | 5 |
| product-lengthscale_r2d2 | interaction_g0.00 | 10 | 0.867 | 0.78 | 0.838 | 0.76 | 0.865 | 0.44 | 0.633 | 2.5 |
| product-lengthscale_r2d2 | interaction_g0.25 | 10 | 0.861 | 0.82 | 0.781 | 0.66 | 0.792 | 0.56 | 0.737 | 4 |
| product-lengthscale_r2d2 | interaction_g0.50 | 10 | 0.755 | 0.66 | 0.648 | 0.54 | 0.737 | 0.64 | 0.623 | 4.5 |
| product-lengthscale_r2d2 | interaction_g0.75 | 10 | 0.684 | 0.58 | 0.501 | 0.48 | 0.647 | 0.58 | 0.531 | 4 |
| additive-lengthscale_r2d2 | aligned10 | 10 | 0.824 | 0.72 | 0.841 | 0.74 | 0.349 | 0.86 | 0.478 | 29.5 |
| additive-lengthscale_r2d2 | aligned3 | 10 | 0.937 | 0.9 | 0.945 | 0.867 | 0.406 | 0.833 | 0.549 | 7 |
| additive-lengthscale_r2d2 | anti_aligned | 10 | 0.717 | 0.65 | 0.689 | 0.633 | 0.307 | 0.833 | 0.388 | 22 |
| additive-lengthscale_r2d2 | decoupled | 10 | 0.591 | 0.525 | 0.653 | 0.575 | 0.15 | 0.988 | 0.24 | 93 |
| additive-lengthscale_r2d2 | interaction_g0.00 | 10 | 0.871 | 0.76 | 0.865 | 0.76 | 0.382 | 0.8 | 0.54 | 12 |
| additive-lengthscale_r2d2 | interaction_g0.25 | 10 | 0.673 | 0.54 | 0.757 | 0.66 | 0.297 | 0.84 | 0.414 | 12.5 |
| additive-lengthscale_r2d2 | interaction_g0.50 | 10 | 0.466 | 0.34 | 0.446 | 0.36 | 0.205 | 0.62 | 0.3 | 14.5 |
| additive-lengthscale_r2d2 | interaction_g0.75 | 10 | 0.428 | 0.38 | 0.476 | 0.42 | 0.205 | 0.46 | 0.27 | 7 |
| product-amplitude_r2d2 | aligned10 | 10 | 0.859 | 0.78 | 0.858 | 0.77 | 1 | 0.01 | 0.182 | 0 |
| product-amplitude_r2d2 | aligned3 | 10 | 0.987 | 0.967 | 0.992 | 0.967 | 1 | 0.4 | 0.637 | 1 |
| product-amplitude_r2d2 | anti_aligned | 10 | 0.891 | 0.833 | 0.844 | 0.733 | 1 | 0.117 | 0.321 | 1 |
| product-amplitude_r2d2 | decoupled | 10 | 0.835 | 0.738 | 0.833 | 0.762 | 1 | 0.075 | 0.389 | 0 |
| product-amplitude_r2d2 | interaction_g0.00 | 10 | 0.971 | 0.92 | 0.977 | 0.9 | 1 | 0.16 | 0.413 | 1 |
| product-amplitude_r2d2 | interaction_g0.25 | 10 | 0.889 | 0.8 | 0.881 | 0.82 | 1 | 0.36 | 0.659 | 2 |
| product-amplitude_r2d2 | interaction_g0.50 | 10 | 0.888 | 0.82 | 0.848 | 0.74 | 0.933 | 0.48 | 0.615 | 2.5 |
| product-amplitude_r2d2 | interaction_g0.75 | 10 | 0.878 | 0.76 | 0.867 | 0.78 | 1 | 0.54 | 0.692 | 3 |
| additive-amplitude_r2d2 | aligned10 | 10 | 0.928 | 0.84 | 0.924 | 0.83 | 1 | 0.01 | 0.182 | 0 |
| additive-amplitude_r2d2 | aligned3 | 10 | 0.972 | 0.933 | 0.983 | 0.933 | 1 | 0.567 | 0.733 | 1.5 |
| additive-amplitude_r2d2 | anti_aligned | 10 | 0.925 | 0.85 | 0.906 | 0.833 | 1 | 0.267 | 0.472 | 1 |
| additive-amplitude_r2d2 | decoupled | 10 | 0.816 | 0.713 | 0.817 | 0.713 | 0.857 | 0.138 | 0.308 | 1 |
| additive-amplitude_r2d2 | interaction_g0.00 | 10 | 0.862 | 0.78 | 0.873 | 0.8 | 1 | 0.2 | 0.661 | 0 |
| additive-amplitude_r2d2 | interaction_g0.25 | 10 | 0.766 | 0.76 | 0.839 | 0.78 | 0.906 | 0.36 | 0.57 | 2 |
| additive-amplitude_r2d2 | interaction_g0.50 | 10 | 0.729 | 0.7 | 0.811 | 0.74 | 0.77 | 0.52 | 0.605 | 4 |
| additive-amplitude_r2d2 | interaction_g0.75 | 10 | 0.66 | 0.62 | 0.715 | 0.66 | 0.711 | 0.58 | 0.605 | 4 |
| dsp_map | aligned10 | 10 | 0.588 | 0.54 | 0.51 | 0.49 | 0.436 | 0.64 | 0.506 | 13 |
| dsp_map | aligned3 | 10 | 0.877 | 0.833 | 0.85 | 0.833 | 0.418 | 0.9 | 0.55 | 8 |
| dsp_map | anti_aligned | 10 | 0.598 | 0.6 | 0.526 | 0.55 | 0.276 | 0.733 | 0.389 | 16.5 |
| dsp_map | decoupled | 10 | 0.637 | 0.525 | 0.457 | 0.487 | 0.3 | 0.775 | 0.425 | 23.5 |
| dsp_map | interaction_g0.00 | 10 | 0.677 | 0.64 | 0.663 | 0.66 | 0.395 | 0.82 | 0.516 | 9.5 |
| dsp_map | interaction_g0.25 | 10 | 0.608 | 0.62 | 0.544 | 0.56 | 0.337 | 0.78 | 0.457 | 11 |
| dsp_map | interaction_g0.50 | 10 | 0.616 | 0.58 | 0.557 | 0.56 | 0.262 | 0.76 | 0.375 | 14 |
| dsp_map | interaction_g0.75 | 10 | 0.66 | 0.64 | 0.477 | 0.48 | 0.353 | 0.76 | 0.439 | 11 |
| oracle_S | aligned10 | 10 | 1 | 1 | 0.973 | 0.97 | 1 | 0.55 | 0.686 | 5 |
| oracle_S | aligned3 | 10 | 1 | 1 | 1 | 1 | 1 | 0.967 | 0.98 | 3 |
| oracle_S | anti_aligned | 10 | 1 | 1 | 0.984 | 0.983 | 1 | 0.633 | 0.765 | 4 |
| oracle_S | decoupled | 10 | 1 | 1 | 0.931 | 0.925 | 1 | 0.55 | 0.673 | 4 |
| oracle_S | interaction_g0.00 | 10 | 1 | 1 | 1 | 1 | 1 | 0.66 | 0.785 | 3 |
| oracle_S | interaction_g0.25 | 10 | 1 | 1 | 1 | 1 | 1 | 0.88 | 0.931 | 4.5 |
| oracle_S | interaction_g0.50 | 10 | 1 | 1 | 0.962 | 0.96 | 1 | 0.86 | 0.922 | 4 |
| oracle_S | interaction_g0.75 | 10 | 1 | 1 | 0.962 | 0.96 | 1 | 0.84 | 0.911 | 4 |


Best cell per family:

| family | best_ap_native | best_ap_sobol | best_f1 | order_by_ap_native |
|---|---|---|---|---|
| aligned3 | product-lengthscale | product-lengthscale | product-amplitude | product-lengthscale > product-amplitude > additive-amplitude > product-amplitude_r2d2 > additive-amplitude_r2d2 > additive-lengthscale_r2d2 > product-lengthscale_r2d2 > additive-lengthscale |
| aligned10 | additive-lengthscale | additive-lengthscale | additive-lengthscale | additive-lengthscale > additive-amplitude_r2d2 > additive-amplitude > product-amplitude > product-lengthscale > product-amplitude_r2d2 > additive-lengthscale_r2d2 > product-lengthscale_r2d2 |
| anti_aligned | additive-amplitude | additive-lengthscale | additive-amplitude | additive-amplitude > additive-lengthscale > product-amplitude > additive-amplitude_r2d2 > product-amplitude_r2d2 > product-lengthscale > additive-lengthscale_r2d2 > product-lengthscale_r2d2 |
| decoupled | additive-lengthscale | additive-lengthscale | additive-lengthscale | additive-lengthscale > additive-amplitude > product-amplitude > product-lengthscale > product-amplitude_r2d2 > additive-amplitude_r2d2 > product-lengthscale_r2d2 > additive-lengthscale_r2d2 |
| interaction_g0.00 | additive-lengthscale | additive-lengthscale | additive-lengthscale | additive-lengthscale > product-amplitude > additive-amplitude > product-amplitude_r2d2 > product-lengthscale > additive-lengthscale_r2d2 > product-lengthscale_r2d2 > additive-amplitude_r2d2 |
| interaction_g0.25 | product-lengthscale | product-amplitude | product-lengthscale | product-lengthscale > product-amplitude > additive-amplitude > product-amplitude_r2d2 > product-lengthscale_r2d2 > additive-lengthscale > additive-amplitude_r2d2 > additive-lengthscale_r2d2 |
| interaction_g0.50 | product-lengthscale | product-amplitude | product-lengthscale | product-lengthscale > product-amplitude > product-amplitude_r2d2 > additive-amplitude > product-lengthscale_r2d2 > additive-lengthscale > additive-amplitude_r2d2 > additive-lengthscale_r2d2 |
| interaction_g0.75 | product-lengthscale | product-lengthscale | product-lengthscale | product-lengthscale > product-amplitude > product-amplitude_r2d2 > additive-amplitude > additive-lengthscale > product-lengthscale_r2d2 > additive-amplitude_r2d2 > additive-lengthscale_r2d2 |


Runs whose t=199 readout came from a gate-excluded fit, per method: {'additive-amplitude': 80, 'additive-amplitude_r2d2': 80, 'additive-lengthscale': 40, 'additive-lengthscale_r2d2': 4, 'dsp_map': 4, 'oracle_S': 0, 'product-amplitude': 80, 'product-amplitude_r2d2': 80, 'product-lengthscale': 23, 'product-lengthscale_r2d2': 0}.


## 8. R2 readouts from the npz draws (amplitude cells)

Per fit, medians over its 16 retained draws: `r2d2_r2` = S/(1 + S) with S = sum_i a_sq_i, what the R2-D2 prior's Beta is on (about R^2/(1 + R^2) on a standardized target, so at most about 1/2); `first_order_r2` = S/(S + sigma^2) for the additive cells and S/(prod_i(1 + a_sq_i) - 1 + sigma^2) for the product cells (ruling R20), with sigma^2 the raw noise draw clamped at 0.0001 (the manifest's fixed_noise where the noise is fixed). Read at t = 20, 25, 50, 75, 100, 125, 150, 175, 199 (`tables/r2_readouts_vs_t_median.csv`); at t = 199, median [Q1, Q3] over seeds:

| method | family | n | r2d2_r2 | first_order_r2 |
|---|---|---|---|---|
| product-amplitude | interaction_g0.25 | 10 | 0.335 [0.315, 0.341] | 0.832 [0.822, 0.84] |
| product-amplitude | decoupled | 10 | 0.193 [0.178, 0.204] | 0.9 [0.89, 0.906] |
| product-amplitude | anti_aligned | 10 | 0.233 [0.201, 0.236] | 0.885 [0.873, 0.894] |
| product-amplitude | aligned3 | 10 | 0.241 [0.21, 0.254] | 0.88 [0.876, 0.888] |
| product-amplitude | aligned10 | 10 | 0.157 [0.15, 0.185] | 0.908 [0.895, 0.915] |
| product-amplitude | interaction_g0.50 | 10 | 0.376 [0.361, 0.383] | 0.808 [0.804, 0.823] |
| product-amplitude | interaction_g0.75 | 10 | 0.378 [0.364, 0.412] | 0.812 [0.799, 0.828] |
| product-amplitude | interaction_g0.00 | 10 | 0.205 [0.191, 0.215] | 0.888 [0.885, 0.899] |
| additive-amplitude | aligned10 | 10 | 0.288 [0.267, 0.3] | 0.991 [0.988, 0.992] |
| additive-amplitude | interaction_g0.75 | 10 | 0.682 [0.635, 0.723] | 0.999 [0.998, 0.999] |
| additive-amplitude | interaction_g0.50 | 10 | 0.637 [0.551, 0.682] | 0.996 [0.986, 0.998] |
| additive-amplitude | interaction_g0.25 | 10 | 0.512 [0.464, 0.549] | 0.992 [0.986, 0.997] |
| additive-amplitude | interaction_g0.00 | 10 | 0.421 [0.346, 0.461] | 0.99 [0.986, 0.992] |
| additive-amplitude | decoupled | 10 | 0.291 [0.258, 0.321] | 0.989 [0.986, 0.99] |
| additive-amplitude | anti_aligned | 10 | 0.348 [0.317, 0.397] | 0.988 [0.986, 0.992] |
| additive-amplitude | aligned3 | 10 | 0.52 [0.455, 0.547] | 0.99 [0.987, 0.993] |
| product-amplitude_r2d2 | interaction_g0.00 | 10 | 0.136 [0.123, 0.145] | 0.891 [0.88, 0.902] |
| product-amplitude_r2d2 | anti_aligned | 10 | 0.143 [0.129, 0.153] | 0.89 [0.881, 0.9] |
| product-amplitude_r2d2 | aligned3 | 10 | 0.131 [0.128, 0.139] | 0.86 [0.853, 0.873] |
| product-amplitude_r2d2 | aligned10 | 10 | 0.125 [0.114, 0.132] | 0.912 [0.907, 0.916] |
| product-amplitude_r2d2 | interaction_g0.25 | 10 | 0.182 [0.163, 0.227] | 0.869 [0.855, 0.898] |
| product-amplitude_r2d2 | decoupled | 10 | 0.137 [0.134, 0.148] | 0.902 [0.884, 0.908] |
| product-amplitude_r2d2 | interaction_g0.75 | 10 | 0.234 [0.206, 0.251] | 0.857 [0.844, 0.87] |
| product-amplitude_r2d2 | interaction_g0.50 | 10 | 0.214 [0.184, 0.247] | 0.859 [0.845, 0.886] |
| additive-amplitude_r2d2 | interaction_g0.50 | 10 | 0.305 [0.268, 0.358] | 0.983 [0.973, 0.988] |
| additive-amplitude_r2d2 | interaction_g0.25 | 10 | 0.24 [0.174, 0.284] | 0.975 [0.963, 0.978] |
| additive-amplitude_r2d2 | interaction_g0.00 | 10 | 0.149 [0.133, 0.171] | 0.96 [0.946, 0.964] |
| additive-amplitude_r2d2 | decoupled | 10 | 0.161 [0.151, 0.198] | 0.973 [0.968, 0.977] |
| additive-amplitude_r2d2 | anti_aligned | 10 | 0.18 [0.174, 0.188] | 0.962 [0.957, 0.97] |
| additive-amplitude_r2d2 | aligned3 | 10 | 0.16 [0.153, 0.171] | 0.934 [0.919, 0.953] |
| additive-amplitude_r2d2 | aligned10 | 10 | 0.15 [0.136, 0.164] | 0.979 [0.973, 0.981] |
| additive-amplitude_r2d2 | interaction_g0.75 | 10 | 0.412 [0.364, 0.472] | 0.986 [0.974, 0.994] |


## 9. Diagnostics

![diagnostics](figures/diagnostics.png)

Per method, all iterations of all runs (gate: split-R-hat <= 1.1, ESS >= 16 on every sampled site, <= 5 divergences):


| method | n_iter | gate_excluded_rate | exception_rows | r_hat_max_q10 | r_hat_max_median | r_hat_max_q90 | n_eff_min_q10 | n_eff_min_median | n_eff_min_q90 | divergences_mean | divergences_gt5_frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| product-lengthscale | 14400 | 0.512 | 0 | 1.04 | 1.09 | 1.38 | 6.08 | 23.9 | 61.6 | 0.192 | 0.00375 |
| additive-lengthscale | 14400 | 0.597 | 0 | 1.04 | 1.12 | 1.56 | 4.39 | 18.2 | 54 | 0.353 | 0.00937 |
| product-amplitude | 14400 | 1 | 0 | 1.36 | 1.49 | 1.68 | 4.82 | 6.13 | 7.75 | 0.0673 | 0.00111 |
| additive-amplitude | 14400 | 1 | 0 | 1.37 | 1.51 | 1.77 | 4.41 | 5.86 | 7.47 | 0.232 | 0.00403 |
| product-lengthscale_r2d2 | 14400 | 0.00667 | 0 | 1.02 | 1.03 | 1.05 | 61.6 | 94.8 | 124 | 0.00736 | 0 |
| additive-lengthscale_r2d2 | 14400 | 0.0556 | 0 | 1.02 | 1.03 | 1.07 | 40.5 | 87.1 | 122 | 1.5 | 0.0218 |
| product-amplitude_r2d2 | 14400 | 1 | 0 | 1.37 | 1.51 | 1.71 | 4.75 | 5.96 | 7.57 | 0.0224 | 0 |
| additive-amplitude_r2d2 | 14400 | 1 | 0 | 1.39 | 1.53 | 1.76 | 4.51 | 5.71 | 7.23 | 0.00264 | 0 |
| sobol | 14400 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | 0 |
| dsp_map | 14400 | 0 | 65 | nan | nan | nan | nan | nan | nan | nan | 0 |
| oracle_S | 14400 | 0 | 0 | nan | nan | nan | nan | nan | nan | nan | 0 |


Sampler effort (num_steps_mean = leapfrog steps per NUTS iteration, max 63 at max_tree_depth = 6; fit_calls > 1 means a retry):

| method | num_steps_mean_median | frac_at_tree_cap | fit_calls_mean | nuts_attempts_mean |
|---|---|---|---|---|
| product-lengthscale | 62.9 | 0.949 | 1 | 1 |
| additive-lengthscale | 62.9 | 0.965 | 1 | 1 |
| product-amplitude | 63 | 0.999 | 1 | 1 |
| additive-amplitude | 63 | 0.997 | 1 | 1 |
| product-lengthscale_r2d2 | 60.4 | 0.141 | 1 | 1 |
| additive-lengthscale_r2d2 | 60.4 | 0.164 | 1 | 1 |
| product-amplitude_r2d2 | 63 | 1 | 1 | 1 |
| additive-amplitude_r2d2 | 63 | 1 | 1 | 1 |


Identification quality of readouts at t >= 100 split by whether that fit passed the gate (cells):

| method | gate_excluded | n_readouts | ap_native_mean | ap_native_median | f1_mean |
|---|---|---|---|---|---|
| product-lengthscale | passed | 212 | 0.943 | 1 | 0.813 |
| product-lengthscale | excluded | 188 | 0.812 | 0.85 | 0.704 |
| additive-lengthscale | passed | 172 | 0.896 | 1 | 0.886 |
| additive-lengthscale | excluded | 228 | 0.792 | 0.848 | 0.777 |
| product-amplitude | excluded | 400 | 0.911 | 1 | 0.743 |
| additive-amplitude | excluded | 400 | 0.836 | 0.892 | 0.762 |
| product-lengthscale_r2d2 | passed | 398 | 0.651 | 0.683 | 0.509 |
| product-lengthscale_r2d2 | excluded | 2 | 0.703 | 0.703 | 0.36 |
| additive-lengthscale_r2d2 | passed | 389 | 0.568 | 0.597 | 0.333 |
| additive-lengthscale_r2d2 | excluded | 11 | 0.32 | 0.28 | 0.177 |
| product-amplitude_r2d2 | excluded | 400 | 0.767 | 0.806 | 0.493 |
| additive-amplitude_r2d2 | excluded | 400 | 0.719 | 0.745 | 0.476 |


Which site fails (medians of the per-site diagnostics):

| method | r_hat_max_native_median | r_hat_max_ell_median | r_hat_max_global_median | n_eff_min_native_median | n_eff_min_ell_median | n_eff_min_global_median | frac_r_hat_below_1_05_median |
|---|---|---|---|---|---|---|---|
| product-lengthscale | 1.09 | nan | 1.01 | 24 | nan | 111 | 0.962 |
| additive-lengthscale | 1.12 | nan | 1.02 | 18.3 | nan | 77.8 | 0.952 |
| product-amplitude | 1.4 | 1.44 | 1.05 | 7.02 | 6.69 | 35.8 | 0.635 |
| additive-amplitude | 1.43 | 1.45 | 1.04 | 6.58 | 6.52 | 37.2 | 0.626 |
| product-lengthscale_r2d2 | 1.03 | nan | 1 | 95.4 | nan | 180 | 1 |
| additive-lengthscale_r2d2 | 1.03 | nan | 1 | 90.4 | nan | 160 | 1 |
| product-amplitude_r2d2 | 1.31 | 1.5 | 1.03 | 9.09 | 6.02 | 56.3 | 0.655 |
| additive-amplitude_r2d2 | 1.31 | 1.53 | 1.02 | 8.92 | 5.74 | 85.6 | 0.65 |


Gate reason breakdown (iterations):

| method | divergences | exception | fit_gpytorch_mll+Adam | n_eff_min | n_eff_min+divergences | ok | r_hat_max | r_hat_max+divergences | r_hat_max+n_eff_min | r_hat_max+n_eff_min+divergences |
|---|---|---|---|---|---|---|---|---|---|---|
| product-lengthscale | 2 | 0 | 0 | 482 | 1 | 7031 | 2131 | 5 | 4702 | 46 |
| additive-lengthscale | 3 | 0 | 0 | 443 | 3 | 5798 | 2058 | 2 | 5966 | 127 |
| product-amplitude | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14384 | 16 |
| additive-amplitude | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14342 | 58 |
| product-lengthscale_r2d2 | 0 | 0 | 0 | 1 | 0 | 14304 | 89 | 0 | 6 | 0 |
| additive-lengthscale_r2d2 | 102 | 0 | 0 | 71 | 21 | 13600 | 236 | 13 | 179 | 178 |
| product-amplitude_r2d2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14400 | 0 |
| additive-amplitude_r2d2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14400 | 0 |
| dsp_map | 0 | 65 | 225 | 0 | 0 | 14110 | 0 | 0 | 0 | 0 |


Per method and family:

| method | family | n_iter | gate_excluded_rate | exception_rows | r_hat_max_median | n_eff_min_median | divergences_mean |
|---|---|---|---|---|---|---|---|
| product-lengthscale | aligned10 | 1800 | 0.719 | 0 | 1.14 | 15.1 | 0.00222 |
| product-lengthscale | aligned3 | 1800 | 0.263 | 0 | 1.06 | 44.7 | 0.462 |
| product-lengthscale | anti_aligned | 1800 | 0.478 | 0 | 1.09 | 26.2 | 0.157 |
| product-lengthscale | decoupled | 1800 | 0.618 | 0 | 1.12 | 19.5 | 0.139 |
| product-lengthscale | interaction_g0.00 | 1800 | 0.471 | 0 | 1.09 | 26.4 | 0.129 |
| product-lengthscale | interaction_g0.25 | 1800 | 0.513 | 0 | 1.1 | 24.7 | 0.227 |
| product-lengthscale | interaction_g0.50 | 1800 | 0.518 | 0 | 1.1 | 21.9 | 0.0406 |
| product-lengthscale | interaction_g0.75 | 1800 | 0.513 | 0 | 1.1 | 24.2 | 0.379 |
| additive-lengthscale | aligned10 | 1800 | 0.665 | 0 | 1.12 | 17.1 | 0.0422 |
| additive-lengthscale | aligned3 | 1800 | 0.383 | 0 | 1.08 | 35.8 | 1.22 |
| additive-lengthscale | anti_aligned | 1800 | 0.604 | 0 | 1.12 | 18.3 | 0.102 |
| additive-lengthscale | decoupled | 1800 | 0.69 | 0 | 1.14 | 14.3 | 0.0228 |
| additive-lengthscale | interaction_g0.00 | 1800 | 0.453 | 0 | 1.09 | 28 | 0.0611 |
| additive-lengthscale | interaction_g0.25 | 1800 | 0.635 | 0 | 1.13 | 15.9 | 0.377 |
| additive-lengthscale | interaction_g0.50 | 1800 | 0.695 | 0 | 1.16 | 11.9 | 0.232 |
| additive-lengthscale | interaction_g0.75 | 1800 | 0.654 | 0 | 1.15 | 13 | 0.769 |
| product-amplitude | aligned10 | 1800 | 1 | 0 | 1.5 | 6.01 | 0 |
| product-amplitude | aligned3 | 1800 | 1 | 0 | 1.47 | 6.28 | 0.0272 |
| product-amplitude | anti_aligned | 1800 | 1 | 0 | 1.48 | 6.2 | 0 |
| product-amplitude | decoupled | 1800 | 1 | 0 | 1.5 | 6.04 | 0.00278 |
| product-amplitude | interaction_g0.00 | 1800 | 1 | 0 | 1.48 | 6.16 | 0 |
| product-amplitude | interaction_g0.25 | 1800 | 1 | 0 | 1.49 | 6.07 | 0.0183 |
| product-amplitude | interaction_g0.50 | 1800 | 1 | 0 | 1.49 | 6.11 | 0.00111 |
| product-amplitude | interaction_g0.75 | 1800 | 1 | 0 | 1.49 | 6.13 | 0.489 |
| additive-amplitude | aligned10 | 1800 | 1 | 0 | 1.52 | 5.74 | 0 |
| additive-amplitude | aligned3 | 1800 | 1 | 0 | 1.49 | 6.11 | 1.35 |
| additive-amplitude | anti_aligned | 1800 | 1 | 0 | 1.5 | 5.94 | 0 |
| additive-amplitude | decoupled | 1800 | 1 | 0 | 1.52 | 5.81 | 0.00111 |
| additive-amplitude | interaction_g0.00 | 1800 | 1 | 0 | 1.51 | 5.86 | 0.161 |
| additive-amplitude | interaction_g0.25 | 1800 | 1 | 0 | 1.52 | 5.79 | 0.0378 |
| additive-amplitude | interaction_g0.50 | 1800 | 1 | 0 | 1.52 | 5.87 | 0.229 |
| additive-amplitude | interaction_g0.75 | 1800 | 1 | 0 | 1.51 | 5.83 | 0.0817 |
| product-lengthscale_r2d2 | aligned10 | 1800 | 0.00333 | 0 | 1.03 | 91.5 | 0.00611 |
| product-lengthscale_r2d2 | aligned3 | 1800 | 0.00222 | 0 | 1.03 | 97.8 | 0.00667 |
| product-lengthscale_r2d2 | anti_aligned | 1800 | 0.00556 | 0 | 1.03 | 95.8 | 0.00722 |
| product-lengthscale_r2d2 | decoupled | 1800 | 0.00889 | 0 | 1.03 | 94.5 | 0.005 |
| product-lengthscale_r2d2 | interaction_g0.00 | 1800 | 0.00611 | 0 | 1.03 | 95.3 | 0.00222 |
| product-lengthscale_r2d2 | interaction_g0.25 | 1800 | 0.00889 | 0 | 1.03 | 94.6 | 0.0144 |
| product-lengthscale_r2d2 | interaction_g0.50 | 1800 | 0.00944 | 0 | 1.03 | 94.4 | 0.00778 |
| product-lengthscale_r2d2 | interaction_g0.75 | 1800 | 0.00889 | 0 | 1.03 | 93.7 | 0.00944 |
| additive-lengthscale_r2d2 | aligned10 | 1800 | 0.0256 | 0 | 1.04 | 81.2 | 0.201 |
| additive-lengthscale_r2d2 | aligned3 | 1800 | 0.0256 | 0 | 1.03 | 89.3 | 0.19 |
| additive-lengthscale_r2d2 | anti_aligned | 1800 | 0.0261 | 0 | 1.03 | 92.9 | 0.502 |
| additive-lengthscale_r2d2 | decoupled | 1800 | 0.0417 | 0 | 1.04 | 82.9 | 0.851 |
| additive-lengthscale_r2d2 | interaction_g0.00 | 1800 | 0.0422 | 0 | 1.03 | 86.1 | 0.341 |
| additive-lengthscale_r2d2 | interaction_g0.25 | 1800 | 0.0544 | 0 | 1.03 | 86.2 | 1.33 |
| additive-lengthscale_r2d2 | interaction_g0.50 | 1800 | 0.0944 | 0 | 1.03 | 88.6 | 3 |
| additive-lengthscale_r2d2 | interaction_g0.75 | 1800 | 0.134 | 0 | 1.03 | 91.5 | 5.59 |
| product-amplitude_r2d2 | aligned10 | 1800 | 1 | 0 | 1.5 | 5.95 | 0.00556 |
| product-amplitude_r2d2 | aligned3 | 1800 | 1 | 0 | 1.5 | 6.01 | 0.0272 |
| product-amplitude_r2d2 | anti_aligned | 1800 | 1 | 0 | 1.51 | 5.95 | 0.0183 |
| product-amplitude_r2d2 | decoupled | 1800 | 1 | 0 | 1.5 | 6 | 0.00778 |
| product-amplitude_r2d2 | interaction_g0.00 | 1800 | 1 | 0 | 1.5 | 6 | 0.035 |
| product-amplitude_r2d2 | interaction_g0.25 | 1800 | 1 | 0 | 1.51 | 5.94 | 0.0272 |
| product-amplitude_r2d2 | interaction_g0.50 | 1800 | 1 | 0 | 1.51 | 5.97 | 0.0233 |
| product-amplitude_r2d2 | interaction_g0.75 | 1800 | 1 | 0 | 1.52 | 5.9 | 0.0344 |
| additive-amplitude_r2d2 | aligned10 | 1800 | 1 | 0 | 1.54 | 5.62 | 0.000556 |
| additive-amplitude_r2d2 | aligned3 | 1800 | 1 | 0 | 1.53 | 5.77 | 0.00167 |
| additive-amplitude_r2d2 | anti_aligned | 1800 | 1 | 0 | 1.53 | 5.74 | 0.00444 |
| additive-amplitude_r2d2 | decoupled | 1800 | 1 | 0 | 1.53 | 5.8 | 0.00222 |
| additive-amplitude_r2d2 | interaction_g0.00 | 1800 | 1 | 0 | 1.54 | 5.68 | 0.00167 |
| additive-amplitude_r2d2 | interaction_g0.25 | 1800 | 1 | 0 | 1.54 | 5.65 | 0.00722 |
| additive-amplitude_r2d2 | interaction_g0.50 | 1800 | 1 | 0 | 1.54 | 5.66 | 0.00111 |
| additive-amplitude_r2d2 | interaction_g0.75 | 1800 | 1 | 0 | 1.53 | 5.73 | 0.00222 |
| sobol | aligned10 | 1800 | 0 | 0 | nan | nan | nan |
| sobol | aligned3 | 1800 | 0 | 0 | nan | nan | nan |
| sobol | anti_aligned | 1800 | 0 | 0 | nan | nan | nan |
| sobol | decoupled | 1800 | 0 | 0 | nan | nan | nan |
| sobol | interaction_g0.00 | 1800 | 0 | 0 | nan | nan | nan |
| sobol | interaction_g0.25 | 1800 | 0 | 0 | nan | nan | nan |
| sobol | interaction_g0.50 | 1800 | 0 | 0 | nan | nan | nan |
| sobol | interaction_g0.75 | 1800 | 0 | 0 | nan | nan | nan |
| dsp_map | aligned10 | 1800 | 0 | 6 | nan | nan | nan |
| dsp_map | aligned3 | 1800 | 0 | 7 | nan | nan | nan |
| dsp_map | anti_aligned | 1800 | 0 | 2 | nan | nan | nan |
| dsp_map | decoupled | 1800 | 0 | 0 | nan | nan | nan |
| dsp_map | interaction_g0.00 | 1800 | 0 | 15 | nan | nan | nan |
| dsp_map | interaction_g0.25 | 1800 | 0 | 14 | nan | nan | nan |
| dsp_map | interaction_g0.50 | 1800 | 0 | 8 | nan | nan | nan |
| dsp_map | interaction_g0.75 | 1800 | 0 | 13 | nan | nan | nan |
| oracle_S | aligned10 | 1800 | 0 | 0 | nan | nan | nan |
| oracle_S | aligned3 | 1800 | 0 | 0 | nan | nan | nan |
| oracle_S | anti_aligned | 1800 | 0 | 0 | nan | nan | nan |
| oracle_S | decoupled | 1800 | 0 | 0 | nan | nan | nan |
| oracle_S | interaction_g0.00 | 1800 | 0 | 0 | nan | nan | nan |
| oracle_S | interaction_g0.25 | 1800 | 0 | 0 | nan | nan | nan |
| oracle_S | interaction_g0.50 | 1800 | 0 | 0 | nan | nan | nan |
| oracle_S | interaction_g0.75 | 1800 | 0 | 0 | nan | nan | nan |


Gate-exclusion rate vs t (bins of 20), cells:

| t_bin | product-lengthscale | additive-lengthscale | product-amplitude | additive-amplitude | product-lengthscale_r2d2 | additive-lengthscale_r2d2 | product-amplitude_r2d2 | additive-amplitude_r2d2 |
|---|---|---|---|---|---|---|---|---|
| 20 | 0.401 | 0.422 | 1 | 1 | 0.0194 | 0.0437 | 1 | 1 |
| 40 | 0.674 | 0.718 | 1 | 1 | 0.0131 | 0.106 | 1 | 1 |
| 60 | 0.728 | 0.751 | 1 | 1 | 0.00562 | 0.0875 | 1 | 1 |
| 80 | 0.715 | 0.698 | 1 | 1 | 0.00375 | 0.06 | 1 | 1 |
| 100 | 0.612 | 0.667 | 1 | 1 | 0.00562 | 0.0556 | 1 | 1 |
| 120 | 0.495 | 0.578 | 1 | 1 | 0.00375 | 0.0488 | 1 | 1 |
| 140 | 0.403 | 0.554 | 1 | 1 | 0.00187 | 0.0325 | 1 | 1 |
| 160 | 0.316 | 0.494 | 1 | 1 | 0.00313 | 0.0281 | 1 | 1 |
| 180 | 0.263 | 0.494 | 1 | 1 | 0.00375 | 0.0381 | 1 | 1 |


## 10. Cost

Wall-times grouped by the GPU that ran the iteration (attributed from the resume segments in log.txt); the reference methods ran on CPU. **Do not compare across GPU types.** `hours_fit_plus_acq` sums the measured fit and acquisition time only: it excludes the per-iteration readout (coords.csv), checkpointing, JAX compilation at resume, and Slurm queue time, so it is a lower bound on allocated GPU-hours.


Each R2-D2 cell against its half-Cauchy twin on the same GPU type (medians over iterations; ratio = R2-D2 / twin):

| cell | twin | device | n_iter | n_iter_twin | fit_wall_s_median | fit_wall_s_median_twin | fit_wall_s_ratio | acq_wall_s_median | acq_wall_s_median_twin | acq_wall_s_ratio | iter_wall_s_median | iter_wall_s_median_twin | iter_wall_s_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| product-lengthscale_r2d2 | product-lengthscale | NVIDIA H200 | 14400 | 7200 | 44.3 | 37.3 | 1.19 | 7.21 | 7.64 | 0.943 | 51.7 | 46.3 | 1.12 |
| additive-lengthscale_r2d2 | additive-lengthscale | NVIDIA H200 | 14400 | 8404 | 49.9 | 39 | 1.28 | 189 | 175 | 1.08 | 240 | 216 | 1.11 |
| product-amplitude_r2d2 | product-amplitude | NVIDIA H200 | 14400 | 7245 | 52.7 | 43.3 | 1.22 | 301 | 242 | 1.24 | 356 | 288 | 1.23 |
| additive-amplitude_r2d2 | additive-amplitude | NVIDIA H200 | 14400 | 9360 | 51.6 | 42.8 | 1.2 | 345 | 298 | 1.16 | 399 | 341 | 1.17 |


| method | device | n_iter | fit_wall_s_median | acq_wall_s_median | iter_wall_s_median | fit_wall_s_q90 | acq_wall_s_q90 | hours_fit_plus_acq |
|---|---|---|---|---|---|---|---|---|
| product-lengthscale | NVIDIA A100-SXM4-80GB | 3822 | 48 | 10.8 | 60.8 | 57.3 | 18.4 | 62.8 |
| product-lengthscale | NVIDIA H100 80GB HBM3 | 3378 | 38 | 7.77 | 46.7 | 46.8 | 13.4 | 43.6 |
| product-lengthscale | NVIDIA H200 | 7200 | 37.3 | 7.64 | 46.3 | 46.2 | 12.6 | 91.2 |
| additive-lengthscale | NVIDIA A100-SXM4-80GB | 5457 | 52 | 258 | 311 | 64.9 | 424 | 506 |
| additive-lengthscale | NVIDIA H100 80GB HBM3 | 539 | 41.6 | 185 | 228 | 53.8 | 240 | 34.1 |
| additive-lengthscale | NVIDIA H200 | 8404 | 39 | 175 | 216 | 50.1 | 268 | 523 |
| product-amplitude | NVIDIA A100-SXM4-80GB | 4292 | 59.1 | 399 | 458 | 76.2 | 501 | 528 |
| product-amplitude | NVIDIA H100 80GB HBM3 | 2863 | 46.6 | 243 | 291 | 60.5 | 336 | 231 |
| product-amplitude | NVIDIA H200 | 7245 | 43.3 | 242 | 288 | 55.7 | 310 | 562 |
| additive-amplitude | NVIDIA A100-SXM4-80GB | 3354 | 57.6 | 471 | 529 | 73.2 | 510 | 456 |
| additive-amplitude | NVIDIA H100 80GB HBM3 | 1686 | 45.3 | 271 | 317 | 57.6 | 320 | 143 |
| additive-amplitude | NVIDIA H200 | 9360 | 42.8 | 298 | 341 | 54.1 | 336 | 833 |
| product-lengthscale_r2d2 | NVIDIA H200 | 14400 | 44.3 | 7.21 | 51.7 | 58.2 | 11.7 | 212 |
| additive-lengthscale_r2d2 | NVIDIA H200 | 14400 | 49.9 | 189 | 240 | 62.7 | 275 | 988 |
| product-amplitude_r2d2 | NVIDIA H200 | 14400 | 52.7 | 301 | 356 | 68.5 | 427 | 1.43e+03 |
| additive-amplitude_r2d2 | NVIDIA H200 | 14400 | 51.6 | 345 | 399 | 67.2 | 443 | 1.57e+03 |
| sobol | cpu | 14400 | nan | 3.5e-06 | nan | nan | 4.91e-06 | 0 |
| dsp_map | cpu | 14400 | 5.31 | 10 | 15.9 | 18.2 | 13.7 | 76.6 |
| oracle_S | cpu | 14400 | 0.791 | 2.7 | 3.58 | 1.73 | 3.48 | 15.1 |


Totals per method (all devices, all rows including incomplete runs):

| method | n_iter | hours_fit | hours_acq | hours_fit_plus_acq | median_run_hours | max_run_hours |
|---|---|---|---|---|---|---|
| product-lengthscale | 14400 | 161 | 36.8 | 198 | 2.34 | 3.14 |
| additive-lengthscale | 14400 | 178 | 884 | 1.06e+03 | 12.5 | 21.5 |
| product-amplitude | 14400 | 195 | 1.13e+03 | 1.32e+03 | 14.9 | 25.7 |
| additive-amplitude | 14400 | 185 | 1.25e+03 | 1.43e+03 | 16.5 | 26.6 |
| product-lengthscale_r2d2 | 14400 | 181 | 30.8 | 212 | 2.66 | 2.76 |
| additive-lengthscale_r2d2 | 14400 | 200 | 788 | 988 | 12.3 | 17.6 |
| product-amplitude_r2d2 | 14400 | 213 | 1.21e+03 | 1.43e+03 | 18 | 22.1 |
| additive-amplitude_r2d2 | 14400 | 209 | 1.36e+03 | 1.57e+03 | 20.2 | 23.3 |
| sobol | 14400 | 0 | 1.53e-05 | 0 | 0 | 0 |
| dsp_map | 14400 | 39.6 | 38.9 | 76.6 | 0.966 | 1.28 |
| oracle_S | 14400 | 3.95 | 11.2 | 15.1 | 0.19 | 0.328 |


![cost vs t](figures/cost_vs_t.png)


## 11. Anomalies and things that look wrong

- DEGENERATE readout: aligned3/additive-lengthscale/seed02 declares 100/100 coordinates active at t=199 (AP by native_median 0.02, by sobol_hat 1.00; min native_median on true S 16.4, max on inactive 907; fit status excluded)
- DEGENERATE readout: aligned3/additive-lengthscale/seed08 declares 100/100 coordinates active at t=199 (AP by native_median 0.04, by sobol_hat 0.92; min native_median on true S 26.9, max on inactive 105; fit status ok)
- DEGENERATE readout: aligned10/additive-lengthscale_r2d2/seed01 declares 52/100 coordinates active at t=199 (AP by native_median 0.83, by sobol_hat 0.86; min native_median on true S 0.481, max on inactive 2.53; fit status ok)
- DEGENERATE readout: aligned10/additive-lengthscale_r2d2/seed09 declares 78/100 coordinates active at t=199 (AP by native_median 0.77, by sobol_hat 0.89; min native_median on true S 1.12, max on inactive 3.71; fit status ok)
- DEGENERATE readout: anti_aligned/additive-lengthscale_r2d2/seed01 declares 99/100 coordinates active at t=199 (AP by native_median 0.59, by sobol_hat 0.59; min native_median on true S 7.88, max on inactive 37; fit status ok)
- DEGENERATE readout: anti_aligned/additive-lengthscale_r2d2/seed04 declares 99/100 coordinates active at t=199 (AP by native_median 0.55, by sobol_hat 0.60; min native_median on true S 6.3, max on inactive 22.6; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed00 declares 64/100 coordinates active at t=199 (AP by native_median 0.63, by sobol_hat 0.74; min native_median on true S 0.558, max on inactive 3.41; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed02 declares 99/100 coordinates active at t=199 (AP by native_median 0.61, by sobol_hat 0.57; min native_median on true S 3.67, max on inactive 18; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed03 declares 97/100 coordinates active at t=199 (AP by native_median 0.51, by sobol_hat 0.83; min native_median on true S 2.5, max on inactive 17.9; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed05 declares 60/100 coordinates active at t=199 (AP by native_median 0.69, by sobol_hat 0.71; min native_median on true S 0.252, max on inactive 4.77; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed06 declares 96/100 coordinates active at t=199 (AP by native_median 0.28, by sobol_hat 0.33; min native_median on true S 0.833, max on inactive 22.2; fit status excluded)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed07 declares 100/100 coordinates active at t=199 (AP by native_median 0.46, by sobol_hat 0.51; min native_median on true S 6.92, max on inactive 25.4; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed08 declares 94/100 coordinates active at t=199 (AP by native_median 0.51, by sobol_hat 0.64; min native_median on true S 0.857, max on inactive 11.1; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed09 declares 92/100 coordinates active at t=199 (AP by native_median 0.72, by sobol_hat 0.78; min native_median on true S 2.45, max on inactive 8.08; fit status ok)
- DEGENERATE readout: interaction_g0.25/additive-lengthscale_r2d2/seed00 declares 73/100 coordinates active at t=199 (AP by native_median 0.64, by sobol_hat 0.69; min native_median on true S 1.01, max on inactive 2.86; fit status ok)
- DEGENERATE readout: interaction_g0.25/additive-lengthscale_r2d2/seed06 declares 100/100 coordinates active at t=199 (AP by native_median 0.10, by sobol_hat 0.67; min native_median on true S 12.4, max on inactive 131; fit status ok)
- DEGENERATE readout: interaction_g0.50/additive-lengthscale_r2d2/seed06 declares 100/100 coordinates active at t=199 (AP by native_median 0.06, by sobol_hat 0.16; min native_median on true S 130, max on inactive 901; fit status ok)
- DEGENERATE readout: interaction_g0.50/additive-lengthscale_r2d2/seed09 declares 99/100 coordinates active at t=199 (AP by native_median 0.12, by sobol_hat 0.12; min native_median on true S 6.7, max on inactive 527; fit status excluded)
- DEGENERATE readout: interaction_g0.75/additive-lengthscale_r2d2/seed00 declares 100/100 coordinates active at t=199 (AP by native_median 0.26, by sobol_hat 0.50; min native_median on true S 13.4, max on inactive 116; fit status ok)
- DEGENERATE readout: interaction_g0.75/additive-lengthscale_r2d2/seed09 declares 100/100 coordinates active at t=199 (AP by native_median 0.16, by sobol_hat 0.11; min native_median on true S 563, max on inactive 5.76e+03; fit status ok)
- NUTS hits max_tree_depth=6 (num_steps_mean >= 62 of the 63 possible) in most fits: fraction per cell additive-amplitude 0.997, additive-amplitude_r2d2 1.000, additive-lengthscale 0.965, additive-lengthscale_r2d2 0.164, product-amplitude 0.999, product-amplitude_r2d2 1.000, product-lengthscale 0.949, product-lengthscale_r2d2 0.141


Expected oddities (not flagged):

- Git commits at creation, by prior family: {'HC': ['2866ed2', 'a51a4b9'], 'R2D2': ['5a485f3', '97eda92', 'c4d9c38'], 'reference': ['2866ed2', 'a51a4b9']} (the HC cells and references were launched at a51a4b9 and 2866ed2; the R2-D2 cells' commits are checked by `prior_code_ok`).
- aligned3/dsp_map/seed06: last coords.csv readout is t=198, last iteration t=199 (the last iteration was an exception row, so no surrogate and no readout; scored at t=198)
- aligned3/dsp_map/seed08: last coords.csv readout is t=198, last iteration t=199 (the last iteration was an exception row, so no surrogate and no readout; scored at t=198)
- interaction_g0.00/dsp_map/seed00: last coords.csv readout is t=198, last iteration t=199 (the last iteration was an exception row, so no surrogate and no readout; scored at t=198)
- interaction_g0.75/dsp_map/seed08: last coords.csv readout is t=198, last iteration t=199 (the last iteration was an exception row, so no surrogate and no readout; scored at t=198)
- dsp_map: 225 iterations used the Adam fallback fitter (status ok, reason 'fit_gpytorch_mll failed; Adam fallback'); these are neither gate exclusions nor exception rows.


## Caveats

- status="excluded" is a label: the draws were still used to choose the next point, so regret curves are valid for every run. It does mean the posterior readouts (native_median, p_active, sobol_hat) from those iterations are less trustworthy.
- Rows whose reason starts with "exception:" queried a seeded random point instead; they are counted separately from gate exclusions.
- GPU runs (A100/H100/H200) reproduce only in distribution, not bit-for-bit; wall-times are only comparable within one GPU type.
- The per-family tests have at most 10 seeds (5 in the pilot, whose smallest two-sided Wilcoxon p is 0.0625).
- The regret plots floor the log axis at a small positive value; seeds that reached regret 0 sit on that floor.
