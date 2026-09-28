# R2-D2 replay: gate G2

Stored runs `runs`, replays `runs_replay`; written 2026-09-28T22:39:04+00:00 by `python -m experiments.replay compare`.

## Verdict

**G2: FAIL: the half-Cauchy amplitude signature -- nothing in-loop is launched; Quan decides the budget for all eight cells (D10).**

Fits: 960 of 960 R2-D2 fits and 240 of 240 control refits at their stored budget (the full array's counts).

PASS and FAIL are read only on a complete tree written at one commit on clean checkouts: every (family, seed, t) of the design below replayed, a same-device cost pair at t = 199 for every R2-D2 cell, and every session of every replay directory at one commit with a clean checkout (Provenance). Anything less is INCONCLUSIVE, naming what is missing; a stop is reported whatever the tree holds.

## Completeness

The design (slurm/r2d2_replay.sbatch): each R2-D2 cell on its twin's stored runs of every family under the stored root, each control on its own runs of aligned10 and decoupled, seeds 0-9, t = 50, 100, 199. A control's fits count at its stored budget only.

| cell | design fits | replayed | missing | cost pair at t = 199 | first missing |
|---|---|---|---|---|---|
| additive/amplitude_r2d2 | 240 | 240 | 0 | yes |  |
| additive/lengthscale_r2d2 | 240 | 240 | 0 | yes |  |
| product/amplitude_r2d2 | 240 | 240 | 0 | yes |  |
| product/lengthscale_r2d2 | 240 | 240 | 0 | yes |  |
| additive/amplitude (control) | 60 | 60 | 0 |  |  |
| additive/lengthscale (control) | 60 | 60 | 0 |  |  |
| product/amplitude (control) | 60 | 60 | 0 |  |  |
| product/lengthscale (control) | 60 | 60 | 0 |  |  |

## R2-D2 cells

FAIL at an exclusion rate of 0.9 or more or a median n_eff_min of 10 or less, read first; PASS for every R2-D2 cell at an exclusion rate of E* or less and a median n_eff_min of N* or more (ruling R31b), with E* = max(0.6, the half-Cauchy lengthscale twins' stored exclusion rates) = 0.7 and N* = min(16, their stored median n_eff_min) = 14, over the stored fits their R2-D2 cells were joined to: additive/lengthscale: exclusion rate 0.7, median n_eff_min 14, over 240 fits; product/lengthscale: exclusion rate 0.579, median n_eff_min 22.7, over 240 fits. A cell between the bar and FAIL reads neither. The twin's columns are its stored fits of the same (family, seed, t).

| cell | twin | fits | exclusion rate | median n_eff_min | twin's exclusion rate | twin's median n_eff_min | reading |
|---|---|---|---|---|---|---|---|
| additive/amplitude_r2d2 | additive/amplitude | 240 | 1 | 5.76 | 1 | 5.77 | FAIL |
| additive/lengthscale_r2d2 | additive/lengthscale | 240 | 0.0958 | 77.9 | 0.7 | 14 | PASS |
| product/amplitude_r2d2 | product/amplitude | 240 | 1 | 5.89 | 1 | 6.2 | FAIL |
| product/lengthscale_r2d2 | product/lengthscale | 240 | 0.00833 | 92.7 | 0.579 | 22.7 | PASS |

## Harness: control refits against their stored fits

Holds when, over a control cell's refits and the stored fits of the same (family, seed, t), the exclusion rates differ by at most 0.15, the medians of r_hat_max by at most 0.1 and the medians of AP_native by at most 0.1 (ruling R31a: a GPU refit reproduces a fit in distribution only). The paired medians of the absolute differences are information, not gated. Refits at a budget other than the stored fits' are left out.

| cell | fits | exclusion rate | stored | abs delta | median r_hat_max | stored | abs delta | median AP_native | stored | abs delta | paired median abs delta r_hat_max | paired median abs delta AP_native | holds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| additive/amplitude | 60 | 1 | 1 | 0 | 1.51 | 1.53 | 0.0189 | 0.639 | 0.681 | 0.0423 | 0.116 | 0.0468 | yes |
| additive/lengthscale | 60 | 0.7 | 0.817 | 0.117 | 1.13 | 1.17 | 0.0423 | 0.834 | 0.78 | 0.0541 | 0.0787 | 0.0449 | yes |
| product/amplitude | 60 | 1 | 1 | 0 | 1.49 | 1.5 | 0.0114 | 0.66 | 0.665 | 0.00577 | 0.0845 | 0.0536 | yes |
| product/lengthscale | 60 | 0.65 | 0.75 | 0.1 | 1.15 | 1.15 | 0.000916 | 0.61 | 0.576 | 0.0343 | 0.041 | 0.0299 | yes |

## Cost at t = 199: R2-D2 against its twin's control refits on the same device

Medians of fit_wall_s at t = 199 alone, over the (family, seed) where the R2-D2 fit and its twin's control refit ran on one device at one budget; holds at a ratio of 1.5 or less. By then every replay process has fitted its cell once, so the reading holds a fit and its compilation at that n and no one-time start-up (ruling R28); `tables/cost.csv` keeps every t.

| cell | device | pairs | median fit_wall_s | control's median | ratio | holds |
|---|---|---|---|---|---|---|
| additive/amplitude_r2d2 | NVIDIA H200 | 20 | 59.9 | 55 | 1.09 | yes |
| additive/lengthscale_r2d2 | NVIDIA H200 | 20 | 55.4 | 53.8 | 1.03 | yes |
| product/amplitude_r2d2 | NVIDIA H200 | 20 | 61 | 59.7 | 1.02 | yes |
| product/lengthscale_r2d2 | NVIDIA H200 | 20 | 45 | 43.9 | 1.03 | yes |

## Per-group attribution

The share of fits whose group's r_hat_max exceeds 1.1 or whose n_eff_min is below 16; n/a where the cell has no site in the group. If the ell group fails under both prior families, the prior cannot be the lever.

| method | kind | fits | r_hat native | r_hat ell | r_hat global | n_eff native | n_eff ell | n_eff global |
|---|---|---|---|---|---|---|---|---|
| additive/amplitude_r2d2 | replay | 240 | 1 | 1 | 0.1 | 0.988 | 1 | 0.0833 |
| additive/amplitude | stored | 240 | 1 | 1 | 0.25 | 1 | 1 | 0.154 |
| additive/lengthscale_r2d2 | replay | 240 | 0.0417 | n/a | 0.05 | 0.0333 | n/a | 0.05 |
| additive/lengthscale | stored | 240 | 0.629 | n/a | 0.179 | 0.554 | n/a | 0.192 |
| product/amplitude_r2d2 | replay | 240 | 1 | 1 | 0.171 | 0.971 | 1 | 0.0875 |
| product/amplitude | stored | 240 | 1 | 1 | 0.283 | 1 | 1 | 0.167 |
| product/lengthscale_r2d2 | replay | 240 | 0.00833 | n/a | 0.00417 | 0.00417 | n/a | 0 |
| product/lengthscale | stored | 240 | 0.542 | n/a | 0.0625 | 0.412 | n/a | 0.025 |

## Provenance

One row per replay directory and session: its creation, then each resume, with the commit it ran at and whether its checkout was dirty. One commit, every checkout clean.

| replay directory | session | time | commit | dirty |
|---|---|---|---|---|
| aligned10/additive-amplitude/seed00 | created | 2026-09-28T20:24:04.966595+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude/seed01 | created | 2026-09-28T20:26:24.336600+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude/seed02 | created | 2026-09-28T20:28:43.993330+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude/seed03 | created | 2026-09-28T20:31:03.887820+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude/seed04 | created | 2026-09-28T20:33:23.150678+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude/seed05 | created | 2026-09-28T20:35:43.005306+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude/seed06 | created | 2026-09-28T20:38:03.726536+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude/seed07 | created | 2026-09-28T20:40:22.784995+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude/seed08 | created | 2026-09-28T20:42:44.302315+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude/seed09 | created | 2026-09-28T20:45:03.439602+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude_r2d2/seed00 | created | 2026-09-28T18:50:35.276882+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude_r2d2/seed01 | created | 2026-09-28T18:53:39.798450+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude_r2d2/seed02 | created | 2026-09-28T18:56:18.247382+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude_r2d2/seed03 | created | 2026-09-28T18:58:55.992327+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude_r2d2/seed04 | created | 2026-09-28T19:01:33.576857+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude_r2d2/seed05 | created | 2026-09-28T19:04:12.081257+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude_r2d2/seed06 | created | 2026-09-28T19:06:47.434349+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude_r2d2/seed07 | created | 2026-09-28T19:09:22.919230+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude_r2d2/seed08 | created | 2026-09-28T19:11:58.184252+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-amplitude_r2d2/seed09 | created | 2026-09-28T19:14:35.165643+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale/seed00 | created | 2026-09-28T20:28:09.308039+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale/seed01 | created | 2026-09-28T20:30:23.843100+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale/seed02 | created | 2026-09-28T20:32:38.769128+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale/seed03 | created | 2026-09-28T20:34:52.470983+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale/seed04 | created | 2026-09-28T20:37:05.751995+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale/seed05 | created | 2026-09-28T20:39:19.196908+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale/seed06 | created | 2026-09-28T20:41:34.829050+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale/seed07 | created | 2026-09-28T20:43:48.653432+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale/seed08 | created | 2026-09-28T20:46:06.167375+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale/seed09 | created | 2026-09-28T20:48:23.255401+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale_r2d2/seed00 | created | 2026-09-28T18:54:41.005712+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale_r2d2/seed01 | created | 2026-09-28T18:57:11.234090+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale_r2d2/seed02 | created | 2026-09-28T18:59:41.303851+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale_r2d2/seed03 | created | 2026-09-28T19:02:10.649962+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale_r2d2/seed04 | created | 2026-09-28T19:04:39.206552+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale_r2d2/seed05 | created | 2026-09-28T19:07:07.636826+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale_r2d2/seed06 | created | 2026-09-28T19:09:36.254124+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale_r2d2/seed07 | created | 2026-09-28T19:12:05.555327+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale_r2d2/seed08 | created | 2026-09-28T19:14:33.609149+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/additive-lengthscale_r2d2/seed09 | created | 2026-09-28T19:17:00.563896+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude/seed00 | created | 2026-09-28T20:27:39.469764+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude/seed01 | created | 2026-09-28T20:30:16.232966+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude/seed02 | created | 2026-09-28T20:32:52.401847+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude/seed03 | created | 2026-09-28T20:35:28.111143+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude/seed04 | created | 2026-09-28T20:38:05.591919+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude/seed05 | created | 2026-09-28T20:40:42.428339+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude/seed06 | created | 2026-09-28T20:43:21.876147+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude/seed07 | created | 2026-09-28T20:45:59.721497+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude/seed08 | created | 2026-09-28T20:48:38.843668+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude/seed09 | created | 2026-09-28T20:51:16.992193+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude_r2d2/seed00 | created | 2026-09-28T18:52:07.689143+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude_r2d2/seed01 | created | 2026-09-28T18:54:54.230716+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude_r2d2/seed02 | created | 2026-09-28T18:57:41.343097+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude_r2d2/seed03 | created | 2026-09-28T19:00:23.651922+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude_r2d2/seed04 | created | 2026-09-28T19:03:09.061996+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude_r2d2/seed05 | created | 2026-09-28T19:05:51.490997+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude_r2d2/seed06 | created | 2026-09-28T19:08:34.203387+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude_r2d2/seed07 | created | 2026-09-28T19:11:19.715497+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude_r2d2/seed08 | created | 2026-09-28T19:14:06.298393+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-amplitude_r2d2/seed09 | created | 2026-09-28T19:16:51.769545+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale/seed00 | created | 2026-09-28T20:29:13.215945+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale/seed01 | created | 2026-09-28T20:32:01.627170+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale/seed02 | created | 2026-09-28T20:34:21.810667+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale/seed03 | created | 2026-09-28T20:36:41.484062+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale/seed04 | created | 2026-09-28T20:39:02.432270+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale/seed05 | created | 2026-09-28T20:41:22.791619+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale/seed06 | created | 2026-09-28T20:43:44.158764+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale/seed07 | created | 2026-09-28T20:46:05.193570+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale/seed08 | created | 2026-09-28T20:48:25.685779+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale/seed09 | created | 2026-09-28T20:50:46.032567+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale_r2d2/seed00 | created | 2026-09-28T18:56:09.677817+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale_r2d2/seed01 | created | 2026-09-28T18:59:04.370383+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale_r2d2/seed02 | created | 2026-09-28T19:01:31.221272+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale_r2d2/seed03 | created | 2026-09-28T19:03:57.538815+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale_r2d2/seed04 | created | 2026-09-28T19:06:25.080110+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale_r2d2/seed05 | created | 2026-09-28T19:08:50.798485+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale_r2d2/seed06 | created | 2026-09-28T19:11:17.087701+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale_r2d2/seed07 | created | 2026-09-28T19:13:43.862923+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale_r2d2/seed08 | created | 2026-09-28T19:16:10.834638+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned10/product-lengthscale_r2d2/seed09 | created | 2026-09-28T19:18:37.723976+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-amplitude_r2d2/seed00 | created | 2026-09-28T19:10:54.492747+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-amplitude_r2d2/seed01 | created | 2026-09-28T19:13:30.058877+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-amplitude_r2d2/seed02 | created | 2026-09-28T19:16:04.806067+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-amplitude_r2d2/seed03 | created | 2026-09-28T19:18:38.879432+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-amplitude_r2d2/seed04 | created | 2026-09-28T19:21:10.985725+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-amplitude_r2d2/seed05 | created | 2026-09-28T19:23:45.792939+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-amplitude_r2d2/seed06 | created | 2026-09-28T19:26:17.205292+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-amplitude_r2d2/seed07 | created | 2026-09-28T19:28:48.093000+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-amplitude_r2d2/seed08 | created | 2026-09-28T19:31:20.509575+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-amplitude_r2d2/seed09 | created | 2026-09-28T19:33:51.767701+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-lengthscale_r2d2/seed00 | created | 2026-09-28T19:10:54.777015+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-lengthscale_r2d2/seed01 | created | 2026-09-28T19:13:18.783194+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-lengthscale_r2d2/seed02 | created | 2026-09-28T19:15:43.839717+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-lengthscale_r2d2/seed03 | created | 2026-09-28T19:18:07.738883+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-lengthscale_r2d2/seed04 | created | 2026-09-28T19:20:30.513436+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-lengthscale_r2d2/seed05 | created | 2026-09-28T19:22:51.954043+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-lengthscale_r2d2/seed06 | created | 2026-09-28T19:25:11.685828+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-lengthscale_r2d2/seed07 | created | 2026-09-28T19:27:34.080716+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-lengthscale_r2d2/seed08 | created | 2026-09-28T19:29:56.687437+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/additive-lengthscale_r2d2/seed09 | created | 2026-09-28T19:32:17.455784+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-amplitude_r2d2/seed00 | created | 2026-09-28T19:10:54.700151+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-amplitude_r2d2/seed01 | created | 2026-09-28T19:13:41.505068+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-amplitude_r2d2/seed02 | created | 2026-09-28T19:16:25.941478+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-amplitude_r2d2/seed03 | created | 2026-09-28T19:19:10.958199+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-amplitude_r2d2/seed04 | created | 2026-09-28T19:21:54.975392+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-amplitude_r2d2/seed05 | created | 2026-09-28T19:24:38.361560+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-amplitude_r2d2/seed06 | created | 2026-09-28T19:27:20.203066+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-amplitude_r2d2/seed07 | created | 2026-09-28T19:30:07.587417+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-amplitude_r2d2/seed08 | created | 2026-09-28T19:32:49.515414+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-amplitude_r2d2/seed09 | created | 2026-09-28T19:35:33.425135+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-lengthscale_r2d2/seed00 | created | 2026-09-28T19:10:54.777307+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-lengthscale_r2d2/seed01 | created | 2026-09-28T19:13:23.440283+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-lengthscale_r2d2/seed02 | created | 2026-09-28T19:15:53.140967+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-lengthscale_r2d2/seed03 | created | 2026-09-28T19:18:20.245223+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-lengthscale_r2d2/seed04 | created | 2026-09-28T19:20:48.404466+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-lengthscale_r2d2/seed05 | created | 2026-09-28T19:23:13.885538+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-lengthscale_r2d2/seed06 | created | 2026-09-28T19:25:40.656299+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-lengthscale_r2d2/seed07 | created | 2026-09-28T19:28:07.833232+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-lengthscale_r2d2/seed08 | created | 2026-09-28T19:30:35.969804+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| aligned3/product-lengthscale_r2d2/seed09 | created | 2026-09-28T19:33:02.503627+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-amplitude_r2d2/seed00 | created | 2026-09-28T19:12:28.533828+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-amplitude_r2d2/seed01 | created | 2026-09-28T19:15:36.602838+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-amplitude_r2d2/seed02 | created | 2026-09-28T19:18:16.887605+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-amplitude_r2d2/seed03 | created | 2026-09-28T19:20:56.259692+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-amplitude_r2d2/seed04 | created | 2026-09-28T19:23:36.052504+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-amplitude_r2d2/seed05 | created | 2026-09-28T19:26:17.836791+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-amplitude_r2d2/seed06 | created | 2026-09-28T19:28:58.272693+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-amplitude_r2d2/seed07 | created | 2026-09-28T19:31:39.099871+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-amplitude_r2d2/seed08 | created | 2026-09-28T19:34:19.067847+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-amplitude_r2d2/seed09 | created | 2026-09-28T19:36:59.216724+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-lengthscale_r2d2/seed00 | created | 2026-09-28T19:35:49.346550+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-lengthscale_r2d2/seed01 | created | 2026-09-28T19:38:13.779816+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-lengthscale_r2d2/seed02 | created | 2026-09-28T19:40:38.829901+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-lengthscale_r2d2/seed03 | created | 2026-09-28T19:43:02.469913+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-lengthscale_r2d2/seed04 | created | 2026-09-28T19:45:25.768552+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-lengthscale_r2d2/seed05 | created | 2026-09-28T19:47:49.733876+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-lengthscale_r2d2/seed06 | created | 2026-09-28T19:50:11.740800+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-lengthscale_r2d2/seed07 | created | 2026-09-28T19:52:36.609175+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-lengthscale_r2d2/seed08 | created | 2026-09-28T19:54:58.560315+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/additive-lengthscale_r2d2/seed09 | created | 2026-09-28T19:57:21.565376+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-amplitude_r2d2/seed00 | created | 2026-09-28T19:13:56.656748+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-amplitude_r2d2/seed01 | created | 2026-09-28T19:16:39.648765+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-amplitude_r2d2/seed02 | created | 2026-09-28T19:19:23.627375+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-amplitude_r2d2/seed03 | created | 2026-09-28T19:22:10.187945+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-amplitude_r2d2/seed04 | created | 2026-09-28T19:24:56.396162+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-amplitude_r2d2/seed05 | created | 2026-09-28T19:27:42.204950+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-amplitude_r2d2/seed06 | created | 2026-09-28T19:30:25.001387+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-amplitude_r2d2/seed07 | created | 2026-09-28T19:33:10.977841+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-amplitude_r2d2/seed08 | created | 2026-09-28T19:35:57.364887+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-amplitude_r2d2/seed09 | created | 2026-09-28T19:38:43.406794+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-lengthscale_r2d2/seed00 | created | 2026-09-28T19:36:19.136145+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-lengthscale_r2d2/seed01 | created | 2026-09-28T19:38:44.347566+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-lengthscale_r2d2/seed02 | created | 2026-09-28T19:41:10.179283+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-lengthscale_r2d2/seed03 | created | 2026-09-28T19:43:35.409255+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-lengthscale_r2d2/seed04 | created | 2026-09-28T19:45:59.001866+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-lengthscale_r2d2/seed05 | created | 2026-09-28T19:48:22.754340+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-lengthscale_r2d2/seed06 | created | 2026-09-28T19:50:48.484159+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-lengthscale_r2d2/seed07 | created | 2026-09-28T19:53:14.010545+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-lengthscale_r2d2/seed08 | created | 2026-09-28T19:55:37.596659+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| anti_aligned/product-lengthscale_r2d2/seed09 | created | 2026-09-28T19:58:03.020682+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude/seed00 | created | 2026-09-28T20:32:58.619682+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude/seed01 | created | 2026-09-28T20:35:26.426525+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude/seed02 | created | 2026-09-28T20:37:54.710598+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude/seed03 | created | 2026-09-28T20:40:20.387592+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude/seed04 | created | 2026-09-28T20:42:49.539750+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude/seed05 | created | 2026-09-28T20:45:14.660770+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude/seed06 | created | 2026-09-28T20:47:44.517773+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude/seed07 | created | 2026-09-28T20:50:10.592354+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude/seed08 | created | 2026-09-28T20:52:40.667397+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude/seed09 | created | 2026-09-28T20:55:09.840307+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude_r2d2/seed00 | created | 2026-09-28T18:56:48.353012+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude_r2d2/seed01 | created | 2026-09-28T18:59:45.130586+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude_r2d2/seed02 | created | 2026-09-28T19:02:22.465755+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude_r2d2/seed03 | created | 2026-09-28T19:04:59.582419+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude_r2d2/seed04 | created | 2026-09-28T19:07:37.279665+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude_r2d2/seed05 | created | 2026-09-28T19:10:13.965164+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude_r2d2/seed06 | created | 2026-09-28T19:12:51.045284+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude_r2d2/seed07 | created | 2026-09-28T19:15:28.487047+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude_r2d2/seed08 | created | 2026-09-28T19:18:04.998245+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-amplitude_r2d2/seed09 | created | 2026-09-28T19:20:42.473707+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale/seed00 | created | 2026-09-28T20:39:28.216172+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale/seed01 | created | 2026-09-28T20:41:43.222219+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale/seed02 | created | 2026-09-28T20:43:55.844883+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale/seed03 | created | 2026-09-28T20:46:11.869202+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale/seed04 | created | 2026-09-28T20:48:27.502957+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale/seed05 | created | 2026-09-28T20:50:43.892839+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale/seed06 | created | 2026-09-28T20:53:00.922143+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale/seed07 | created | 2026-09-28T20:55:17.410579+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale/seed08 | created | 2026-09-28T20:57:29.792636+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale/seed09 | created | 2026-09-28T20:59:44.620131+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale_r2d2/seed00 | created | 2026-09-28T19:00:14.852844+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale_r2d2/seed01 | created | 2026-09-28T19:02:41.540746+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale_r2d2/seed02 | created | 2026-09-28T19:05:08.853950+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale_r2d2/seed03 | created | 2026-09-28T19:07:35.021080+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale_r2d2/seed04 | created | 2026-09-28T19:10:02.672918+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale_r2d2/seed05 | created | 2026-09-28T19:12:29.522916+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale_r2d2/seed06 | created | 2026-09-28T19:14:57.260358+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale_r2d2/seed07 | created | 2026-09-28T19:17:20.600261+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale_r2d2/seed08 | created | 2026-09-28T19:19:48.924232+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/additive-lengthscale_r2d2/seed09 | created | 2026-09-28T19:22:16.804804+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude/seed00 | created | 2026-09-28T20:38:21.208145+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude/seed01 | created | 2026-09-28T20:41:00.283555+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude/seed02 | created | 2026-09-28T20:43:40.987899+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude/seed03 | created | 2026-09-28T20:46:20.745494+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude/seed04 | created | 2026-09-28T20:48:59.095962+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude/seed05 | created | 2026-09-28T20:51:37.098655+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude/seed06 | created | 2026-09-28T20:54:14.944968+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude/seed07 | created | 2026-09-28T20:56:55.645934+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude/seed08 | created | 2026-09-28T20:59:38.172642+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude/seed09 | created | 2026-09-28T21:02:20.078885+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude_r2d2/seed00 | created | 2026-09-28T18:56:48.582136+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude_r2d2/seed01 | created | 2026-09-28T18:59:45.817472+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude_r2d2/seed02 | created | 2026-09-28T19:02:34.395013+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude_r2d2/seed03 | created | 2026-09-28T19:05:24.379112+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude_r2d2/seed04 | created | 2026-09-28T19:08:15.108320+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude_r2d2/seed05 | created | 2026-09-28T19:11:00.963834+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude_r2d2/seed06 | created | 2026-09-28T19:13:51.478431+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude_r2d2/seed07 | created | 2026-09-28T19:16:41.918026+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude_r2d2/seed08 | created | 2026-09-28T19:19:29.332344+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-amplitude_r2d2/seed09 | created | 2026-09-28T19:22:19.916189+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale/seed00 | created | 2026-09-28T20:39:29.141517+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale/seed01 | created | 2026-09-28T20:41:50.588182+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale/seed02 | created | 2026-09-28T20:44:11.324131+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale/seed03 | created | 2026-09-28T20:46:31.219895+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale/seed04 | created | 2026-09-28T20:48:51.852069+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale/seed05 | created | 2026-09-28T20:51:13.692911+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale/seed06 | created | 2026-09-28T20:53:34.881330+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale/seed07 | created | 2026-09-28T20:55:55.431990+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale/seed08 | created | 2026-09-28T20:58:15.663514+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale/seed09 | created | 2026-09-28T21:00:35.532204+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale_r2d2/seed00 | created | 2026-09-28T19:05:18.904124+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale_r2d2/seed01 | created | 2026-09-28T19:07:49.579449+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale_r2d2/seed02 | created | 2026-09-28T19:10:17.659709+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale_r2d2/seed03 | created | 2026-09-28T19:12:44.989398+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale_r2d2/seed04 | created | 2026-09-28T19:15:12.282140+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale_r2d2/seed05 | created | 2026-09-28T19:17:40.842755+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale_r2d2/seed06 | created | 2026-09-28T19:20:08.588696+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale_r2d2/seed07 | created | 2026-09-28T19:22:37.688704+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale_r2d2/seed08 | created | 2026-09-28T19:25:03.441003+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| decoupled/product-lengthscale_r2d2/seed09 | created | 2026-09-28T19:27:32.731761+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-amplitude_r2d2/seed00 | created | 2026-09-28T19:41:54.673175+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-amplitude_r2d2/seed01 | created | 2026-09-28T19:44:30.408545+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-amplitude_r2d2/seed02 | created | 2026-09-28T19:47:05.975291+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-amplitude_r2d2/seed03 | created | 2026-09-28T19:49:41.619032+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-amplitude_r2d2/seed04 | created | 2026-09-28T19:52:17.947316+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-amplitude_r2d2/seed05 | created | 2026-09-28T19:54:53.592566+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-amplitude_r2d2/seed06 | created | 2026-09-28T19:57:30.108029+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-amplitude_r2d2/seed07 | created | 2026-09-28T20:00:05.518692+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-amplitude_r2d2/seed08 | created | 2026-09-28T20:02:41.816288+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-amplitude_r2d2/seed09 | created | 2026-09-28T20:05:17.939515+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-lengthscale_r2d2/seed00 | created | 2026-09-28T19:44:26.386066+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-lengthscale_r2d2/seed01 | created | 2026-09-28T19:46:50.499799+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-lengthscale_r2d2/seed02 | created | 2026-09-28T19:49:14.573349+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-lengthscale_r2d2/seed03 | created | 2026-09-28T19:51:37.809718+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-lengthscale_r2d2/seed04 | created | 2026-09-28T19:54:01.635635+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-lengthscale_r2d2/seed05 | created | 2026-09-28T19:56:26.631670+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-lengthscale_r2d2/seed06 | created | 2026-09-28T19:58:51.047301+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-lengthscale_r2d2/seed07 | created | 2026-09-28T20:01:14.989712+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-lengthscale_r2d2/seed08 | created | 2026-09-28T20:03:39.116489+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/additive-lengthscale_r2d2/seed09 | created | 2026-09-28T20:06:02.709899+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-amplitude_r2d2/seed00 | created | 2026-09-28T19:41:55.400339+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-amplitude_r2d2/seed01 | created | 2026-09-28T19:44:44.873489+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-amplitude_r2d2/seed02 | created | 2026-09-28T19:47:34.834842+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-amplitude_r2d2/seed03 | created | 2026-09-28T19:50:24.293191+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-amplitude_r2d2/seed04 | created | 2026-09-28T19:53:14.621362+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-amplitude_r2d2/seed05 | created | 2026-09-28T19:56:04.158349+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-amplitude_r2d2/seed06 | created | 2026-09-28T19:58:49.943488+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-amplitude_r2d2/seed07 | created | 2026-09-28T20:01:39.586762+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-amplitude_r2d2/seed08 | created | 2026-09-28T20:04:26.749012+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-amplitude_r2d2/seed09 | created | 2026-09-28T20:07:11.437663+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-lengthscale_r2d2/seed00 | created | 2026-09-28T19:46:58.598004+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-lengthscale_r2d2/seed01 | created | 2026-09-28T19:49:30.622595+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-lengthscale_r2d2/seed02 | created | 2026-09-28T19:52:02.794082+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-lengthscale_r2d2/seed03 | created | 2026-09-28T19:54:34.093970+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-lengthscale_r2d2/seed04 | created | 2026-09-28T19:57:06.488630+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-lengthscale_r2d2/seed05 | created | 2026-09-28T19:59:38.031298+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-lengthscale_r2d2/seed06 | created | 2026-09-28T20:02:10.266895+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-lengthscale_r2d2/seed07 | created | 2026-09-28T20:04:41.735328+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-lengthscale_r2d2/seed08 | created | 2026-09-28T20:07:12.375874+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.00/product-lengthscale_r2d2/seed09 | created | 2026-09-28T20:09:45.190397+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-amplitude_r2d2/seed00 | created | 2026-09-28T19:50:00.936673+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-amplitude_r2d2/seed01 | created | 2026-09-28T19:52:37.874111+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-amplitude_r2d2/seed02 | created | 2026-09-28T19:55:14.053712+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-amplitude_r2d2/seed03 | created | 2026-09-28T19:57:49.640089+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-amplitude_r2d2/seed04 | created | 2026-09-28T20:00:26.138173+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-amplitude_r2d2/seed05 | created | 2026-09-28T20:03:02.495664+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-amplitude_r2d2/seed06 | created | 2026-09-28T20:05:38.637624+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-amplitude_r2d2/seed07 | created | 2026-09-28T20:08:16.491307+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-amplitude_r2d2/seed08 | created | 2026-09-28T20:10:52.804016+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-amplitude_r2d2/seed09 | created | 2026-09-28T20:13:28.850042+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-lengthscale_r2d2/seed00 | created | 2026-09-28T20:00:39.919683+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-lengthscale_r2d2/seed01 | created | 2026-09-28T20:03:01.564051+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-lengthscale_r2d2/seed02 | created | 2026-09-28T20:05:24.362877+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-lengthscale_r2d2/seed03 | created | 2026-09-28T20:07:44.047450+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-lengthscale_r2d2/seed04 | created | 2026-09-28T20:10:02.617561+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-lengthscale_r2d2/seed05 | created | 2026-09-28T20:12:24.764800+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-lengthscale_r2d2/seed06 | created | 2026-09-28T20:14:43.260676+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-lengthscale_r2d2/seed07 | created | 2026-09-28T20:17:04.223351+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-lengthscale_r2d2/seed08 | created | 2026-09-28T20:19:26.625692+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/additive-lengthscale_r2d2/seed09 | created | 2026-09-28T20:21:46.447972+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-amplitude_r2d2/seed00 | created | 2026-09-28T20:00:09.890654+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-amplitude_r2d2/seed01 | created | 2026-09-28T20:02:58.091153+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-amplitude_r2d2/seed02 | created | 2026-09-28T20:05:44.797714+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-amplitude_r2d2/seed03 | created | 2026-09-28T20:08:29.465747+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-amplitude_r2d2/seed04 | created | 2026-09-28T20:11:19.883255+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-amplitude_r2d2/seed05 | created | 2026-09-28T20:14:04.295879+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-amplitude_r2d2/seed06 | created | 2026-09-28T20:16:50.410449+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-amplitude_r2d2/seed07 | created | 2026-09-28T20:19:39.461135+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-amplitude_r2d2/seed08 | created | 2026-09-28T20:22:24.938813+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-amplitude_r2d2/seed09 | created | 2026-09-28T20:25:08.271424+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-lengthscale_r2d2/seed00 | created | 2026-09-28T20:14:52.461580+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-lengthscale_r2d2/seed01 | created | 2026-09-28T20:17:18.742645+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-lengthscale_r2d2/seed02 | created | 2026-09-28T20:19:47.412718+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-lengthscale_r2d2/seed03 | created | 2026-09-28T20:22:12.952456+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-lengthscale_r2d2/seed04 | created | 2026-09-28T20:24:38.874089+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-lengthscale_r2d2/seed05 | created | 2026-09-28T20:27:02.963789+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-lengthscale_r2d2/seed06 | created | 2026-09-28T20:29:29.144760+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-lengthscale_r2d2/seed07 | created | 2026-09-28T20:31:55.495233+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-lengthscale_r2d2/seed08 | created | 2026-09-28T20:34:19.696997+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.25/product-lengthscale_r2d2/seed09 | created | 2026-09-28T20:36:45.326994+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-amplitude_r2d2/seed00 | created | 2026-09-28T20:15:56.803418+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-amplitude_r2d2/seed01 | created | 2026-09-28T20:19:02.236251+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-amplitude_r2d2/seed02 | created | 2026-09-28T20:21:42.198247+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-amplitude_r2d2/seed03 | created | 2026-09-28T20:24:21.885431+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-amplitude_r2d2/seed04 | created | 2026-09-28T20:26:59.913503+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-amplitude_r2d2/seed05 | created | 2026-09-28T20:29:38.813935+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-amplitude_r2d2/seed06 | created | 2026-09-28T20:32:15.818606+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-amplitude_r2d2/seed07 | created | 2026-09-28T20:34:51.809350+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-amplitude_r2d2/seed08 | created | 2026-09-28T20:37:29.788573+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-amplitude_r2d2/seed09 | created | 2026-09-28T20:40:07.820383+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-lengthscale_r2d2/seed00 | created | 2026-09-28T20:16:57.197045+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-lengthscale_r2d2/seed01 | created | 2026-09-28T20:19:18.792203+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-lengthscale_r2d2/seed02 | created | 2026-09-28T20:21:42.610710+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-lengthscale_r2d2/seed03 | created | 2026-09-28T20:24:03.913860+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-lengthscale_r2d2/seed04 | created | 2026-09-28T20:26:25.448887+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-lengthscale_r2d2/seed05 | created | 2026-09-28T20:28:45.393954+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-lengthscale_r2d2/seed06 | created | 2026-09-28T20:31:06.675853+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-lengthscale_r2d2/seed07 | created | 2026-09-28T20:33:27.746078+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-lengthscale_r2d2/seed08 | created | 2026-09-28T20:35:49.118663+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/additive-lengthscale_r2d2/seed09 | created | 2026-09-28T20:38:13.539270+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-amplitude_r2d2/seed00 | created | 2026-09-28T20:16:26.367454+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-amplitude_r2d2/seed01 | created | 2026-09-28T20:19:10.456925+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-amplitude_r2d2/seed02 | created | 2026-09-28T20:21:58.863219+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-amplitude_r2d2/seed03 | created | 2026-09-28T20:24:42.138451+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-amplitude_r2d2/seed04 | created | 2026-09-28T20:27:28.095751+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-amplitude_r2d2/seed05 | created | 2026-09-28T20:30:10.018592+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-amplitude_r2d2/seed06 | created | 2026-09-28T20:32:51.866770+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-amplitude_r2d2/seed07 | created | 2026-09-28T20:35:38.734167+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-amplitude_r2d2/seed08 | created | 2026-09-28T20:38:25.164308+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-amplitude_r2d2/seed09 | created | 2026-09-28T20:41:12.855624+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-lengthscale_r2d2/seed00 | created | 2026-09-28T20:17:28.373747+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-lengthscale_r2d2/seed01 | created | 2026-09-28T20:19:55.385622+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-lengthscale_r2d2/seed02 | created | 2026-09-28T20:22:24.247024+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-lengthscale_r2d2/seed03 | created | 2026-09-28T20:24:54.090937+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-lengthscale_r2d2/seed04 | created | 2026-09-28T20:27:23.741153+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-lengthscale_r2d2/seed05 | created | 2026-09-28T20:29:52.374628+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-lengthscale_r2d2/seed06 | created | 2026-09-28T20:32:20.673617+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-lengthscale_r2d2/seed07 | created | 2026-09-28T20:34:49.551793+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-lengthscale_r2d2/seed08 | created | 2026-09-28T20:37:19.424334+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.50/product-lengthscale_r2d2/seed09 | created | 2026-09-28T20:39:45.131095+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-amplitude_r2d2/seed00 | created | 2026-09-28T20:17:28.373750+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-amplitude_r2d2/seed01 | created | 2026-09-28T20:20:03.478274+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-amplitude_r2d2/seed02 | created | 2026-09-28T20:22:35.548754+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-amplitude_r2d2/seed03 | created | 2026-09-28T20:25:11.953948+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-amplitude_r2d2/seed04 | created | 2026-09-28T20:27:45.204498+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-amplitude_r2d2/seed05 | created | 2026-09-28T20:30:16.833281+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-amplitude_r2d2/seed06 | created | 2026-09-28T20:32:51.502025+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-amplitude_r2d2/seed07 | created | 2026-09-28T20:35:25.839357+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-amplitude_r2d2/seed08 | created | 2026-09-28T20:37:57.877677+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-amplitude_r2d2/seed09 | created | 2026-09-28T20:40:28.755204+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-lengthscale_r2d2/seed00 | created | 2026-09-28T20:21:02.113215+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-lengthscale_r2d2/seed01 | created | 2026-09-28T20:23:18.493303+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-lengthscale_r2d2/seed02 | created | 2026-09-28T20:25:35.164454+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-lengthscale_r2d2/seed03 | created | 2026-09-28T20:27:52.126941+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-lengthscale_r2d2/seed04 | created | 2026-09-28T20:30:08.188497+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-lengthscale_r2d2/seed05 | created | 2026-09-28T20:32:25.916718+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-lengthscale_r2d2/seed06 | created | 2026-09-28T20:34:42.897039+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-lengthscale_r2d2/seed07 | created | 2026-09-28T20:36:58.850880+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-lengthscale_r2d2/seed08 | created | 2026-09-28T20:39:17.181004+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/additive-lengthscale_r2d2/seed09 | created | 2026-09-28T20:41:38.999825+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-amplitude_r2d2/seed00 | created | 2026-09-28T20:17:28.220706+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-amplitude_r2d2/seed01 | created | 2026-09-28T20:20:12.603042+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-amplitude_r2d2/seed02 | created | 2026-09-28T20:22:53.868893+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-amplitude_r2d2/seed03 | created | 2026-09-28T20:25:35.203602+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-amplitude_r2d2/seed04 | created | 2026-09-28T20:28:15.992938+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-amplitude_r2d2/seed05 | created | 2026-09-28T20:30:56.690150+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-amplitude_r2d2/seed06 | created | 2026-09-28T20:33:38.031268+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-amplitude_r2d2/seed07 | created | 2026-09-28T20:36:18.275277+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-amplitude_r2d2/seed08 | created | 2026-09-28T20:38:59.199154+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-amplitude_r2d2/seed09 | created | 2026-09-28T20:41:44.899399+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-lengthscale_r2d2/seed00 | created | 2026-09-28T20:23:35.408727+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-lengthscale_r2d2/seed01 | created | 2026-09-28T20:26:04.628542+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-lengthscale_r2d2/seed02 | created | 2026-09-28T20:28:36.809480+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-lengthscale_r2d2/seed03 | created | 2026-09-28T20:31:04.258185+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-lengthscale_r2d2/seed04 | created | 2026-09-28T20:33:34.239134+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-lengthscale_r2d2/seed05 | created | 2026-09-28T20:36:04.209325+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-lengthscale_r2d2/seed06 | created | 2026-09-28T20:38:33.383430+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-lengthscale_r2d2/seed07 | created | 2026-09-28T20:41:05.037273+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-lengthscale_r2d2/seed08 | created | 2026-09-28T20:43:36.251316+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |
| interaction_g0.75/product-lengthscale_r2d2/seed09 | created | 2026-09-28T20:46:04.129703+00:00 | 76cbb7994ff0c82405a9462d037ec8674fd8ac9c | no |

## Tables

- `tables/replay_vs_stored.csv`: one row per replayed (cell, family, seed, t) -- the replayed fit, its source's stored fit (`stored_`), and for an R2-D2 cell its twin's control refit where one exists (`control_`).
- `tables/gate_by_cell_t.csv`: the gate statistics per cell and t, and over all t (`at_tree_cap`: every sampling iteration at 2^depth - 1 leapfrog steps).
- `tables/identification.csv`: AP of native_median and of sobol_hat, and precision, recall and F1 of p_active > 0.5, against S.
- `tables/cost.csv`: fit_wall_s by cell, device and t, every t (the cost criterion reads t = 199).
