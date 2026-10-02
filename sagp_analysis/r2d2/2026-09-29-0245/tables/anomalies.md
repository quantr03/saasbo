- DEGENERATE readout: aligned10/additive-lengthscale_r2d2/seed01 declares 52/100 coordinates active at t=199 (AP by native_median 0.83, by sobol_hat 0.86; min native_median on true S 0.481, max on inactive 2.53; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed00 declares 64/100 coordinates active at t=199 (AP by native_median 0.63, by sobol_hat 0.74; min native_median on true S 0.558, max on inactive 3.41; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed02 declares 99/100 coordinates active at t=199 (AP by native_median 0.61, by sobol_hat 0.57; min native_median on true S 3.67, max on inactive 18; fit status ok)
- DEGENERATE readout: decoupled/additive-lengthscale_r2d2/seed03 declares 97/100 coordinates active at t=199 (AP by native_median 0.51, by sobol_hat 0.83; min native_median on true S 2.5, max on inactive 17.9; fit status ok)
- NUTS hits max_tree_depth=6 (num_steps_mean >= 62 of the 63 possible) in most fits: fraction per cell additive-amplitude 1.000, additive-amplitude_r2d2 1.000, additive-lengthscale 0.990, additive-lengthscale_r2d2 0.200, product-amplitude 1.000, product-amplitude_r2d2 1.000, product-lengthscale 0.993, product-lengthscale_r2d2 0.156

Expected oddities:
- Git commits at creation, by prior family: {'HC': ['a51a4b9'], 'R2D2': ['97eda92', 'c4d9c38'], 'reference': ['a51a4b9']} (the HC cells and references were launched at a51a4b9 and 2866ed2; the R2-D2 cells' commits are checked by `prior_code_ok`).
