- DEGENERATE readout: aligned3/additive-lengthscale/seed02 declares 100/100 coordinates active at t=199 (AP by native_median 0.02, by sobol_hat 1.00; min native_median on true S 16.4, max on inactive 907; fit status excluded)
- DEGENERATE readout: aligned3/additive-lengthscale/seed08 declares 100/100 coordinates active at t=199 (AP by native_median 0.04, by sobol_hat 0.92; min native_median on true S 26.9, max on inactive 105; fit status ok)
- NUTS hits max_tree_depth=6 (num_steps_mean >= 62 of the 63 possible) in most fits: fraction per cell additive-amplitude 0.996, additive-lengthscale 0.965, product-amplitude 1.000, product-lengthscale 0.953

Expected oddities:
- aligned3/dsp_map/seed06: last coords.csv readout is t=198, last iteration t=199 (the last iteration was an exception row, so no surrogate and no readout; scored at t=198)
- aligned3/dsp_map/seed08: last coords.csv readout is t=198, last iteration t=199 (the last iteration was an exception row, so no surrogate and no readout; scored at t=198)
