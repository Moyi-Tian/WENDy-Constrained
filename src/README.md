# src — constrained WENDy solver and weak-form machinery

This folder is derived from the WENDy implementation at [**MathBioCU/WENDy**](https://github.com/MathBioCU/WENDy) (main branch, as downloaded on April 3, 2025), which accompanies:

> Bortz, D. M., Messenger, D. A. and Dukic, V.
> **Direct Estimation of Parameters in ODE Models Using WENDy: Weak-Form Estimation of Nonlinear Dynamics.**
> *Bulletin of Mathematical Biology* **85**, 110 (2023).
> [doi:10.1007/s11538-023-01208-6](https://doi.org/10.1007/s11538-023-01208-6)

Every file keeps its original copyright header, and every change carries a note in that header describing it. The files with key changes are:

| file | change |
| :-- | :-- |
| `wendy_fcn.m` | derived from upstream `wendy_fcn_0.m` and renamed, since this repository keeps only one solver; full coefficient vector is the affine map `w = S*theta + C`, with `S` and `C` both optional (`S = []` defaults to the identity, not to per-equation libraries); whitened solve simplified; resolved `S` and `C` returned |
| `display_wendy_results_no_noise.m` | adapted from `display_wendy_results.m`: starred quantities built from the noise-free library and corrected for `C`; bounds drawn with `errorbar` |
| `display_wendy_results_with_noise.m` | as above, for the noisy case |
| `get_Lfac.m` | takes `numeq` explicitly; indexes one shared feature library across all equations |
| `get_RT.m` | removed a duplicated `chol()` call |
| `wendy_snf_params.m` | test-function exponent renamed `eta` to `phi_eta` to avoid clobbering a model parameter of the same name |

Elsewhere, the code is upstream's, apart from small bug fixes and typo corrections in `VVp_svd.m`, `wendy_opt.m` and `els.m`. Each change is commented where it occurs, and the file headers note that a fix was applied.

One thing is documented rather than changed: `VVp_svd.m` defines a local `getcorner` that shadows `getcorner.m`, and the two score the corner differently. Changing it would move the SVD truncation point and alter results, so it is left as upstream wrote it.

Upstream's own driver and demonstration scripts have been removed; this repository's demos live in the root folder and are described in the parent `README.md`.
