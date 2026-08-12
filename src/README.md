# src — WENDy solver and weak-form machinery

This folder is derived from the WENDy implementation at [**MathBioCU/WENDy**](https://github.com/MathBioCU/WENDy) (main branch, as downloaded on April 3, 2025), which accompanies:

> Bortz, D. M., Messenger, D. A. and Dukic, V.
> **Direct Estimation of Parameters in ODE Models Using WENDy: Weak-Form Estimation of Nonlinear Dynamics.**
> *Bulletin of Mathematical Biology* **85**, 110 (2023).
> [doi:10.1007/s11538-023-01208-6](https://doi.org/10.1007/s11538-023-01208-6)

Every file keeps its original copyright header, and every change carries a note in that header describing it. The key files changed, where the constrained coefficient map is implemented, are:

| file | change |
| :-- | :-- |
| `wendy_fcn.m` | derived from upstream `wendy_fcn_0.m` and renamed, since this repository keeps only one solver; full coefficient vector is the affine map `w = S*theta + C`, with `S` and `C` both optional; whitened solve simplified; resolved `S` and `C` returned |
| `display_wendy_results_no_noise.m` | adapted from `display_wendy_results.m`: starred quantities built from the clean library and corrected for `C`; bounds drawn with `errorbar` |
| `display_wendy_results_with_noise.m` | as above, for the noisy case |
| `get_Lfac.m` | takes `numeq` explicitly; indexes one shared feature library across all equations |
| `get_RT.m` | removed a duplicated `chol()` call |
| `wendy_snf_params.m` | test-function exponent renamed `eta` to `phi_eta` to avoid clobbering a model parameter of the same name |

Elsewhere the code is upstream's, with small bug fixes and cosmetic changes that do not alter the main functionality. Each is commented at the point of the change, and the file header notes that a fix was applied. These are `VVp_svd.m` (a branch referenced an undefined variable and errored when reached), `wendy_opt.m` (the declared function name was misspelled, the documented `meth` options disagreed with the branches, and an unknown option failed with an unhelpful message) and `els.m` (an unknown averaging method failed the same way).

One upstream quirk is documented rather than changed: `VVp_svd.m` defines a local `getcorner` that shadows `getcorner.m`, and the two score the corner differently (relative $\ell_1$ against relative $\ell_2$), so the criterion depends on which function calls it. Altering that would move the SVD truncation point and change results, so it is left as upstream wrote it, with a comment at the call site.

Upstream driver and demonstration scripts that this repository does not use have been removed; see the parent `README.md` for the examples provided here instead.
