# WENDy-Constrained

This repository extends WENDy with two ways to constrain the coefficients of an ODE system: a **structure matrix** that ties coefficients together across equations, and an **offset vector** that fixes coefficients known in advance. Both are demonstrated on the fully-mixed two-layer online-offline engagement model of [Tian, Brantingham, and Rodríguez (2026)](https://doi.org/10.1093/comnet/cnaf057).


## Contents

- [Dependency](#dependency)
- [The Example Model](#the-example-model)
- [Constrained Coefficients: the Structure Matrix and the Offset Vector](#constrained-coefficients-the-structure-matrix-and-the-offset-vector)
  - [Weak-form setup](#weak-form-setup)
  - [Dimensions](#dimensions)
  - [Structure matrix S](#structure-matrix-s)
  - [Offset vector C](#offset-vector-c)
  - [Supported cases](#supported-cases)
- [Running the Examples](#running-the-examples)
- [Repository Structure](#repository-structure)
- [Citation](#citation)


## Dependency

This repository uses the WENDy implementation from [**MathBioCU/WENDy**](https://github.com/MathBioCU/WENDy), by Bortz, Messenger, and Dukic (full reference under [Citation](#citation)).

The experiments were run using the main branch of `WENDy` as downloaded on April 3, 2025.


## The Example Model

The demonstration uses the fully-mixed two-layer online-offline engagement model. Writing $\vec{x}(t) = \big(U(t), E(t), D(t), P(t), R(t)\big)$ for the aggregated population fractions, the deterministic system is

$$
\begin{aligned}
\frac{dU}{dt} &= -\beta U E - \theta U P, \\
\frac{dE}{dt} &= \beta U E + \theta U P - (\eta + \gamma_i) E, \\
\frac{dD}{dt} &= (\eta + \gamma_i) E, \\
\frac{dP}{dt} &= \eta E - \gamma_p P, \\
\frac{dR}{dt} &= \gamma_p P,
\end{aligned}
$$

with $U, E, D, P, R \in [0,1]$.

| state | meaning |
| :-- | :-- |
| *U* | not engaged / uninterested online |
| *E* | engaged online |
| *D* | done engaging / disengaged online |
| *P* | participating offline |
| *R* | done participating offline |

| parameter | meaning |
| :-- | :-- |
| β | transmission rate online, *U* → *E* |
| θ | transmission rate from offline participation feedback, *U* → *E* |
| η | spillover rate from online engagement to offline participation, *E* → *P* |
| γ<sub>i</sub> | recovery rate online, *E* → *D* |
| γ<sub>p</sub> | recovery rate offline, *P* → *R* |

This system motivates **both features of** this package. First, the coefficient $-(\eta + \gamma_i)$ appears as a **sum**, so the same two parameters are tied together across the $\dot{E}$ and $\dot{D}$ equations. Second, several coefficients are related by sign or shared outright between equations, because mass leaving one compartment must enter another. An unconstrained solver cannot see any of that — **hence the structure matrix**. Lastly, as is common in epidemiological models, a practical situation is that some parameters are known while others are to be inferred from data — **as in one of the demos below, where the recovery rates are treated as known. This is what the offset vector provides for.**


## Constrained Coefficients: the Structure Matrix and the Offset Vector

### Weak-form setup

Rather than differentiating noisy data, WENDy multiplies the ODE by smooth, compactly supported test functions and integrates by parts, moving the time derivative off the data and onto the test functions. Discretising that operation produces two matrices: $\Phi$, whose rows hold the test functions sampled on the time grid, and $\dot{\Phi}$, holding their derivatives. Write $X$ for the observed trajectory (the $M \times d$ array of $M$ time points by $d$ states, `xobs` in the code) and $\Theta(X)$ for the **feature library**, the $M \times J$ matrix whose columns are the $J$ distinct nonlinear terms appearing anywhere on the right-hand side, evaluated at the data.

In this notation the weak-form residual of the unconstrained problem is linear in the coefficient vector. This package replaces that vector by an **affine map**

$$
\vec{w} = S \vec{\theta} + C ,
$$

where $\vec{w} \in \mathbb{R}^{Jd}$ collects every right-hand-side coefficient of the system, $\vec{\theta} \in \mathbb{R}^{q}$ is the shorter vector that actually gets estimated, $S$ encodes dependencies among coefficients, and $C$ holds coefficients that are known in advance. The residual becomes

$$
\vec{r}(X, \vec{\theta}) = \big[\mathbb{I}_d \otimes \big(\Phi \Theta(X)\big)\big]\big(S\vec{\theta} + C\big) + \mathrm{vec}\big(\dot{\Phi} X\big),
$$

with $\mathbb{I}_d$ the $d \times d$ identity, $\otimes$ the Kronecker product — which applies the same test-function integration to each of the $d$ equations — and $\mathrm{vec}(\cdot)$ the operation stacking a matrix's columns into a single long vector. Setting $\vec{r} = 0$ gives the linear system $G \vec{\theta} = b$ with

$$
G = \big[\mathbb{I}_d \otimes (\Phi\Theta)\big] S,
\qquad
b = - \mathrm{vec}\big(\dot{\Phi} X\big) - \big[\mathbb{I}_d \otimes (\Phi\Theta)\big] C .
$$

Note where each ingredient lands: $S$ multiplies into the system matrix, because it says how the unknowns combine, while $C$ is fully known and so is evaluated once and moved to the right-hand side.

Because $X$ carries noise, the residual is correlated and unequally scaled across rows, so least squares on $G\vec{\theta} = b$ is not efficient. WENDy corrects for this by modelling the residual through a covariance factor $L_{\vec{w}}$, which is built from the **full** coefficient vector, not from $\vec{\theta}$:

$$
L_{\vec{w}} = \big[\mathrm{mat}(S\vec{\theta} + C)^{\top} \otimes \Phi\big] \nabla\Theta \Pi + \big[\mathbb{I}_d \otimes \dot{\Phi}\big],
\qquad
\vec{r} \sim \mathcal{N}\big(0, \sigma^2 L_{\vec{w}} L_{\vec{w}}^{\top}\big).
$$

Here $\mathrm{mat}(\cdot)$ inverts $\mathrm{vec}$, reshaping the $Jd$ coefficients back into a $J \times d$ array; $\nabla\Theta$ is the Jacobian of the feature library with respect to the states; $\Pi$ is a fixed permutation aligning the Kronecker index ordering; and $\sigma$ is the noise standard deviation. This factor is what makes the estimator a generalized least squares problem, and it is why $S$ and $C$ have to be resolved before the covariance is formed — the noise propagates through the coefficients that multiply the data, which are $\vec{w}$, not $\vec{\theta}$.

### Dimensions

| symbol | meaning | value for this example |
| :-- | :-- | :--: |
| *d* | number of states (= number of equations) | 5 |
| *J* | number of distinct features in Θ | 4 |
| *q* | number of weights to learn | 5 (or 3 with offsets) |
| *S* | structure matrix, *Jd* × *q* | 20 × 5 |
| *C* | offset vector, *Jd* × 1 | 20 × 1 |

The feature library holds the $J = 4$ distinct nonlinearities appearing anywhere on the right-hand side,

$$
\Theta(X) = \big[ U E \big| U P \big| E \big| P \big],
$$

and the full coefficient vector $\vec{w} \in \mathbb{R}^{Jd} = \mathbb{R}^{20}$ stacks, for each of the $5$ equations, the coefficient of each of the $4$ features. Most of those $20$ entries are zero, several are equal or opposite, and only $q = 5$ numbers are free.

### Structure matrix $S$

$S$ maps the $q$ free parameters onto the $Jd$ coefficients. Row $(i-1)J + j$ of $S$ gives the coefficient of feature $j$ in equation $i$, so reading a row against $\vec{\theta} = (\beta, \theta, \eta, \gamma_i, \gamma_p)^{\top}$ recovers the model above. For this example $S$ is $20 \times 5$:

| equation | feature | β | θ | η | γ<sub>i</sub> | γ<sub>p</sub> |
| :-- | :-- | :--: | :--: | :--: | :--: | :--: |
| d*U*/d*t* | *UE* | -1 | 0 | 0 | 0 | 0 |
|  | *UP* | 0 | -1 | 0 | 0 | 0 |
|  | *E* | 0 | 0 | 0 | 0 | 0 |
|  | *P* | 0 | 0 | 0 | 0 | 0 |
| d*E*/d*t* | *UE* | 1 | 0 | 0 | 0 | 0 |
|  | *UP* | 0 | 1 | 0 | 0 | 0 |
|  | *E* | 0 | 0 | -1 | -1 | 0 |
|  | *P* | 0 | 0 | 0 | 0 | 0 |
| d*D*/d*t* | *UE* | 0 | 0 | 0 | 0 | 0 |
|  | *UP* | 0 | 0 | 0 | 0 | 0 |
|  | *E* | 0 | 0 | 1 | 1 | 0 |
|  | *P* | 0 | 0 | 0 | 0 | 0 |
| d*P*/d*t* | *UE* | 0 | 0 | 0 | 0 | 0 |
|  | *UP* | 0 | 0 | 0 | 0 | 0 |
|  | *E* | 0 | 0 | 1 | 0 | 0 |
|  | *P* | 0 | 0 | 0 | 0 | -1 |
| d*R*/d*t* | *UE* | 0 | 0 | 0 | 0 | 0 |
|  | *UP* | 0 | 0 | 0 | 0 | 0 |
|  | *E* | 0 | 0 | 0 | 0 | 0 |
|  | *P* | 0 | 0 | 0 | 0 | 1 |

Three things this encodes that an unconstrained fit cannot:

- **Structural zeros.** Most rows are identically zero: $UE$ and $UP$ never appear in $\dot{D}$, $\dot{P}$ or $\dot{R}$. The solver is never given the chance to fit them.
- **Sign-flipped sharing.** Rows $(\dot{U}, UE)$ and $(\dot{E}, UE)$ are $-1$ and $+1$ on the same column, so one number $\beta$ serves both equations with opposite sign, exactly as conservation demands.
- **Parameter tying.** Row $(\dot{E}, E)$ has $-1$ in *both* the $\eta$ and $\gamma_i$ columns, expressing the single coefficient $-(\eta + \gamma_i)$ as a sum of two parameters. This is why $q = 5$ rather than $20$.

This is implemented in `FullyMixedModel_Structured.m`.

### Offset vector $C$

$C$ handles coefficients that are **known a priori and should not be estimated**. Where $S$ says *how* coefficients depend on the free parameters, $C$ says *which parts of the answer are already fixed*. In the linear system above, $C$ contributes a known term that is simply moved to the right-hand side — the same role an offset plays in a generalized linear model.

Suppose the two recovery rates $\gamma_i$ and $\gamma_p$ have been measured independently, so only $\vec{\theta} = (\beta, \theta, \eta)^{\top}$ remains unknown. Then $S$ shrinks to $20 \times 3$ — its $\gamma_i$ and $\gamma_p$ columns are dropped — and their contributions reappear in $C$:

| equation | feature | *C* entry |
| :-- | :-- | :--: |
| d*E*/d*t* | *E* | −γ<sub>i</sub> |
| d*D*/d*t* | *E* | +γ<sub>i</sub> |
| d*P*/d*t* | *P* | −γ<sub>p</sub> |
| d*R*/d*t* | *P* | +γ<sub>p</sub> |

with every other entry zero. Row $(\dot{E}, E)$ then reads $S_{(\dot{E},E)}\vec{\theta} + C_{(\dot{E},E)} = -\eta - \gamma_i$, reproducing the original coefficient exactly. This is implemented in `FullyMixedModel_Structured_with_Offset.m`.

**$S$ cannot absorb $C$.** The range of $S$ is a linear subspace and therefore always contains the origin, whereas a nonzero $C$ is a translation off that subspace; no choice of $S$ produces it. The only workaround is homogeneous coordinates — append $C$ as an extra column of $S$ and pin its parameter to $1$ — but pinning a parameter to a known value *is* the offset feature. $S$ and $C$ are the homogeneous and inhomogeneous halves of one affine map, and neither subsumes the other.

Fixing a parameter is also not merely a smaller problem. Because $\eta$ and $\gamma_i$ appear only as the sum $\eta + \gamma_i$ in the $\dot{E}$ and $\dot{D}$ equations; they are separated solely by the $\dot{P}$ equation and are therefore weakly identifiable. Declaring $\gamma_i$ known collapses that degeneracy outright.

### Supported cases

Both matrices are optional. Pass `[]` for either one and `src/wendy_fcn.m` fills in the default — the identity for $S$, the zero vector for $C$ — so all four cases run through a single code path.

| `S` | `C` | **w** | meaning |
| :--: | :--: | :-- | :-- |
| `[]` | `[]` | **w** = θ | plain WENDy: every coefficient free |
| given | `[]` | **w** = *S*θ | structured only |
| `[]` | given | **w** = θ + *C* | known offsets only |
| given | given | **w** = *S*θ + *C* | both |

The resolved `S` and `C` are returned as outputs, so downstream diagnostics never have to re-derive them.


## Running the Examples

Generate the synthetic data first, then run any demo. Each demo `cd`s to its own directory and adds `./src` and `./Generate_Data/data` to the path, so no manual setup is needed.

```matlab
% 1. generate synthetic trajectories (writes into Generate_Data/data/)
run('Generate_Data/Driver.m')          % clean
run('Generate_Data/Driver_noisy.m')    % noisy, controlled by the noise ratio `sigma`

% 2. run a demonstration
InferPars_NoNoise_Structured                 % S only,  no noise
InferPars_NoNoise_Structured_with_Offset     % S and C, no noise
InferPars_WithNoise_Structured               % S only,  with noise
InferPars_WithNoise_Structured_with_Offset   % S and C, with noise
```

| demo | estimates | noise |
| :-- | :-- | :-- |
| `InferPars_NoNoise_Structured` | β, θ, η, γ<sub>i</sub>, γ<sub>p</sub> | none |
| `InferPars_NoNoise_Structured_with_Offset` | β, θ, η | none |
| `InferPars_WithNoise_Structured` | β, θ, η, γ<sub>i</sub>, γ<sub>p</sub> | σ<sub>NR</sub> = 0.05 |
| `InferPars_WithNoise_Structured_with_Offset` | β, θ, η | σ<sub>NR</sub> = 0.05 |

Each script prints the estimated against the true parameters and produces the nine-panel WENDy diagnostic figure. The toggles `save_results`, `write_to_txt`, `save_fig` and `write_to_csv` near the top of each script control whether results, logs and figures are written to disk; the output folders are created automatically on first run.

The diagnostic figure follows the original WENDy code: the same nine panels, showing the iterate errors, confidence bounds, Shapiro–Wilk *p*-values, the data against the learned dynamics, the residual decomposition, and the Fourier content of the data and test functions. The panels here are adapted to the constrained setting — starred quantities account for the offset vector $C$, and the Jacobian is evaluated at the full coefficient vector $S\vec{\theta} + C$ — but the layout and the diagnostics themselves are unchanged, so the figure can be read exactly as in the original.

**Noise convention.** In `Driver_noisy.m` the variable `sigma` is the *noise ratio*: the standard deviation of the additive Gaussian noise expressed as a fraction of each state's RMS amplitude,

$$
X_{\text{noisy}}(t_m) = X_{\text{true}}(t_m) + \varepsilon_m,
\qquad
\varepsilon_m \overset{iid}{\sim} \mathcal{N}(0, \sigma_X^2),
\qquad
\sigma_X = \sigma_{NR} \mathrm{RMS}(X_{\text{true}}).
$$

It is dimensionless, and $\sigma_X$ is computed separately for each state variable, so every component is perturbed relative to its own amplitude.


## Repository Structure

**`src/`** holds the core code: the WENDy solver and the weak-form machinery it depends on.
The entry point is `wendy_fcn.m`, which implements the constrained coefficient map $\vec{w} = S\vec{\theta} + C$ described below. Solver defaults live in `wendy_snf_params.m`, and the two `display_wendy_results_*.m` scripts build the diagnostic figure. The remaining files are test-function construction, covariance factors and utilities. Every file carries the original WENDy copyright header. Six files carry the constrained coefficient map and are listed in `src/README.md`; elsewhere there are only small bug fixes and cosmetic changes that leave the main functionality intact, each noted in the file header and commented at the point of the change.

**`Generate_Data/`** produces the synthetic data the demos consume, and must be run first. `FullyMixedModel_ODE.m` is the right-hand side of the model, while `Driver.m` and `Driver_noisy.m` simulate clean and noisy trajectories respectively and save them to `Generate_Data/data/`.

**Model definitions.** `FullyMixedModel_Structured.m` supplies the feature library, equations and structure matrix $S$ for the five-parameter problem. `FullyMixedModel_Structured_with_Offset.m` supplies $S$ and the offset vector $C$ for the three-parameter problem, in which $\gamma_i$ and $\gamma_p$ are treated as known.

**Demonstrations.** The four `InferPars_*.m` scripts in the repository root are the worked examples, covering the structure matrix alone and the structure matrix with offsets, each without and with noise.


## Citation

If you use the code in this repository, please cite this GitHub page:

```bibtex
@misc{tian2026wendyconstrained,
  author       = {Tian, Moyi},
  title        = {{WENDy-Constrained}},
  year         = {2026},
  howpublished = {\url{https://github.com/Moyi-Tian/WENDy-Constrained}},
  note         = {GitHub repository}
}
```

The code is based on [**MathBioCU/WENDy**](https://github.com/MathBioCU/WENDy), which accompanies:

> Bortz, D. M., Messenger, D. A. and Dukic, V.
> **Direct Estimation of Parameters in ODE Models Using WENDy: Weak-Form Estimation of Nonlinear Dynamics.**
> *Bulletin of Mathematical Biology* **85**, 110 (2023).
> [doi:10.1007/s11538-023-01208-6](https://doi.org/10.1007/s11538-023-01208-6)

The example model is from:

> Tian, M., Brantingham, P. J. and Rodríguez, N.
> **Modelling the spillover from online engagement to offline protest: stochastic dynamics and mean-field approximations on networks.**
> *Journal of Complex Networks* **14**(2), cnaf057 (2026).
> [doi:10.1093/comnet/cnaf057](https://doi.org/10.1093/comnet/cnaf057)
