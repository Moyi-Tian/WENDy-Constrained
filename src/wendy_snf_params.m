%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% WENDy: covariance-corrected ODE parameter estimation
%%%%%%%%%%%% Copyright 2023, All Rights Reserved
%%%%%%%%%%%% Code by Daniel Ames Messenger
%%%%%%%%%%%%
%%%%%%%%%%%% Modified 2026 by Moyi Tian for WENDy-Constrained:
%%%%%%%%%%%%   test-function exponent renamed eta -> phi_eta, since this file runs as a
%%%%%%%%%%%%   script and a bare 'eta' would clobber a model parameter of that name.

%% set and forget (iteration params)

%%% smooth data
toggle_smooth = 0;

%%% set optimization params
optim_params = {'meth','LS'};

%%% set weak integration
% Test-function exponent. Named phi_eta, NOT eta, because this file is run as
% a script and writes into the caller's workspace -- a bare 'eta' would silently
% clobber the model's eta parameter.
phi_eta = 9;
phifuns = {@(x) exp(-phi_eta*(1-x.^2).^(-1)), @(x) (1-x.^2).^phi_eta};
center_scheme = 'uni';
toggle_VVp_svd = NaN; % 0, no SVD reduction; in (0,1), truncates Frobenious norm; NaN, truncates SVD according to cornerpoint of cumulative sum of singular values
% NB: this truncates the TEST-FUNCTION basis (V depends on tobs and the radii,
% not on xobs), so it is a basis-conditioning setting, not a noise filter.
% Stacking several radii makes the basis highly redundant; keep NaN for every
% noise level so the sigma_NR -> 0 case is the same estimator as the rest.
submt = 2.1;

%%% set jacobian correction params
max_iter = 100;
iter_diff_tol = 10^-10;
err_norm = 2;
diag_reg = 10^-10; %% arbitrary low value to avoid warnings
check_pval_it = 10;
pvalmin = 10^-4;
w0 = []; % [], weak-form OLS initial guess
