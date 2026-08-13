% Learning coefficients in the fully mixed model example using WENDy with a
% SPARSITY-ONLY structure matrix
% With Noise
%
% Uses the unified WENDy package in ./src, where the full coefficient vector is
% w = S*theta + C. Here S records only which feature appears in which equation:
% one column per (equation, feature) pair present in the model, holding a single
% 1 in that row. It carries no parameter tying and no sign sharing. That is
% exactly the information the original WENDy gets from per-equation term lists,
% so this run estimates the same 9 coefficients the original solver would.
%
% Passing S = [] would NOT be equivalent. The package then fills S in as the
% identity, which hands every equation the full feature library and estimates
% all 20 coefficients -- a larger problem than the original WENDy ever poses.
%
% This is a counterexample rather than a demonstration. It runs on the same data
% as InferPars_WithNoise_Structured.m, so the two can be compared directly: this
% script gets only the sparsity, while the structured demo also gets the tying
% and sign sharing. Nothing is written to file.

clear
close all

%% cd to current directory
scriptFullName = matlab.desktop.editor.getActiveFilename();
scriptDir = fileparts(scriptFullName);

% change to current folder
if ~isempty(scriptDir)
    cd(scriptDir);
end

%% Add path and load data
cur_dir=pwd;
addpath('./Generate_Data/data',...
    './src')

% numbers in data file to be loaded
pars(1) = 0.6; % beta: transmission rate online NotEngaged/Uninterested(U) -> Engaged(E)
pars(2) = 0.5; % theta: transmission rate due to offline feedback NotEngaged/Uninterested(U) -> Engaged(E)
pars(3) = 0.4; % eta: fraction of Engaged(E) online from UnParticipating(UP) -> Participating(P) offline
pars(4) = 0.1; % gamma_i: recovery rate online Engaged(E) -> DoneEngaging/DisEngaged(D)
pars(5) = 0.2; % gamma_p: recovery rate offline Participating(P) -> DoneParticipating(R)
t_max = 25;
t_num = 100;
E0 = 0.02;

sigma = 0.05; % noise ratio: additive Gaussian noise std as a fraction of each state's RMS amplitude

% load data
fname = sprintf('Noisy Fully-Mixed Dynamics sigma=%.5g, beta=%.3g, theta=%.3g, eta=%.3g, gamma_i=%.3g, gamma_p=%.3g, tmax=%u, t_grid_num=%u, E0=%.3g.mat',sigma,pars(1),pars(2),pars(3),pars(4),pars(5),t_max,t_num,E0);
load(fname)

%% Prepare data
[features,equation_terms,numeq,params,~,rhs_p,~] = FullyMixedModel_Structured(pars(1),pars(2),pars(3),pars(4),pars(5));

% Sparsity-only structure matrix, built from equation_terms so that it stays in
% step with the model file. Features are matched on their text, and the columns
% follow the order equation_terms lists the terms in.
nfeat   = numel(features);
featstr = cellfun(@(f) func2str(f{1}), features, 'uni', 0);
nterms  = length([equation_terms{:}]);
rows    = zeros(nterms,1);
c = 0;
for i = 1:numeq
    for k = 1:numel(equation_terms{i})
        c = c + 1;
        rows(c) = (i-1)*nfeat + find(strcmp(func2str(equation_terms{i}{k}),featstr),1);
    end
end
S = zeros(numeq*nfeat, nterms);
S(sub2ind(size(S), rows, (1:nterms)')) = 1;

C = [];   % no known offsets: src/wendy_fcn fills this in as zeros

% params lists the true coefficient of each term in the same order, so it is the
% truth for the vector being estimated here.
true_vec = [params{:}]';

tobs = time_array;
xobs = X_noisy;  % WENDy sees the noisy trajectories

subsamp = 1; % subsample data in time

xsub = X(1:subsamp:end,:);         % noise-free reference trajectory, for diagnostics only
noise = noise(1:subsamp:end,:);    % noise realization, needed by the with-noise display script
[M,nstates] = size(xsub);
x0 = xsub(1,:)';

%% set WENDy params

%%% set-and-forget params
wendy_snf_params;

%%% noise handling
% toggle_smooth: 0 = no pre-smoothing (WENDy's covariance correction handles
% the noise); 1 = automatic moving-average pre-filter; >1 = fixed half-width.
toggle_smooth = 0;

%%% set weak integration
phifun = phifuns{1};                % defined in wendy_snf_params.m
meth = 'mtmin';                     % 'mtmin','FFT','direct','timefrac'
mt_params = 2.^(0:3);               % see get_rad.m
K_max = 5000;
% Kept identical to the structured demos so that both use the same
% test-function basis: K_min also sets mt_max, hence the radii.
K_min = length([equation_terms{:}]);
mt_max = max(floor((M-1)/2)-K_min,1);
mt_min = rad_select(tobs,xobs,phifun,1,submt,0,1,2,mt_max,[]);
mt_cell = cellfun(@(x,y) [x,{y}], repmat({{phifun,meth}},length(mt_params),1),num2cell(mt_params(:)),'uni',0);

%% post-processing options

toggle_plot = 1;
toggle_ddd = 1;

%% run WENDy
% S and C come back resolved, so the display scripts always receive well-formed
% matrices.

tic;
[w_hat,res,res_0,w_hat_its,V_cell,Vp_cell,...
    Theta_cell,mt,xobs,Jac_mat,G_0,b_0,RT,stdW,mseW,CovW,S,C] = wendy_fcn(...
    xobs,tobs,features,numeq,S,C,toggle_smooth,...
    mt_cell,mt_min,mt_max,K_min,K_max,center_scheme,toggle_VVp_svd,...
    w0,optim_params,iter_diff_tol,max_iter,diag_reg,pvalmin,check_pval_it);
total_time = toc;

%% display results

if toggle_ddd
    ind=inf; % choose which weights to generate from
    w_hat_its_final = w_hat_its(:,min(ind,end));
    % theta holds one coefficient per term, ordered as equation_terms lists them,
    % so it splits straight back into the per-equation cell that rhs_p expects.
    w_jac_cell = cell(numeq,1);
    o = 0;
    for i = 1:numeq
        n = numel(equation_terms{i});
        w_jac_cell{i} = w_hat_its_final(o+(1:n));
        o = o + n;
    end
end

if exist('true_vec','var')
    err_norm = 2;
    errs = arrayfunvec(w_hat_its,@(w)norm(w-true_vec,err_norm)/norm(true_vec,err_norm),1);
    err_wendy = errs(end);
    disp(['err by iterate: ',num2str(errs(:)',' %.4g')])
    disp(['err of returned w_hat: ',num2str(norm(w_hat-true_vec,err_norm)/norm(true_vec,err_norm))])
    disp(['cond(G_0): ',num2str(cond(G_0))])
    disp(['-----------------'])
    disp([' '])
    disp(['err WENDy:',num2str(err_wendy)])
    disp('coefficients: true, WENDy')
    disp([true_vec w_hat])
    G = RT \ G_0;
    b = RT \ b_0;
    res_true = G*true_vec-b;
    res_0_true = G_0*true_vec-b_0;
    display_wendy_results_with_noise;
end
