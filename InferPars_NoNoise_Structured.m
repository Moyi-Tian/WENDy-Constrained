% Learning coefficients in the fully mixed model example using WENDy with
% Structure Matrix
% No Noise
%
% Uses the unified WENDy package in ./src, where the full coefficient vector is
% w = S*theta + C. Here S is supplied and C is left empty (no known offsets),
% so the package fills C in as zeros.

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

% load data
fname = sprintf('Fully-Mixed Dynamics beta=%.3g, theta=%.3g, eta=%.3g, gamma_i=%.3g, gamma_p=%.3g, tmax=%u, t_grid_num=%u, E0=%.3g.mat',pars(1),pars(2),pars(3),pars(4),pars(5),t_max,t_num,E0);
load(fname)

%% Prepare data
[features,equation_terms,numeq,params,true_vec,rhs_p,S] = FullyMixedModel_Structured(pars(1),pars(2),pars(3),pars(4),pars(5));
C = [];   % no known offsets: src/wendy_fcn fills this in as zeros

tobs = time_array;
xobs = X;

subsamp = 1; % subsample data in time

xsub = xobs(1:subsamp:end,:);
[M,nstates] = size(xsub);
x0 = xsub(1,:)';

% setup what to save onto file
save_results = 1; % = 1 save inference; = 0 not save inference
write_to_txt = 1; % = 1 write results to txt; = 0 do not write
save_fig = 1; % = 1 save figure; = 0 not save figure

%% set WENDy params

%%% set-and-forget params
wendy_snf_params;

%%% set weak integration
phifun = phifuns{1};                % defined in wendy_snf_params.m
meth = 'mtmin';                     % 'mtmin','FFT','direct','timefrac'
mt_params = 2.^(0:3);               % see get_rad.m
K_max = 5000;
K_min = length([equation_terms{:}]);
mt_max = max(floor((M-1)/2)-K_min,1);
mt_min = rad_select(tobs,xobs,phifun,1,submt,0,1,2,mt_max,[]);
mt_cell = cellfun(@(x,y) [x,{y}], repmat({{phifun,meth}},length(mt_params),1),num2cell(mt_params(:)),'uni',0);

%% post-processing options

toggle_plot = 1;
toggle_ddd = 1;

%% run WENDy
% S and C come back resolved (C filled in as zeros here), so the display
% scripts always receive well-formed matrices.

tic;
[w_hat,res,res_0,w_hat_its,V_cell,Vp_cell,...
    Theta_cell,mt,xobs,Jac_mat,G_0,b_0,RT,stdW,mseW,CovW,S,C] = wendy_fcn(...
    xobs,tobs,features,numeq,S,C,toggle_smooth,...
    mt_cell,mt_min,mt_max,K_min,K_max,center_scheme,toggle_VVp_svd,...
    w0,optim_params,iter_diff_tol,max_iter,diag_reg,pvalmin,check_pval_it);
total_time = toc;

%% Save results
if save_results == 1
    result_path = [cur_dir, '/inference_results'];
    if ~exist(fullfile(cur_dir,'inference_results'), 'dir')
        mkdir inference_results
    end
    fname_results = sprintf('WENDy with Structure Matrix Fully-Mixed Dynamics subsamp=%u, beta=%.3g, theta=%.3g, eta=%.3g, gamma_i=%.3g, gamma_p=%.3g, tmax=%u, t_grid_num=%u, E0=%.3g NO NOISE.mat',subsamp,pars(1),pars(2),pars(3),pars(4),pars(5),t_max,t_num,E0);
    save(fullfile(result_path, fname_results), 'true_vec','w_hat','res','res_0','w_hat_its','V_cell','Vp_cell',...
        'Theta_cell','mt','tobs','xobs','xsub','x0','Jac_mat','G_0','b_0','RT','stdW','mseW','CovW','total_time','toggle_plot','toggle_ddd','phifun','M','nstates','features','rhs_p','equation_terms','S','C','numeq');
end

%% display results

if write_to_txt == 1
    % Write output to txt
    logFile = 'results_fully_mixed_structured_NoNoise.txt';
    fid = fopen(logFile,'a');
    if fid < 0
        error('Couldn''t open %s for writing.', logFile);
    end
    % Write one formatted line, then a newline
    fprintf(fid, '\n%s\n', sprintf('Model on Fully-Mixed beta=%.3g, theta=%.3g, eta=%.3g, gamma_i=%.3g, gamma_p=%.3g, tmax=%u, t_grid_num=%u, E0=%.3g',pars(1),pars(2),pars(3),pars(4),pars(5),t_max,t_num,E0));
    fclose(fid);
    diary(logFile);
end

if toggle_ddd
    ind=inf; % choose which weights to generate from
    w_hat_its_final = w_hat_its(:,min(ind,end));
    [~,~,~,w_jac_cell,~,~,~] = FullyMixedModel_Structured(w_hat_its_final(1),w_hat_its_final(2),w_hat_its_final(3),w_hat_its_final(4),w_hat_its_final(5));
end

if exist('true_vec','var')
    err_norm = 2;
    errs = arrayfunvec(w_hat_its,@(w)norm(w-true_vec,err_norm)/norm(true_vec,err_norm),1);
    err_wendy = errs(end);
    disp(['-----------------'])
    disp([' '])
    disp(['err WENDy:',num2str(err_wendy)])
    disp('parameters: true, WENDy')
    disp([true_vec w_hat])
    G = RT \ G_0;
    b = RT \ b_0;
    res_true = G*true_vec-b;
    res_0_true = G_0*true_vec-b_0;
    display_wendy_results_no_noise;
end

if write_to_txt == 1
    % turn diary off when done
    diary off;
end


%% save plots
if save_fig == 1
    fig_path = [cur_dir, '/figures'];
    if ~exist(fullfile(cur_dir,'figures'), 'dir')
        mkdir figures
    end

    fig = gcf;
    fig.Units = 'inches';
    fig.Position(3:4) = [12 18];
    fig_name = sprintf('Inference Results with Structure Matrix for Fully-Mixed Model beta=%.3g, theta=%.3g, eta=%.3g, gamma_i=%.3g, gamma_p=%.3g, tmax=%u, t_grid_num=%u, E0=%.3g NO NOISE.pdf',pars(1),pars(2),pars(3),pars(4),pars(5),t_max,t_num,E0);
    exportgraphics(gcf, fullfile(fig_path,fig_name), 'ContentType','vector');
end
