% Learning coefficients in the fully mixed model example using WENDy with
% Structure Matrix and selected unknown parameters to be inferred
% With Noise
%
% Uses the unified WENDy package in ./src, where the full coefficient vector is
% w = S*theta + C. Here both S and C are supplied: gamma_i and gamma_p are
% known and carried by C, so only beta, theta and eta are estimated.

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
% true_vec is [beta; theta; eta] only: gamma_i and gamma_p are known and are
% carried by the offset vector C, not estimated.
[features,equation_terms,numeq,params,true_vec,rhs_p,S,C] = FullyMixedModel_Structured_with_Offset(pars(1),pars(2),pars(3),pars(4),pars(5));
tobs = time_array;
xobs = X_noisy;  % WENDy sees the noisy trajectories

subsamp = 1; % subsample data in time

xsub = X(1:subsamp:end,:);         % noise-free reference trajectory, for diagnostics only
noise = noise(1:subsamp:end,:);    % noise realization, needed by the with-noise display script
[M,nstates] = size(xsub);
x0 = xsub(1,:)';

% setup what to save onto file
save_results = 1; % = 1 save inference; = 0 not save inference
write_to_txt = 1; % = 1 write results to txt; = 0 do not write
save_fig = 1; % = 1 save figure; = 0 not save figure
write_to_csv = 1; % = 1 write error to csv; = 0 do not write

%% WENDy arguments
% wendy_default_args fills in every argument after the feature library, using
% wendy_snf_params.m for the set-and-forget values and the same recipes the
% original wendy_script.m uses for the derived ones. Any of them can be
% overridden by name.
% toggle_smooth: 0 = no pre-smoothing (WENDy's covariance correction handles
% the noise); 1 = automatic moving-average pre-filter; >1 = fixed half-width.
toggle_smooth = 0;

[args,opts] = wendy_default_args(xobs,tobs,features, ...
    'numeq',numeq,'S',S,'C',C,'toggle_smooth',toggle_smooth);
phifun = opts.phifun;   % the display scripts plot the test function

%% post-processing options

toggle_plot = 1;
toggle_ddd = 1;

%% run WENDy
% S and C come back resolved, so the display scripts always receive
% well-formed matrices.

tic;
[w_hat,res,res_0,w_hat_its,V_cell,Vp_cell,...
    Theta_cell,mt,xobs,Jac_mat,G_0,b_0,RT,stdW,mseW,CovW,S,C] = wendy_fcn(...
    xobs,tobs,features,args{:});
total_time = toc;

%% Save results
if save_results == 1
    result_path = [cur_dir, '/inference_results'];
    if ~exist(fullfile(cur_dir,'inference_results'), 'dir')
        mkdir inference_results
    end
    fname_results = sprintf('WENDy with Structure Matrix and Offset Fully-Mixed Noisy Dynamics subsamp=%u, sigma=%.5g, beta=%.3g, theta=%.3g, eta=%.3g, gamma_i=%.3g, gamma_p=%.3g, tmax=%u, t_grid_num=%u, E0=%.3g WITH NOISE.mat',subsamp,sigma,pars(1),pars(2),pars(3),pars(4),pars(5),t_max,t_num,E0);
    save(fullfile(result_path, fname_results), 'true_vec','w_hat','res','res_0','w_hat_its','V_cell','Vp_cell',...
        'Theta_cell','mt','tobs','xobs','xsub','x0','Jac_mat','G_0','b_0','RT','stdW','mseW','CovW','total_time','toggle_plot','toggle_ddd','phifun','M','nstates','features','rhs_p','equation_terms','S','C','numeq','sigma','noise','toggle_smooth');
end

%% display results

if write_to_txt == 1
    % Write output to txt
    logFile = 'results_fully_mixed_structured_with_offset_WithNoise.txt';
    fid = fopen(logFile,'a');
    if fid < 0
        error('Couldn''t open %s for writing.', logFile);
    end
    % Write one formatted line, then a newline
    fprintf(fid, '\n%s\n', sprintf('Model on Fully-Mixed sigma=%.5g, beta=%.3g, theta=%.3g, eta=%.3g, gamma_i=%.3g, gamma_p=%.3g, tmax=%u, t_grid_num=%u, E0=%.3g',sigma,pars(1),pars(2),pars(3),pars(4),pars(5),t_max,t_num,E0));
    fclose(fid);
    diary(logFile);
end

if toggle_ddd
    ind=inf; % choose which weights to generate from
    w_hat_its_final = w_hat_its(:,min(ind,end));
    % Only beta, theta, eta are estimated; gamma_i and gamma_p are known
    % offsets carried in C, so they come from pars, not from the estimate.
    [~,~,~,w_jac_cell,~,~,~] = FullyMixedModel_Structured_with_Offset( ...
        w_hat_its_final(1),w_hat_its_final(2),w_hat_its_final(3),pars(4),pars(5));
end

if exist('true_vec','var')
    err_norm = 2;
    errs = arrayfunvec(w_hat_its,@(w)norm(w-true_vec,err_norm)/norm(true_vec,err_norm),1);
    err_wendy = errs(end);

    % Diagnostic: if the p-value gate fires, wendy_fcn rolls w_hat back to
    % the iterate with the highest normality p-value, which is not always the
    % most accurate one. Compare the trajectory against the returned answer.
    disp(['err by iterate: ',num2str(errs(:)',' %.4g')])
    disp(['err of returned w_hat: ',num2str(norm(w_hat-true_vec,err_norm)/norm(true_vec,err_norm))])
    disp(['-----------------'])
    disp([' '])
    disp(['err WENDy:',num2str(err_wendy)])
    disp('parameters: true, WENDy')
    disp([true_vec w_hat])
    G = RT \ G_0;
    b = RT \ b_0;
    res_true = G*true_vec-b;
    res_0_true = G_0*true_vec-b_0;
    display_wendy_results_with_noise;

    if write_to_csv == 1
        fname_csv = 'noise_errors_with_offset.csv';

        T = table(sigma, err_wendy, 'VariableNames', {'noise','error'});

        if isfile(fname_csv)
            % if file exists, append without writing the header again
            writetable(T, fname_csv, ...
                'WriteMode','append', ...
                'WriteVariableNames', false);
        else
            % if file doesn't exist, write it (this creates it) and include header
            writetable(T, fname_csv, ...
                'WriteVariableNames', true);
        end
    end
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
    fig_name = sprintf('Inference Results with Structure Matrix and Offset for Noisy Fully-Mixed Model sigma=%.5g, beta=%.3g, theta=%.3g, eta=%.3g, gamma_i=%.3g, gamma_p=%.3g, tmax=%u, t_grid_num=%u, E0=%.3g WITH NOISE.pdf',sigma,pars(1),pars(2),pars(3),pars(4),pars(5),t_max,t_num,E0);
    exportgraphics(gcf, fullfile(fig_path,fig_name), 'ContentType','vector');
end
