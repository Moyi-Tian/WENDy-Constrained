% Simulate dynamics for all compartments with noise

close all
clear
tic

%% cd to current directory
scriptFullName = matlab.desktop.editor.getActiveFilename();
scriptDir = fileparts(scriptFullName);
% change to current folder
if ~isempty(scriptDir)
    cd(scriptDir);
end

%% Set Parameters
pars(1) = 0.6; % beta: transmission rate online NotEngaged/Uninterested(U) -> Engaged(E)
pars(2) = 0.5; % theta: transmission rate due to offline feedback NotEngaged/Uninterested(U) -> Engaged(E)
pars(3) = 0.4; % eta: fraction of Engaged(E) online from UnParticipating(UP) -> Participating(P) offline
pars(4) = 0.1; % gamma_i: recovery rate online Engaged(E) -> DoneEngaging/DisEngaged(D)
pars(5) = 0.2; % gamma_p: recovery rate offline Participating(P) -> DoneParticipating(R)

time_range = [0 25];
t_num = 100;

sigma = 0.05; % noise ratio: additive Gaussian noise std as a fraction of each state's RMS amplitude

rng_seed = 123;  % fixed so the noisy data, and therefore the demo results, are
rng(rng_seed); % reproducible; set rng('shuffle') for a fresh realization

save_data = 1; % = 1 save simulation data; = 0 not save data

%% Initialization
E0 = 0.02;
U0 = 1-E0;
D0 = 0;
P0 = 0;
R0 = 0;

y0 = [U0; E0; D0; P0; R0];

%% Solve
opts = odeset('AbsTol',1e-15,'RelTol',1e-12,'Stats','on','OutputFcn',@odeplot);
odeFunc = @(t,y) FullyMixedModel_ODE(t,y,pars);
[tsol,ysol] = ode45(odeFunc, linspace(time_range(1),time_range(2),t_num), y0, opts);


%% Modify Plot
legend('U','E','D','P','R');
xlabel('Time');
ylabel('Ratio');

%% Add noise to data
X = [ysol(:,1),ysol(:,2),ysol(:,3),ysol(:,4),ysol(:,5)];
sigma_X = sigma*rms(X);          % rms acts column-wise: one std per state
noise = randn(size(X)).*sigma_X; % iid across states and time points

X_noisy = X + noise;

% Plot noisy trajectories
figure(2)
plot(tsol,  X_noisy(:,1),"LineWidth",2)
hold on
plot(tsol,X_noisy(:,2),"LineWidth",2)
hold on
plot(tsol, X_noisy(:,3),"LineWidth",2)
hold on
plot(tsol, X_noisy(:,4),"LineWidth",2)
hold on
plot(tsol, X_noisy(:,5),"LineWidth",2)
legend('U','E','D','P','R');
xlabel('Time',FontSize=12);
ylabel('Ratio',FontSize=12);
ylim([0,1]);
title('Noisy Fully-Mixed Model Dynamics',FontSize=14)

%% Save Data
if save_data == 1
    tempdir=pwd;
    save_path=[tempdir,'/data'];
    if ~exist(save_path, 'dir'), mkdir(save_path); end

    parameters.beta = pars(1);
    parameters.theta = pars(2);
    parameters.eta = pars(3);
    parameters.gamma_i = pars(4);
    parameters.gamma_p = pars(5);
    parameters.t_max = time_range(2);
    parameters.E0 = E0;
    parameters.sigma = sigma;

    time_array = tsol;

    fname = sprintf('Noisy Fully-Mixed Dynamics sigma=%.5g, beta=%.3g, theta=%.3g, eta=%.3g, gamma_i=%.3g, gamma_p=%.3g, tmax=%u, t_grid_num=%u, E0=%.3g.mat',sigma,pars(1),pars(2),pars(3),pars(4),pars(5),time_range(2),t_num,E0);
    save(fullfile(save_path, fname), 'parameters', 'time_array', 'X', 'X_noisy','noise');
end

toc
