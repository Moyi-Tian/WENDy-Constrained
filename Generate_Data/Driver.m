% Simulate dynamics for all compartments

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
    parameters.t_max = time_range(end);
    parameters.E0 = E0;

    time_array = tsol;
    X = [ysol(:,1),ysol(:,2),ysol(:,3),ysol(:,4),ysol(:,5)];

    fname = sprintf('Fully-Mixed Dynamics beta=%.3g, theta=%.3g, eta=%.3g, gamma_i=%.3g, gamma_p=%.3g, tmax=%u, t_grid_num=%u, E0=%.3g.mat',pars(1),pars(2),pars(3),pars(4),pars(5),time_range(end),t_num,E0);
    save(fullfile(save_path, fname), 'parameters', 'time_array', 'X');
end

toc
