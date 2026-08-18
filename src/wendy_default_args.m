% Default arguments for wendy_fcn.
%
% Written 2026 by Moyi Tian for WENDy-Constrained. This file is not derived from
% MathBioCU/WENDy; it only collects that package's own recommended settings so
% they do not have to be written out every time wendy_fcn is called.
%
%   args = wendy_default_args(xobs,tobs,features)
%   args = wendy_default_args(xobs,tobs,features,'S',S,'C',C)
%   [args,opts] = wendy_default_args(...)
%
% ARGS is every argument wendy_fcn takes after the feature library, in order, so
% a complete call is
%
%   args = wendy_default_args(xobs,tobs,features,'S',S);
%   [w_hat,...] = wendy_fcn(xobs,tobs,features,args{:});
%
% OPTS is a struct of the resolved settings, including the derived ones
% (K_min, mt_min, mt_max, mt_cell) and phifun, which the display scripts plot.
%
% Any argument can be overridden by name, for example
%
%   args = wendy_default_args(xobs,tobs,features,'max_iter',50,'K_min',9);
%
% The set-and-forget values come from wendy_snf_params.m unchanged: they are the
% settings the WENDy authors use for all five benchmark problems in the original
% wendy_script.m, with no per-model tuning. The derived quantities follow the
% same recipes that script uses.
%
% K_min defaults to the number of coefficients the structure permits to be
% nonzero. Upstream sets K_min = length(true_vec), which is the same count when
% every coefficient is free; counting through S and C generalizes it to the
% constrained case.

function [args,opts] = wendy_default_args(xobs,tobs,features,varargin)

    [M,nstates] = size(xobs);

    % wendy_snf_params.m is a script, so its assignments land in the workspace
    % of whoever runs it. Running it here keeps them inside this function
    % instead of the caller's workspace, and keeps one source of truth for the
    % values. If a MATLAB release objects to calling a script from a function,
    % paste that file's assignments in place of this line.
    wendy_snf_params;

    p = inputParser;
    p.FunctionName = 'wendy_default_args';
    addParameter(p,'numeq',         nstates);
    addParameter(p,'S',             []);
    addParameter(p,'C',             []);
    addParameter(p,'toggle_smooth', toggle_smooth);
    addParameter(p,'phifun',        phifuns{1});
    addParameter(p,'meth',          'mtmin');
    addParameter(p,'mt_params',     2.^(0:3));
    addParameter(p,'submt',         submt);
    addParameter(p,'K_max',         5000);
    addParameter(p,'K_min',         []);           % [] -> derived below
    addParameter(p,'mt_max',        []);           % [] -> derived below
    addParameter(p,'mt_min',        []);           % [] -> derived below
    addParameter(p,'mt_cell',       []);           % [] -> derived below
    addParameter(p,'center_scheme', center_scheme);
    addParameter(p,'toggle_VVp_svd',toggle_VVp_svd);
    addParameter(p,'w0',            w0);
    addParameter(p,'optim_params',  optim_params);
    addParameter(p,'iter_diff_tol', iter_diff_tol);
    addParameter(p,'max_iter',      max_iter);
    addParameter(p,'diag_reg',      diag_reg);
    addParameter(p,'pvalmin',       pvalmin);
    addParameter(p,'check_pval_it', check_pval_it);
    parse(p,varargin{:});
    o = p.Results;

    %%% resolve S and C exactly as wendy_fcn does, so K_min can be counted
    nfull = o.numeq*numel(features);
    if isempty(o.S), S_res = eye(nfull);      else, S_res = full(o.S);    end
    if isempty(o.C), C_res = zeros(nfull,1);  else, C_res = full(o.C(:)); end
    if size(S_res,1) ~= nfull
        error('wendy_default_args:badS', ...
            'S has %d rows but the full coefficient vector has length %d (numeq*numel(features)).', ...
            size(S_res,1), nfull);
    end
    if numel(C_res) ~= nfull
        error('wendy_default_args:badC', ...
            'C has %d entries but the full coefficient vector has length %d (numeq*numel(features)).', ...
            numel(C_res), nfull);
    end

    %%% derived quantities
    K_min = o.K_min;
    if isempty(K_min)
        K_min = nnz( any(S_res~=0,2) | C_res~=0 );
    end

    mt_max = o.mt_max;
    if isempty(mt_max)
        mt_max = max(floor((M-1)/2)-K_min,1);
    end

    mt_min = o.mt_min;
    if isempty(mt_min)
        mt_min = rad_select(tobs,xobs,o.phifun,1,o.submt,0,1,2,mt_max,[]);
    end

    mt_cell = o.mt_cell;
    if isempty(mt_cell)
        mt_cell = cellfun(@(x,y) [x,{y}], ...
            repmat({{o.phifun,o.meth}},numel(o.mt_params),1), ...
            num2cell(o.mt_params(:)),'uni',0);
    end

    args = {o.numeq, o.S, o.C, o.toggle_smooth, ...
            mt_cell, mt_min, mt_max, K_min, o.K_max, ...
            o.center_scheme, o.toggle_VVp_svd, ...
            o.w0, o.optim_params, o.iter_diff_tol, o.max_iter, ...
            o.diag_reg, o.pvalmin, o.check_pval_it};

    opts = o;
    opts.K_min   = K_min;
    opts.mt_max  = mt_max;
    opts.mt_min  = mt_min;
    opts.mt_cell = mt_cell;
    opts.M       = M;
    opts.nstates = nstates;
end
