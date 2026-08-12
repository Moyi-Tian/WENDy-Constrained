%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% WENDy: covariance-corrected ODE parameter estimation
%%%%%%%%%%%% Copyright 2023, All Rights Reserved
%%%%%%%%%%%% Code by Daniel Ames Messenger
%%%%%%%%%%%%
%%%%%%%%%%%% Small bug fixes applied for WENDy-Constrained; structure unchanged.

%%% 'meth' options: 'LS', 'TLS', 'ensLS'
%%% 'LS' options: cov
%%% 'TLS' options: (none)
%%% 'ensLS' options: batch_size, num_runs, avg_meth
%%% (was: options listed as 'LS','elsLS'. The code tests for 'ensLS', and 'TLS'
%%%  was undocumented, so the header disagreed with the branches below.)

function w = wendy_opt(G,b,varargin)   % was: windy_opt, a typo; MATLAB
                                       % dispatches on the filename, so the
                                       % mismatch only tripped Code Analyzer

    defaultmeth = 'LS';
    defaultbatch_size = 1;
    defaultnum_runs = 1;
    defaultavg_meth = 'mean';
    defaultcov = [];

    inp = inputParser;
    addParameter(inp,'meth',defaultmeth);
    addParameter(inp,'batch_size',defaultbatch_size);
    addParameter(inp,'num_runs',defaultnum_runs);
    addParameter(inp,'avg_meth',defaultavg_meth);
    addParameter(inp,'cov',defaultcov);

    parse(inp,varargin{:});

    meth = inp.Results.meth;
    batch_size = inp.Results.batch_size;
    num_runs = inp.Results.num_runs;
    avg_meth = inp.Results.avg_meth;
    cov = inp.Results.cov;

    % Reject an unknown meth up front. Previously w was only assigned inside the
    % branches below, so a bad option failed with "Output argument w is not
    % assigned" rather than naming the actual problem.
    if ~any(strcmp(meth,{'LS','TLS','ensLS'}))
        error('wendy_opt:badMeth', ...
            'Unknown ''meth'' option ''%s''. Use ''LS'', ''TLS'' or ''ensLS''.',meth);
    end

    if isequal(meth,'LS')
        if isempty(cov)
            w = G \ b;
        else
            w = lscov(G,b,cov);
        end
    end

    if isequal(meth,'TLS')
        [~,~,V] = svd([G b],0);
        [~,n]=size(G);
        w = V(1:n,n+1:end);
    end

    if isequal(meth,'ensLS')
        w = els(G,b,batch_size,num_runs,avg_meth);
    end

end
