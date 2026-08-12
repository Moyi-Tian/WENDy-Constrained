%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% WENDy: covariance-corrected ODE parameter estimation
%%%%%%%%%%%% Copyright 2023, All Rights Reserved
%%%%%%%%%%%% Code by Daniel Ames Messenger
%%%%%%%%%%%%
%%%%%%%%%%%% Small bug fix applied for WENDy-Constrained; structure unchanged.

function w=els(G,b,batch_size,num_runs,meth)
    [K,J] = size(G);
    W = zeros(J,num_runs);
    for rr=1:num_runs
        inds = randperm(K,floor(K/batch_size));
        W(:,rr) = G(inds,:) \ b(inds);
    end
    % w was only assigned inside the two branches, so an unrecognised meth
    % failed with "Output argument w is not assigned" instead of saying why.
    if isequal(meth,'median')
        w = median(W,2);
    elseif isequal(meth,'mean')
        w = mean(W,2);
    else
        error('els:badMeth','Unknown averaging method ''%s''. Use ''mean'' or ''median''.',meth);
    end
end
