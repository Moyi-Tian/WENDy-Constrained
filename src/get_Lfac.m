%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% WENDy: covariance-corrected ODE parameter estimation
%%%%%%%%%%%% Copyright 2023, All Rights Reserved
%%%%%%%%%%%% Code by Daniel Ames Messenger
%%%%%%%%%%%%
%%%%%%%%%%%% Modified 2026 by Moyi Tian for WENDy-Constrained:
%%%%%%%%%%%%   takes numeq explicitly and indexes one shared feature library across all
%%%%%%%%%%%%   equations, as required by the structured formulation.

function [L0,L1] = get_Lfac(Jac_mat,Js,V_cell,Vp_cell,numeq)
    [~,d,M] = size(Jac_mat);
    Jac_mat = permute(Jac_mat,[2 3 1]);
    L0 = blkdiag(Vp_cell{:});


    L1 = repmat(L0*0,1,1,sum(Js));


    Ktot = 0;
    Jtot = 0;
    for i=1:numeq
        [K,~] = size(V_cell{i});
        J = Js(i);
        for ell=1:d
            L1(Ktot+(1:K),(ell-1)*M+(1:M),Jtot+(1:J)) = Jac_mat(ell,:,1:J).*V_cell{i};
        end
        Ktot = Ktot+K;
        Jtot = Jtot+J;
    end
end
