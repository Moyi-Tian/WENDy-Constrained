% Fully-Mixed Model with Coefficient Dependencies Defined by a Selection Matrix

function [features,equation_terms,numeq,params,true_vec,rhs_p,S] = FullyMixedModel_Structured(beta,theta,eta,gamma_i,gamma_p)

    numeq = 5;

    features = cell(4,1); % 4 distinct monomial terms on the right-hand side
    features{1} = {@(U,E,D,P,R) U.*E};
    features{2} = {@(U,E,D,P,R) U.*P};
    features{3} = {@(U,E,D,P,R) E};
    features{4} = {@(U,E,D,P,R) P};

    equation_terms = cell(numeq,1); % what terms live in which equation
    equation_terms{1} = {@(U,E,D,P,R) U.*E, @(U,E,D,P,R) U.*P};
    equation_terms{2} = {@(U,E,D,P,R) U.*E, @(U,E,D,P,R) U.*P, @(U,E,D,P,R) E};
    equation_terms{3} = {@(U,E,D,P,R) E};
    equation_terms{4} = {@(U,E,D,P,R) E, @(U,E,D,P,R) P};
    equation_terms{5} = {@(U,E,D,P,R) P};

    % what coefficient corresponding to each term in the equations
    params = {[-beta -theta],[beta theta -(eta+gamma_i)],[eta+gamma_i],[eta -gamma_p],[gamma_p]};

    rhs_p = @(x,params) rhs_fun(equation_terms,params,x);

    true_vec = [beta;theta;eta;gamma_i;gamma_p];

    % Selection Matrix
    % Size = # of distinct monomial terms (4) x # of equations (5) by # of
    % independent parameters (5) = 20 by 5 for this model
    S = [-1 0 0 0 0;
        0 -1 0 0 0;
        0 0 0 0 0;
        0 0 0 0 0;
        1 0 0 0 0;
        0 1 0 0 0;
        0 0 -1 -1 0;
        0 0 0 0 0;
        0 0 0 0 0;
        0 0 0 0 0;
        0 0 1 1 0;
        0 0 0 0 0;
        0 0 0 0 0;
        0 0 0 0 0;
        0 0 1 0 0;
        0 0 0 0 -1;
        0 0 0 0 0;
        0 0 0 0 0;
        0 0 0 0 0;
        0 0 0 0 1];
end
