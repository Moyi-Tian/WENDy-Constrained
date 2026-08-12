% Fully-Mixed Model with Coefficient Dependencies Defined by a Structure Matrix
% As an example, suppose that recovery parameters gamma_i and gamma_p
% are known. Here we define a function to be provided back to the inference
% process where only unknown parameters beta, theta and eta will be
% learned.

function [features,equation_terms,numeq,params,true_vec,rhs_p,S,C] = FullyMixedModel_Structured_with_Offset(beta,theta,eta,gamma_i,gamma_p)

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

    true_vec = [beta;theta;eta];

    % Structure Matrix
    % Size = 20 (5 equations x 4 features) by 3 (beta, theta, eta)
    S = [-1  0  0;
          0 -1  0;
          0  0  0;
          0  0  0;
          1  0  0;
          0  1  0;
          0  0 -1;
          0  0  0;
          0  0  0;
          0  0  0;
          0  0  1;
          0  0  0;
          0  0  0;
          0  0  0;
          0  0  1;
          0  0  0;
          0  0  0;
          0  0  0;
          0  0  0;
          0  0  0];

    % Offset vector
    C = [0;
        0;
        0;
        0;
        0;
        0;
        -gamma_i;
        0;
        0;
        0;
        gamma_i;
        0;
        0;
        0;
        0;
        -gamma_p;
        0;
        0;
        0;
        gamma_p];
end
