function dydt = FullyMixedModel_ODE(t,y,pars)

    U = y(1);
    E = y(2);
    D = y(3);
    P = y(4);
    R = y(5);

    beta = pars(1);
    theta = pars(2);
    eta = pars(3);
    gamma_i = pars(4);
    gamma_p = pars(5);


    dUdt = -beta*U*E - theta*U*P;
    dEdt = beta*U*E + theta*U*P - (eta+gamma_i)*E;
    dDdt = (eta+gamma_i)*E;
    dPdt = eta*E - gamma_p*P;
    dRdt = gamma_p*P;

    dydt = [dUdt; dEdt; dDdt; dPdt; dRdt];

end
