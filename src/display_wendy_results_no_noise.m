%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% WENDy: covariance-corrected ODE parameter estimation
%%%%%%%%%%%% Copyright 2023, All Rights Reserved
%%%%%%%%%%%% Code by Daniel Ames Messenger
%%%%%%%%%%%%
%%%%%%%%%%%% Adapted 2026 by Moyi Tian from display_wendy_results.m:
%%%%%%%%%%%%   starred quantities built from the noise-free library Theta_cell_true and
%%%%%%%%%%%%   corrected for the known-offset vector C; confidence bounds drawn with
%%%%%%%%%%%%   errorbar rather than unparented rectangle/line primitives.

%% print results
disp(['-----------------'])
disp([' '])
disp(['time WENDy: ',num2str(total_time), '(sec)'])
disp(['Num its: ', num2str(size(w_hat_its,2))])
disp(['K,d,M:',num2str(size(V_cell{1},1)),',',num2str(length(x0)),',',num2str(length(tobs))])
disp(['test function radii: (',num2str(unique(mt)'),')'])
disp([' '])
disp(['-----------------'])

%% compute quantities of interest

%%% compute (G*,b*)
x_cell = mat2cell(xsub,M,ones(1,nstates));
Theta_cell_true = cellfun(@(x) cell2mat(cellfun(@(y) y(x_cell{:}), x, 'uni',0)), features,'uni',0);

V_big = blkdiag( V_cell{:} );

% Starred quantities must be built from the CLEAN data (Theta_cell_true),
% not the observed library Theta_cell. These coincide in the no-noise case
% but diverge as soon as noise is added.
Theta_mat_true = [ Theta_cell_true{:} ];
Theta_mat_pre_true = repmat({Theta_mat_true},1,numeq);
Theta_big_true = blkdiag(Theta_mat_pre_true{:});

G_0_true = V_big*Theta_big_true*S;

% Known-offset contribution evaluated on noise-free data. b_0 returned by
% wendy_fcn already has its own C_0 subtracted, so b_0_true needs the
% matching term to stay comparable.
C_0_true  = V_big*Theta_big_true*C;
b_0_true  = -blkdiag(Vp_cell{:})*xsub(:) - C_0_true;
K = size(b_0,1);

%%% Phi'*ep
response_error = b_0_true - b_0;

%%% Phi(Theta(U)-Theta(U-ep))
F_obs_error = G_0*w_hat - G_0_true*w_hat;

%%% Phi*u' + Phi'*u
int_error = (RT/norm(RT)) \ (G_0_true*true_vec - b_0_true);

%%% G*(w-w*)
w_error_response = (RT/norm(RT)) \ G_0_true*(w_hat - true_vec);

%%% linear part of the residual. With no noise there is no grad(F)*eps term,
%%% so this is the response error alone. Upstream also built a weighted library
%%% Jacobian here (via a get_jac helper); without noise nothing consumes it, so
%%% that call, its helper param_length_vec, and get_jac itself are removed. The
%%% with-noise script computes the contraction it needs inline.
lin_approx = (RT/norm(RT)) \ (response_error);

%%% nonlinear portion of residual
nonlin_res = (RT/norm(RT)) \ (F_obs_error);

%%% full residual (same as res(:,end) up to scaling)
Res_full = w_error_response + int_error + lin_approx + nonlin_res;

%%% compute data-driven dynamics
if toggle_ddd
    tol_ode = 1e-12;
    options_ode_sim = odeset('RelTol',tol_ode,'AbsTol',tol_ode*ones(1,length(x0)));
    [t_learned,x_learned]=ode45(@(t,x)rhs_p(x,w_jac_cell),tobs,x0,options_ode_sim);
end

%%% compute FFTs
tau = 10^-10;
tauhat = 1;
[~,~,~,corners] = findcorners(xobs,tobs,tau,tauhat,phifun);
xfft = abs(fft(xobs)); xfft = xfft./max(xfft);
phifft = cell2mat(cellfun(@(V)abs(fft(full(V(1,:)))),Vp_cell,'uni',0));
phifft = phifft./max(phifft,[],2);

%%% compute conf intervals
c = 0.05; % <(100)c chance of not containing true val
stdW = max(sqrt(diag(CovW)),eps);
conf_int = arrayfun(@(x)norminv(1 - c/2,0,x),stdW);

if toggle_plot
    figure(11)
    set(gcf,'Renderer','painters')

    % Compact number formatting for titles: two significant figures, and
    % exponents without zero padding, so 6.3e-06 prints as 6.3e-6. Keeps the
    % two-quantity titles on the top row short enough to fit the axes width.
    fmtnum = @(v) regexprep(num2str(v,'%.2g'),'e([+-])0*(\d+)','e$1$2');

    %%% wendy iterates
    subplot(3,3,1)
    semilogy(1:length(errs),errs,'bo-')
    legend({'err({\bf w^{(n)}})'},'location','best','box','off')
    ylabel('||\bf w^{(n)}-w^*||_2/||w^*||_2')
    title(['err(OLS) = ',fmtnum(100*errs(1)),'%,   ',...
        'err(WENDy) = ',fmtnum(100*errs(end)),'%'],'fontsize',9)
    ylims = [10^(log10(min(errs)))*0.9 10^(log10(max(errs)))*1.2];
    ylim(ylims)
    xlim([1 length(errs)])
    % Y ticks left to MATLAB. Upstream forced 8 geometrically spaced ticks here;
    % when errs spans less than one decade (as in the noise-free case, where it
    % sits flat at machine precision) those are not decade values, so each label
    % is written out in full AND the axis keeps its own shared exponent - the
    % x10^-8 then appears twice and eight long labels crowd the panel.
    xlabel('iter')
    grid on

    %%% confidence intervals
    % Upstream carried a commented-out alternative version of this panel that
    % drew the bounds as a shaded band via fill(xflip,confbounds,...). It was
    % never enabled, and the xflip / confbounds vectors it needed were the only
    % reason those two existed, so both the block and the variables are removed.
    % errorbar is a parented chart object, so it clips correctly under both
    % the hardware and painters renderers. The rectangle/line primitives it
    % replaces were unparented and escaped the axes when conf_int underflowed.
    ax2 = subplot(3,3,2);
    h1 = plot(ax2,1:length(w_hat),w_hat,'ro',1:length(w_hat),true_vec,'bx');
    hold(ax2,'on')
    he = errorbar(ax2,1:length(w_hat),w_hat(:)',conf_int(:)', ...
        'k','LineStyle','none','Marker','none','CapSize',12);
    hold(ax2,'off')
    set([h1(:); he(:)],'Clipping','on')
    legend(h1(1:2),{'{\bf w}_{WENDy}','\bf w^*'},'box','off','location','best')
    title(ax2,[num2str((1-c)*100),'% confidence bounds'],'fontsize',9)
    set(ax2,'Xtick',1:length(w_hat))
    xlim(ax2,[0 length(w_hat)+1])
    grid(ax2,'on')


    %%% p-values
    subplot(3,3,3)
    pvals = arrayfunvec(res,@(v)outn(@swtest,v,2),1);
    pvals_0 = arrayfunvec(res_0,@(v)outn(@swtest,v,2),1);
    plot(pvals,'o-')
    xlim([1 length(errs)])
    title(['p-val(OLS) = ',fmtnum(pvals_0(1)),'   ',...
        'p-val(WENDy) = ',fmtnum(pvals(end))],'fontsize',9)
    xlabel('iter')
    legend('p-val({\bf w^{(n)}})','location','best','box','off')
    grid on

    %%% data
    subplot(3,3,4)
    h1=plot(1:length(tobs),xsub,'k-','linewidth',2);
    hold on;
    h2=plot(1:length(tobs),xobs,'r.','markersize',8);
    xlim([1 length(tobs)])
    if toggle_ddd
        h3=plot(1:length(t_learned),x_learned,'--g','linewidth',2);
        ylim([min(xobs(:)) max(xobs(:))+0.5*range(xobs(:))])
        try
            err_dd = norm(xsub(:) - x_learned(:))/norm(xsub(:));
        catch
            err_dd = Inf;
        end
        legend([h1(1);h2(1);h3(1)],{'u^*','{\bfU}',['{\bfU_{dd}}: ',fmtnum(100*err_dd),'% rel. err']},'location','best','box','off')
    else
        legend([h1(1);h2(1)],{'u^*','{\bfU}'},'location','best','box','off')
    end
    title(['data: (K,d,M)=(',num2str(K),',',num2str(nstates),',',num2str(M),')'],'fontsize',9)
    hold off
    xlabel('timeindex')
    grid on

    %%% GLS residual: WENDy final to true
    subplot(3,3,5)
    plot(norm(RT)*res(:,end),'LineWidth',2);
    hold on;
    plot(norm(RT)*res_true(:,end),'r--','LineWidth',2);
    hold off;
    xlim([1 length(b_0)])
    legend({'{\bf w}_{WENDy}','{\bf w}^*'},'box','off')
    title(['\bf C^{-1/2}r(U,w),  p-val ',fmtnum(pvals(end))],'fontsize',9)
    xlabel('row num (k)')
    % Scale to what this panel actually plots. Upstream used max(abs(Res_full)),
    % which is a different (unscaled) quantity, so the traces could overflow.
    yl = norm(RT)*max(abs([res(:,end); res_true(:,end)]));
    if yl>0, ylim(yl*[-1.5 1.5]), end
    grid on


    %%% residual components
    subplot(3,3,6)
    plot(1:K,Res_full,'k-','LineWidth',2)
    hold on
    plot(1:K,lin_approx,'r--','LineWidth',2);
    xlim([1 length(b_0)])
    hold off
    title('\bf C^{-1/2}r(U,w) vs. C^{-1/2}L_w\epsilon','fontsize',9)
    xlabel('row num (k)')
    legend({'\bfr(U,w)','\bfL_w\epsilon'},'location','best','box','off')
    % Both traces must fit: lin_approx can exceed Res_full.
    yl = max(abs([Res_full(:); lin_approx(:)]));
    if yl>0, ylim(yl*[-1.5 1.5]), end
    grid on


    %%% FFT data, FFT test function
    subplot(3,3,7)
    h1=semilogy(xfft(1:floor(end/2),:),'r');hold on;
    h2=semilogy(phifft(:,1:floor(end/2))'); hold off
    ylim([min(min(xfft))/10 1])
    xlim([1 length(xfft(1:floor(end/2),:))])
    set(gca,'Ytick',10.^(floor(log10(min(min(xfft))/10)):0))
    legend([h1(1);h2(1)],{'F({\bfU})','F(\Phi(1,:))'},'fontsize',10,'box','off')
    title('Fourier content of data','fontsize',9)
    xlabel('wavenumber')
    grid on


    %%% OLS residual: WENDy final to true
    subplot(3,3,8)
    plot(res_0(:,1),'LineWidth',2);hold on;
    plot(res_0_true(:,1),'r--','LineWidth',2)
    hold off;
    xlim([1 length(b_0)])
    legend({'{\bf w}_{WENDy}','{\bf w}^*'},'box','off')
    title(['\bf r(U,w),  p-val ',fmtnum(outn(@swtest,res_0(:,1),2))],'fontsize',9)
    xlabel('row num (k)')
    % Both traces must fit: res_0_true can exceed res_0.
    yl = max(abs([res_0(:,1); res_0_true(:,1)]));
    if yl>0, ylim(yl*[-1.5 1.5]), end
    grid on


    %%% residual and error of linear approx
    subplot(3,3,9)
    plot(1:length(Res_full),[w_error_response int_error nonlin_res],'-.','LineWidth',2.5);
    xlim([1 length(b_0)])
    % These three components sum to Res_full, so when they cancel each one can
    % be far larger than the total. Scaling by Res_full (as upstream did) clips
    % them; scale by the components themselves.
    yl = max(abs([w_error_response(:); int_error(:); nonlin_res(:)]));
    if yl>0, ylim(yl*[-1.5 1.5]), end
    legend({'{\bfr}_0','{\bfe}_{int}','\bfh'},'location','best','box','off')
    title('other residual components','fontsize',9)
    xlabel('row num (k)')
    grid on

    %%% tidy the 3x3 grid: lift every title clear of the axes box and give the
    %%% panels a little more vertical room, so long titles are not cramped
    %%% against the plot or clipped by the neighbouring subplot.
    all_ax = findobj(gcf,'Type','axes');
    for k = 1:numel(all_ax)
        all_ax(k).Title.Units    = 'normalized';
        all_ax(k).Title.Position(2) = 1.12;
        all_ax(k).Title.FontSize = 9;
        if isprop(all_ax(k),'Toolbar') && ~isempty(all_ax(k).Toolbar)
            all_ax(k).Toolbar.Visible = 'off';   % the "..." button sat on the title
        end
        p = all_ax(k).Position;
        all_ax(k).Position = [p(1) p(2)+0.02*p(4) p(3) p(4)*0.88];
    end

end
