% TSA Project, Not very succesful attempt of modelling temperature with ARMAX

load('climate67.dat')
Mdldata=climate67(3400:5000,:);
%%  ARMAX model with input u=net radiation
y=Mdldata(:,8);
u=Mdldata(:,6);
plot(u)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ARMA modelling of input u

%% Transforming u with log and making zero mean

OptLambda=bcNormPlot(climate67(3000:5800,6)) % =0.14 so log transformation seems reasonable
min(Mdldata(:,6)) % =92 thus we need to shift u up before taking log
u=Mdldata(:,6);
u=u+100; % Shifts u 100 units up to ensure positivity
u=log(u);
u=u-mean(u); 

%% Model u as ARMA
figure(1)

phi = pacf( u, 100,0.05, 1, 1 );
title("PACF for u with 95% confidence interval (asymptotic interval)");
figure(2)
rho = acf( u, 100,0.05, 1, 1 );
title("ACF for u with 95% confidence interval (asymptotic interval)");
%%
A24=[1 zeros(1,23) -1];
A=conv([1 1 1],A24)
C=[1 zeros(1,23) 1];
M1u = idpoly ( A,[],C);
M1u.Structure.a.Free = A;
M1u.Structure.c.Free = C;
 data = iddata(u);
modelU = pem(data,M1u); 
r=resid(data,modelU);
figure(1)
phi = pacf( r.y, 100,0.05, 1, 1 );

figure(2)
rho = acf( r.y, 100,0.05, 1, 1 );
figure(3)
whitenessTest(r.y,0.01)
figure(4)
plot(r.y)
present(modelU)

% Looks sufficiently white to proceed
%% Transform (pre-whiten) y and u with inverse input arma model
upw=filter(modelU.a,modelU.c,u);
ypw=filter(modelU.a,modelU.c,y);


M= 40; stem(-M:M,crosscorr(upw,ypw,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on

plot(-M:M, 2/sqrt(length(upw))*ones(1,2*M+1),'--') 
plot(-M:M, -2/sqrt(length(upw))*ones(1,2*M+1),'--') 
hold off

% Looks like d=0, r=1 and we guess s=0 Looking at CCF between vhat and upw tells us
% that s=1 is in fact better 
% the b1 coefficient is however not significant but still improves model a
% lot
%% 
A2 = [1 1];
B =[1 1];
H = idpoly (1 ,B ,[] ,[] ,A2);
H.Structure.b.Free = B;
 zpw = iddata(ypw,upw);
Mba2 = pem(zpw,H); 
present(Mba2)
vhat = resid(Mba2,zpw); 
%% CCF between vhat=ypw-H(z)upw  and upw 
M= 40; stem(-M:M,crosscorr(vhat.y,upw,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on

plot(-M:M, 2/sqrt(length(upw))*ones(1,2*M+1),'--') 
plot(-M:M, -2/sqrt(length(upw))*ones(1,2*M+1),'--') 
hold off

%% Modelling residual res=y-H(z)u as ARMA
res=y-filter(Mba2.b,Mba2.f,u);

%% CCF between res=y-H(z)u and u 
M= 40; stem(-M:M,crosscorr(res,u,M)); 
title('Cross correlation function'), xlabel('Lag')
hold on


plot(-M:M, 2/sqrt(length(u))*ones(1,2*M+1),'--') 
plot(-M:M, -2/sqrt(length(u))*ones(1,2*M+1),'--') 
hold off
%%
phi = pacf( res, 100,0.05, 1, 1 );
title("PACF for u with 95% confidence interval (asymptotic interval)");
figure(2)
rho = acf( res, 1000,0.05, 1, 1 );
title("ACF for u with 95% confidence interval (asymptotic interval)");
%% x looks like AR(2)
A2=[1 1 1];
C1=[1];
ar2=idpoly(1,[],C1,A2,[]);
data=iddata(res);
NoiseMdl=pem(data,ar2);
% r=filter(NoiseMdl.d,NoiseMdl.c,res); % Alternative way of constr. residual 
r=resid(data,NoiseMdl);
present(NoiseMdl)
%% Plots of ehat
phi = pacf( r.y, 100,0.05, 1, 1 );
title("PACF for e with 95% confidence interval (asymptotic interval)");
figure(2)
rho = acf( r.y, 100,0.05, 1, 1 );
title("ACF for e with 95% confidence interval (asymptotic interval)");
figure(3)
normplot(phi)
title("Normal probability plot of pacf");
whitenessTest(r.y)
% relatively white

%%  Finally estimate all paramters together, with PEM

A1=[1 1 1];
C1=1;
B =[1 1];
A2 = [1 1];
Mi = idpoly (1 ,B ,C1 ,A1 ,A2);
z = iddata(y,u);
ARMAX= pem(z,Mi); 
present(ARMAX)
ehat=resid(ARMAX,z);

M= 40; stem(-M:M,xcorr(ehat.y,u,M,'biased')); 
title('Cross correlation function'), xlabel('Lag')
hold on

plot(-M:M, 2/sqrt(length(u))*ones(1,2*M+1),'--') 
plot(-M:M, -2/sqrt(length(u))*ones(1,2*M+1),'--') 
hold off

%%
phi = pacf( ehat.y, 100,0.05, 1, 1 );
title("PACF for e with 95% confidence interval (asymptotic interval)");
figure(2)
rho = acf( ehat.y, 100,0.05, 1, 1 );
title("ACF for e with 95% confidence interval (asymptotic interval)");
figure(3)

whitenessTest(r.y)
