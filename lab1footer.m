%%%%%%%% LAB1 TSA

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3.1 Working with time series in Matlab
clear
clc
A1 = [ 1 -1.79 0.84 ]; 
C1 = [ 1 -0.18 -0.11 ];
A2 = [ 1 -1.79 ];
C2 = [ 1 -0.18 -0.11 ];
arma1=idpoly(A1,[],C1); %% Define ARMA Systems with V[et]=1
arma2=idpoly(A2,[],C2); %% and samling period=1
figure(1)
pzmap( arma1) %% Pole zero plots
figure(2)
pzmap( arma2)

%% Simulation of arma1 and arma2 

sigma2=1.5;
N=200; %% Nbr of realizations
e=sqrt(sigma2)*randn(N,1); %%Generate innovation seq e(t) with V[e(t)]=sigma2
y1=filter(arma1.c,arma1.a,e);
y2=filter(arma2.c,arma2.a,e);
subplot(211)
plot(y1)
title("arma1 stable") 
subplot(212)
plot(y2)
title('arma2 unstable')

%% Covariance of arma1 
figure(1)
m=200;
rtheo = kovarians( arma1.c, arma1.a, m ); 
stem( 0:m, rtheo*sigma2, 'b')
hold on
rest = covf( y1, m+1 );
stem( 0:m, rest ,'r')
legend('True covariance','Estimated covariance')

%% Re-estimate model based on arma1 realization data y1
figure(1)
data=iddata(y1);
phi = pacf( y1, 100,0.05, 1, 1 );
title("PACF for y1 with 95% confidence interval (asymptotic interval)");
figure(2)
rho = acf( y1, 10,0.05, 1, 1 );
title("ACF for y1 with 95% confidence interval (asymptotic interval)");
figure(3)
normplot(y1)
title("Normal probability plot");

%%% It seems to be a gaussian process according to normplot thus the
%%% confidence intervals used in pacf and acf are at least correct
%%% (asympoticically!)

%%
%%% Let us test the simplest model that seems probable, an AR(2)
model1 = armax( y1, [2 0]);
poly1=idpoly(model1.a,[],1);


%%% Let us check the resiudual of this model (filter yt through inverse model, output
%%% should resemble white noise if the ARMA model is "correct":
 residual1 = filter( poly1.a, 1, y1 ); 
 
 figure(1)
 plot(residual1)
 
 figure(2)
phi = pacf( residual1, 10,0.05, 1, 1 );
title("PACF for r1 with 95% confidence interval (asymptotic interval)");
figure(3)
rho = acf( residual1, 10,0.05, 1, 1 );
title("ACF for r1 with 95% confidence interval (asymptotic interval)");
figure(4)
normplot(residual1)
title("Normal probability plot r1");

model2 = armax( y1, [2 1]);
model3 = armax( y1, [2 2]);

present(model1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3.2 Model order estimation of an AR-process
clc

n = 500;
n2=1000000;
A = [1 -1.35 0.43];
sigma2 = 4;
 n_order = zeros(100,1);
n_aic = zeros(100,1);
n_order2 = zeros(100,1);
n_aic2 = zeros(100,1);
n_est = floor(2/3 * n);
n_est2 = floor(2/3 * n2);
NN= [1:10]'; %We want to compare AR(1) up to AR(10)
%% Simulate AR(2) process 100 times and for each decide optimal order estimate
for i=1:100
noise = sqrt(sigma2) * randn(n + 100 ,1); 
noise2 = sqrt(sigma2) * randn(n2 + 100 ,1);

y = filter(1,A,noise);
y = y(101:end); % Why do we remove?
y2 = filter(1,A,noise2);
y2 = y2(101:end); 

y_est = iddata(y(1: n_est ));
y_val = iddata(y(n_est + 1:end));
V = arxstruc(y_est,y_val,NN);
y_est2 = iddata(y2(1: n_est2 ));
y_val2 = iddata(y2(n_est2 + 1:end));
V2 = arxstruc(y_est2,y_val2,NN);
 n_order(i) = selstruc(V,0);
n_aic(i) = selstruc(V, 'aic');
n_order2(i) = selstruc(V2,0);
n_aic2(i) = selstruc(V2, 'aic');
end

n_order
%%
figure(1)
histogram(n_order)
title("Histogram over the optimal model order (LS) for each of 100 realizations of an AR(2) ");
figure(2)
histogram(n_aic)
title("Histogram over the optimal model order (AIC) for each of 100 realizations of an AR(2) ");

%%
figure(3)
histogram(n_order2)
figure(4)
histogram(n_aic2)

%%
ar_model = arx(y, n_order(end)); 
ar_model.NoiseVariance
ar_model.CovarianceMatrix %% Why is this only 2*2
present(ar_model)

%We see that the noise variance and polynomial coefficents could be well estimated with only 500 data
%points for modelling and validation given that right model order was
%choosen!!!!


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3.3 Model order estimation of an ARMA-process
data1;
data=iddata(data);
ar1 = armax( data, [1 0]);
ar2 = armax( data, [2 0]);
rar1=resid(ar1,data); 
rar2=resid(ar2,data);
resid(ar1,data)
figure(2)
resid(ar2,data)
present(ar2)

arma11 = armax(data ,[1 1]); 
arma22 = armax(data ,[2 2]);
resid(arma11,data)
figure(2)
resid(arma22,data)
present(arma11)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3.4 Estimation of a SARIMA-process

A = [1 -1.5 0.7];
C = [1 zeros(1,11) -0.5]; 
A12 = [1 zeros(1,11) -1];
A_star = conv(A,A12);
e = randn(600 ,1);
y = filter(C,A_star,e); y = y(100:end);
plot(y)
figure(2)
phi = pacf( y, 25,0.05, 1, 1 );
title("pacf");

figure(3)
rho = acf( y, 25,0.05, 1, 1 );
title("acf");
figure(4)
normplot(y)

y_s=filter(A12,1,y);
y_s=y_s(13:end); %% Remove first 12 since they are wrong
data=iddata(y_s);
size(data)
%%
figure(1)
phi = pacf( y_s, 25,0.05, 1, 1 );
title("pacf");

figure(2)
rho = acf( y_s, 25,0.05, 1, 1 );
title("acf");
figure(3)
normplot(y_s)

% PACF is nonzero up until lag 2 so we begin with AR(2)
Polyar2 = idpoly([1 0 0] ,[] ,[]);
ar2=pem(data , Polyar2 );
present(ar2)

%%
r=resid(ar2,data);
rAr2=r.y;
figure(1)
phi = pacf( rAr2, 25,0.05, 1, 1 );
title("pacf");

figure(2)
rho = acf( rAr2, 25,0.05, 1, 1 );
title("acf");
figure(3)
normplot(rAr2)

%%
arma1 = idpoly([1 0 0],[],[1 zeros(1,12)]); 
arma1.Structure.c.Free = [zeros(1,12) 1]; 
Model1 = pem(data,arma1);

r=resid(Model1,data);
res=r.y;
figure(1)
phi = pacf( res, 25,0.05, 1, 1 );
title("pacf");

figure(2)
rho = acf( res, 25,0.05, 1, 1 );
title("acf");
figure(3)
normplot(res)

%%
present(Model1)
% Looks reasonably white!

%% Real data
load("svedala")
y=svedala;
plot(y)
figure(1)
bcNormPlot(y) % We see that the data does not need to be transformed
%%
[ rejectMean, tRatio, tLimit] = testMean( y); % T-ratio test says that we can not reject zero mean hypothesis.
figure(1)
phi = pacf( y, 100,0.05, 1, 1 );
title("pacf");

figure(2)
rho = acf( y, 100,0.05, 1, 1 );
title("acf");
figure(3)
normplot(y)

% ACF displays a periodicity of 24, try to filter data with season 24
% operator
As= [1 zeros(1,23) -1];
y_s=filter(As,1,y);
y_s=y_s(24:end); %% Remove first 24 since they are wrong
data=iddata(y_s);

%% DESEASONALIZED with stochastic trend
figure(1)
phi = pacf( y_s, 50,0.05, 1, 1 );
title("pacf");

figure(2)
rho = acf( y_s, 100,0.05, 1, 1 );
title("acf");
figure(3)
plot(y_s)

%% DESEASONALIZED with deterministic trend


%% TEST AR(1)
ar1 = armax( data, [1 0]);
r=resid(ar2,data);
pacf(r.y,100,0.05, 1, 1 );

%% TEST ARMA with a1 a2 and a24 and c24 coefficients being nonzero

data=iddata(y_s);
As=[1 zeros(1,23) -1 ];
A=[1  1  1 zeros(1,20) 1 -1 -1 ];
arma1 = idpoly(As,[],[1 zeros(1,23) 1]); 
arma1.Structure.a.Free = [ 0 1 1 zeros(1,21) 1 ]; 
arma1.Structure.c.Free = [ 0 zeros(1,23) 1]; 
Model1 = pem(data,arma1);
r=resid(Model1,data);

whitenessTest(r.y,0.0001) % At this significance level it looks white :)

present(Model1)

figure(1)
phi = pacf( r.y, 25,0.05, 1, 1 );
title("pacf");

figure(2)
rho = acf( r.y, 100,0.05, 1, 1 );
title("acf");
figure(3)
normplot(r.y)
figure(4)
[ C, x1, x2, Ka ] = plotCumPer( r.y, 0.05, 1 ); %% Looks good spectral density is relatively flat
jbtest(r.y) % Not  so good which we also see in normplot 
% The residual has larger tails then the normal distribution.
%%











 
 


