%% 3.1 

%rng('default')

A1 = [ 1 -1.79 0.84 ];
C1 = [ 1 -0.18 -0.11 ];

A2 = [ 1 -1.79 ] ;
C2 = [ 1 -0.18 -0.11 ] ;

arma1 = idpoly( A1, [], C1);
arma2 = idpoly( A2, [], C2);

sigma2 = 1.5; N = 200;
e = sqrt(sigma2)*randn(N,1);

y1 = filter(arma1.c, arma1.a, e);
y2 = filter(arma2.c, arma2.a, e);

figure1 = figure('Name','3.1 ARMA processes','NumberTitle','on');

subplot(211)
plot(y1)
title('arma1')
subplot(212)
plot(y2)
title('arma2')


m = 20;

rtheo = kovarians(arma1.c, arma1.a, m);

figure2 = figure('Name','3.1 Theoretical and estimated covariance functions','NumberTitle','on');
stem(0:m, rtheo*sigma2);
hold on
rest = covf(y1,m+1);
stem(0:m, rest, 'r');
xlabel('lags')
ylabel('r_y(k)')
title('blue = theoretical, red = estimated')

figure3 = figure('Name','3.1 Re-estimation of data vector y1 plots','NumberTitle','on');
set(gcf, 'Position',  [150, 50, 800, 700]);
subplot(311);
pacf(y1, m, 0.05, 1, 1)
title('Estimated PACF')
subplot(312)
acf(y1, m, 0.05, 1, 1)
title('Estimated ACF')
subplot(313)
normplot(y1);
title('Normplot of output')

data = iddata(y1);
arma_model = armax(y1, [2 2]);


%close all

%% 3.2

n = 500;
A = [1 -1.35 0.43];
sigma2 = 4;
noise = sqrt(sigma2)*randn(n+100,1);
y = filter(1, A, noise);
y = y(101:end); % Why do we do this 
subplot(211)
plot(y)
title('AR(2) process')
subplot(212)
plot(noise)
title('Driving noise')

n_est = floor(2/3 * n);
y_est = iddata(y(1:n_est)); % Modelling data
y_val = iddata(y(n_est + 1:end)); % Validation data

NN = [1:10]';

V = arxstruc(y_est, y_val, NN);
n_order = selstruc(V,0);
n_aic = selstruc(V, 'aic');

%% Looping model selection order 
n = 500;

A = [1 -1.35 0.43];
sigma2 = 4;

k=100;
n_order = zeros(k,1);
n_aic = zeros(k,1);
NN = [1:10]';

rng('default')
for i=1:k
   noise = sqrt(sigma2)*randn(n+100,1);
   y = filter(1, A, noise);
   y = y(101:end);
   
   n_est = floor(2/3 * n);
   y_est = iddata(y(1:n_est));
   y_val = iddata(y(n_est + 1:end));

  

   V = arxstruc(y_est, y_val, NN);
   n_order(i) = selstruc(V,0);
   n_aic(i) = selstruc(V, 'aic');
   
end

figure3 = figure('Name','3.2 Model order selection histogram - LS vs. AIC','NumberTitle','on');
set(gcf, 'Position',  [150, 150, 800, 700]);
subplot(221)
histogram(n_order)
title('Histogram LS, n = 500')
subplot(222)
histogram(n_aic)
title('Histogram AIC, n = 500')

n = 2*500;

A = [1 -1.35 0.43];
sigma2 = 4;

k=100;
n_order = zeros(k,1);
n_aic = zeros(k,1);
NN = [1:10]';

rng('default')
for i=1:k
   noise = sqrt(sigma2)*randn(n+100,1);
   y = filter(1, A, noise);
   y = y(101:end);
   
   n_est = floor(2/3 * n);
   y_est = iddata(y(1:n_est));
   y_val = iddata(y(n_est + 1:end));

  

   V = arxstruc(y_est, y_val, NN);
   n_order(i) = selstruc(V,0);
   n_aic(i) = selstruc(V, 'aic');
   
end


subplot(223)
histogram(n_order)
title('Histogram LS, n = 1000')
subplot(224)
histogram(n_aic)
title('Histogram AIC, n = 1000')

%% 3.3 - Model order estimation of an ARMA-process
load data.dat

ar1_model = arx(data, [1]);
ar2_model = arx(data, [2]);

data = iddata(data);
rar1 = resid(ar1_model, data);
rar2 = resid(ar2_model, data);

figure3 = figure('Name','3.3 - Residuals vs. noise for AR(p) model for ARMA(1,1) data','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);

load noise.dat;
subplot(221)
plot(noise, rar1.y, '*')
xlabel('Driving noise')
ylabel('Residuals')
title('AR(1)')
subplot(222)
plot(noise, rar2.y, '*')
xlabel('Driving noise')
ylabel('Residuals')
title('AR(2)')
subplot(223)
resid(ar1_model, data);
title('AR(1)')
subplot(224)
resid(ar2_model, data);
title('AR(2)')

% SECTION 5.5

am11_model = armax(data, [1 1]);
am22_model = armax(data, [2 2]);

%% 3.4 Estimation fo a SARIMA-process

A =[1 1.5 0.7];
C = [1 zeros(1,11) -0.5];
A12 = [1 zeros(1,11) -1]; %Why -1? 
A_star = conv(A, A12);
e = randn(600,1);
y = filter(C, A_star, e);
y = y(100:end);

m = 20;
figure3 = figure('Name','3.4 Basic analysis of SARIMA output','NumberTitle','on');
set(gcf, 'Position',  [150, 50, 800, 700]);
subplot(311);
pacf(y, m, 0.05, 1, 1);
title('Estimated PACF')
subplot(312)
acf(y, m, 0.05, 1, 1);
title('Estimated ACF')
subplot(313)
normplot(y);
title('Normplot of output')

y_s = filter(A12, 1, y); %Removing season - HOW?
data = iddata(y_s);

yy = data.OutputData;


m = 20;
figure3 = figure('Name','3.4 Basic analysis of SARIMA output with season removed','NumberTitle','on');
set(gcf, 'Position',  [150, 50, 800, 700]);
subplot(311);
pacf(yy, m, 0.05, 1, 1);
title('Estimated PACF')
subplot(312)
acf(yy, m, 0.05, 1, 1);
title('Estimated ACF')
subplot(313)
normplot(yy);
title('Normplot of output')


model_init = idpoly([1 0 0], [], []);
model_armax = pem(data, model_init);

rar = resid(model_armax, data);


figure4 = figure('Name','3.4 Residual of SARIMA output','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
subplot(221)
resid(model_armax,data)
subplot(222)
acf(rar.OutputData,m,0.05,1,1);
title('ACF')
subplot(223)
pacf(rar.OutputData,m,0.05,1,1);
title('PACF')
subplot(224)
normplot(rar.OutputData);
title('normplot')


model_init = idpoly([1 0 0], [], [1 zeros(1,12)]);
model_init.Structure.c.Free = [zeros(1,12) 1];
model_armax = pem(data, model_init);


% model_init = idpoly([1 0 0 0], [], [1 zeros(1,12)]);
% model_init.Structure.c.Free = [0 1 zeros(1,10) 1];
% model_init.Structure.a.Free = [0 1 0 1];
% model_armax = pem(data, model_init);


rar = resid(model_armax, data);


figure5 = figure('Name','3.4 Residual of SARIMA output with C12 free','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
subplot(221)
resid(model_armax,data)
subplot(222)
acf(rar.OutputData,m,0.05,1,1);
title('ACF')
subplot(223)
pacf(rar.OutputData,m,0.05,1,1);
title('PACF')
subplot(224)
normplot(rar.OutputData);
title('normplot')


close all;
%% Estimation on real data
load('/Users/joelbluhme/Dokument/hd_SKOLA/MATLAB/Tidsserier FMSN45/Course files/data/svedala.mat')
y = svedala;
figure(1)
subplot(121)
plot(y)
subplot(122)
lambda = bcNormPlot(y) % = 0.8687 -> No Transformation needed


m = 100;
figure3 = figure('Name','3.5 Basic data anlysis','NumberTitle','on');
set(gcf, 'Position',  [150, 50, 800, 700]);
subplot(311);
pacf(y, m, 0.05, 1, 1);
title('Estimated PACF')
subplot(312)
acf(y, m, 0.05, 1, 1);
title('Estimated ACF')
subplot(313)
normplot(y);
title('Normplot of output')

A24 = [1 zeros(1,23) -1]; %Why -1 
y_s=filter(A24,1,y);
y_s=y_s(25:end); %Fråga Carl vrf. fel?
data=iddata(y_s);


m = 50;
figure3 = figure('Name','3.5 Basic data anlysis - season removed','NumberTitle','on');
set(gcf, 'Position',  [150, 50, 800, 700]);
subplot(311);
pacf(y_s, m, 0.05, 1, 1);
title('Estimated PACF')
subplot(312)
acf(y_s, m, 0.05, 1, 1);
title('Estimated ACF')
subplot(313)
normplot(y_s);
title('Normplot of output')


model_init = idpoly([1 zeros(1,24)],[],[1 zeros(1,24)]);
model_init.Structure.a.Free = [0 1 1 zeros(1,21) 1];
model_init.Structure.c.Free = [0 0 0 zeros(1,21) 1];
model_armax = pem(data, model_init)
armax_a = model_armax.a;
armax_c = model_armax.c;

% y_res = filter(armax_a,armax_c,y_s); Fråga om denna
rar = resid(model_armax, data);

figure5 = figure('Name','3.5 Residual Analysis','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
subplot(221)
resid(model_armax,data)
subplot(222)
acf(rar.OutputData,m,0.05,1,1);
title('ACF')
subplot(223)
pacf(rar.OutputData,m,0.05,1,1);
title('PACF')
subplot(224)
normplot(rar.OutputData);
title('normplot')


%% Förberedelse

y = input_data;
A5 = [1 zeros(1,4) -1]; % Setting season
y_s = filter(A5, 1, y); % Removing season


model_init = idpoly([1 0 0 0], [], [1 zeros(1,12)]); % Initial structure
model_init.Structure.c.Free = [0 1 zeros(1,10) 1]; %Setting c_1 and c_12 as free, rest fixed
model_init.Structure.a.Free = [0 1 0 1]; % Setting a_1 and a_3 as free, rest fixed
model_armax = pem(data, model_init); % Solve for parameters
