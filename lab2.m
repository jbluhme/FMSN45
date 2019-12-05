%% 3.1 Modeling of an exogenous input signal

rng('default')

clear; 

n = 500;
A1 = [1 -0.65]; A2 =[1 0.90 0.78];
C = 1; B = [0 0 0 0 0.4];
e = sqrt(1.5)*randn(n + 100,1);
w = sqrt(2)*randn(n + 200,1);
A3 = [1 0.5]; C3 = [1 -0.3 0.2]; % Structure for input u ARMA(1,2)
u = filter(C3, A3, w); u = u(101:end); % Generate input
y = filter(C, A1,e) + filter(B, A2, u);
u = u(101:end) ; y = y(101:end);
clear A1 A2 C B e A3 C3;

%% 3.1 Pre-whitening of u_t by finding suitible model

m = 20;
figure1 = figure('Name','3.1 u_t non-white data analysis','NumberTitle','on');
set(gcf, 'Position',  [150, 50, 800, 700]);
basicanalysis(u, m)

u_data = iddata(u);

model_init = idpoly([1 0], [], [1 0 0]); %ARMA(1,2)
model_u = pem(u_data, model_init);
upw = filter(model_u.a,model_u.c,u);
upw = upw(3:end);

rar = resid(model_u, u);

figure2 = figure('Name','3.1 Residuals from model of u_t (pw)','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
residualanalysis(model_u,u,20);

ypw = filter(model_u.a, model_u.c, y);
ypw = ypw(3:end);
figure3 = figure('Name','3.1 Residuals from model of y_t (pw)','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
basicanalysis(ypw,20);

figure4 = figure('Name','3.1 CCF from upw to ypw','NumberTitle','on');

M = 40; stem(-M:M, crosscorr(upw,ypw,M));
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(length(w))*ones(1,2*M+1), '--')
plot(-M:M, -2/sqrt(length(w))*ones(1,2*M+1), '--')
hold off

d = 4; r = 2; s = 1; %s is uncertain



%% 3.1 Modelling transfer function

A2 = [1 ones(1,r)];
B = [zeros(1,d) ones(1, 1+s)];

Mi = idpoly([1], [B], [], [], [A2]);
zpw = iddata(ypw,upw);
Mba2 = pem(zpw, Mi); 
%present(Mba2);
vhat = resid(Mba2,zpw);

figure100 = figure('Name','3.1 u_pw vs. residual v(t) correlation','NumberTitle','on');
M = 40; stem(-M:M, crosscorr(upw,vhat.y,M));
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(length(w))*ones(1,2*M+1), '--')
plot(-M:M, -2/sqrt(length(w))*ones(1,2*M+1), '--')
hold off


%% 3.1 Modelling BJ-ARMA
x = y - filter(Mba2.b, Mba2.f, u);
x_data = iddata(x);
figure5 = figure('Name','3.1 x_t data','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
basicanalysis(x, 20); % AR(1) initial guess

x_poly = idpoly([1 0], [], [1]);
x_model = pem(x_data, x_poly);
rar = resid(x_model, x_data);

figure6 = figure('Name','3.1 u_t and x_t correlation','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
M = 40; stem(-M:M, crosscorr(u,rar.y,M));
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(length(w))*ones(1,2*M+1), '--')
plot(-M:M, -2/sqrt(length(w))*ones(1,2*M+1), '--')
hold off

figure7 = figure('Name','Residual-analysis for x_t','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
residualanalysis(x_model,x_data,M)


%% 3.1  Final Model

A1 = [1 0];
A2 = [1 0 0];
B = [0 0 0 0 1]; % ASK how setting fixed vs. free works
C = [1];
Mi = idpoly(1, B, C, A1, A2);
z = iddata(y, u);
MboxJ = pem(z, Mi);
present(MboxJ);
ehat = resid(MboxJ,z);

figure8 = figure('Name','Residual-analysis for final BJ model','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
residualanalysis(MboxJ,z,M)

figure9 = figure('Name','3.1 u_t and e_t correlation','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
M = 40; stem(-M:M, crosscorr(u, ehat.y,M));
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(length(w))*ones(1,2*M+1), '--')
plot(-M:M, -2/sqrt(length(w))*ones(1,2*M+1), '--')
hold off



%% 3.2 Hairdryer initialization 
clear;
load tork.dat
tork = tork - repmat(mean(tork), length(tork),1);
y = tork(:,1); u = tork(:,2);
z = iddata(y,u);
% u = u(1:300); ???
% y = y(1:300); ???

figure('Name','3.2 Input and output data','NumberTitle','on');
plot(z(1:300))


%% 3.2 Pre-whitening of u_t by finding suitible model

m = 20;
figure1 = figure('Name','3.2 u_t non-white data analysis','NumberTitle','on');
set(gcf, 'Position',  [150, 50, 800, 700]);
basicanalysis(z.u, m) %%Looks like AR(1)

model_init = idpoly([1 0], [], [1]);
model_u = pem(z.u, model_init);

upw = filter(model_u.a,model_u.c,z.u);
upw = upw(2:end);

rar = resid(model_u, z.u);

figure2 = figure('Name','3.2 Residuals from model of u_t (pw)','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
residualanalysis(model_u,z.u,20); %Looks white

ypw = filter(model_u.a, model_u.c, z.y);
ypw = ypw(2:end);
figure3 = figure('Name','3.1 Residuals from model of y_t (pw)','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
basicanalysis(ypw,20);

figure4 = figure('Name','3.1 CCF from upw to ypw','NumberTitle','on');
M = 40; stem(-M:M, crosscorr(upw,ypw,M));
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(length(z.u))*ones(1,2*M+1), '--')
plot(-M:M, -2/sqrt(length(z.u))*ones(1,2*M+1), '--')
hold off

d = 3; r = 2; s=2;


%% 3.2 Modelling transfer function

A2 = [1 zeros(1,r)];
B = [zeros(1,d) ones(1, 1+s)];

Mi = idpoly([1], [B], [], [], [A2]);
zpw = iddata(ypw,upw);
Mba2 = pem(zpw, Mi); 
%present(Mba2);
vhat = resid(Mba2,zpw);

figure5 = figure('Name','3.1 u_pw vs. residual v(t) correlation','NumberTitle','on');
crosscorre(upw, vhat.y, M);

%% 3.2 Modelling BJ-ARMA
x = y - filter(Mba2.b, Mba2.f, u);
x_data = iddata(x);
figure5 = figure('Name','3.2 x_t data','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
basicanalysis(x, 20); % AR(1) initial guess

x_poly = idpoly([1 0], [], [1]);
x_model = pem(x_data, x_poly);
rar = resid(x_model, x_data);

figure6 = figure('Name','3.2 u_t and x_t correlation','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
M = 40; stem(-M:M, crosscorr(u,rar.y,M));
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(length(u))*ones(1,2*M+1), '--')
plot(-M:M, -2/sqrt(length(u))*ones(1,2*M+1), '--')
hold off

figure7 = figure('Name','3.2 Residual-analysis for x_t','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
residualanalysis(x_model,x_data,M)

%% 3.2  Final Model
%d = 3; r = 2; s=2;

p=1;
q=0;
r=2;
s=2;
d=3;


A1 = [1 0];
A2 = [1 0 0];
B = [0 0 0 1 1]; % ASK how setting fixed vs. free works
C = [1];
Mi = idpoly(1, B, C, A1, A2);
z = iddata(y, u);
MboxJ = pem(z, Mi);
present(MboxJ);
ehat = resid(MboxJ,z);

figure8 = figure('Name','Residual-analysis for final BJ model','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
residualanalysis(MboxJ,z,M)

figure9 = figure('Name','3.2 u_t and e_t correlation','NumberTitle','on');
set(gcf, 'Position',  [250, 150, 800, 700]);
M = 40; stem(-M:M, crosscorr(u, ehat.y,M));
title('Cross correlation function'), xlabel('Lag')
hold on
plot(-M:M, 2/sqrt(length(u))*ones(1,2*M+1), '--')
plot(-M:M, -2/sqrt(length(u))*ones(1,2*M+1), '--')
hold off

%% 3.3 Prediction of ARMA-processes
clear;
load svedala
y = svedala;
A = [1 -1.79 0.84]; 
C = [1 -0.18 -0.11];


k1 = 1;
[CS, AS] = equalLength(C, A);
[Fk1, Gk1] = deconv(conv([1, zeros(1,k1-1)], CS), AS);

yhat_k1 = filter(Gk1, C, y);
y1 = y(max(length(Gk1),length(C)):end);
yhat_k1 = yhat_k1(max(length(Gk1),length(C)):end);

% plot(linspace(1,length(y1),length(y1)),yhat_k1, 'r-')
% hold on
% plot(linspace(1,length(y1),length(y1)),y1, 'b-')
% hold off
% figure

ehat_1 = y1 - yhat_k1;
noise_var = var(ehat_1);


k3 = 3;
[CS, AS] = equalLength(C, A);
[Fk3, Gk3] = deconv(conv([1, zeros(1,k3-1)], CS), AS);

yhat_k3 = filter(Gk3, C, y);
y3 = y(max(length(Gk3),length(C)):end);
yhat_k3 = yhat_k3(max(length(Gk3),length(C)):end);

ehat_3 = y3 - yhat_k3;
mean_ehat_3 = mean(ehat_3);
Vehat_3_th = (norm(Fk3))^2*noise_var;
Vehat_3_emp = var(ehat_3);


k26 = 26;
[CS, AS] = equalLength(C, A);
[Fk26, Gk26] = deconv(conv([1, zeros(1,k26-1)], CS), AS);

yhat_k26 = filter(Gk26, C, y);
y26 = y(max(length(Gk26),length(C)):end);
yhat_k26 = yhat_k26(max(length(Gk26),length(C)):end);

ehat_26 = y26 - yhat_k26;
mean_ehat_26 = mean(ehat_26);
Vehat_26_th = norm(Fk26)^2*noise_var;
Vehat_26_emp = var(ehat_26);

quanti = norminv([0.025 0.975]);
y_k26_confi = yhat_k26 + quanti*noise_var*norm(Fk26);
y_k3_confi = yhat_k3 + quanti*noise_var*norm(Fk3);

% plot(linspace(1,length(y3),length(y3)),yhat_k3, 'r-')
% hold on
% plot(linspace(1,length(y3),length(y3)),y3, 'b-')

CI_ehat_3 = mean_ehat_3 + norminv([0.025 0.975])*sqrt(Vehat_3_th);
CI_ehat_26 = mean_ehat_26 + norminv([0.025 0.975])*sqrt(Vehat_26_th);

nbrOutside_3 = sum(ehat_3 > CI_ehat_3(2)) + sum(ehat_3 < CI_ehat_3(1));
prctOutside_3 = nbrOutside_3/length(ehat_3);
nbrOutside_26 = sum(ehat_26 > CI_ehat_26(2)) + sum(ehat_26 < CI_ehat_26(1));
prctOutside_26 = nbrOutside_26/length(ehat_26);


% nbrOutside_3_test = 0;
% 
% for i=1:length(ehat_3)
%     temp = ehat_3(i);
%     if temp > 0 && temp > CI_ehat_3(2)
%         nbrOutside_3_test = nbrOutside_3_test + 1;
%     end
%     
%     if temp < 0 && temp < CI_ehat_3(1)
%         nbrOutside_3_test = nbrOutside_3_test + 1;
%     end
%     
%     
% end

figure9 = figure('Name','3 and 26 step predictions','NumberTitle','on');
set(gcf, 'Position',  [50, 150, 1200, 700]);

% subplot(121)
% grid_3 = linspace(3,length(y3)+2,length(y3));
% plot(grid_3,yhat_k3, 'r-')
% hold on
% plot(grid_3,y3, 'b-')
% hold off
% legend('3-step prediction','Process')
% title('3-step prediction')
% 
% subplot(122)
% grid_26 = linspace(26,length(y3)+25,length(y26));
% plot(grid_26,yhat_k26, 'r-')
% hold on
% plot(grid_26,y26, 'b-')
% hold off
% legend('26-step prediction','Process')
% title('26-step prediction')

subplot(121)
plot(yhat_k3, 'r-')
hold on
plot(y3, 'b-')
hold off
legend('3-step prediction','Process')
title('3-step prediction')

subplot(122)
plot(yhat_k26, 'r-')
hold on
plot(y26, 'b-')
hold off
legend('26-step prediction','Process')
title('26-step prediction')

figure10 = figure('Name','3 step prediction residuals','NumberTitle','on');
set(gcf, 'Position',  [50, 150, 1200, 700]);
basicanalysis(ehat_3,40)
title('3-step prediction')
figure11 = figure('Name','26 step prediction residuals','NumberTitle','on');
set(gcf, 'Position',  [50, 150, 1200, 700]);
basicanalysis(ehat_26,40)
title('26-step prediction')

%% 3.4 Prediction of ARMAX-processes

load sturup;
u = sturup;
y = svedala;

A_u = [1 -1.49 0.57];
B_u = [0 0 0 0.28 -0.26];
C_u = [1];

%% 3.4 Svedala initialization

k3 = 3;
k26 = 26;

[Fk3, Gk3] = diophantine(C_u, A_u, k3);
[Fk26, Gk26] = diophantine(C_u, A_u, k26);

BF3 = conv(B_u,Fk3);
BF26 = conv(B_u,Fk26);

[Fu3_hat, Gu3_hat] = diophantine(BF3,C_u,k3);
[Fu26_hat, Gu26_hat] = diophantine(BF26,C_u,k26);

u3_hat = filter(Gu3_hat, C_u, u);
u3_hat = u3_hat(max([length(Gu3_hat), length(C_u), length(Gk3)]):end);
u26_hat = filter(Gu26_hat, C_u,u);
u26_hat = u26_hat(max([length(Gu26_hat), length(C_u), length(Gk26)]):end);

y1_3_hat = filter(Gk3,C_u, y);
y1_3_hat = y1_3_hat(max([length(Gu3_hat), length(C_u), length(Gk3)]):end);
y1_26_hat = filter(Gk26,C_u, y);
y1_26_hat = y1_26_hat(max([length(Gu26_hat), length(C_u), length(Gk26)]):end);

y_3_hat = u3_hat + y1_3_hat;
y_26_hat = u26_hat + y1_26_hat;


figure9 = figure('Name','3 step predictions','NumberTitle','on');
set(gcf, 'Position',  [50, 150, 1200, 700]);

y3 = y(length(y)-length(yhat_k3):length(y));
plot(yhat_k3, 'r-')
hold on
plot(y3,'b-')
hold off
legend('3-step prediction','Process')
title('3-step prediction')

e_3_hat = y(max([length(Gu3_hat), length(C_u), length(Gk3)]):end) - y_3_hat;
var_3_hat_emp = var(e_3_hat);

%% 3.5 Prediction of SARIMA-process

close all;
clear;
load svedala

figure11 = figure('Name','First analysis Svedala data','NumberTitle','on');
set(gcf, 'Position',  [50, 150, 1200, 700]);
subplot(121)
plot(svedala)
title('Raw data')
S = 24;
AS = [1 zeros(1,S-1) -1];
ys = filter(AS,1,svedala);
ys = ys(length(AS):end); % remove samples
subplot(122)
plot(ys);
title('Season of 24 removed')

figure12 = figure('Name','Data analysis Svedala season model','NumberTitle','on');
set(gcf, 'Position',  [50, 150, 1200, 700]);
basicanalysis(ys,40)


ys_poly = idpoly([1 0 0], [], [1 zeros(1,23) 1],[],[]); %% ARMA with a1 a2 and c24
ys_poly.Structure.c.Free = [0 zeros(1,23) 1];

ysdata = iddata(ys);
model_init = pem(ysdata,ys_poly);
ys_res = resid(model_init,ysdata);
figure13 = figure('Name','Residual analysis initial model','NumberTitle','on'); %Looks white!
set(gcf, 'Position',  [50, 150, 1200, 700]);
residualanalysis(model_init, ys, 40)

AxAS = conv(model_init.A, AS);
C = model_init.C;

nonStableARMA = model_init; %Copy of original model
nonStableARMA.a = AxAS; %Replacing A with SAR

svedala_data = iddata(svedala);
svedala_res = resid(nonStableARMA,svedala_data);


A = nonStableARMA.a;
C = nonStableARMA.c;

k3 = 3; 
k26 = 26;

y = svedala;

[Fk3s,Gk3s] = diophantine(C,A,k3);
yhat_sk3 = filter(Gk3s,C,svedala);
yhat_sk3 = yhat_sk3(max(length(Gk3s),length(C)):end);
y_sk3 = y(max(length(Gk3s),length(C)):end);

[Fk26s,Gk26s] = diophantine(C,A,k26);
yhat_sk26 = filter(Gk26s,C,svedala);
yhat_sk26 = yhat_sk26(max(length(Gk26s),length(C)):end);
y_sk26 = y(max(length(Gk26s),length(C)):end);

figure12 = figure('Name','Data analysis Svedala season model','NumberTitle','on');
set(gcf, 'Position',  [50, 150, 1200, 700]);

subplot(121)
plot(yhat_sk3, 'r-')
hold on
plot(y_sk3,'b-')
hold off
legend('3-step prediction','Process')
title('3-step prediction')

subplot(122)
plot(yhat_sk26, 'r-')
hold on
plot(y_sk26,'b-')
hold off
legend('3-step prediction','Process')
title('3-step prediction')

e_3_hat = y_sk3 - yhat_sk3;
var_emp_e_3 = var(e_3_hat);

e_26_hat = y_sk26 - yhat_sk26;
var_emp_e_26 = var(e_26_hat);



