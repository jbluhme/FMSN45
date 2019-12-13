% TSA Kalman filter test of kalman_armax function


%% Simulate non-dynamic ARMA(2,1) process
clc
clear
sigma2=1.5; 
N=1000; %% Nbr of realizations
e=sqrt(sigma2)*randn(N,1); %%Generate innovation seq e(t) with V[e(t)]=sigma2
a1=0.5;
a2=0.8;
A=[1 a1 a2];
C=[1 -0.4 0.3];
y=filter(C,A,e); % Generate ARMA(2,1)
Re = zeros(4);
V0 = 10^-2*eye(4)%100*eye(4); % Initial variance 
V0(1,1)=0;
m0 = [0.5 0.8 -0.4 0.3]';
Rw=1.5;

%% Simulate ARMAX model
clc 
clear

sigma2=1.5; 
sigma1=3;
N=1000; %% Nbr of realizations
e=sqrt(sigma2)*randn(N,1); %%Generate innovation seq e(t) with V[e(t)]=sigma2
w=sqrt(sigma1)*randn(N,1);
u=filter([1 0.3], [1 0.6],w);  %% Generate input seq



A=[1 0.5 0.4 0.3];
C=[1 -0.4 0.8];
B=[0.5 -0.2];

y=filter(C,A,e)+filter(B,A,u); % Generate ARMAX(3,2,1)
Re = zeros(7);
V0 = 10^-3*eye(7); % Initial variance

m0 = [0.5 0.4 0.3 -0.4 0.8 0.5 -0.2]';
Rw=1.5;


%% Test kalman_armax with ARMAX model simulated above




k=1; % prediction step size

% function call:
[param,pred]=kalman_armax(y,u,3,1,2,Re,Rw,V0,m0,k);
size(pred)

% Plots prediction and param estimates:

plot(pred(400:600,2),pred(400:600,1));
hold on
plot(pred(400:600,2),y(pred(400:600,2)))
legend('pred','true')
hold off
figure(2)
plot(param(1,:))
hold on
plot(param(2,:))
plot(param(3,:))
hold off

pe=y(pred(100:600,2))-pred(100:600,1);


figure(3)
rho = acf( pe, 100,0.05, 1, 1 );
title("ACF for pe with 95% confidence interval (asymptotic interval)");
figure(4)



param

whitenessTest(pe,0.01)

