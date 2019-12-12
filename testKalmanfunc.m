% TSA Kalman filter test of kalman_armax function
clc
clear

% Simulate non-dynamic ARMA(2,1) process
sigma2=1.5; 
N=1000; %% Nbr of realizations
e=sqrt(sigma2)*randn(N,1); %%Generate innovation seq e(t) with V[e(t)]=sigma2
a1=0.5;
a2=0.8;
A=[1 a1 a2];
C=[1 -0.4];
y=filter(C,A,e); % Generate ARMA(2,1)
Re = zeros(3);
V0 = 1000*eye(3); % Initial variance 
m0 = [0 0 0]';
Rw=10;

k=1; % prediction step size

% function call:
[param,pred]=kalman_armax(y,0,2,0,1,Re,Rw,V0,m0,k);
size(pred)

% Plots prediction and param estimates:

plot(pred(500:600,2),pred(500:600,1));
hold on
plot(pred(500:600,2),y(pred(500:600,2)))
legend('pred','true')
hold off
figure(2)
plot(param(1,:))
hold on
plot(param(2,:))
plot(param(3,:))
hold off






