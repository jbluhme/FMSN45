%% 3.1 RLS Estimation
clear; clc;

load tar2.dat;
load thx.dat;

figure('Name','3.1 tar2 data and thx parameters','NumberTitle','on');
set(gcf, 'Position',  [50, 150, 1200, 700]);
subplot(131)
plot(tar2)
title('tar2 data')
subplot(132)
plot(thx(:,1), 'r')
hold on
plot(thx(:,2), 'b')
title('thx parameters')
legend('Parameter 1','Parameter 2')
hold off

na = 2;
nb = 0; %Degree of input
nk = 0; %Delay of input
model = [na]; %[na,nb,nk]

lambda = 0.95;
[Aest,yhat,covAest,yprev]= rarx(tar2,model,'ff',lambda);
subplot(133)
plot(Aest(:,1),'r');
hold on
plot(Aest(:,2),'b');
legend('Parameter 1 est.','Parameter 2 est.')
title('thx parameter estimates')

%% 3.1 Choosing lambda
n = 100;
lambda_line = linspace(0.85,1,n);
ls2 = zeros(n,1);
for i=1:length(lambda_line)
    [Aest,yhat,CovAest,trash] = rarx(tar2,[2],'ff',lambda_line(i));
    ls2(i) = sum((tar2-yhat).^2);
end
figure
plot(lambda_line,ls2)


%% 3.2 Kalman filtering of time series

%   Simulate data
y=tar2;

% Length of process
N = length(y);

% Define the state space equations
A = [1 0; 0 1]; %Initialization
Re = 0.001*[1 0; 0 0]; % Hidden state noise covariance matrix (Only one allowed to vary)
Rw = 1.25; %Observation variance - how noisy are measurements?

%usually C should be set here to
%but in this case C is a function of time.

% Set initial values

Raa_1 = 0.04*eye(2); % Initial variance
att_1 = [0 0]'; % Initial state att_1(1) is the varying

% Vector to store values in
xsave=zeros(2,N); %parameter estimates

% Kalman filter. Start from k=3, since we need old values of y.
for k=3:N
  % C is a function of time.
  C = [-y(k-1) -y(k-2)];
   
  % Update
  Ryy = C*Raa_1*C' + Rw; % 8.116
  Kt = (Raa_1*C')*(Ryy)^(-1); % 8.111
  att = att_1 + (Kt*(y(k) - C*att_1)); % 8.109
  Raa = (eye(2)-Kt*C)*Raa_1; % 8.114
 
  % Save
   xsave(:,k) = att;
  
  %1-step prediction
  
  att_1 = A*att; % 8.110
  Raa_1 = A*Raa*A' + Re; % 8.115
  
end


%% 3.2 Plotting true, RLS and Kalman 
figure('Name','3.2 True parameters, RLS estimates, Kalman estimates','NumberTitle','on');
subplot(311) 
plot(thx)
title('True parameters')

subplot(312) %RLS lambda = 0.95
plot(Aest)
title('RLS estimates, lambda = 0.95')

subplot(313) %Kalman
plot(xsave')
title('Kalman estimates')


%% 3.3 Quality control of a process - Init 
clear;
close all;

P = [7/8 1/8; 1/8 7/8];
n = 500;
u_t = markov(P, n);


b = 20;

%Generating white noises
e_t = sqrt(1)*randn(500,1); %Variance is 1
v_t = sqrt(4)*randn(500,1); %Variance is 4

%Generating x series
x = zeros(n, 1);
x(1) = e_t(1);
for i=2:length(x)
    x(i) = x(i-1) + e_t(i);
end

%Generating y series
y = zeros(n, 1);
for i=1:length(y)
    y(i) = x(i) + b*u_t(i) + v_t(i);
end

%% 3.3 QUality control of a process - Kalman

N = length(y); 

% Define the state space equations
A = [1 0; 0 1]; %Standard def.? 
Re = [1 0; 0 0]; %Only x_t corrupted by process noise (B is not), scaling this?
Rw = 1.25; 

% Set initial values

Rzz_1 = var(y(1:5))*eye(2); % Initial variance, why y? (8.107 initialization?)
ztt_1 = [0 0]'; % Initial states 

% Vector to store values in
z=zeros(2,N);

% Kalman filter. Start from k=3, since we need old values of y.
for k=2:N
  % C is a function of time.
  C = [1 u_t(k)]; %?
   
  % Update
  Ryy = C*Rzz_1*C' + Rw;
  Kt = (Rzz_1*C')*(Ryy)^(-1);
  ztt = ztt_1 + (Kt*(y(k) - C*ztt_1));
  Rzz = (eye(2)-Kt*C)*Rzz_1;
 
  % Save
  z(:,k) = ztt;
  
  %1-step prediction
  
  ztt_1 = A*ztt;
  Rzz_1 = A*Rzz*A' + Re;
  
  
end

%% 3.3 Plotting to compare
b_s = zeros(1,500)
for i=1:500
b_s(i) = b;    
end

figure('Name','3.3 True parameters, Kalman estimates','NumberTitle','on');
subplot(211) 
plot(x)
hold on
plot(b_s')
hold off
title('True parameters')

subplot(212) 
plot(z')
title('Kalman estimates')


