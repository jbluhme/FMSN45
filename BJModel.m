


%% Pretty succesful BJ attempt

%% Working BJ Model and prediction with it

%% Define data
clear
clc

load('climate67.dat')
Mdldata=climate67(3400:5000,:); 
Valdata=climate67(5001:5600,:);


Test1data=climate67(5601:5800,:);
Test2data=climate67(8000-500:9200-500,:);


%% Transforming u with log and making zero mean

OptLambda=bcNormPlot(climate67(1:8500,6)) % =0.02 so log transformation seems reasonable
 % We need to shift u up before taking log

u=Mdldata(:,6);
u=u+150; % Shifts u 150 units up to ensure positivity
u=log(u);
MeanLogu=mean(u);
u=u-mean(u);

y=Mdldata(:,8); %Output modelling vector
meanY=mean(y);
y=y-meanY; % Makes y zero mean



%%  Finally estimate all paramters together, with PEM


A1=[1 1 1 zeros(1,20) 0 1 0];
C1=[1 zeros(1,23) 1];
B =[1 1 1];
A2 = [1 1 1];
Mi = idpoly (1 ,B ,C1 ,A1 ,A2);
Mi.Structure.d.Free =A1;
Mi.Structure.c.Free =C1;
z = iddata(y,u);
BJ= pem(z,Mi); 
present(BJ)
ehat=resid(BJ,z);

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

whitenessTest(ehat.y,0.05)

%% 1 step pred
k=1;

% Transform BJ into ARMAX with:
B=conv(BJ.b,BJ.d);
A=conv(BJ.f,BJ.d);
C=conv(BJ.c,BJ.f);
ufut1=u(k+1:end);


SF=50; % Safety factor, begin predicting SF time units before val to handle the initial corruptness of the data

% yhat_1(1)=prediction of y(end-SF+1)=ynew(2) thus
% yhat_1(1+SF)=prediction of ynew(SF+2)=yval(1) ie the first "wanted" prediction
% yhat_1(end)=prediction of yval(end) ie the last "wanted" pred.

y=Mdldata(:,8); % Our zero mean output modelling and val. data vector
yval=Valdata(:,8);
y=y-meanY; 
yval=yval-meanY; 

u=Mdldata(:,6); % Our shifted,log,zero mean input modelling and val. data vector
uval=Valdata(:,6);
u=u+150; 
uval=uval+150;
u=log(u);
uval=log(uval);
u=u-MeanLogu;
uval=uval-MeanLogu;

ynew=[y(end-SF:end); yval];
unew=[u(end-SF:end); uval];
unewfu1=unew(k+1:end); % Future u vector ie unefu1(i)=unew(i+1)

[F,G]=Diophantine(C,A,k);
[Fhat,Ghat]=Diophantine(conv(B,F),C,k)
yhat_7=filter(Ghat,C,unew(1:end-k))+filter(G,C,ynew(1:end-k))+filter(Fhat,1,unewfu1);



figure(1)
plot(yhat_7(2+SF-k:end)+meanY) 
hold on 
plot(yval(1:end)+meanY)
legend('1-step pred','True value')
hold off
pe7=yval(1:end)-yhat_7(2+SF-k:end); % 1-step pred error 



figure(2)
rho = acf( pe7, 100,0.05, 1, 1 );
title("ACF for pe1");

figure(3)
whitenessTest(pe7,0.01)



V_pe7=var(pe7) %=0.2304
mean(pe7)  % =0.0647

%% 7 step pred.
k=7;





SF=50; % Safety factor, begin predicting SF time units before val to handle the initial corruptness of the data

% yhat_1(1)=prediction of y(end-SF+1)=ynew(2) thus
% yhat_1(1+SF)=prediction of ynew(SF+2)=yval(1) ie the first "wanted" prediction
% yhat_1(end)=prediction of yval(end) ie the last "wanted" pred.


unewfu7=unew(k+1:end); % Future u vector ie unefu7(i)=unew(i+7)

[F,G]=Diophantine(C,A,k);
[Fhat,Ghat]=Diophantine(conv(B,F),C,k)
yhat_7=filter(Ghat,C,unew(1:end-k))+filter(G,C,ynew(1:end-k))+filter(Fhat,1,unewfu7);



figure(1)
plot(yhat_7(2+SF-k:end)+meanY) 
hold on 
plot(yval(1:end)+meanY)
legend('7-step pred','True value')
hold off
pe7=yval(1:end)-yhat_7(2+SF-k:end); % 7-step pred error 




figure(2)
rho = acf( pe7, 100,0.05, 1, 1 );
title("ACF for pe7"); % Should me MA(7-1) which seems reasonable





V_pe7=var(pe7) % = 2,00
mean(pe7)  % 0.58 Slight underestimation again