
%% PROJECT TSA 2019 with Joel Bluhme cont...


%% PART B Conclusions:
% A good Box-Jenkins-model is saved in BJ and the steps for getting to it
% and the predictions with it on val and test data is found below

% BJ =                                                                          
% Discrete-time BJ model: y(t) = [B(z)/F(z)]u(t) + [C(z)/D(z)]e(t)              
%   B(z) = 1.672 (+/- 0.05988) + 1.179 (+/- 0.05664) z^-1                       
%                                                                               
%   C(z) = 1 + 0.0531 (+/- 0.02778) z^-24                                       
%                                                                               
%                                                                               
%   D(z) = 1 - 1.34 (+/- 0.02336) z^-1 + 0.3929 (+/- 0.02343) z^-2              
%           - 0.1327 (+/- 0.02484) z^-23 + 0.1078 (+/- 0.02423) z^-24           
%                                                                               
%   F(z) = 1 - 0.4636 (+/- 0.02291) z^-2   

% As seen are all parameters highly significant except c24 which is right
% on edge of the confidence interval (estimate+-2 stdv). It does however improve
% val.data predictions and is therefore kept.


% This BJ model yields much better pred. error variance on the validation data set
% then the ARMA model M1 (in TSAprojPartA). Especially the 7 step and
% 26-step predictions are extraordinary improvement


clear
clc

%% Pretty succesful BJ attempt

%% Working BJ Model and prediction with it

%% Define data
clear
clc

load('climate67.dat')
Mdldata=climate67(3400:5000,:); 
Valdata=climate67(5001:5600,:);


Test1data=climate67(5601:5768,:);
Test2data=climate67(7501:7668,:);


%% Transforming u with log and making zero mean & making y zero mean

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


%%
%% Model u as ARMA
figure(1)

phi = pacf( u, 100,0.05, 1, 1 );
title("PACF for u with 95% confidence interval (asymptotic interval)");
figure(2)
rho = acf( u, 100,0.05, 1, 1 );
title("ACF for u with 95% confidence interval (asymptotic interval)");
%%
A24=[1 zeros(1,23) -1];

A=conv([1 1 1],A24);
C=[1 zeros(1,23) 1];
M1u = idpoly ( A,[],C);
M1u.Structure.a.Free =A;
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

%% Transform (pre-whiten) y and u with inverse input arma model
upw=filter(modelU.a,modelU.c,u);
ypw=filter(modelU.a,modelU.c,y);
upw=upw(25:end);
ypw=ypw(25:end);
M=300;

crosscorre(upw,ypw,M)

%% CCF above -> r=2 s=1 d=0 seems good choice of orders
% We tried s=0 first but the resulting CCF between res and u was bad then


A2 = [1 1 1]; % r=2
B =[1 1];  % s=1
H = idpoly ([1] ,[B] ,[] ,[] ,[A2]);

 zpw = iddata(ypw,upw);
Mba2 = pem(zpw,H); 
present(Mba2)
vhat = resid(Mba2,zpw); 
%% CCF between vhat=ypw-H(z)upw  and upw 

crosscorre(vhat.y,upw,M) % looks relatively good

%% Modelling residual res=y-H(z)u as ARMA
res=y-filter(Mba2.b,Mba2.f,u);
z = iddata(y,u);
r=resid(z,Mba2);
%% CCF between res=y-H(z)u and u 
M=100
crosscorre(res,u,M) % not perfect uncorrelation but looks good enough to move on
%% ACF and PACF of Res
phi = pacf( res, 50,0.05, 1, 1 );
title("PACF for u with 95% confidence interval (asymptotic interval)");
figure(2)
rho = acf( res, 50,0.05, 1, 1 );
title("ACF for u with 95% confidence interval (asymptotic interval)");
%% Seems to be a dependency at lag 1,2 and 23, 24 in PACF 
% 
A1=[1 1 1 zeros(1,20) 1 1];
C1=[1 zeros(1,23) 1];
ar2=idpoly(1,[],C1,A1,[]);
ar2.Structure.d.Free =A1;
ar2.Structure.c.Free =C1;
data=iddata(res);
NoiseMdl=pem(data,ar2);
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

% we remove lag 1 in A2 polynomial since it is severely unsignificant

A1=[1 1 1 zeros(1,20) 1 1 0];
C1=[1 zeros(1,23) 1];
B =[1 1];
A2 = [1 0 1];
Mi = idpoly (1 ,B ,C1 ,A1 ,A2);
Mi.Structure.d.Free =A1;
    Mi.Structure.c.Free =C1;
Mi.Structure.b.Free =B;
Mi.Structure.f.Free =A2;
z = iddata(y,u);
BJ= pem(z,Mi); 
present(BJ) 
ehat=resid(BJ,z);

M= 100; stem(-M:M,xcorr(ehat.y,u,M,'biased')); 
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

whitenessTest(ehat.y,0.01)




%% Predictions on val. data:
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
yhat_1=filter(Ghat,C,unew(1:end-k))+filter(G,C,ynew(1:end-k))+filter(Fhat,1,unewfu1);



figure(1)
plot(yhat_1(2+SF-k:end)+meanY) 
hold on 
plot(yval(1:end)+meanY)
legend('1-step pred','True value')
hold off
pe1=yval(1:end)-yhat_1(2+SF-k:end); % 1-step pred error 



figure(2)
rho = acf( pe1, 100,0.05, 1, 1 );
title("ACF for pe1");

figure(3)
whitenessTest(pe1,0.01)



V_pe1=var(pe1) %=0.2261
mean(pe1)  % =0.0566

%% 7 step pred.
k=7;

% Same SF as for 1-step





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





V_pe7=var(pe7) % = 1.96
mean(pe7)  % 0.5049 Slight underestimation again
%% 26 step pred.
k=26;



unewfu26=unew(k+1:end); % Future u vector ie unefu26(i)=unew(i+26)

[F,G]=Diophantine(C,A,k);
[Fhat,Ghat]=Diophantine(conv(B,F),C,k)
yhat_26=filter(Ghat,C,unew(1:end-k))+filter(G,C,ynew(1:end-k))+filter(Fhat,1,unewfu26);



figure(1)
plot(yhat_26(2+SF-k:end)+meanY) 
hold on 
plot(yval(1:end)+meanY)
legend('26-step pred','True value')
hold off
pe26=yval(1:end)-yhat_26(2+SF-k:end); % 26-step pred error 




figure(2)
rho = acf( pe26, 100,0.05, 1, 1 );
title("ACF for pe26"); % Should me MA(26-1) which seems reasonable





V_pe26=var(pe26) % =2.9880 
mean(pe26)  %  =1.0774



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pred on test data
Test1data=climate67(5601:5768,:);
Test2data=climate67(7501:7668,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 7 step pred. test1 data
k=7;

% Same SF as for 1-step

yval=Valdata(:,8); % Our val. data vector and test1 data
yval=yval-meanY; 

ytest1=Test1data(:,8);
ytest1=ytest1-mean(Mdldata(:,8)); % Our zero mean test1 data 

uval=Valdata(:,6);
uval=uval+150;
uval=log(uval);
uval=uval-MeanLogu;

utest1=Test1data(:,6);
utest1=utest1+150;
utest1=log(utest1);
utest1=utest1-MeanLogu;

ynew=[yval(end-SF:end); ytest1]; % Concatenate last SF-1 values of val data and test1 data
unew=[uval(end-SF:end); utest1];

unewfu7=unew(k+1:end); % Future u vector ie unefu7(i)=unew(i+7)

% 
[F,G]=Diophantine(C,A,k);
[Fhat,Ghat]=Diophantine(conv(B,F),C,k)
yhat_7=filter(Ghat,C,unew(1:end-k))+filter(G,C,ynew(1:end-k))+filter(Fhat,1,unewfu7);



figure(1)
plot(yhat_7(2+SF-k:end)+meanY) 
hold on 
plot(ytest1(1:end)+meanY)
legend('7-step pred','True value')
hold off
pe7=ytest1(1:end)-yhat_7(2+SF-k:end); % 7-step pred error 




figure(2)
rho = acf( pe7, 100,0.05, 1, 1 );
title("ACF for pe7"); % Should me MA(7-1) which seems reasonable





V_pe7=var(pe7) % = 1,8770 lower than on validation data
mean(pe7)  % 0.4650 Slight underestimation again

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 7 step pred. test2 data
k=7;

% Same SF as for 1-step

yprev=climate67(7401:7500,8); % Our zero mean adj. output data vector of the 100 data points right before ytest2
yprev=yprev-mean(Mdldata(:,8)); 

ytest2=Test2data(:,8);
ytest2=ytest2-mean(Mdldata(:,8)); % Our zero mean test1 data 

uprev=climate67(7401:7500,8); % Our input data vector of the 100 data points right before utest2
uprev=uprev+150;
uprev=log(uprev);
uprev=uprev-MeanLogu; 

utest2=Test2data(:,6);
utest2=utest2+150;
utest2=log(utest2);
utest2=utest2-MeanLogu;


ynew=[yprev(end-SF:end); ytest2]; % Concatenate last SF-1 values of prev data and test2 data
unew=[uprev(end-SF:end); utest2];

unewfu7=unew(k+1:end); % Future u vector ie unefu7(i)=unew(i+7)

% 
[F,G]=Diophantine(C,A,k);
[Fhat,Ghat]=Diophantine(conv(B,F),C,k)
yhat_7=filter(Ghat,C,unew(1:end-k))+filter(G,C,ynew(1:end-k))+filter(Fhat,1,unewfu7);



figure(1)
plot(yhat_7(2+SF-k:end)+meanY) 
hold on 
plot(ytest2(1:end)+meanY)
legend('7-step pred','True value')
hold off
pe7=ytest2(1:end)-yhat_7(2+SF-k:end); % 7-step pred error 




figure(2)
rho = acf( pe7, 100,0.05, 1, 1 );
title("ACF for pe7"); % Should me MA(7-1) which seems reasonable





V_pe7=var(pe7) % = 1,2443 lower than on test1, most likely due to lower temperatures and thus lower absolut variability
mean(pe7)  % -1.2711 Slight overestimation again