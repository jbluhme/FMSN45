% TSA project part C: Recursive estiamtion of BJ-model (from part B) param

%% Define data and define the BJ model derived in TSAprojPartB
clear
clc

load('climate67.dat')
Mdldata=climate67(3400:5000,:); 
Valdata=climate67(5001:5600,:);


Test1data=climate67(5601:5800,:);
Test2data=climate67(8000-500:9200-500,:);


u=Mdldata(:,6);
u=u+150; % Shifts u 150 units up to ensure positivity
u=log(u);
MeanLogu=mean(u);
u=u-mean(u);

y=Mdldata(:,8); %Output modelling vector
meanY=mean(y);
y=y-meanY; % Makes y zero mean
%% Define the earlier derived BJ model 
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



%% Concatenate modelling and validation data
% Mdl data has 1601 values, thus val data is found in ynew(1602:end)

yval=Valdata(:,8);
yval=yval-meanY; 

uval=Valdata(:,6);
uval=uval+150;
uval=log(uval);
uval=uval-MeanLogu;

ynew=[y; yval];
unew=[u; uval];

%% Initliaze kalman filter with the following parameter values
% We use our non-recursive param estimate of BJ above as m0;
% We set the initial variance to zero for parameters that are zero and a
% relatively low variance for those non-zero.
% Rw we set close to the MSE of the BJ estimate above, which should be an
% unbiased estimate of the variance of the driving noise.
% Re has zero diag for zero param. and relatively low for the non-zero param.

% BJ model on ARMAX form:

A=conv(BJ.d,BJ.f);
C=conv(BJ.c,BJ.f);
B=conv(BJ.d,BJ.b);
p=length(A)-1; %=27
q=length(C)-1; %=26
s=length(B)-1; %=26
% In total 27+26+26+1=80 parameters but a lot are of course fixed to zero

Re = 10^-6*eye(80); % Choose system error variability
Rw=0.2228;             % Choose measurement error variability which should be around MSE of model
m0=[A(2:end) C(2:end) B]'; % Our BJ non-recursive estimate as initial
diagOfV0=zeros(1,length(m0)); 

for i=1:length(m0) % Makes sure that zero param stay zero by setting their variance to zero
if m0(i)~=0
    diagOfV0(i)=1;
else
   Re(i,i)=0; 
end
end
V0=10^-3*diag(diagOfV0); % Initial variance of m0, should be pretty low


%% Use kalman to rec. estimate param. over ynew and unew and predict k step on val. data 



k=7; % desired prediction step size

% function call:
[param,pred]=kalman_armax(ynew,unew,p,s,q,Re,Rw,V0,m0,k);
size(pred)

% Plots prediction and param estimates:

% pred(1,1)=ynewhat(max(p,s)+predlength|max(p,s))=ynewhat(27+k|27)

% Thus pred(1575,1)=ynewhat(1602|1601)=prediction of yval(1)
% pred(end,1)=prediction of yval(end)

plot(pred(1575:end,2),pred(1575:end,1));
hold on
plot(pred(1575:end,2),ynew(pred(1575:end,2)))
legend('pred','true')
hold off


figure(2)
plot(param(1,:))
hold on
plot(param(2,:))
plot(param(3,:))
plot(param(4,:))

plot(ones(2201,1)*A(2))
plot(ones(2201,1)*A(3))
plot(ones(2201,1)*A(4))
plot(ones(2201,1)*A(5))
hold off
title("A-polynomial param")
legend('a1', 'a2', 'a3', 'a4')
pe=ynew(pred(1575:end,2))-pred(1575:end,1);


figure(3)
rho = acf( pe, 100,0.05, 1, 1 );
title("ACF for pe with 95% confidence interval (asymptotic interval)");
figure(4)

VarianceOfPredError=var(pe)
meanOfPredError=mean(pe)

whitenessTest(pe,0.01)

