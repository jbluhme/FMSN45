% TSA project part C: Recursive estiamtion of BJ-model (from part B) param

%% Define data and define the BJ model derived in TSAprojPartB
clear
clc

load('climate67.dat')
Mdldata=climate67(3400:5000,:); 
Valdata=climate67(5001:5600,:);


Test1data=climate67(5601:5768,:); 
Test2data=climate67(7501:7668,:);


u=Mdldata(:,6);
u=u+150; % Shifts u 150 units up to ensure positivity
u=log(u);
MeanLogu=mean(u);
u=u-MeanLogu;

y=Mdldata(:,8); %Output modelling vector
meanY=mean(y);
y=y-meanY; % Makes y zero mean

%% Vectors containing all 1967 input and output data
totY=climate67(1:end,8);
totY=totY-meanY;

totU=climate67(1:end,6);
totU=totU+150; 
totU=log(totU);
totU=totU-MeanLogu;


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
Rw=0.25;             % Choose measurement error variability which should be around MSE of model
m0=[A(2:end) C(2:end) B]'; % Our BJ non-recursive estimate as initial
diagOfV0=zeros(1,length(m0)); 

for i=1:length(m0) % Makes sure that zero param stay zero by setting their variance to zero
if m0(i)~=0
    diagOfV0(i)=1;
else
   Re(i,i)=0; 
end
end
V0=10^-10*diag(diagOfV0); % Initial variance of m0, should be pretty low

V0=zeros(80);
Re=zeros(80);
%% Use kalman to rec. estimate param. over ynew and unew and predict k step on val. data 

k=7; % desired prediction step size

% function call:
[param,pred]=kalman_armax(ynew,unew,p,s,q,Re,Rw,V0,m0,k);
size(pred)

% Plots prediction and param estimates:

% pred(1,1)=ynewhat(max(p,s)+predlength|max(p,s))=ynewhat(27+k|27)

% Thus pred(1576-k,1)=ynewhat(1602|1602-k)=prediction of yval(1)
% pred(end,1)=prediction of yval(end)

plot(pred(1576-k:end,2),pred(1576-k:end,1)+meanY*ones(600,1)); % Plots k-step pred against true values over validation set
hold on
plot(pred(1576-k:end,2),ynew(pred(1576-k:end,2))+meanY*ones(600,1))
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

pe=ynew(pred(1576-k:end,2))-pred(1576-k:end,1); % k-step pred error over val.


figure(3)
rho = acf( pe, 100,0.05,1 , k-1,0 );
title("ACF for pe over val with 95% confidence interval (asymptotic interval)");
figure(4)

VarianceOfPredError=var(pe)
meanOfPredError=mean(pe)

whitenessTest(pe,0.01)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prediction and recursive param. estimation over entire 1967 

% Choose kalman param:
m0=[A(2:end) C(2:end) B]'; % Our BJ non-recursive estimate as initial

Re = 10^-5*eye(80); % Choose system error variability
Rw=5;             % Choose measurement error variability which should be around MSE of model
diagOfV0=zeros(1,length(m0)); 

for i=1:length(m0) % Makes sure that zero param stay zero by setting their variance to zero
if m0(i)~=0
    diagOfV0(i)=1;
else
   Re(i,i)=0; 
end
end






V0=10^-8*diag(diagOfV0); % Initial variance of m0, should be pretty low


k=7; % desired prediction step size

% function call:
y=totY(3400:end); 
u=totU(3400:end);
[param,pred]=kalman_armax(y,u,p,s,q,Re,Rw,V0,m0,k);


%% Plot k-step (defined above) pred over entire data, the correpsonding prediction error acf 
% and the recursive param estimates of the ARMAX corresponding to our BJ
% model in TSAProjPartB
figure(1)
plot(pred(1:end,2),pred(1:end,1)+meanY*ones(length(pred),1));
hold on
plot(pred(1:end,2),y(pred(1:end,2))+meanY*ones(length(pred),1))
title("k-step pred over entire data set") 
legend('pred','true')
hold off

% Converting our BJ model into ARMAX yields:
% A includes 1,2,3,4,23,24,25,26 lags
% C includes 2,24,26 lags
% B includes 0,1,2,3,23,24,25 lags (0 included since b0 is not=1)

figure(2)
plot(param(1,:)) % a1 
hold on
plot(param(2,:))
plot(param(3,:))
plot(param(4,:))
plot(param(23,:)) 
plot(param(24,:))
plot(param(25,:))
plot(param(26,:)) 

hold off
title("A-polynomial")
legend('a1', 'a2', 'a3', 'a4','a23', 'a24', 'a25', 'a26')



figure(3)
plot(param(29,:)) 
hold on
plot(param(51,:))
plot(param(53,:))
title("C-polynomial")
legend('c2', 'c24', 'c26')
hold off

figure(5)
plot(param(54,:)) 
hold on
plot(param(55,:))
plot(param(56,:))
plot(param(57,:))
plot(param(77,:)) 
plot(param(78,:))
plot(param(79,:))
title("B-polynomial")
legend('b0', 'b1', 'b2','b3','b23', 'b24', 'b25')
hold off


figure(6)
peTot=y(pred(1:end,2))-pred(1:end,1);
rho = acf( peTot, 100,0.05, 1, k-1,0 );
title("ACF for peTot with 95% confidence interval (asymptotic interval)");

VarianceOfTotalDataSetPredError=var(peTot)
meanOfTotalDataSetPredError=mean(peTot)
figure(7)
whitenessTest(peTot,0.01)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
% Idea: Try to find which parameters that seem to be constant over the year
% and set their variance to zero so that they stay constant. Then we allow the
% truely changing parameters to tune in to their "right" values.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Test data 7-step pred. (initialize filter at totY(3400) to make m0 a more certain initial state estimate)

k=7; % desired prediction step size

% When trying to predict the test data sets 
% we do not want to initialize the kalman already at the beginning of the
% year ie at totY(1), since the initial estimates m0 are not correct for
% that part of the year since they are derived from modelling data.
% It is much more logical to initilize the filter at beginning of modelling
% data, ie at totY(3400). 

% Test data found at:
Test1data=climate67(5601:5768,:); % ie ytest1=totY(5601:5768)
Test2data=climate67(7501:7668,:);
m0=[A(2:end) C(2:end) B]';
% Define kalman param: 
Re = 10^-4*eye(80); % Choose system error variability
Rw=10;             % Choose measurement error variability which should be around MSE of model
diagOfV0=zeros(1,length(m0)); 

for i=1:length(m0) % Makes sure that zero param stay zero by setting their variance to zero
if m0(i)~=0
    diagOfV0(i)=1;
else
   Re(i,i)=0; 
end
end



V0=10^-8*diag(diagOfV0); % Initial variance of m0, should be pretty low




ydata=totY(3400:end); % Initialize filter at beginning of modelling data, ie use these vectors.
udata=totU(3400:end);

% function call:
[param,pred]=kalman_armax(ydata,udata,p,s,q,Re,Rw,V0,m0,k);

% Thus we find the predictions of the test sets:

% pred(2176-k,1)=ydatahat(2202|2202-k)=prediction of ytest1(1)
% pred(2343,1)=prediction of ytest1(end)

% pred(4076-k,1)=ydatahat(4102|4102-k)=prediction of ytest2(1)
% pred(4243,1)=prediction of ytest2(end)

figure(1)

plot(pred(2169:2336,2),pred(2169:2336,1)+meanY*ones(168,1));
hold on
plot(pred(2169:2336,2),ydata(pred(2169:2336,2))+meanY*ones(168,1))
title("7-step pred with rec. kalman estimation of test1 data") 
legend('pred','true')
hold off

figure(2)
peTest1=ydata(pred(2169:2336,2))-pred(2169:2336,1);
rho = acf( peTest1, 100,0.05, 1, k-1,1 );
title("ACF for peTest1 with 95% confidence interval (asymptotic interval)");

VarianceOfPredErrorTest1=var(peTest1)
meanOfPredErrorTest1=mean(peTest1)
peTest1SS=peTest1'*peTest1/length(peTest1)

figure(3)

plot(pred(4069:4236,2),pred(4069:4236,1)+meanY*ones(168,1));
hold on
plot(pred(4069:4236,2),ydata(pred(4069:4236,2))+meanY*ones(168,1))
title("7-step pred with rec. kalman estimation of test2 data") 
legend('pred','true')
hold off

figure(4)
peTest2=ydata(pred(4069:4236,2))-pred(4069:4236,1);
rho = acf( peTest2, 100,0.05, 1, k-1,1 );
title("ACF for peTest2 with 95% confidence interval (asymptotic interval)");

VarianceOfPredErrorTest2=var(peTest2)
meanOfPredErrorTest2=mean(peTest2)
peTest2SS=peTest2'*peTest2/length(peTest2)

% Prediction error should be zero mean let us look at the sum of squared
% deviations from zero rather than the sample variance of the pred. error
% as a measurement of the performance of the model


% BJ w. constant parameters:

% Test 1
% V[pe7]=1,7438
% mean[pe7]=0.4265
% pe7SS=1.9154

% Test 2
% V[pe7]=1,3152
% mean[pe7]=-1.2544
% pe7SS=2.8810


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Test data 7-step pred. Constant BJ model but allow for varying mean of process



k=7; % desired prediction step size

% When trying to predict the test data sets 
% we do not want to initialize the kalman already at the beginning of the
% year ie at totY(1), since the initial estimates m0 are not correct for
% that part of the year since they are derived from modelling data.
% It is much more logical to initilize the filter at beginning of modelling
% data, ie at totY(3400). 

% Test data found at:
Test1data=climate67(5601:5768,:); % ie ytest1=totY(5601:5768)
Test2data=climate67(7501:7668,:);

% Define kalman param: 

Rw=1; 

m0=[A(2:end) C(2:end) B 0]'; % Add initial mean estimate=0 last

diagOfRe=[zeros(1,length(m0)-1) 8]; 
diagOfV0=[zeros(1,length(m0)-1) 1]; 

for i=1:length(m0)-1 % Makes sure that zero param stay zero by setting their variance to zero
if m0(i)~=0
    diagOfV0(i)=1;
   diagOfRe(i)=1; 
end
end



%diagOfRe=[zeros(1,length(m0)-1) 1]; 
%diagOfV0=[zeros(1,length(m0)-1) 1]; 
V0=10^-8*diag(diagOfV0); 

Re = 10^-4*diag(diagOfRe);
 




ydata=totY(3400:end); % Initialize filter at beginning of modelling data, ie use these vectors.
udata=totU(3400:end);

% function call:
[param,pred]=kalman_armaxVM(ydata,udata,p,s,q,Re,Rw,V0,m0,k);

% Thus we find the predictions of the test sets:

% pred(2176-k,1)=ydatahat(2202|2202-k)=prediction of ytest1(1)
% pred(2343,1)=prediction of ytest1(end)

% pred(4076-k,1)=ydatahat(4102|4102-k)=prediction of ytest2(1)
% pred(4243,1)=prediction of ytest2(end)

figure(1)

plot(pred(2169:2336,2),pred(2169:2336,1)+meanY*ones(168,1));
hold on
plot(pred(2169:2336,2),ydata(pred(2169:2336,2))+meanY*ones(168,1))
title("7-step pred with rec. kalman estimation of test1 data") 
legend('pred','true')
hold off

figure(2)
peTest1=ydata(pred(2169:2336,2))-pred(2169:2336,1);
rho = acf( peTest1, 100,0.05, 1, k-1,1 );
title("ACF for peTest1 with 95% confidence interval (asymptotic interval)");

VarianceOfPredErrorTest1=var(peTest1)
meanOfPredErrorTest1=mean(peTest1)
peTest1SS=peTest1'*peTest1/length(peTest1)

figure(3)

plot(pred(4069:4236,2),pred(4069:4236,1)+meanY*ones(168,1));
hold on
plot(pred(4069:4236,2),ydata(pred(4069:4236,2))+meanY*ones(168,1))
title("7-step pred with rec. kalman estimation of test2 data") 
legend('pred','true')
hold off

figure(4)
peTest2=ydata(pred(4069:4236,2))-pred(4069:4236,1);
rho = acf( peTest2, 100,0.05, 1, k-1,1 );
title("ACF for peTest2 with 95% confidence interval (asymptotic interval)");

VarianceOfPredErrorTest2=var(peTest2)
meanOfPredErrorTest2=mean(peTest2)
peTest2SS=peTest2'*peTest2/length(peTest2)

figure(5)
plot(param(end,:))
title("Varying mean");
% Prediction error should be zero mean let us look at the sum of squared
% deviations from zero rather than the sample variance of the pred. error
% as a measurement of the performance of the model


% BJ w. constant parameters:

% Test 1
% V[pe7]=1,7438
% mean[pe7]=0.4265
% pe7SS=1.9154

% Test 2
% V[pe7]=1,3152
% mean[pe7]=-1.2544
% pe7SS=2.8810




figure(6)
plot(param(1,:)) % a1 
hold on
plot(param(2,:))
plot(param(3,:))
plot(param(4,:))
plot(param(23,:)) 
plot(param(24,:))
plot(param(25,:))
plot(param(26,:)) 

hold off
title("A-polynomial")
legend('a1', 'a2', 'a3', 'a4','a23', 'a24', 'a25', 'a26')



figure(7)
plot(param(29,:)) 
hold on
plot(param(51,:))
plot(param(53,:))
title("C-polynomial")
legend('c2', 'c24', 'c26')
hold off

figure(8)
plot(param(54,:)) 
hold on
plot(param(55,:))
plot(param(56,:))
plot(param(57,:))
plot(param(77,:)) 
plot(param(78,:))
plot(param(79,:))
title("B-polynomial")
legend('b0', 'b1', 'b2','b3','b23', 'b24', 'b25')
hold off