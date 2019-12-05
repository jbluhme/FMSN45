% PROJECT TSA 2019 with Joel Bluhme
clear
clc
%%
load('climate66.dat')
load('climate67.dat')
load('climate68.dat')
load('climate69.dat')
plot(climate67(7000:7200,8))
 





%% SUGGESTION:
%% Following (appr 16 weeks) of temp. data from 1967 seems relatively stationary, we choose it to make up our 
% modelling (11 weeks) and validation (3 weeks) and test data 1 and 2 (1 week each).
figure(2)
plot(climate67(3200:5800,8)) % Plots modelling, validation, and test 1 sets

Mdldata=climate67(3200:5000,:); % Defines the data sets
Valdata=climate68(3200:5000,:);
Test1data=climate67(5601:5800,:);
Test2data=climate67(7000:7200,:);


%% ARMA modelling
%% Step 1a Check whether transformation of data is reasonable

OptLambda=bcNormPlot(climate67(3200:5800,8)) % =0.95 so no transformation seems necessary
% looks at the entire choosen data set
%% Step 1b make the process y zero mean
y=Mdldata(:,8);
y=y-mean(y); 

%% Step 2 Examine the pacf and acf of y
Ad= [1 -1];
y_d=filter(Ad,1,y);
y_d=y_d(2:end); %%
figure(1)
phi = pacf( y_d, 100,0.05, 1, 1 );
title("PACF for y");
figure(2)
rho = acf( y_d, 100,0.05, 1, 1 );
title("ACF for y");
figure(3)
normplot(phi)
title("Normplot of pacf"); % estimated pacf seems gaussian -> confidence intervals are reliable

%% Step 2 Examine the pacf and acf of y
figure(1)
phi = pacf( y, 100,0.05, 1, 1 );
title("PACF for y");
figure(2)
rho = acf( y, 100,0.05, 1, 1 );
title("ACF for y");
figure(3)
normplot(phi)
title("Normplot of pacf"); % estimated pacf seems gaussian -> confidence intervals are reliable

%% Step 3 From ACF we see a clear 24 periodicity and from pacf two prominent peaks at lag 1 and 2
%%% Let us therefore try model 1, M1: AR(2) combined with 24-differentiator
% After removing insign. lag 26 and adding lag 23
% and adding two MA lags we conclude that
% M1 is a reasonably good model 

data=iddata(y);

AS=[1 zeros(1,23) -1]; % Define our model polynomials
A=conv([1 1 0],AS)
C=[1 1 1];
ar=idpoly(A,[],C); % Estimate model M1 and the resulting residual 
ar.Structure.a.Free = [0 1 1 A(4:end-4) 1 1 1 0];
M1 = pem(data,ar);
r1=resid(M1,data);


figure(1)
phi = pacf( r1.y, 100,0.05, 1, 1 );
title("PACF for res");
figure(2)
rho = acf( r1.y, 100,0.05, 1, 1 );
title("ACF for res");
figure(3)
whitenessTest(r1.y,0.01)

present(M1)

% The test statistics in he whiteness test are relatively good for being
% real data and both acf, pacf and cumPer of residuals look good


%% Step 4 Prediction on validation set


%% 1-step prediction 
k=1;
[F,G]=Diophantine(M1.c,M1.a,k)
SF=50; % Safety factor, begin predicting SF time units before val to handle the initial corruptness of the data

% yhat_1(1)=prediction of y(end-SF+1)=ynew(2) thus
% yhat_1(1+SF)=prediction of ynew(SF+2)=yval(1) ie the first "wanted" prediction
% yhat_1(end-1)=prediction of yval(end) ie the last "wanted" pred.

y=Mdldata(:,8); % Our zero mean modelling data vector
y=y-mean(y); 

yval=Valdata(:,8); % Our zero mean adj. validation data vector
yval=yval-mean(Mdldata(:,8)); % We subtract the modelling mean not validation mean since the former is the mean we assume to be true 
% ie our model is that the difference Temperature-mean(Modelling set) is an ARMA process

ynew=[y(end-SF:end); yval]; % We filter this concatenated vector  
yhat_1=filter(G,M1.c,ynew);

figure(1) % We plot the predicted vs true with the mean added
plot(yhat_1(1+SF:end-1)+mean(Mdldata(:,8))) % discard the first SF predictions 
hold on 
plot(yval(1:end)+mean(Mdldata(:,8)))
legend('1-step pred','True value')
pe1=yval(1:end)-yhat_1(1+SF:end-1); % 1-step pred error 
hold off

figure(2)
rho = acf( pe1, 100,0.05, 1, 1 );
title("ACF for pe1");
figure(3)
whitenessTest(pe1,0.01)

% The 1-step prediction residuals of validation data set look extremely white
% thus our model probably is very good

V_pe1=var(pe1) % =0.3689
mean(pe1)



%% 7-step prediction 
k=7;
[F,G]=Diophantine(M1.c,M1.a,k)

SF=50; % Safety factor, begin predicting SF time units before val to handle the initial corruptness of the data

y=Mdldata(:,8); % Our zero mean modelling data vector
y=y-mean(y); 

yval=Valdata(:,8); % Our zero mean adj. validation data vector
yval=yval-mean(Mdldata(:,8)); 

% Crucial that the predictions and true values are in line, it can be seen that:

% yhat_7(1)= prediction of y(end-SF+k) = y(end-SF+(k-1)+1)=ynew(1+k) 
% yhat_7(2)= prediction of y(end-SF+k+1) = ynew(2+k)
% ...
% yhat_7(2+SF-k)=prediction of ynew(2+SF)=yval(1) ie the first "wanted" prediction

% yhat_7(end-k)=prediction of yval(end) ie the last "wanted" pred.


ynew=[y(end-SF:end); yval]; % We concatenate the last SF+1 values of the modelling data vector and the validation vector
yhat_7=filter(G,M1.c,ynew); % We filter this concatenated vector
figure(1)
plot(yhat_7(2+SF-k:end-k)+mean(Mdldata(:,8))) 
hold on 
plot(yval(1:end)+mean(Mdldata(:,8)))
legend('7-step pred','True value')
pe7=yval(1:end)-yhat_7(2+SF-k:end-k); % 7-step pred error 
hold off

% Since not even SMHI perfectly can tell the weather one week from now I would say
% that the 7-step prediction from a simple SARIMA model M1 is absolutely acceptable


figure(2)
rho = acf( pe7, 100,0.05, 1, 1 );
title("ACF for pe7");
%%
V_pe7=var(pe7) % =3.95
TV_pe7=F'*F*V_pe1 % =5.17 ie higher than actual
mean(pe7) % =0.78 thus we see that our model underestimate the temperature,
% but not by much. This is which is logical since
% we modelled during late spring early summer and our validation is over
% end of july and august which usually brings higher temperatures.
% This can be fixed with a state space model that continously reestimates
% model paramters

% We could also try to make a slighlty smaller modelling set



