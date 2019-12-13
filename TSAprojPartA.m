%% PROJECT TSA 2019 with Joel Bluhme


%% PART A Conclusions:
% A good ARMA-model is saved in M1 and the steps for getting to it
% and the predictions with it on val and test data is found below

% M1:
% A(z) = 1 - 1.342 (+/- 0.03405) z^-1                                         
%           + 0.4108 (+/- 0.03324) z^-2                                         
%           - 0.167 (+/- 0.02166) z^-23                                         
%           - 0.2308 (+/- 0.05447) z^-24                                        
%           + 0.3484 (+/- 0.0399) z^-25                                         
%                                                                               
%                                                                               
%   C(z) = 1 + 0.05144 (+/- 0.03654) z^-1                                       
%           - 0.3417 (+/- 0.04016) z^-24  
% 

% As seen are all parameters highly significant except c1 which is a bit  whithin
% the confidence interval (estimate+-2 stdv). It does however improve
% val.data predictions and is therefore kept.

clear
clc
%%
load('climate66.dat')
load('climate67.dat')
load('climate68.dat')
load('climate69.dat')

 






%% Following (appr 16 weeks) of temp. data from 1967 seems relatively stationary, we choose it to make up our 
% modelling (11 weeks) and validation (3 weeks) and test data 1 and 2 (1 week each).

Mdldata=climate67(3400:5000,:); % Defines the data sets

Valdata=climate67(5001:5600,:);


Test1data=climate67(5601:5800,:);
Test2data=climate67(8000-500:9200-500,:);

%% ARMA modelling
%% Step 1a Check whether transformation of data is reasonable

OptLambda=bcNormPlot(climate67(1:8500,8)) % =1.07 so no transformation seems necessary
% looks at the entire choosen data set
%% Step 1b make the process y zero mean
y=Mdldata(:,8);
y=y-mean(y); 

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
% and adding 1 and 24 MA (lag 2 was insignificant) lags we conclude that
% M1 is a reasonably good model 


data=iddata(y);

AS=[1 zeros(1,23) -1]; % Define our model polynomials
A=conv([1 0 0],AS);
C=[1 1 0 zeros(1,21) 1];
ar=idpoly(A,[],C); % Estimate model M1 and the resulting residual 
ar.Structure.a.Free = [0 1 1 A(4:end-4) 1 1 1 0];
ar.Structure.c.Free = C;
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

% The test statistics in the whiteness test are relatively good for being
% real data and both acf, pacf and cumPer of residuals look good
% The val-data predictions below look good with this model too
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

V_pe1=var(pe1) % =0.3621
mean(pe1) % =0.0455

% Passes almost all whitenessTests!!!!!

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



figure(2)
rho = acf( pe7, 100,0.05, 1, 1 );
title("ACF for pe7");

V_pe7=var(pe7) % =3.7104
TV_pe7=F'*F*V_pe1 % =3.9 ie higher than actual
mean(pe7) % =0.4101 thus we see that our model underestimate the temperature,
% but not by much. This is which is logical since
% we modelled during late spring early summer and our validation is over
% end of july and august which usually brings higher temperatures.
% This can be fixed with a state space model that continously reestimates
% model paramters

% We could also try to make a slighlty smaller modelling set

%% 26-step prediction
k=26;
[F,G]=Diophantine(M1.c,M1.a,k)

SF=50; % Safety factor, begin predicting SF time units before val to handle the initial corruptness of the data

y=Mdldata(:,8); % Our zero mean modelling data vector
y=y-mean(y); 

yval=Valdata(:,8); % Our zero mean adj. validation data vector
yval=yval-mean(Mdldata(:,8)); 

% Crucial that the predictions and true values are in line, it can be seen that:

% yhat_k(1)= prediction of y(end-SF+k) = y(end-SF+(k-1)+1)=ynew(1+k) 
% yhat_k(2)= prediction of y(end-SF+k+1) = ynew(2+k)
% ...
% yhat_k(2+SF-k)=prediction of ynew(2+SF)=yval(1) ie the first "wanted" prediction

% yhat_k(end-k)=prediction of yval(end) ie the last "wanted" pred.


ynew=[y(end-SF:end); yval]; % We concatenate the last SF+1 values of the modelling data vector and the validation vector
yhat_26=filter(G,M1.c,ynew); % We filter this concatenated vector
figure(1)
plot(yhat_26(2+SF-k:end-k)+mean(Mdldata(:,8))) 
hold on 
plot(yval(1:end)+mean(Mdldata(:,8)))
legend('26-step pred','True value')
pe26=yval(1:end)-yhat_26(2+SF-k:end-k); % 26-step pred error 
hold off

% 26 step pred. follows the wiggling of the data amazingly well


figure(2)
rho = acf( pe26, 100,0.05, 1, 1 );
title("ACF for pe26");

V_pe26=var(pe26) % =4.7964
TV_pe26=F'*F*V_pe1 % =5.2584 ie higher than actual
mean(pe26) % =0.7774 underestimates temperature

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prediction on test sets

% ...


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
