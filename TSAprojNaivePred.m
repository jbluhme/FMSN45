%% TSA Project naive estimators on val and test data

clear
clc

load('climate67.dat')
Mdldata=climate67(3400:5000,:); 
Valdata=climate67(5001:5600,:);


Test1data=climate67(5601:5768,:); 
Test2data=climate67(7501:7668,:);

meanY=mean(Mdldata(:,8));

y=climate67(1:7668,8); % All data up to and including Test2 in one vector

%% Naive 1 step prediction yhat(t+1|t)=y(t) on val. data

yhat=y(1:end-1); % This vector contains the 1-step predictions for y(2:end)

% Val.
peVal=y(5001:5600)-yhat(5000:5599);
figure(1)
plot(y(5001:5600))
hold on
plot(yhat(5000:5599));
hold off
legend('true y','naive 1-step pred')
title("Naive 1-step prediction on validation data");

VarVal=var(peVal)
meanVal=mean(peVal)

%% Naive 7-step val. data and test data, yhat(t+7|t)=y(t+7-24)

yhat=y(1:7644); % This vector contains the 7-step predictions yhat(25|18)...yhat(7668|7661)

% Val. data

peVal=y(5001:5600)-yhat(4977:5576);
figure(1)
plot(y(5001:5600))
hold on
plot(yhat(4977:5576));
hold off
legend('true y','naive 7-step pred')
title("Naive 7-step prediction on validation data");

VarVal=var(peVal) %=5.4563
meanVal=mean(peVal) %=-0.1633

% Test1 

figure(2)
peTest1=y(5601:5768)-yhat(5577:5744);

plot(y(5601:5768))
hold on
plot(yhat(5577:5744));
hold off
legend('true y','naive 7-step pred')
title("Naive 7-step prediction on test1 data");

VarTest1=var(peTest1) %=1.8964
meanTest1=mean(peTest1) %=0.1817

% Test2
figure(3)
peTest2=y(7501:7668)-yhat(7477:7644);

plot(y(7501:7668))
hold on
plot(yhat(7477:7644));
hold off
legend('true y','naive 7-step pred')
title("Naive 7-step prediction on test2 data");

VarTest2=var(peTest2) % =3.7225
meanTest2=mean(peTest2) % =0.3704

%% Naive 26-step val. data data, yhat(t+26|t)=y(t+26-48)=y(t-22)

yhat=y(1:7620); % This vector contains the 7-step predictions yhat(49|23)...yhat(7668|7661)

% Val. data

peVal=y(5001:5600)-yhat(4953:5552);
figure(1)
plot(y(5001:5600))
hold on
plot(yhat(4953:5552));
hold off
legend('true y','naive 26-step pred')
title("Naive 26-step prediction on validation data");

VarVal=var(peVal)
meanVal=mean(peVal)