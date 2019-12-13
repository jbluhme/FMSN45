%% Kalman filter for recursive parameter estimation of SISO ARMAX model
% on form A(z)y(t)=B(z)u(t)+C(z)e(t).

%% Function with output:

% param=optimal linear reconstruction of state vector x(t|y(t)&u(t)) 
% where param(x) is the reconstruc. at time corresponding to time of observ y(x) and u(x) in
% input vectors u and y.

% pred= matrix containing predlength-step prediction (col 1) and their times (col 2), of ARMAX with recur. estim. param. 
% through kalman filter, future input assumed known 

% Ie pred(i,1) is prediction of y(pred(i,2)) 
 


%% Requires input:
% y=output data vector [modelleing data...testdata]
% u=input data vector [modelleing data...testdata] (length(y)=length(u))
% if no input set u=0
% Aord=order A(z)=p
% Cord=order C(z)=q
% Bord=order B(z)=s
% predlength= desired prediction length

% Re:
% Usually assume independet param estimates which means that
% Re should be an n*n diagonal matrix (n=order(A)+order(B)+1+order(C)), 
% where the first p diag elements corrsponds to the assumed variance of the
% A parameters, the next q correpsonds to C parameters and final s+1 to B param variance. 
% If you do not want a parameter to change, set it's diag element in Re to zero.
% Set diag in V0 for that parameter to zero too then it will not change


% Example: A=[1 a1 a2] B=[b0 b1] C=[1 c1] where you believe that a1 and b1 should be constant
% then you set Re=zeros(5) and Re(2,2)=V[a2], Re[3,3]=V[b0] and Re[5,5]=V[c1]

% Rw: 
% Since we are looking at single output system Rw is scalar and should of course theoretically 
% be equal to the variance of the underlying noise seq. for the examined ARMAX model

% V0 and m0:
% Idea about recursive estimation: Estimate a good model non-recursively
% for modelling data and then initialize the kalman filter function
% at beginning of modelling data with (m0=non recursive param estimate
% =[a1 ... ap c1 ... cq b0 ... bq])
% (V0=low, like eye(length(m0))*10^-4) and Re and Rw choosen according to how fast you think the 
% system fluctuates between different seasons. Fast systems need higher Re
% and lower Rw and vice versa for slow changing systems



function [param,pred]=kalman_armax(y,u,Aord,Bord,Cord,Re,Rw,V0,m0,predlength)

% Data length
N = length(y); 

predictions=zeros(N-max([Bord Aord])-predlength+1,2); % Vector to save predictions in


% State space equations w. states being ARMAX param: x(t)=[a1(t) ... ap(t) c1(t) ... cq(t) b0(t) ... bs(t)]'

% x(t) = A*x(t-1)+e = I*x(t-1)+e(t)
% y(t) = C(t)*x(t) = [-y(t-1)...-y(t-p) ehat(t-1|t-2)...ehat(t-q|t-q-1) u(t)...u(t-s)]*x(t)+w(t)
 
if isrow(y)
y=y';
end

if isrow(u)
u=u';
end

if isrow(m0)
m0=m0';
end
    
Totparam=Aord+Bord+Cord; 


if sum(u)~=0 % If we have input then we allow the first coeff. b0 in B(z) to vary 
    Totparam=Totparam+1;
end
 

A =eye(Totparam); % Define system matrices
C=zeros(1,Totparam); 

ehat=zeros(1,Cord); % ehat=[pe(t|t-1) ... pe(t-q+1|t-q)]
% Vector to store 1-step pred errors (pe) in, since we do not have any
% idea about their size in the beginning we assume that the pred errors for 
% times max([s p])-q+1 ... max([s p]) are zero.


Rxx_1 = V0; % Initial variance 
xtt=m0; %  Initial state, necessary for first pred.
xtt_1 = xtt; 

% Example AR(2):
% Then V0=Rxx_1(3|2) 
% m0=xtt(2|2)=xtt_1(3|2)
 

% Vector to store values in
xsave=zeros(Totparam,N);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Actual Kalman filter algorithm  

for t=max([Bord Aord])+1:N % We must have enough values to define C(k)

 
 
for i=1:Totparam % Update C(k) 
   
    
    if i<=Aord % Update AR part with previous y values
 C(i)=-y(t-(i));
    
    elseif i<=Aord+Cord % Update MA part with previous prediction errors
        C(i)=ehat(i-Aord);
       
    else   % Update input part with prev. u 
    C(i)= u(t-(i-(Aord+Cord)-1));      
   
    end
     
   
    
end % C(k) is updated
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates prediction:
% yhat(t-1+predlength|t-1)=C(t-1+predlength|t-1)*xt-1t-1
% predictions(1,1)=yhat(max([Bord Aord])+predlength|max([Bord Aord]))
% predictions(1,2)=time of predictions(1,1)=max([Bord Aord])
% predictions(2,1)=yhat(max([Bord Aord])+1+predlength|max([Bord Aord])+1)
% predictions(2,2)=time of predictions(2,1)=max([Bord Aord])+1
% ...


if t<=N-predlength+1 % Our last prediction is made at time t=N-predlength and is for y(N)
yhat=zeros(1,predlength);   
Chat=C;

for k=1:predlength
    
    yhat(k)=Chat*xtt; % yhat(k)=yhat(t-1+k|t+k-2)
    
    if k==predlength
        break;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if Aord>0 % If we have an AR part in model
    
       for j=1:min([k Aord]) % 
      Chat(j)=yhat(k-j+1);  
        end
    
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if Cord>0 % If we have an MA part in model
    
    if k>Cord % Chat contains only future noise which are best predicted by zero
        
        Chat(Aord+1:Aord+Cord)=zeros(1,Cord);
        
    else % Chat contains estimates (pred. errors) of previous noise
        
        if k>1  
        Chat(Aord+1:Aord+k-1)=zeros(1,k-1);
        end
        Chat(Aord+k:Aord+Cord)=ehat(1:Cord-k+1);
        
    end
    
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    if Bord>=0 && sum(u)~=0  % If we have an input part with B(z) in model
        
      Chat(Aord+Cord+1:Aord+Cord+Bord+1)=flip(u(t+k-Bord:t+k)'); % Assume future input known 
   
    end
    
end
  predictions(t-max([Bord Aord]),1)=yhat(end); % The desired prediction ( for t-1+predlength) is found at end of yhat
  predictions(t-max([Bord Aord]),2)=t-1+predlength; % Time that is predicted

end









% Update Covariance, variance predictions, kalman gain K and pred. error
 % of y(k|k-1) and optimal reconstruction of state (xtt) and one step state prediction (xtt_1).
Ryy =C*Rxx_1*C'+Rw; 

K =Rxx_1*C'*inv(Ryy);

pe=y(t)-C*xtt_1; 

xtt =xtt_1+K*pe;

Rxx =(eye(Totparam)-K*C)*Rxx_1;

Rxx_1 = A*Rxx*A'+Re; 

xsave(:,t)=xtt; % Save x(t|t) estimate

xtt_1=A*xtt; 

ehat=circshift(ehat,1); % Oldest pred. error is now first vector element
ehat(1)=pe; % Put the most recent pred. error in that spot



end 

param=xsave;
pred=predictions;
end