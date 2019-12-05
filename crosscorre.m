% FUNCTION crosscorr(x,y,m)
%
% Returns plot of the cross-covariance of data series x and y, evaluated in
% lags -m, -m + 1, ..., 0, ..., m

function crosscorre(x,y,m) 
    M = m; 
    stem(-M:M, crosscorr(x,y,M));
    title('Cross correlation function'), xlabel('Lag')
    hold on
    confi = min(length(x), length(y));
    plot(-M:M, 2/sqrt(confi)*ones(1,2*M+1), '--')
    plot(-M:M, -2/sqrt(confi)*ones(1,2*M+1), '--')
    hold off
end