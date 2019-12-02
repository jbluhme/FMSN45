% FUNCTION basicanalysis(x,m)
%
% Performs basic analysis of data vector data, calculations for lags 0, 1, ... m

function basicanalysis(data,m) 
    m = m;
    x = data;
    subplot(311);
    acf(x, m, 0.05, 1, 1);
    title('Estimated ACF')
    subplot(312)
    pacf(x, m, 0.05, 1, 1);
    title('Estimated PACF')
    subplot(313)
    normplot(x);
    title('Normplot of output')
end

