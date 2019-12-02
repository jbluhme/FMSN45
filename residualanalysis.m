% FUNCTION residualanalysis(model, data, m)
%
% Performs basic residual analysis of modelling data vector data with model, calculations for lags 0, 1, ... m

function residualanalysis(model, data, m)
    rar = resid(model, data);
    subplot(221)
    resid(model,data)
    subplot(222)
    acf(rar.OutputData,m,0.05,1,1);
    title('ACF')
    subplot(223)
    pacf(rar.OutputData,m,0.05,1,1);
    title('PACF')
    subplot(224)
    normplot(rar.OutputData);
    title('normplot')
end
