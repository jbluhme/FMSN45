function [Fk, Gk] = diophantine(C, A, k)
%DIOPHANTINE performs diophantine polynominal division
%   Makes vectors of equal length and then performs polynominal division

[CS, AS] = equalLength(C,A); 
[Fk, Gk] = deconv( conv( [1 zeros(1, k-1)], CS), AS);

end