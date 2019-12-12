function u = markov(P, n)
%MARKOV Summary of this function goes here
%   Detailed explanation goes here

mc = dtmc(P);
mc.StateNames = ["State 1" "State 2"];

u = simulate(mc, n);

end

