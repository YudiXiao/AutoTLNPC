function [k,a,b] = SteinmetzConst(f,B,Pv)
% This function calculates the constants for using in Steinmetz equation
%   f in kHz, B in T (Tesla), Pv in kW/m^3
% x(1) is k in kW/m^3, x(2) is a, x(3) is b.

% f = [300;500;1e3];
% B = [100e-3;50e-3;50e-3];
% Pv = [330;80;475];

fun = @(x) [x(1) * (f(1) ^ x(2)) .* (B(1) ^ x(3)) - Pv(1);...
            x(1) * (f(2) ^ x(2)) .* (B(2) ^ x(3)) - Pv(2);...
            x(1) * (f(3) ^ x(2)) .* (B(3) ^ x(3)) - Pv(3)];

x0 = [0.1;0.1;0.1];

options = optimoptions('fsolve'); 
options.MaxIterations = 1000;
options.MaxFunctionEvaluations = 5000;
options.Diagnostics = 'off';
options.Display = 'off';
sol = fsolve(fun,x0,options);

k = sol(1);
a = sol(2);
b = sol(3);

end

