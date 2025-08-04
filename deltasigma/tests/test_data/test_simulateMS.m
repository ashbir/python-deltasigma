% MATLAB script to generate test_simulateMS.mat

% Add the delsig directory to the MATLAB path
addpath('../../delsig');

% Test parameters
M = 16;
order = 1;
mtf = zpk(1, 0, 1, 1); % A simple z-1 MTF
d = 0.1;
dw = ones(M, 1);
sx0 = zeros(order, M);
N = 256;
v = round( (M-1) * sin(2*pi*(1:N)/N) ); % Example input
% Ensure v has the correct parity
if mod(M, 2) == 0 % M is even
    v(mod(v,2) ~= 0) = v(mod(v,2) ~= 0) + 1;
else % M is odd
    v(mod(v,2) == 0) = v(mod(v,2) == 0) + 1;
end


% Run the simulation
[sv, sx, sigma_se, max_sx, max_sy] = simulateMS(v, M, mtf, d, dw, sx0);

% Save the results
save('test_simulateMS.mat', 'v', 'M', 'mtf', 'd', 'dw', 'sx0', 'sv', 'sx', 'sigma_se', 'max_sx', 'max_sy');

fprintf('test_simulateMS.mat generated successfully.\n');
