n = 20 ;
m = 100;
% Data generation
x = randn(n,1) + 1i*randn(n,1); % True Signal
A = 1/sqrt(2)*randn(m,n) + 1i/sqrt(2)*randn(m,n); % measurement matrix
b = abs(A*x).^2; % measurements

% WF solver
max_iter= 2500;                           % Max number of iterations
tau0 = 330;                         % Time constant for step size
opts = [max_iter, tau0];
[z, errs] = phase_WF(A, b, x, opts);
 
% Reuslts
fprintf('Relative error after initialization: %f\n', errs(1))
fprintf('Relative error after %d iterations: %f\n', T, errs(T+1))
 
figure, semilogy(0:T,errs, 'linewidth', 2) 
xlabel('Iteration'), ylabel('Relative error (log10)')
title('Relative error vs. iteration count')