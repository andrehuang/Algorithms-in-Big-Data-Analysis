function [z, errs] = phase_WF(A, b, x, opts)
% Input arguments:
%       A: the measurement matrix, size m*n
%       b: the measurements, size m*1
%       x: the true signal, size m*1, used to calculate the relative error
%       opts:
%           (1) max_iter: the maximum iteration times in the loop
%           (2) tau0: the parameter to control the schedule for step size
%
% Output:
%       z: the recovered signal using Wirtinger Flow
%       Relerrs: relative errors in each iteration
%
[m, n] = size(A);
max_iter = opts(1);
tau0 = opts(2);

% Initialization stage
npower_iter = 50; % Number of power iterations 
z0 = randn(n,1); z0 = z0/norm(z0,'fro');    
for tt = 1:npower_iter                      
    z0 = A'*(b.* (A*z0)); z0 = z0/norm(z0,'fro');
end

normest = sqrt(sum(b)/numel(b));    % Estimate norm to scale eigenvector  
z = normest * z0;                   % Apply scaling 
errs = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro'); % Initial rel. error

% Gradient Descent Stage
mu = @(t) min(1-exp(-t/tau0), 0.2); % Schedule for step size
for t = 1:max_iter
    yz = A*z;
    grad  = 1/m* A'*( ( abs(yz).^2-b ) .* yz ); % Wirtinger gradient
    z = z - mu(t)/normest^2 * grad;             % Gradient update 
    errs = [errs, norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro')];  
end
 

    
  
