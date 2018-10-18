clear all; 
n = 30;
rho = 0.1; %10, 0.1, 0.001
P = 30; %60, 90 ,120, 200
nesterov = 0;  % using first-order method or not
model = 1; % Choose one model to generate S

if model == 1
    S = model1(n);
else
    S = model2(n, P);
end
% original_inv = inv(S);

if nesterov
    tic;
    [X1, out] = Nesterov(S, 0.005, rho, 150, 0.01); % S, mu, rho, L, sigma1
% Recommended Parameters
% rho = 0.001, epsilon=0.001, L=30 (For Model1 only; Model2 doesn't
% converge, has not feasible point)
% rho = 0.1, epsilon=0.005, L=150
% rho = 10, epsilon = 0.0001, L=10000, dualgap=0.0787 for Model1;
% dualgap=0.535 for Model2
    iter_times = out.iter
    DualGap = out.dualgap
    t = toc
else
    tic;
    [X1, out] = sparse_inverse(S, rho);
    t = toc
end

dual_gap = out.dugap

% imagesc(original_inv)
imagesc(X1)



