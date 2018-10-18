% min ||x||_1, s.t. Ax=b
clear all;

% generate data
rng(1)
n = 1024;
m = 512;

A = randn(m,n);
u = sprandn(n,1,0.1);
b = A*u;


x0 = rand(n,1);

errfun = @(x1, x2) norm(x1-x2)/(1+norm(x1));
resfun = @(x) norm(A*x-b);
nrm1fun = @(x) norm(x,1);



subplot(4,1,1); plot(1:n, u);
title('exact solu');

opts2 = []; 
tic; 
[x2, out2] = l1_cvx_mosek(x0, A, b, opts2);
t2 = toc;
subplot(4,1,2); plot(1:n, x2);
title('cvx-mosek solu');

opts4 = []; %modify options
tic; 
[x4, out4] = l1_Augmented_Lagrangian(x0, A, b, [25, 1e-4, 20, 50, 0]);  % parameters: mu, tao, max_iter1, max_iter2,ifFista
t4 = toc;
subplot(4,1,3); plot(1:n, x4);
title('Augmented Lagrangian solu');

opts5 = []; %modify options
tic; 
[x5, out5] = l1_Augmented_Lagrangian(x0, A, b, [25, 1e-4, 20, 17, 1]);  % parameters: mu, tao, max_iter1, max_iter2,ifFista
t5 = toc;
subplot(4,1,4); plot(1:n, x5);
title('Augmented Lagrangian with FISTA solu');

fprintf('cvx_mosek: nrm1: %3.2e, res: %3.2e, cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', ...
        nrm1fun(x2), resfun(x2), t2, errfun(x2, x2));
fprintf('Augmented Lagrangian: nrm1: %3.2e, res: %3.2e, cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', ...
        nrm1fun(x4), resfun(x4), t4, errfun(x2, x4));
fprintf('Augmented Lagrangian with FISTA: nrm1: %3.2e, res: %3.2e, cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', ...
        nrm1fun(x5), resfun(x5), t5, errfun(x2, x5));