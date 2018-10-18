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



subplot(7,1,1); plot(1:n, u);
title('exact solu');

opts1 = []; 
tic; 
[x1, out1] = l1_cvx(x0, A, b, opts1);
t1 = toc;
subplot(7,1,2); plot(1:n, x1);
title('cvx solu');

opts2 = []; 
tic; 
[x2, out2] = l1_cvx_mosek(x0, A, b, opts2);
t2 = toc;
subplot(7,1,3); plot(1:n, x2);
title('cvx-mosek solu');

tic; 
[x3, out3] = l1_mosek(x0, A, b, opts1);
t3 = toc;
subplot(7,1,4); plot(1:n, x3);
title('mosek solu');


opts4 = []; %modify options
tic; 
[x4, out4] = l1_Augmented_Lagrangian(x0, A, b, [25, 1e-4, 50, 25, 0]);  % parameters: mu, tao, max_iter1, max_iter2,ifFista
t4 = toc;
subplot(7,1,5); plot(1:n, x4);
title('Augmented Lagrangian solu');

opts5 = []; %modify options
tic; 
[x5, out5] = l1_Augmented_Lagrangian(x0, A, b, [25, 1e-4, 50, 10, 1]);  % parameters: mu, tao, max_iter1, max_iter2,ifFista
t5 = toc;
subplot(7,1,6); plot(1:n, x5);
title('Augmented Lagrangian with FISTA solu');

opts6 = []; %modify options
tic; 
[x6, out6] = l1_ADMM(x0, A, b, [25 , 100]);  % parameters: mu, max_iter
t6 = toc;
subplot(7,1,7); plot(1:n, x6);
title('l1 ADMM solu');


fprintf('cvx:        nrm1: %3.2e, res: %3.2e, cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', nrm1fun(x1), resfun(x1), t1, errfun(x2, x1));
fprintf('cvx_mosek: nrm1: %3.2e, res: %3.2e, cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', ...
        nrm1fun(x2), resfun(x2), t2, errfun(x2, x2));
fprintf('mosek: nrm1: %3.2e, res: %3.2e, cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', ...
        nrm1fun(x3), resfun(x3), t3, errfun(x2, x3));   
fprintf('Augmented Lagrangian: nrm1: %3.2e, res: %3.2e, cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', ...
        nrm1fun(x4), resfun(x4), t4, errfun(x2, x4));
fprintf('Augmented Lagrangian with FISTA: nrm1: %3.2e, res: %3.2e, cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', ...
        nrm1fun(x5), resfun(x5), t5, errfun(x2, x5));
fprintf('ADMM: nrm1: %3.2e, res: %3.2e, cpu: %5.2f, err-to-cvx-mosek: %3.2e\n', ...
        nrm1fun(x6), resfun(x6), t6, errfun(x2, x6));