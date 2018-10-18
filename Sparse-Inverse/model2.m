function X2 = model2(n, P)
% INPUT: n is matrix size
%        P is condition number of output matrix, can be 30, 60, 90, 120
p = 0.9;
B = rand(n);
B = 0.5 * (B>p);
B = B - diag(diag(B));
L = triu(B);
B = L + L';
f = @(x)cond(B+x*eye(n))-P;
delta = fsolve(f, 1);
X2 = B + delta*eye(n);
X2 = 1/delta * X2;
end
