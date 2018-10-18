% cvx callinxg mosek
function [X1, out] = sparse_inverse(S, rho)

% OUTPUT: X1: the solution
%         out: the optimality conditions (need to derive the dual problem)
% s = size(X0);
n = length(S);
% cvx_solver mosek
cvx_begin 
    variable X(n,n) semidefinite
    maximize log_det(X) - trace(S*X) - rho* norm(vec(X), 1)
cvx_end
X1 = X;
out.dugap = n-trace(S*X)  - rho*norm(vec(X), 1);
end