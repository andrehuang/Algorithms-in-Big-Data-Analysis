function X = phase_cvx(A, b)
% Input arguments:
%       A: the measurement matrix, size m*n
%       b: the measurements, size m*1
%
% Output:
%       X: the solution to the sdp problem of phase lift
[~, n] = size(A);
cvx_begin sdp
    cvx_solver mosek
    variable X(n,n) hermitian
    
    minimize  trace(X)

    subject to
        X >= 0;
        diag(A*X*A')== b;
cvx_end
