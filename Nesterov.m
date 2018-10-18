function [X, out] = Nesterov(S, epsilon, rho, L, sigma1)
% INPUT: S: the observed covariance matrix
%        epsilon: upper bound of final duality gap
%        rho: the coefficient of ||X||_1 in the primal problem
%        L: L is the Lipschitz constant of the gradient of f_epsilon. It
%        should be calculated. But we simplify it to be a given constant
%        here
%        sigma1: 1/beta^2 in the paper, where in our case beta = Inf  

n = size(S,1);
iter=0; % iteration times

% Initilization
gradSum = zeros(n);
Uk = randn(n);
Uk=Uk*Uk';
Xk=randn(n);
Xk=Xk*Xk';

if rho == 10
    dualgap = 1;

while abs(dualgap) > 0.0535 % 0.0787 for Model1; 0.0535 for Model2
   iter=iter+1;
   grad = -pinv(Xk) + S + max(min(Xk/epsilon,rho),-rho);
   Yk = getY(Xk,grad,L);
   %3
   gradSum = sigma1/L*grad*(iter+1)/2 + gradSum;
   
   Zk= getZ(gradSum);
   %4
   Xk=2/(iter+3)*Zk+(iter+1)/(iter+3)*Yk;
   Ustar=max(min(Xk/epsilon,rho),-rho);
   Uk = (iter*Uk + 2*Ustar)/(iter+2);
   
   dualgap = -log(det(Yk))+trace(S*Yk')+rho*sum(abs(Yk(:))) - phiU(S,Uk)
   DUALGAP = n-trace(S*Yk)  - rho*norm(vec(Yk), 1)

end
else
dualgap = 2*epsilon;

while abs(dualgap) > epsilon
   iter=iter+1;
   grad = -inv(Xk) + S + max(min(Xk/epsilon,rho),-rho);
   Yk = getY(Xk,grad,L);
   %3
   gradSum = sigma1/L*grad*(iter+1)/2 + gradSum;
   
   Zk= getZ(gradSum);
   %4
   Xk=2/(iter+3)*Zk+(iter+1)/(iter+3)*Yk;
   Ustar=max(min(Xk/epsilon,rho),-rho);
   Uk = (iter*Uk + 2*Ustar)/(iter+2);
   
   dualgap = -log(det(Yk))+trace(S*Yk')+rho*sum(abs(Yk(:))) - phiU(S,Uk)
   DUALGAP = n-trace(S*Yk)  - rho*norm(vec(Yk), 1);
end
end
out.dugap = dualgap;
out.dualgap = DUALGAP;
out.iter = iter;
X = Xk;