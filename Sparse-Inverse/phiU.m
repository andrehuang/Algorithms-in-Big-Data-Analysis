function output = phiU(S,U)
A=inv(S+U);
[V,D]=eig(A);
Lambda=max(D,0);
X=V*Lambda*V';
output=-log(det(X))+trace((S+U)*X);
end

