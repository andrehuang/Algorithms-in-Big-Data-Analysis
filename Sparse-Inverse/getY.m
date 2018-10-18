function Y = getY(X,grad,L)
G=X - 1/L * grad;
[V, D]=eig(G); % G = VDV^-1 = VDV' 
Lambda=max(D,0);  % also a diagonal matrix
Y=V*Lambda*V';
end

