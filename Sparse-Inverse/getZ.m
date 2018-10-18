function Z = getZ(S)
[V,D]=eig(S);
n=size(D,1);
invD = D\eye(n);
Lambda=max(invD,0);
Z=V*Lambda*V';
end

