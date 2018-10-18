function X1 = model1(n)
X1 = zeros([n,n]);
for i = 1:n
    for j = 1:n
        X1(i,j)= 0.6^(abs(i-j));
    end
end
end