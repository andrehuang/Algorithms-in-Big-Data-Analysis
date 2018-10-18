m = 2048;
n = 512;
p = 20;
A = randn(m,p)*randn(p,n); % m * n
d = svd(A);

[error1,s1, U1] = prototype(A, 5);
[error2,s2, U2] = prototype(A, 10);
[error3,s3, U3] = prototype(A, 15);
[error4,s4, U4] = prototype(A, 20);

figure
plot(1:5, d(1:5), '+', 1:5, s1, 'o', 'LineWidth', 1.2)
legend('svd', 'r=5')
figure
plot(1:10, d(1:10), '+', 1:10, s2,'o', 'LineWidth', 1.2)
legend('svd', 'r=10')
figure
plot(1:15, d(1:15), '+',1:15, s3,'o', 'LineWidth', 1.2)
legend('svd', 'r=15')
figure
plot(1:20, d(1:20,1), '+',1:20, s4,'o', 'LineWidth', 1.2)
legend('svd', 'r=20')
% plot(1:20, d(1:20))
% figure
% imagesc(U1)
% figure
% imagesc(U2)
% figure
% imagesc(U3)
% figure
% imagesc(U4)
