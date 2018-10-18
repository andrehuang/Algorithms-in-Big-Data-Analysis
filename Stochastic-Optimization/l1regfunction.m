function [loss, grad] = l1regfunction(x, y, w, lambda)
% x [784, batchSize]
% y [1, batchSize]
% w [784, 1]
y_hat = w'*x; % [1, batchSize]

f = log(1+exp(-y .* y_hat));  % [1 batchSize]
% f = sum(f, 1);
loss = mean(f) + lambda*norm(w, 1);
grad = (exp(-y .* y_hat) .* (-y .* x)) ./ (1+ exp(-y .* y_hat));
grad = mean(grad, 2) + lambda*sign(w);
end
