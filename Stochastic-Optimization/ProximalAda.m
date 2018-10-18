function [loss, w, g_sqr] = ProximalAda(ObjFunc, x, y, w, g_sqr, opts)
%ProximalAda
lr = opts(1); 
if length(opts) == 1
    epsilon = 1e-10;
else
    epsilon = opts(2);
end
lambda = opts(3);

[loss, grad] = ObjFunc(x, y, w, lambda);

g_sqr = g_sqr + grad.^2 ;

w = Soft_thresholding(w - lr * grad ./ (sqrt(g_sqr) + epsilon), lambda*lr./ (sqrt(g_sqr))) ;

end