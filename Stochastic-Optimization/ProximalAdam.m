function [loss, w, state] = ProximalAdam(ObjFunc, x, y, w, state, opts)
%ADAM

lr = opts(1); 
epsilon = opts(2);
beta1 = opts(3);
beta2 = opts(4);
lambda = opts(5);
[loss, grad] = ObjFunc(x, y, w, lambda);


if isequal(state, 0) % start off with state = 0 so as to get default state
  state = struct('m', 0, 'v', 0, 't', 0);
end

% update first moment vector `m`
state.m = beta1 * state.m + (1 - beta1) * grad ;

% update second moment vector `v`
state.v = beta2 * state.v + (1 - beta2) * grad.^2 ;

% update the time step
state.t = state.t + 1 ;

% This implicitly corrects for biased estimates of first and second moment
% vectors
lr_t = lr * (((1 - beta2^state.t)^0.5) / (1 - beta1^state.t)) ;

% Update `w`
w = Soft_thresholding(w - lr_t * state.m ./ (state.v.^0.5 + epsilon), lambda*lr_t  ./ (state.v.^0.5 + epsilon)) ;
