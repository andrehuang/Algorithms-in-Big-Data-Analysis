function res = Soft_thresholding(y, v)
res = sign(y) .* max(abs(y) - v, 0);
end