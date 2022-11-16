function [y] = Eadd(x1, x2);
% y = -log(exp(-x1) + exp(-x2))

sz1 = size(x1);
sz2 = size(x2);

if length(sz1) ~= length(sz2),
   error('Eadd called with arrays of different dimensionality');
end

if any(sz1 ~= sz2),
   error('Eadd called with unequal sizes');
end

y = zeros(sz1);

ind1 = find(x1 < x2);
y(ind1) = x1(ind1) - log(1 + exp(x1(ind1) - x2(ind1)));
ind2 = find(x1 >= x2);
y(ind2) = x2(ind2) - log(1 + exp(x2(ind2) - x1(ind2)));
ind3 = find(x1 == x2 & isinf(x1));
y(ind3) = x1(ind3);

return;


