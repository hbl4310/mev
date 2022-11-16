function [y] = Esum(x, n);
% y = -log(sum(exp(-x), n))

shift = min(x, [], n);
ind = find(isinf(shift) | isnan(shift));
shift(ind) = 0;

sz = size(x);
repper = ones(size(sz));
repper(n) = sz(n);

x = x - repmat(shift, repper);

y = -lognw(sum(exp(-x), n));

y = y + shift;

return;


% Local Variables: 
% indent-line-function: indent-relative
% eval: (auto-fill-mode 0)
% End:

