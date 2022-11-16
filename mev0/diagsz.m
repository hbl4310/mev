function [A] = diagsz(d, sz);
% Usage: [A] = diagsz(d, sz);
% Constructs A as a 2-d matrix of size sz with the (initial) elements of
%  the vector d down the main diagonal.

A = diag(d);
N = length(d);
x = sz(1);
y = sz(2);

if x > N,
   A = [A; zeros(x - N, N)];
end
if y > N,
   A = [A, zeros(x, y - N)];
end

A = A(1:x, 1:y);



% Local Variables: 
% indent-line-function: indent-relative
% eval: (auto-fill-mode 0)
% End:


