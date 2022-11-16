function [d] = diagfrom(A);
% Usage: [d] = diagfrom(A);
% Delivers the main diagonal of the 2-D matrix A, even if it is 1xN !

S = size(A);
N = min(S(1:2));
d = diag(A(1:N, 1:N));




% Local Variables: 
% indent-line-function: indent-relative
% eval: (auto-fill-mode 0)
% End:


