function [out] = randbeta(alpha, beta, sz);
% Usage: [out] = randbeta(alpha, beta, sz);
% Makes a matrix of size sz of samples from the Beta distribution
%  with parameters alpha and beta (which must be scalars);

% Reference:
%  Dagpunar, John, "Principles of Random Variate Generation", Clarenden Press, Oxford, 1988.

% The method is simple: we use randgamma to give us two Gamma variates X1, X2, one with parameters 
%  (alpha, 1) and the other (beta, 1). We then deliver X1 / (X1 + X2).

% Change Log:
%
%     1.1          30:sep:98    rfs      First version submitted to SCCS, not
%                                        yet all written.
%     1.2          30:sep:98    rfs      First complete version.
%
%  @(#)randbeta.m	1.2  09/30/98

if nargin < 3,
   sz = [1 1];
end

X1 = randgamma(alpha, 1, sz);
X2 = randgamma(beta, 1, sz);

out = X1 ./ (X1 + X2);


% Local Variables: 
% indent-line-function: indent-relative
% eval: (auto-fill-mode 0)
% End:
