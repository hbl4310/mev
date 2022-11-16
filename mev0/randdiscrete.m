function [out] = randdiscrete(sz, probs, nowarning);
% Usage: [out] = randdiscrete(sz, probs, nowarning);
% Draws random samples from a finite discrete distribution.
% The output array out has size sz.
% The probability of each outcome is given by the array probs,
%  which need not sum to 1 over the appropriate vectors as it
%  will automatically be normalised before use. The possible 
%  outcomes are 1, 2, ..., M, and probs(m, K) gives the (unnormalised)
%  probability of out(K) having value m. If all elements of 
%  such a vector are zero then the output distribution is uniform;
%  under such circumstances a warning is given unless nowarning
%  is passed and non-zero.
% probs should either be of size [M, 1] or of size [M, sz].

% Change Log:
%
%     1.1          08:may:01    rfs  First version.
%     1.2          08:may:01    rfs  Working.
%     1.3          09:may:01    rfs  Bug when sz is [1, 1] fixed.
%     1.4          22:may:01    rfs  Zero probability vector now gives uniform distribution
%                                    and optional warning.
%     1.5          22:may:01    rfs  Bugs fixed.
%     1.6          25:may:01    rfs  Now gives an error if M is zero.
%
%  /home/rfs/matlab/SCCS/s.randdiscrete.m 1.6 01/05/25 12:54:21


if nargin < 2,
   error('Not enough arguments');
end
if nargin < 3,
   nowarning = [];
end

if isempty(nowarning),
   nowarning = 0;
end

M = size(probs, 1);
N = prod(sz);

if M == 0,
   error('randdiscrete cannot take samples from the empty set');
end

ind = find(sz ~= 1);
if isempty(ind),
   sz = [1];
else
   sz = sz(1 : ind(end));
end

if prod(size(probs)) == M,
   probs = repmat(probs(:), [1, sz]);
end

if length(size(probs)) ~= length([M, sz]),
   error('probs is neither a simple vector nor of the same dimensionality as sz');
end
if any(size(probs) ~= [M, sz]),
   error('probs is neither a simple vector nor of size [M, sz] for some M');
end

probs = reshape(probs, [M, N]);

% Normalise.
if any(probs(:) < 0),
   error('randdiscrete called with negative probabilities');
end
probsum = sum(probs, 1);
ind = find(probsum(:) == 0);
if ~isempty(ind),
   if ~nowarning,
      warning('randdiscrete called with zero vector of probabilities');
   end
   probsum(ind) = 1;
end
probs = probs ./ repmat(probsum, [M, 1]);
probs(:, ind) = 1 / M;

probs = cumsum(probs, 1);

out = max(1, min(M, 1 + sum(repmat(rand(1, N), [M, 1]) > probs, 1)));

out = reshape(out, [sz, 1]);

return;



% Local Variables: 
% indent-line-function: indent-relative
% eval: (auto-fill-mode 0)
% End:


