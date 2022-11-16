function [out] = randgamma(m, r, sz);
% Usage: [out] = randgamma(m, r, sz);
% Makes a matrix of size sz of samples from the gamma distribution
%  with parameters m and r (which must be scalars);

% Reference:
%  Dagpunar, John, "Principles of Random Variate Generation", Clarenden Press, Oxford, 1988.
% Don't hope to understand it without reading the reference!

% Change Log:
%
%     1.1          27:apr:98    rfs      First version submitted to SCCS, not
%                                        yet all written.
%     1.2          27:apr:98    rfs      Appears to be working.
%     1.3          28:apr:98    rfs      Allows sz to be omitted if [1 1] is intended.
%     1.4          21:jun:99    rfs      Bug with small m fixed.
%     1.5          12:nov:99    rfs      Sped up version.
%
%    @(#)randgamma.m	1.5  99/11/12

if nargin < 3,
   sz = [1 1];
end

if m == 1, % we are dealing with negative exponential distribution

   out = rand(sz);
   out = -log(out);

elseif m < 1,

   t = 1 - m;
   p = t / (t + m * exp(-t));
   c = 1 / m;
   d = p * (eps / t) .^ m;

   out = repmat(d, sz);
   W = out;
   R1 = out;
   R2 = out;
   indactive = find(ones(sz));

   while ~isempty(indactive),   

      sznew = size(indactive);
      R1(indactive) = rand(sznew);

      % used to have:      indlep = intersect(indactive, find(R1 <= p));
      indlep = find(R1(indactive) <= p);
      indlep = indactive(indlep);

      out(indlep) = t * (R1(indlep) / p) .^ c;
      W(indlep) = out(indlep);

      % used to have:      indgtp = intersect(indactive, find(R1 > p));
      indgtp = find(R1(indactive) > p);
      indgtp = indactive(indgtp);

      out(indgtp) = t + log((1 - p) ./ (1 - R1(indgtp)));
      W(indgtp) = t * log(out(indgtp) ./ t);
      sznew = size(indactive);
      R2(indactive) = rand(sznew);

      % used to have:
      %              indactive = intersect(indactive, find(W >= 1 - R2));
      %              indactive = intersect(indactive, find(((W + 1) .* R2 >= 1) | (exp(-W) <= R2)));
      Wact = W(indactive);
      R2act = R2(indactive);
      indact = find(Wact >= 1 - R2act);
      indactive = indactive(indact);
      Wact = W(indactive);
      R2act = R2(indactive);
      indact = find(((Wact + 1) .* R2act >= 1) | (exp(-Wact) <= R2act));
      indactive = indactive(indact);

   end % while ~isempty(indactive)

else % m > 1

   b = m - 1;
   c = 3 * m - 0.75;

   out = zeros(sz);
   R1 = out;
   R2 = out;
   f = out;
   g = out;
   d = ones(sz);
   indactive = find(ones(sz));

   while ~isempty(indactive),

      sznew = size(indactive);
      R1(indactive) = rand(sznew);
      g(indactive) = R1(indactive) - R1(indactive) .^ 2;
      f(indactive) = (R1(indactive) - 0.5) .* sqrt(c ./ g(indactive));
      out(indactive) = b + f(indactive);

      indgt0 = find(out(indactive) > 0);
      indgt0 = indactive(indgt0);
      indle0 = find(out(indactive) <= 0);
      indle0 = indactive(indle0);

      sznew = size(indgt0);
      R2(indgt0) = rand(sznew);
      d(indgt0) = 64 * R2(indgt0) .^ 2 .* g(indgt0) .^ 3;
      inds1 = find((1 - d(indgt0)) .* out(indgt0) <= 2 * f(indgt0) .^ 2);
      inds1 = indgt0(inds1);
      indactive = intersect(indactive, union(indle0, inds1));
      inds2 = find(exp((0.5 * log(d(indgt0)) + f(indgt0)) / b) >= out(indgt0) / b);
      inds2 = indgt0(inds2);
      indactive = intersect(indactive, union(indle0, inds2));

   end % while ~isempty(indactive)

end % if m <, ==, or > 1

out = out ./ r;


% Local Variables: 
% indent-line-function: indent-relative
% eval: (auto-fill-mode 0)
% End:
