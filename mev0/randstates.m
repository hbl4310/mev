function [oldstates, newstates] = randstates(newstates);
% Usage: [oldstates, newstates] = randstates(newstates);
% This function is a shortcut to setting all the random generator
%   states in one statement.
% If newstates is a double value, then the new states are generated
%   by using newstates as a seed, then warming up the generators.
% If newstates is a value previously passed out by this function,
%   (in which case it will be a cell array),
%   then the generators are returned to exactly the same states
%   they were in before that previous call.
% In *both* cases, the value of oldstates passed out is the 
%   states of the generators before this call, and that of newstates
%   is that of the generators after this call (and will therefore
%   be a cell array).


if nargin < 1,
   newstates = [];
end

if nargout >= 1,
   randstate = rand('state');
   randnstate = randn('state');
   oldstates = {randstate, randnstate};
end

if ~isempty(newstates),
   if ~iscell(newstates),
      rand('state', newstates);
      randn('state', newstates);
      rand(16, 16);
      randn(16, 16);
   else
      rand('state', newstates{1});
      randn('state', newstates{2});
   end
end

if nargout >= 2,
   randstate = rand('state');
   randnstate = randn('state');
   newstates = {randstate, randnstate};
end





% Local Variables: 
% indent-line-function: indent-relative
% eval: (auto-fill-mode 0)
% End:


