function [outorder, marginfracs, norder, entropy, orderprobs, orders, orderpos] = sim1(marginstableorcreateseed, analysisseed, Nc, Nv, Nvg);
% Usage: [outorder, marginfracs, norder, entropy, orderprobs, orders, orderpos] ...
%           = sim1(marginstableorcreateseed, analysisseed, Nc, Nv, Nvg);
% 
% Function to simulate the maximum entropy election system.
% If marginstableorcreateseed is 1x1 or empty then a set of votes 
%  is created and its analysis simulated; if 1x1 then that value is 
%  used to seed the random number generators used for creation;
%  in this case Nc and Nv are respectively the number
%  of candidates and the number of voters, taken to be 4 and 50
%  if not given, and Nvg is the number of different types of voter,
%  taken to be 3 if not given.
% If marginstableorcreateseed is N x N then Nc is taken to be N and 
%  the passed values of Nc and Nv are ignored; the matrix is 
%  taken to be the marginstable C, such that C(nc1, nc2) is the 
%  number of voters that thought candidate nc1 was better
%  than or equal to candidate nc2.
% If analysisseed is non-empty then it is used to seed the random 
%  number generator used for choosing the output order.
% outorder is a Nc x 1 vector which gives the output ordering
%  of the candidates starting with the most preferred.
% marginfracs is Nc x Nc and marginfracs(nc1, nc2) gives the fraction
%  of voters who preferred nc1 to nc2.
% norder is the index of the output ordering in each of the following
%  items.
% orderprobs is the table of probabilities on each possible order, 
%  sorted into descending order.
% orders(np, norder) is the candidate in position np in order norder.
% orderpos(nc, norder) is the position in order norder of candidate nc.

% positiontable is an Nc x Nc matrix A such that A(nc, np) gives
%  the probability that outorder(np) == nc.



% Deal with the arguments.
if nargin < 1,
   error('Too few arguments');
end
if nargin < 2,
   analysisseed = [];
end
if nargin < 3,
   Nc = [];
end
if nargin < 4,
   Nv = [];
end
if nargin < 5,
   Nvg = [];
end

if isempty(Nc),
   Nc = 4;
end
if isempty(Nv),
   Nv = 50;
end
if isempty(Nvg),
   Nvg = 3;
end

% Global variables to do with the random generator.
global laststartstate lastanalysisstate

if prod(size(marginstableorcreateseed)) <= 1,
   createseed = marginstableorcreateseed;
   synthesising = 1;
else
   C = marginstableorcreateseed;
   synthesising = 0;
end

if ~synthesising,
   Nc = size(C, 1);
end

% Set various parameters for the analysis.

% Overall number of iterations.
Niters = 10000;

% Number of iterations for which it must be stopped before ceasing to iterate.
Nstopped = 10;

% Value of fullmovement which counts as stopped.
stoppedfullmovement = 1e-11;

% Setting this to 1 means the SVD is used when solving overdetermined equations;
% setting it to 0 means that matrix left division (\) is used instead.
usesvd1 = 1; % for the determination of lambda and mu
usesvd2 = 1; % for the determination of A
svdtol = 1e-8; % must be set if either usesvd is 1

% The minimum positive entry permitted in positiontable.
global wall
wall = 1e-300;

% The threshold A value at the end of stage 1 that will be assumed to 
% imply that that value can only be zero due solely to the constraints.
killthreshold = 1e-10;

% Whether to use random weights on the equations for A.
% Weights are drawn from Beta(weightsalpha, weightsbeta).
% If userandomAweights is 1, then a single weight is used for the gradient
% matching component and a second weight for the linear constraint component.
% If userandomAweights is 2, then different weights are used for each
% separate equation.
userandomAweights = 1;
weightsalpha = 12;
weightsbeta = 0.5;
gradweightsalpha = 3;
gradweightsbeta = 1;

% Factor of gradient to make the nudge.
nudgefactor = 0; % i.e. no nudge for now.

% Do the synthesising if necessary
if synthesising,
   
   % Set up the random generator.
   if ~isempty(createseed),
      [oldstate, laststartstate] = randstates(createseed);
   else
      [laststartstate] = randstates;
   end

   % First we must determine the score distributions for each candidate and each voter group
   scoremeans = rand(Nc, Nvg);
   scorestrengths = randgamma(2, 0.1, [Nc, Nvg]);

   % Then we assign a voter group to each voter,
   crits = repmat(1 ./ Nvg, [Nvg, 1]);
   crits = cumsum(crits, 1);
   nvgs = min(Nvg, 1 + sum(repmat(rand(1, Nv), [Nvg, 1]) > repmat(crits, [1, Nv])));

   % and propagate the means and strengths to each voter.
   scoremeans = scoremeans(:, nvgs);
   scorestrengths = scorestrengths(:, nvgs);

   % Then we assign a score to each voter for each candidate
   for nc = 1 : Nc,
      for nv = 1 : Nv,
         scores(nc, nv) = randbeta(scoremeans(nc, nv) .* scorestrengths(nc, nv), ...
                                   (1 - scoremeans(nc, nv)) .* scorestrengths(nc, nv));
      end
   end

   % Then we work out the margins table
   C = zeros(Nc, Nc);
   for nc1 = 1 : Nc,
      for nc2 = 1 : Nc,
         C(nc1, nc2) = sum(scores(nc1, :) > scores(nc2, :), 2) ...
                       + 0.5 * sum(scores(nc1, :) == scores(nc2, :));
      end
   end

   % Restore the random generator if necessary.
   if ~isempty(createseed),
      randstates(oldstate);
   end

end % if synthesising

% Check that C has the desired properties
if length(size(C)) ~= 2,
   error('marginstable is not 2-dimensional');
end
if size(C, 1) ~= Nc | size(C, 2) ~= Nc,
   error('marginstable is not Nc x Nc');
end
Ccheck = C + C.';
Nv = mean(mean(Ccheck));
Ccheck = Ccheck / Nv;
if any(any(Ccheck < 0.9999 | Ccheck > 1.0001)),
   error('marginstable is not valid');
end

C = C / Nv;
marginfracs = C;

% Start the analysis.

% Set up the random generator for analysis.
if ~isempty(analysisseed),
   [oldstate, lastanalysisstate] = randstates(analysisseed);
end

% Apply a random permutation to the candidates to ensure fairness
perm = randperm(Nc);
iperm = zeros(1, Nc);
iperm(perm) = [1 : Nc]; % make the inverse permutation ready for use later
C = C(perm, perm);

% Make all the possible orderings, recursively.
fprintf('Making orderings...\n');
for nc = 1 : Nc,
   if nc == 1,
      orders = [1];
   else
      szsofar = size(orders, 2);
      neworders = zeros(nc, nc * szsofar);
      sofar = 0;
      for ncfirst = 1 : nc,
         if nc > 5,
            fprintf('\rnc %d, ncfirst %d         ', nc, ncfirst);
         end
         neworders(1, sofar + [1 : szsofar]) = ncfirst;
         rest = [[1 : ncfirst - 1], [ncfirst + 1 : nc]];
         neworders(2 : end, sofar + [1 : szsofar]) = rest(orders);
         sofar = sofar + szsofar;
      end
      orders = neworders;
   end
end
fprintf('\r...done                        \n');   
Norders = size(orders, 2);
clear neworders szsofar rest sofar

% Make the inverse table, that translates candidate number into position in ordering.
orderpos = zeros(size(orders));
fprintf('Making inverse order table...\n');
for norder = 1 : Norders,
   if rem(norder, 100) == 0,
      fprintf('\r%d out of %d       ', norder, Norders);
   end
   orderpos(orders(:, norder), norder) = [1 : Nc].';
end
fprintf('\r...done.                       \n');

% Make the matrix B such that if A is a column vector giving the 
% probabilities on the orderings, then reshape(B * A, [Nc, Nc]) is
% a corresponding matrix of the form and meaning of marginfracs.

straight = triu(ones(Nc, Nc), 1) + 0.5 * eye(Nc, Nc);
B = zeros(Nc * Nc, Norders);
fprintf('Making B...\n');
for norder = 1 : Norders,
   if rem(norder, 100) == 0,
      fprintf('\r%d out of %d      ', norder, Norders);
   end
   B(:, norder) = reshape(straight(orderpos(:, norder), orderpos(:, norder)), [Nc * Nc, 1]);
end % for norder = 1 : Norders
clear straight
fprintf('\r...done.                       \n');

% Initialise A (the developing positiontable) and mu
% A = repmat(1 ./ Norders, [Norders, 1]);
A = rand(Norders, 1);
A = A ./ sum(A);

% Check that A meets the wall criterion.
ind = find(A < wall);
A(ind) = wall;

% Reshape C to fit the algorithm.
C = reshape(C, [Nc * Nc, 1]);

% Initialise onepluslogA.
onepluslogA = 1 + lognw(A);

% The analysis proceeds as follows.
% The aim is to satisfy simultaneously the following equations:
%
% 0) all(A >= 0)
% 1) B * A = C
% 2) sum(A) = 1
% 3) dG / dA(n) = 0 for all n where
%             G = - sum(A .* log(A)) ...
%                 + sum(lambda.' .* (B * A - C))
%                 + mu .* (sum(A) - 1)
%
% Old solution proceeds by alternating two operations O1 and O3 below, of which only
% the first happens until equations 1 and 2 are reasonably well satisfied.
% In addition, before first using operation C, we apply operation O2 alternating
% with O1 a few times, and kill from further consideration any indices which despite
% that have A components below killthreshold.
%
% New solution proceeds by alternating O1 and O3 from the start, but with
% O1 modified, so that if we ever get to the wall, we retreat to half way 
% to where we saw the wall.
%
% O1) Find the nearest point to current guess satisfying 0, 1, and 2;
%     proceed towards it using the gradient of the distance from the 
%     hyperplane of solutions (to 0, 1, and 2) to do binary chop; where
%     A threatens to become negative we limit it at wall, and also
%     limit the corresponding elements of the gradient.
%
% O2) Proceed as in O3 below for choosing the direction to move in, but instead
%     of choosing how far to go by when the entropy is maximised, choose
%     how far to go by when sum(log(A)) is maximised.
%
% O3) Calculate the lambda and mu which make the gradient of G smallest; these
%     also make the gradient of G be in the constraint plane defined by 
%     1 and 2. If that constraint plane requires some A to be zero, then 
%     the gradient will theoretically point along that plane; if we have some
%     component of A at zero and the constraint does not require it, that
%     component of the gradient will be positive; therefore if we have 
%     some component of A at wall, and the corresponding gradient component
%     negative, we set the latter to zero.
%
% In both cases the minimum norm solution or minimum norm residual solution
% is found either by Matlab's
% matrix division operators / or \ (otherwise known as mrdivide or mldivide), or
% by using the SVD, according to the setting of usesvd1 and usesvd2. In both
% cases SVD is preferred, indeed potentially the only option.

Nrecords = Niters;
nrecord = 1;
logmaxentnotconstraints = repmat(NaN, [1, Nrecords]);
logepsilon = repmat(NaN, [1, Nrecords]);
logfullmovement = repmat(NaN, [1, Nrecords]);
logmovement = repmat(NaN, [1, Nrecords]);
lognormgradient = repmat(NaN, [1, Nrecords]);
lognormmismatch = repmat(NaN, [1, Nrecords]);
loglowderivative = repmat(NaN, [1, Nrecords]);
logentropy = repmat(NaN, [1, Nrecords]);
logsum = repmat(NaN, [1, Nrecords]);
logCmismatch = repmat(NaN, [Nc, Nc, Nrecords]);
logCmismatchfrac = repmat(NaN, [Nc, Nc, Nrecords]);
loggradient = repmat(NaN, [Norders, Nrecords]);
logsumlogA = repmat(NaN, [1, Nrecords]);

maxCmismatch = 1;
maxgradienterror = 1;
epsilon = 1;

% Set the stage of operation; while we are trying to meet the constraints,
% we are in stage 0. Once we have met the constraints, we move to stage 1, 
% to flee the walls. After a fixed amount of time in stage 1, we move to stage 2,
% to maximise entropy.
stage = 2; % We are now going straight to stage 2.
itersinstage = 0;
Nitersinstage1 = 100;

% Set the type of operation.
% maxentnotconstraints = 1 means we are doing operation O2 or O3 above,
% maxentnotconstraints = 0 means we are doing operation O1 above.
maxentnotconstraints = 0;
fleewalls = 0;

% Initialise the number of dead orders by killing all those with
% zero marginfracs.

indzero = find(reshape(C, [Nc, Nc]) == 0);
[nc1list, nc2list] = ind2sub([Nc, Nc], indzero);
Nlist = length(indzero);
thesetokill = zeros(Norders, 1);
for nlist = 1 : Nlist,
   nc1 = nc1list(nlist);
   nc2 = nc2list(nlist);
   thesetokill = thesetokill | (orderpos(nc1, :) < orderpos(nc2, :)).';
end

killindices = find(thesetokill);
liveindices = setdiff([1 : Norders].', killindices);
Nkilled = length(killindices)

wantedA = NaN;
        
for niter = 1 : Niters,

   if stage == 2 & itersinstage == 0,
      spI = speye(Norders, Norders);
      spI = spI(killindices, :);
      H = [full(spI); ones(1, Norders); B];

      if usesvd1 | usesvd2,
         fprintf('Doing SVD on H...\n');
         Hinv = svdinv(H, svdtol);
         fprintf('...done.\n');
      end
   end

   if stage == 2,
      A(killindices) = wall;
   end

   if stage == 1,
      maxentnotconstraints = ~maxentnotconstraints;
      fleewalls = maxentnotconstraints;
   elseif stage == 2,
      fleewalls = 0;
      if itersinstage == 0,
         maxentnotconstraints = 0;
      else
         maxentnotconstraints = normgradient > sqrt(normmismatch);
      end
   end

   if fleewalls,
      nudge = abs(randn(Norders, 1)) * nudgefactor * sqrt(sum(onepluslogA .^ 2));
   else
      nudge = sparse(Norders, 1);
   end

   wantedlambda = onepluslogA + nudge;
   if usesvd1,
      vec = reshape(wantedlambda, [1, Norders]) * Hinv;
   else
      vec = reshape(wantedlambda, [1, Norders]) / H;
   end
   mu = vec(Nkilled + 1);
   lambda = vec(Nkilled + 2 : end);

   if maxentnotconstraints, % i.e. if we are doing constrained gradient ascent on the entropy

      % Calculate the gradient of G projected onto the constraint hyperplane.
      gradient = - wantedlambda + reshape(vec * H, [Norders, 1]);

      % Apply killing if at stage 2.
      if stage == 2,
         gradient(killindices) = 0;
      end

      % Check that we don't insist on going through the wall.
      ind = find(A <= wall);
      gradient(ind) = max(gradient(ind), 0);

      fullmovement = gradient(liveindices).' * (- wantedlambda(liveindices)) / sqrt(sum(wantedlambda(liveindices) .^ 2));

      % Normalise to length 1.
      gradient = gradient / sqrt(sum(gradient .^ 2));

      % Add it to A to get a potential Anew.
      Anew = gradient + A;

   else % i.e. if trying to meet constraints

      wantedA2 = C;

      wantedA1 = 1;

      wantedA0 = zeros(Nkilled, 1);

      wantedA = [wantedA0; wantedA1; wantedA2];

      wanteddeltaA = wantedA - H * A;

      if usesvd2,
         Anew = Hinv * wanteddeltaA + A;
      else
         Anew = H \ wanteddeltaA + A;
      end

      Anew(killindices) = wall;

      fullmovement = sqrt(sum((Anew - A) .^ 2));
   
   end % if maxentnotconstraints or not
      
   % Now we have to determine whether Anew has all positive elements,
   % and if not not go all the way there.

   % In fact, we will either go for minimising the mismatch, or for
   % maximising entropy. Since both entropy and -mismatch
   % are (non-strictly) convex, we are in either case maximising a convex function,
   % and can do this by finding a zero of the derivative by binary chop.

   gap = H * (Anew - A);

   lowepsilon = 0;
   highepsilon = 1; 

   lowderivative = calcderivative(H, Anew, A, gap, lowepsilon, wantedA, maxentnotconstraints, fleewalls);

   if lowderivative < -1e-15 & stage ~= 1,
      warning('lowderivative is negative');
      % This should never happen, so backwards should never get set except in stage 1.
   end
   backwards = lowderivative < 0;

   if backwards,
      Anew = A - (Anew - A);
      gap = - gap;
   end         

   printing = 0;

   if printing,
      if maxentnotconstraints,
         if fleewalls,
            s = 'fleewalls';
         else
            s = 'maxent';
         end
      else
         s = 'constraints';
      end
      fprintf('iter %d lowderivative %g using %s\n', ...
              niter, lowderivative, s);
   end

   foundepsilon = 0;
   while ~foundepsilon,
      midepsilon = (lowepsilon + highepsilon) / 2;
      if midepsilon == lowepsilon | midepsilon == highepsilon,
         foundepsilon = 1;
      else
         if any(midepsilon * Anew(liveindices) + (1 - midepsilon) * A(liveindices) < wall),
            if printing,
               fprintf('epsilon %g leads to A < wall\n', midepsilon);
            end
            midistoohigh = 1;
            if ~maxentnotconstraints,
               midepsilon = midepsilon / 2;
            end
         else
            midderivative = calcderivative(H, Anew, A, gap, midepsilon, wantedA, maxentnotconstraints, fleewalls);
            if printing,
               fprintf('epsilon %g gives derivative %g\n', midepsilon, midderivative);
            end
            midistoohigh = midderivative < 0;
         end
   
         if midistoohigh,
            highepsilon = midepsilon;
         else
            lowepsilon = midepsilon;
         end
         if lowepsilon > highepsilon,
            lowepsilon = highepsilon;
         end
      end
   end % while ~foundepsilon
   epsilon = lowepsilon;

   if printing,
      fprintf('iter %d epsilon %d accepted\n', ...
              niter, epsilon);
   end

   if fullmovement < 1e-12,
%       fprintf('Iter %d:\n', niter);
%       warning(sprintf('fullmovement only %g', fullmovement));
   elseif epsilon < eps,
      fprintf('Iter %d:\n', niter);
      warning(sprintf('epsilon only %g', epsilon));
   end

   Aold = A;

   A = (1 - epsilon) * Aold + epsilon * Anew;

   ind = find(A < wall);
   A(ind) = wall;
   A(killindices) = wall;

   movement = sqrt(sum((A - Aold) .^ 2));

   % Update onepluslogA.
   onepluslogA = 1 + lognw(A);

   % Display progress by reporting the current entropy and the current
   % misfit to the various equations.
   
   entropy = -A .* lognw(A);
   ind = find(A == 0);
   entropy(ind) = 0;
   entropy = sum(entropy);
   
   actualsum = sum(A);
   
   actualCmismatch = B * A - C;
   actualCmismatchfrac = actualCmismatch ./ max(C, double(C == 0));
   actualCmismatch = reshape(actualCmismatch, [Nc, Nc]);
   actualCmismatchfrac = reshape(actualCmismatchfrac, [Nc, Nc]);
   
   maxCmismatch = squeeze(max(max(abs(actualCmismatch))));

   actualgradient = reshape(lambda * B, [Norders, 1]) ...
                    - onepluslogA ...
                    + mu;

   maxgradienterror = max(abs(actualgradient(liveindices)));

   normgradient = sqrt(sum(actualgradient(liveindices) .^ 2));
   normmismatch = sqrt(sum(sum(actualCmismatch .^ 2)) + (sum(A) - 1) .^ 2);
   sumlogA = sum(lognw(A(liveindices)));

   logmaxentnotconstraints(1, nrecord) = maxentnotconstraints + 2 * fleewalls;
   logepsilon(1, nrecord) = epsilon;
   logfullmovement(1, nrecord) = fullmovement;
   logmovement(1, nrecord) = movement;
   logentropy(1, nrecord) = entropy;
   loglowderivative(1, nrecord) = lowderivative;
   lognormgradient(1, nrecord) = normgradient;
   lognormmismatch(1, nrecord) = normmismatch;
   logsum(:, nrecord) = actualsum;
   logCmismatch(:, :, nrecord) = reshape(actualCmismatch, [Nc, Nc]);
   logCmismatchfrac(:, :, nrecord) = reshape(actualCmismatchfrac, [Nc, Nc]);
   loggradient(:, nrecord) = actualgradient;
   logsumlogA(:, nrecord) = sumlogA;
   nrecord = nrecord + 1;

   itersinstage = itersinstage + 1;

   if normmismatch < 1e-13 & stage == 0,
      stage = 1;
      itersinstage = 0;
   end
   if stage == 1,
      if itersinstage == Nitersinstage1,
         stage = 2;
         itersinstage = 0;
      end
   end

   if niter > Nstopped & itersinstage > Nstopped,
      indices = [nrecord - Nstopped : nrecord - 1];
      if (all(logfullmovement(indices) < stoppedfullmovement) ...
          | all(logepsilon(indices) == 0) ...
          | all(abs(loglowderivative(indices)) < 1e-14) ...
          | all(logmovement(indices) < 1e-14)),
         if stage == 2,
            stopping = 1;
         else
            stage = stage + 1;
            itersinstage = 0;
         end
      else
         stopping = 0;
      end
   else
      stopping = 0;
   end

   if niter > Niters - 10 | any(rem(niter, 50) == [0, 1]) | stopping | itersinstage == 0,
      niter
      epsilon
      entropy
      movement
      fullmovement
      mu
      A
      actualsum
      actualCmismatch
      actualCmismatchfrac
      actualgradient
      stage
      itersinstage
      Nkilled
      maxentnotconstraints
      sumlogA
      normgradient
      normmismatch
   end

   if rem(niter, 100) == 0 | itersinstage == 0,
%       keyboard;
   end

   if stopping,
      break;
   end
   
end % for niter = 1 : Niters

% Curtail the logs
nrecord = nrecord - 1;
logmaxentnotconstraints = logmaxentnotconstraints(1, 1 : nrecord);
logepsilon = logepsilon(1, 1 : nrecord);
logfullmovement = logfullmovement(1, 1 : nrecord);
logmovement = logmovement(1, 1 : nrecord);
loglowderivative = loglowderivative(1, 1 : nrecord);
lognormgradient = lognormgradient(1, 1 : nrecord);
lognormmismatch = lognormmismatch(1, 1 : nrecord);
logentropy = logentropy(1, 1 : nrecord);
logsum = logsum(:, 1 : nrecord);
logCmismatch = logCmismatch(:, :, 1 : nrecord);
logCmismatchfrac = logCmismatchfrac(:, :, 1 : nrecord);
loggradient = loggradient(:, 1 : nrecord);
logsumlogA = logsumlogA(:, 1 : nrecord);

% Set the output values.
orderprobs = A;

% Choose the output order.
norder = min(1 + sum(rand > cumsum(A)), Norders);

outorder = orders(:, norder);

% keyboard;

% Undo all the permutations on the output variables.
outorder = perm(outorder);
orders = perm(orders);
orderpos = orderpos(iperm, :);

% keyboard;

% Sort out the orderings into descending probabilities
fprintf('Sorting order probabilities...\n');
[trash, ind] = sort(-orderprobs);
clear trash
orderprobs = orderprobs(ind);
orders = orders(:, ind);
orderpos = orderpos(:, ind);
norder = find(ind == norder);
fprintf('...done.\n');

% keyboard;

% We've now finished, so restore the random generator if necessary.
if ~isempty(analysisseed),
   randstates(oldstate);
end

return;


function [derivative] = calcderivative(H, Anew, A, gap, epsilon, wantedA, maxentnotconstraints, fleewalls);
% Usage: [derivative] = calcderivative(H, Anew, A, gap, epsilon, wantedA, maxentnotconstraints, fleewalls);
% Calculates the derivative with respect to epsilon of either -mismatch, entropy,
% or sum(log(A)), according to the setting of maxentnotconstraints and fleewalls.

global wall

Atrial = Anew * epsilon + A * (1 - epsilon);

if maxentnotconstraints,

   if fleewalls,
      derivative = sum((Anew - A) .* (1 ./ Atrial));
   else % i.e. if maximising entropy
      derivative = - sum((Anew - A) .* (1 + log(Atrial)));
      if any(Atrial < wall / 2),
         error('Should not be able to want entropy derivative of Atrial values < wall');
      end      
   end
else

   % Wall Atrial in.
   ind = find(Atrial < wall);
   Atrial(ind) = wall;

   % In theory, to calculate the derivative, we need to update gap in the light
   % of the last few lines, as the gradient is no longer pointing directly
   % towards the constraint hyperplane. However we do not bother to do this,
   % as we are really only interested in the sign of the derivative, and as far
   % as I can see this will still be correct.
   if 0, % ~isempty(ind),
      gap = H * (Anew - A);
   end % if ~isempty(ind)

   nowgap = H * Atrial - wantedA;
   nowgapnorm = sqrt(sum(nowgap .^ 2));
   if nowgapnorm == 0,
      derivative = 0;
   else
      derivative = - sum(gap .* nowgap) / nowgapnorm;
   end

end

return;


function [Hinv] = svdinv(H, svdtol);
% Usage: [Hinv] = svdinv(H, svdtol);
% Calculates the pseudoinverse by the SVD, making use of the economy size version
% where appropriate.

if size(H, 1) < size(H, 2),
   H = H.';
   transposing = 1;
else
   transposing = 0;
end

[HL, HD, HR] = svd(H);
hd = diagfrom(HD)
hdinv = zeros(size(hd));
ind = find(hd >= svdtol);
hdinv(ind) = 1 ./ hd(ind);
sz = size(HD);
szinv = [sz(2), sz(1)];
HDinv = diagsz(hdinv, szinv);
HLtransp = HL.';
clear HL HD hd hdinv ind sz szinv
HDinv = sparse(HDinv);
Hinv = HR * (HDinv * HLtransp);
clear HLtransp HDinv HR

if transposing,      
   Hinv = Hinv.';
end

return;


% Local Variables: 
% indent-line-function: indent-relative
% eval: (auto-fill-mode 0)
% End:
