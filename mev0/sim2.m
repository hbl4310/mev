function [outorder, outorderpos, outorderlist, outorderprobs] = sim2(marginstableorcreateseed, Niters, analysisseed, Nc, Nv, Nvg, Nparticles, inorders);
% Usage: [outorder, outorderpos, outorderlist, outorderprobs] = sim2(marginstableorcreateseed, Niters, analysisseed, Nc, Nv, Nvg, Nparticles, inorders);
% 
% Function to simulate the maximum entropy election system 
%  (without voter ability to conjoin candidates) 
%  by MCMC and adaptation of Lagrange multipliers.
% If marginstableorcreateseed is 1x1 or empty then a set of votes 
%  is created and its analysis simulated; if 1x1 then that value is 
%  used to seed the random number generators used for creation;
%  in this case Nc and Nv are respectively the number
%  of candidates and the number of voters, taken to be 3 and 50
%  if not given, and Nvg is the number of different types of voter,
%  taken to be 3 if not given.
% If marginstableorcreateseed is N x N then Nc is taken to be N and 
%  the passed values of Nc and Nv are ignored; the matrix is 
%  taken to be the marginstable C, such that C(nc1, nc2) is the 
%  number of voters that thought candidate nc1 was better
%  than or equal to candidate nc2.
% If marginstableorcreateseed is Nv x Nc or inadmissible as a 
%  (unnormalised) marginstable then it is taken to be a scores
%  table; each voter in it will be represented by 100 voters with
%  orderings derived by taking Gaussianly distributed scores around
%  the given values.
% If analysisseed is non-empty then it is used to seed the random 
%  number generator used for choosing the output order.
% If inorders is passed, of size Nc x Nparticles, and we are not
%  synthesising a problem, then those orders are used to initialise 
%  the particles, and Nparticles is set from this parameter; 
%  inorders should contain randomly drawn orders from the votes.
%  If we are synthesising a problem then inorders is ignored.
% Nparticles is the number of Markov chains that are run; each 
%  starts from the votes of a single randomly chosen voter (if available)
%  or from a randomly chosen ordering otherwise.
% outorder is a Nc x 1 vector which gives the output ordering
%  of the candidates starting with the most preferred.
% outorderpos is a Nc x 1 vector such that outorderpos(nc) is the 
%  position in outorder of candidate nc.
% outorderlist is a Nc x Nparticles x Niters list of all the samples
%  considered.
% outorderprobs is a Niters x 1 list of the probabilities with which
%  each niter in outorderlist was considered for being outorder;
%  the probability for each particle for any given iter is identical.



% Deal with the arguments.
if nargin < 1,
   marginstableorcreateseed = [];
end
if nargin < 2,
   Niters = [];
end
if nargin < 3,
   analysisseed = [];
end
if nargin < 4,
   Nc = [];
end
if nargin < 5,
   Nv = [];
end
if nargin < 6,
   Nvg = [];
end
if nargin < 7,
   Nparticles = [];
end
if nargin < 8,
   inorders = [];
end

if isempty(Niters),
   Niters = 1000;
end
if isempty(Nc),
   Nc = 3;
end
if isempty(Nv),
   Nv = 50;
end
if isempty(Nvg),
   Nvg = 3;
end
if isempty(Nparticles),
   if ~isempty(inorders),
      Nparticles = size(inorders, 2);
   else
      Nparticles = 100;
   end
end

% ****
fprintf('Niters is %d\n', Niters);

if prod(size(marginstableorcreateseed)) <= 1 | iscell(marginstableorcreateseed),
   createseed = marginstableorcreateseed;
   synthesising = 1;
else
   C = marginstableorcreateseed;
   synthesising = 0;
end

% Global variables to do with the random generator.
global laststartstate lastanalysisstate

if ~synthesising,
   % We need to sort out whether the table is a scores table or a margins table.

   marginsnotscores = 0;
   if size(C, 1) == size(C, 2),
      if all(all(abs(1 - (C + C.') ./ (2 * C(1, 1))) < 1e-6, 1), 2),
         marginsnotscores = 1;
         Nc = size(C, 1);
      end
   end

   if ~marginsnotscores,
      Nc = size(C, 2);
      Nv = size(C, 1);
      C = C.';
   end
else
   marginsnotscores = 0;
end

if marginsnotscores,
   fprintf('Input is being taken as margin fractions rather than scores.\n');
else
   fprintf('Input is being taken as scores not margin fractions.\n');
end

% Set various parameters for the analysis.

% Number of candidates to aim to move each time.
Ncdraw = 8;

% Minimum limit for averaging length
minavlen = ceil(100 / Nparticles);

% Amount to nudge betas.
epsilon = 0.1;

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
         scores(nc, nv) = randbeta(scoremeans(nc, nv) * scorestrengths(nc, nv), ...
                                   (1 - scoremeans(nc, nv)) * scorestrengths(nc, nv));
      end
   end

   marginsnotscores = 0;
   betascale = 10;
   C = scores * betascale;

   % Restore the random generator if necessary.
   if ~isempty(createseed),
      randstates(oldstate);
   end

end % if synthesising

% Set up the random generator for analysis.
if ~isempty(analysisseed),
   [oldstate, lastanalysisstate] = randstates(analysisseed);
end

if ~marginsnotscores,

   % Work out the margins table.

   scores = C;
   Nvmult = 100;
   scores = repmat(scores, [1, Nvmult]) + randn(Nc, Nv * Nvmult);
   Nv = Nv * Nvmult;

   % Then we work out the margins table
   C = zeros(Nc, Nc);
   for nc1 = 1 : Nc,
      for nc2 = 1 : Nc,
         C(nc1, nc2) = sum(scores(nc1, :) > scores(nc2, :), 2) ...
                       + 0.5 * sum(scores(nc1, :) == scores(nc2, :));
      end
   end

end

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

% We've now finished synthesising and checking everything, so time to turn to analysis.

if ~marginsnotscores,

   % Then we draw inorders.
   inordervoters = ceil(Nv * rand(1, Nparticles));
   inorders = repmat(NaN, [Nc, Nparticles]);
   for nparticle = 1 : Nparticles,
      [trash, ind] = sort(- scores(:, inordervoters(nparticle)));
      inorders(:, nparticle) = ind;
   end % for nparticle = 1 : Nparticles

end % if ~marginsnotscores

% beta(nc1, nc2) is the number of nepers of favour to give to orderings that prefer nc1 to nc2.
if 0,
   % Initialise the betas at zero.
   betas = zeros(Nc, Nc);
else
   cliplimit = 1e-4;
   betas = 0.5 * log(max(marginfracs, cliplimit) ./ max(marginfracs', cliplimit));
end    

% Initialise the output ordering.
if ~isempty(inorders),
   outorders = inorders;
else
   outorders = repmat(NaN, [Nc, Nparticles]);
   for nparticle = 1 : Nparticles,
      outorders(:, nparticle) = randperm(Nc).';
   end
end

% Record variables.
betaslist = repmat(NaN, [Nc, Nc, Niters]);
acceptlist = repmat(NaN, [Nparticles, Niters]);
outorderlist = repmat(NaN, [Nc, Nparticles, Niters]);
outorderposlist = repmat(NaN, [Nc, Nparticles, Niters]);
epsilonlist = repmat(NaN, [Niters, 1]);
rmserrorslist = repmat(NaN, [Niters, 1]);
decayfactorlist = repmat(NaN, [Niters, 1]);

% Iterate until converged.
niter = 0;
useIIRcurrentmargins = 1;
if useIIRcurrentmargins,
   currentmargins = marginfracs;
else
   currentmargins = repmat(0.5, [Nc, Nc]);
end
while niter < Niters,
   niter = niter + 1;

   % First we nudge the betas.

   % We calculate how far to nudge them based on how far out we are with
   % each marginfrac.

   if 0, % *****
      nudgescale = abs(currentmargins - marginfracs);
   else
      nudgescale = abs(currentmargins - marginfracs) ./ max(1e-3, min(marginfracs, 1 - marginfracs));
   end

   outorderposs = zeros(Nc, Nparticles);
   outorderposs(sub2ind([Nc, Nparticles], ...
                        outorders, ...
                        repmat([1 : Nparticles], [Nc, 1]))) ...
      = repmat([1 : Nc].', [1, Nparticles]);
   thesecomparisons = zeros(Nc, Nc, Nparticles);
   for nparticle = 1 : Nparticles,
      outorderpos = outorderposs(:, nparticle);
      [nn1, nn2] = ndgrid(outorderpos, outorderpos);
      thiscomparison = (nn1 < nn2) + 0.5 * (nn1 == nn2); % Set to 1 if nn1 better than nn2 or 0 otherwise.
      thesecomparisons(:, :, nparticle) = thiscomparison;
   end
   thiscomparison = sum(thesecomparisons, 3) / Nparticles;

% **** question is whether this should have
% **** nudgescale = 1 ./ max(1e-3, min(marginfracs, 1 - marginfracs)); and
% **** update delta = sqrt(Nparticles * epsilon * nudgescale) .* sqrt(abs(thiscomparison - marginfracs)) .* sign(thiscomparison - marginfracs);;

   betas = betas - sqrt(Nparticles * epsilon * abs(thiscomparison - marginfracs) .* nudgescale) .* sign(thiscomparison - marginfracs);
   betas = (betas - betas.') / 2;

   Eolds = reshape(-sum(sum(thesecomparisons .* repmat(betas, [1, 1, Nparticles]), 1), 2), ...
                   [Nparticles, 1]);

   % Redraw outorders, by Metropolis-Hastings, using the proposal distribution outlined in
   % testhypothesis.m .

   for nparticle = 1 : Nparticles,
      outorder = outorders(:, nparticle);
      outorderpos = outorderposs(:, nparticle);
   
      % We will only redraw certain of the candidates.
      moving = rand(Nc, 1) < (Ncdraw / Nc);
      posmoving = find(moving);
      ncsmoving = outorder(posmoving);
   
      % We first calculate the contributions to the proposal dist.
      Econtribs = -betas(ncsmoving, :);
      Econtribs = Econtribs - Eadd(-Econtribs, Econtribs);
   
      % Make a copy for updating in due course to calculate probability of backward move.
      Eoldcontribs = Econtribs;
      oldmoving = moving;

      % We start by considering what's going in the top moving position, and work downwards.
      neworder = outorder;
      Eforward = 0;
      Ebackward = 0;
      for nposmoving = 1 : length(posmoving),
   
         neworderpos = zeros(Nc, 1);
         neworderpos(neworder) = [1 : Nc].';
   
         ncsbelow = find(neworderpos >= posmoving(nposmoving));
         oldncsbelow = find(outorderpos >= posmoving(nposmoving));
   
         Etot = sum(Econtribs(:, ncsbelow), 2);
         Etot(~moving(posmoving)) = Inf; % Kill those that are not up for moving.
         Etot = Etot - min(Etot);
         ptot = exp(-Etot);
   
         ptot = ptot ./ sum(ptot);
   
         chosennposmoving = randdiscrete([1, 1], ptot(:));
         Eforward = Eforward - log(ptot(chosennposmoving));
   
         Eoldtot = sum(Econtribs(:, oldncsbelow), 2);
         Eoldtot(~oldmoving(posmoving)) = Inf;
         Eoldtot = Eoldtot - min(Eoldtot);
         poldtot = exp(-Eoldtot);
         sumpoldtot = sum(poldtot);
         poldtot = poldtot ./ sumpoldtot;
         Eoldtot = Eoldtot + log(sumpoldtot);
   
         Ebackward = Ebackward + Eoldtot(nposmoving);
   
         neworder(neworderpos(ncsmoving(chosennposmoving))) = neworder(posmoving(nposmoving));
         neworder(posmoving(nposmoving)) = ncsmoving(chosennposmoving);
         moving(posmoving(chosennposmoving)) = 0;
         oldmoving(posmoving(nposmoving)) = 0;

      end % for nposmoving = 1 : length(posmoving)
      neworderpos = zeros(Nc, 1);
      neworderpos(neworder) = [1 : Nc].';
   
      [nn1, nn2] = ndgrid(neworderpos, neworderpos);
      newcomparison = (nn1 < nn2);
      Enew = -sum(sum(newcomparison .* betas));
   
      % Now need to decide accept or reject ?
      Eaccept = Enew - Eolds(nparticle) + Ebackward - Eforward;
      accept = rand < exp(-Eaccept);
   
      if accept,
         outorder = neworder;
         outorderpos = neworderpos;
         outorders(:, nparticle) = outorder;
         outorderposs(:, nparticle) = outorderpos;
      end
   
      % Make a record for looking at later.
      betaslist(:, :, niter) = betas;
      acceptlist(nparticle, niter) = accept;
      outorderlist(:, nparticle, niter) = outorder;
      outorderposlist(:, nparticle, niter) = outorderpos;

   end % for nparticle = 1 : Nparticles

   wantedavlen = max(minavlen, 2 / (Nparticles * epsilon .^ 2));
   startav = round(max(niter / 4, niter - wantedavlen));
   avlength = niter - startav + 1;

   if useIIRcurrentmargins,
      updatetimeconstant = max(minavlen, avlength);
      updatealpha = 1 ./ updatetimeconstant;
      currentmargins = (1 - updatealpha) * currentmargins;
      decayfactorlist(niter) = 1 - updatealpha;
      for nc1 = 1 : Nc,
         for nc2 = 1 : Nc,
            currentmargins(nc1, nc2) = currentmargins(nc1, nc2) ...
                                       + updatealpha * sum(outorderposlist(nc1, :, niter) < outorderposlist(nc2, :, niter), 2) / Nparticles;
         end
      end
      currentmargins(sub2ind([Nc, Nc], [1 : Nc], [1 : Nc])) = 0.5;
   else
      if rem(niter, 100) == 0 | niter == Niters,
         currentmargins = repmat(NaN, [Nc, Nc]);
         for nc1 = 1 : Nc,
            for nc2 = 1 : Nc,
               currentmargins(nc1, nc2) = sum(sum(outorderposlist(nc1, :, startav : niter) < outorderposlist(nc2, :, startav : niter), ...
                                                  2), ...
                                              3) ...
                                          / (avlength * Nparticles);
            end
         end
         currentmargins = currentmargins + 0.5 * eye(Nc, Nc);
      end
   end

   ratios = currentmargins ./ marginfracs;
   errors = currentmargins - marginfracs;

   rmserrors = sqrt(mean(errors(:) .^ 2));

   epsilon = rmserrors / sqrt(max(minavlen, niter));

   if rem(niter, 100) == 0 | niter == Niters,

      % Prepare an interim report.

      acceptrate = sum(sum(acceptlist(:, 1 : niter), 1), 2) / (niter * Nparticles)

      if Nc <= 5,
         marginfracs
         currentmargins
         ratios
         errors
      else
         ratioerrs = [max(ratios(:)), min(ratios(:))]
         maxerrors = max(abs(errors(:)))
      end

      rmserrors
      epsilon

   end

   % Record the various control variables.
   epsilonlist(niter) = epsilon;
   rmserrorslist(niter) = rmserrors;
   
end % while niter < Niters

if useIIRcurrentmargins,
   overalldecays = flipud(cumprod(flipud([decayfactorlist(2 : end); 1]))) .* (1 - decayfactorlist);
   sumoveralldecays = sum(overalldecays)
   overalldecays = overalldecays / sumoveralldecays;
   chosenone = randdiscrete([1, 1], overalldecays);
   outorderprobs = overalldecays;
else
   chosenone = ceil(rand * avlength) + startav - 1;
   outorderprobs = [zeros(startav - 1, 1); repmat(1 ./ avlength, [Niters - startav, 1])];
end
   
outorder = outorderlist(:, ceil(rand * Nparticles), chosenone);
outorderpos = outorder;
outorderpos(outorder) = [1 : Nc].';

% We've now finished, so restore the random generator if necessary.
if ~isempty(analysisseed),
   randstates(oldstate);
end

return;



% Local Variables: 
% indent-line-function: indent-relative
% eval: (auto-fill-mode 0)
% End:
