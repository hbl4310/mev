# C: candidates 
# s_i,j: fractino of votes that prefer c_i to c_j ; ties are distributed equally 
# T: set of strong total orderings on C 
#   for each t \in T, define pairwise preference variables d_i,j(t) = 1 if c_i preferred to c_j under t, = 1/2 if i=j, and 0 otherwise 
# U: set of probability distributions on T 
#   for each u \in U, define matrix D(u) whose elements d_i,j(u) are thr probability that under random ordering t drawn from u, c_i is preferred to c_j 
#   d_i,j(u) = \Sum u(t) d_i,j(t) 
#   the map B that maps u to D(u) is linear 
# 

import numpy as np 

"""
sim2.m runs the Markov chain Monte-Carlo version of the
algorithm. This version runs a number of iterations of a sampling
algorithm, whose distribution gradually converges to the output
distribution of the system. How long it takes depends on the settings
of Niters - the larger this variable is set, the longer it takes, and
the more accurately the result distribution will be to the desired
distribution. Niters=1000 seems a reasonable compromise.


Usage: [outorder, outorderpos, outorderlist, outorderprobs] = sim2(marginstableorcreateseed, Niters, analysisseed, Nc, Nv, Nvg, Nparticles, inorders);

Function to simulate the maximum entropy election system 
 (without voter ability to conjoin candidates) 
 by MCMC and adaptation of Lagrange multipliers.
If marginstableorcreateseed is 1x1 or empty then a set of votes 
 is created and its analysis simulated; if 1x1 then that value is 
 used to seed the random number generators used for creation;
 in this case Nc and Nv are respectively the number
 of candidates and the number of voters, taken to be 3 and 50
 if not given, and Nvg is the number of different types of voter,
 taken to be 3 if not given.
If marginstableorcreateseed is N x N then Nc is taken to be N and 
 the passed values of Nc and Nv are ignored; the matrix is 
 taken to be the marginstable C, such that C(nc1, nc2) is the 
 number of voters that thought candidate nc1 was better
 than or equal to candidate nc2.
If marginstableorcreateseed is Nv x Nc or inadmissible as a 
 (unnormalised) marginstable then it is taken to be a scores
 table; each voter in it will be represented by 100 voters with
 orderings derived by taking Gaussianly distributed scores around
 the given values.
If analysisseed is non-empty then it is used to seed the random 
 number generator used for choosing the output order.
If inorders is passed, of size Nc x Nparticles, and we are not
 synthesising a problem, then those orders are used to initialise 
 the particles, and Nparticles is set from this parameter; 
 inorders should contain randomly drawn orders from the votes.
 If we are synthesising a problem then inorders is ignored.
Nparticles is the number of Markov chains that are run; each 
 starts from the votes of a single randomly chosen voter (if available)
 or from a randomly chosen ordering otherwise.
outorder is a Nc x 1 vector which gives the output ordering
 of the candidates starting with the most preferred.
outorderpos is a Nc x 1 vector such that outorderpos(nc) is the 
 position in outorder of candidate nc.
outorderlist is a Nc x Nparticles x Niters list of all the samples
 considered.
outorderprobs is a Niters x 1 list of the probabilities with which
 each niter in outorderlist was considered for being outorder;
 the probability for each particle for any given iter is identical.
"""
def sim2(   marginstableorcreateseed = [], 
            Niters = 1000, 
            analysisseed = [], 
            Nc = 3,  
            Nv = 50, 
            Nvg = 3, 
            Nparticles = 100, 
            inorders = [], 
        ): 
    if inorders: 
        Nparticles = inorders.shape[1] 

    # ****
    print(f'Niters is {Niters}')

    if np.isscalar(marginstableorcreateseed) or np.prod(marginstableorcreateseed.shape) <= 1: 
        createseed = marginstableorcreateseed 
        synthesising = 1
    else: 
        C = marginstableorcreateseed    # TODO check what type C should be 
        synthesising = 0 
    
    # Global variables to do with the random generator. 
    laststartstate, lastanalysisstate = None, None

    if not synthesising: 
        # We need to sort out whether the table is a scores table or a margins table.
        # TODO check this branch 
        marginsnotscores = 0 
        if C.shape[0] == C.shape[1]: 
            if (abs(1 - (C + C.T) / (2 * C[0, 0])) < 1e-6).all():
                marginsnotscores = 1
                Nc = C.shape[0]
        if not marginsnotscores: 
            Nc = C.shape[1] 
            Nv = C.shape[0] 
            C = C.T 
    else: 
        marginsnotscores = 0 

    if marginsnotscores:
        print('Input is being taken as margin fractions rather than scores.')
    else:
        print('Input is being taken as scores not margin fractions. ')
    
    # Set various parameters for the analysis.

    # Number of candidates to aim to move each time.
    Ncdraw = 8

    # Minimum limit for averaging length
    minavlen = int(100 / Nparticles) + 1

    # Amount to nudge betas.
    epsilon = 0.1

    # Do the synthesising if necessary
    if synthesising: 

        # Set up the random generator.
        # TODO randstates sets the random seed and warms up RNG
        # if not createseed:
        #     oldstate, laststartstate = randstates(createseed) 
        # else:
        #     laststartstate = randstates

        # First we must determine the score distributions for each candidate and each voter group
        scoremeans = np.random.rand(Nc, Nvg)
        scorestrengths = np.random.gamma(2, 1/0.1, (Nc, Nvg))

        # Then we assign a voter group to each voter,
        crits = np.tile(1 / Nvg, (Nvg, 1))
        crits = crits.cumsum(0)
        nvgs = np.minimum(Nvg-1, (np.tile(np.random.rand(1, Nv), (Nvg, 1)) > np.tile(crits, (1, Nv))).sum(0))

        # and propagate the means and strengths to each voter.
        scoremeans = scoremeans[:, nvgs]
        scorestrengths = scorestrengths[:, nvgs]

        # Then we assign a score to each voter for each candidate
        scores = np.zeros((Nc, Nv))
        for nc in range(Nc):
            for nv in range(Nv): 
                scores[nc, nv] = np.random.beta(scoremeans[nc, nv] * scorestrengths[nc, nv], \
                                        (1 - scoremeans[nc, nv]) * scorestrengths[nc, nv])

        marginsnotscores = 0
        betascale = 10
        C = scores * betascale

        # Restore the random generator if necessary.
        # TODO not sure if relevant in python
        # if createseed: 
            # randstates(oldstate)
    
    # Set up the random generator for analysis.
    # TODO reset random seed for analysis
    # if analysisseed: 
        # oldstate, lastanalysisstate = randstates(analysisseed)

    if not marginsnotscores:

        # Work out the margins table.

        scores = C
        Nvmult = 100
        scores = np.tile(scores, (1, Nvmult)) + np.random.randn(Nc, Nv * Nvmult)
        Nv = Nv * Nvmult

        # Then we work out the margins table
        C = np.zeros((Nc, Nc))
        for nc1 in range(Nc): 
            for nc2 in range(Nc): 
                C[nc1, nc2] = (scores[nc1, :] > scores[nc2, :]).sum() \
                            + 0.5 * (scores[nc1, :] == scores[nc2, :]).sum()

    # Check that C has the desired properties
    assert len(C.shape) == 2, 'marginstable is not 2-dimensional'
    assert C.shape[0] == Nc and C.shape[1] == Nc, 'marginstable is not Nc x Nc'
    Ccheck = C + C.T
    Nv = Ccheck.mean()
    Ccheck = Ccheck / Nv
    assert (Ccheck > 0.9999).all() and (Ccheck < 1.0001).all(), 'marginstable is not valid'

    C = C / Nv
    marginfracs = C

    # We've now finished synthesising and checking everything, so time to turn to analysis.

    if not marginsnotscores: 

        # Then we draw inorders.
        inordervoters = np.floor(Nv * np.random.rand(Nparticles)).astype(int)  # floor instead of ceil because of 0 vs 1 indexing 
        inorders = np.tile(np.nan, (Nc, Nparticles))
        for nparticle in range(Nparticles):
            ind = np.argsort(- scores[:, inordervoters[nparticle].astype(int)])
            inorders[:, nparticle] = ind
        inorders = inorders.astype(int)

    # beta(nc1, nc2) is the number of nepers of favour to give to orderings that prefer nc1 to nc2.
    if 0:
        # Initialise the betas at zero.
        betas = np.zeros((Nc, Nc))
    else: 
        cliplimit = 1e-4
        betas = 0.5 * np.log(np.maximum(marginfracs, cliplimit) / np.maximum(marginfracs.T, cliplimit))
    
    # Initialise the output ordering.
    if inorders.size != 0: 
        outorders = inorders
    else: 
        outorders = np.tile(np.nan, (Nc, Nparticles))
        for nparticle in range(Nparticles): 
            outorders[:, nparticle] = np.random.permutation(Nc).T 
    outorders = outorders.astype(int)

    # Record variables.
    betaslist = np.tile(np.nan, (Nc, Nc, Niters))
    acceptlist = np.tile(np.nan, (Nparticles, Niters))
    outorderlist = np.tile(np.nan, (Nc, Nparticles, Niters))
    outorderposlist = np.tile(np.nan, (Nc, Nparticles, Niters))
    epsilonlist = np.tile(np.nan, (Niters, 1))
    rmserrorslist = np.tile(np.nan, (Niters, 1))
    decayfactorlist = np.tile(np.nan, (Niters, 1))

    # Iterate until converged.
    niter = 0
    useIIRcurrentmargins = 1
    if useIIRcurrentmargins:
        currentmargins = marginfracs
    else:
        currentmargins = np.tile(0.5, (Nc, Nc))

    while niter < Niters: 
        niter = niter + 1

        # First we nudge the betas.

        # We calculate how far to nudge them based on how far out we are with
        # each marginfrac.

        if 0: # *****
            nudgescale = np.abs(currentmargins - marginfracs)
        else: 
            nudgescale = np.abs(currentmargins - marginfracs) / np.maximum(1e-3, np.minimum(marginfracs, 1 - marginfracs)) 
   
        outorderposs = np.zeros(Nc*Nparticles)
        sub2ind = np.vectorize(lambda x,y: np.ravel_multi_index((x, y), dims=(Nc, Nparticles), order='F'))  # F for Fortran-style arrays (1 indexed / column-major)
        idx = sub2ind(outorders.ravel('F'), np.tile(np.arange(Nparticles, dtype=int), (Nc, 1)).ravel('F'))
        outorderposs[idx] = np.tile(np.arange(Nc).reshape(Nc, 1), (1, Nparticles)).ravel('F')
        thesecomparisons = np.zeros((Nc, Nc, Nparticles))
        # TODO not sure about the reshape back to Nc x Nparticles
        outorderposs = outorderposs.reshape(Nparticles, Nc).T.astype(int)

        for nparticle in range(Nparticles): 
            outorderpos = outorderposs[:, nparticle]
            nn1, nn2 = np.meshgrid(outorderpos, outorderpos, indexing='ij')
            thiscomparison = (nn1 < nn2) + 0.5 * (nn1 == nn2) # Set to 1 if nn1 better than nn2 or 0 otherwise.
            thesecomparisons[:, :, nparticle] = thiscomparison
        thiscomparison = thesecomparisons.sum(2) / Nparticles

    # **** question is whether this should have
    # **** nudgescale = 1 ./ max(1e-3, min(marginfracs, 1 - marginfracs)); and
    # **** update delta = sqrt(Nparticles * epsilon * nudgescale) .* sqrt(abs(thiscomparison - marginfracs)) .* sign(thiscomparison - marginfracs);;

        betas = betas - np.sqrt(Nparticles * epsilon * np.abs(thiscomparison - marginfracs) * nudgescale) * np.sign(thiscomparison - marginfracs)
        betas = (betas - betas.T) / 2

        Eolds = (-(thesecomparisons * np.tile(np.expand_dims(betas, -1), (1, 1, Nparticles))).sum((0, 1))).reshape(Nparticles, 1)

        # Redraw outorders, by Metropolis-Hastings, using the proposal distribution outlined in
        # testhypothesis.m .

        for nparticle in range(Nparticles):
            outorder = outorders[:, nparticle]
            outorderpos = outorderposs[:, nparticle]
        
            # We will only redraw certain of the candidates.
            moving = np.random.rand(Nc, 1) < (Ncdraw / Nc)
            posmoving = np.where(moving)[0]
            ncsmoving = outorder[posmoving]
        
            # We first calculate the contributions to the proposal dist.
            Econtribs = -betas[ncsmoving, :]
            Econtribs = Econtribs + np.log(np.exp(Econtribs) + np.exp(-Econtribs))
        
            # Make a copy for updating in due course to calculate probability of backward move.
            Eoldcontribs = Econtribs
            oldmoving = moving
        
            # We start by considering what's going in the top moving position, and work downwards.
            neworder = outorder
            Eforward = 0
            Ebackward = 0
            for nposmoving in range(posmoving.shape[0]): 
        
                neworderpos = np.zeros((Nc, 1), dtype=int)
                neworderpos[neworder] = np.expand_dims(np.arange(Nc, dtype=int), -1)
        
                ncsbelow = np.where(neworderpos >= posmoving[nposmoving])[0]
                oldncsbelow = np.where(outorderpos >= posmoving[nposmoving])[0]
        
                Etot = Econtribs[:, ncsbelow].sum(1)
                Etot[~moving[posmoving].flatten()] = np.inf  # Kill those that are not up for moving.
                Etot = Etot - Etot.min() if Etot.min() != np.inf else Etot   # TODO this hack prevents errors when Etot is all np.inf
                ptot = np.exp(-Etot)
        
                ptot = ptot / ptot.sum() if ptot.sum() != 0 else ptot + 1/ptot.shape[0]  # TODO related hack 
        
                chosennposmoving = np.random.choice(ptot.shape[0], p=ptot[:])
                Eforward = Eforward - np.log(ptot[chosennposmoving])
        
                Eoldtot = Econtribs[:, oldncsbelow].sum(1)
                Eoldtot[~oldmoving[posmoving].flatten()] = np.inf
                Eoldtot = Eoldtot - Eoldtot.min() if Eoldtot.min() != np.inf else Eoldtot
                poldtot = np.exp(-Eoldtot)
                sumpoldtot = poldtot.sum()
                poldtot = poldtot / sumpoldtot if sumpoldtot != 0 else poldtot + 1/poldtot.shape[0]  # TODO related hack 
                Eoldtot = Eoldtot + np.log(sumpoldtot) if sumpoldtot != 0 else Eoldtot
        
                Ebackward = Ebackward + Eoldtot[nposmoving]
        
                neworder[neworderpos[ncsmoving[chosennposmoving]]] = neworder[posmoving[nposmoving]]
                neworder[posmoving[nposmoving]] = ncsmoving[chosennposmoving]
                moving[posmoving[chosennposmoving]] = 0
                oldmoving[posmoving[nposmoving]] = 0
        
            neworderpos = np.zeros((Nc, 1))
            neworderpos[neworder] = np.expand_dims(np.arange(Nc), -1)
        
            nn1, nn2 = np.meshgrid(neworderpos, neworderpos, indexing='ij')
            newcomparison = (nn1 < nn2)
            Enew = -(newcomparison * betas).sum()
        
            # Now need to decide accept or reject ?
            Eaccept = Enew - Eolds[nparticle] + Ebackward - Eforward
            accept = np.random.rand() < np.exp(-Eaccept)
        
            if accept:
                outorder = neworder
                outorderpos = neworderpos
                outorders[:, nparticle] = outorder
                outorderposs[:, nparticle] = outorderpos.flatten()
            
            # Make a record for looking at later.
            betaslist[:, :, niter-1] = betas
            acceptlist[nparticle, niter-1] = accept
            outorderlist[:, nparticle, niter-1] = outorder
            outorderposlist[:, nparticle, niter-1] = outorderpos.flatten()

        wantedavlen = max(minavlen, 2 / (Nparticles * epsilon ** 2))
        startav = round(max(niter / 4, niter - wantedavlen))
        avlength = niter - startav + 1

        if useIIRcurrentmargins:
            updatetimeconstant = max(minavlen, avlength)
            updatealpha = 1 / updatetimeconstant
            currentmargins = (1 - updatealpha) * currentmargins
            decayfactorlist[niter-1] = 1 - updatealpha
            for nc1 in range(Nc):
                for nc2 in range(Nc): 
                    currentmargins[nc1, nc2] = currentmargins[nc1, nc2] \
                                            + updatealpha * (outorderposlist[nc1, :, niter-1] < outorderposlist[nc2, :, niter-1]).sum() / Nparticles
            np.fill_diagonal(currentmargins, 0.5)
        else: 
            if niter % 100 == 0 or niter == Niters: 
                currentmargins = np.tile(np.nan, [Nc, Nc])
                for nc1 in range(Nc): 
                    for nc2 in range(Nc): 
                        currentmargins[nc1, nc2] = (outorderposlist[nc1, :, startav : niter-1] < outorderposlist[nc2, :, startav : niter-1]).sum() \
                                                / (avlength * Nparticles)
                currentmargins = currentmargins + 0.5 * np.eye(Nc)

        ratios = currentmargins / marginfracs
        errors = currentmargins - marginfracs

        rmserrors = np.sqrt(np.mean(np.power(errors, 2)))

        epsilon = rmserrors / np.sqrt(max(minavlen, niter))

        if niter % 100 == 0 or niter == Niters: 

            # Prepare an interim report.

            acceptrate = sum(sum(acceptlist[:, 1 : niter-1], 1), 2) / (niter * Nparticles)
            print('acceptrate =', acceptrate)

            if Nc <= 5: 
                print('marginfracs =\n', marginfracs)
                print('currentmargins =\n', currentmargins)
                print('ratios =\n', ratios)
                print('errors =\n', errors)
            else: 
                ratioerrs = [ratios.max(), ratios.min()]
                maxerrors = errors.abs().max()
                print('ratioerrs =\n', ratioerrs)
                print('maxerrors =\n', maxerrors)

            print('rmserrors =', rmserrors)
            print('epsilon =', epsilon)

        # Record the various control variables.
        epsilonlist[niter-1] = epsilon
        rmserrorslist[niter-1] = rmserrors

    if useIIRcurrentmargins: 
        overalldecays = np.multiply(np.cumprod(np.append(decayfactorlist[1:], 1.)[::-1])[::-1], (1 - decayfactorlist).flatten())
        sumoveralldecays = overalldecays.sum()
        print('sumoveralldecays =', sumoveralldecays)
        overalldecays = overalldecays / sumoveralldecays
        chosenone = np.random.choice(overalldecays.shape[0], p=overalldecays)
        outorderprobs = overalldecays
    else: 
        chosenone = int(np.random.rand() * avlength) + startav 
        outorderprobs = np.array([np.zeros(startav - 1, 1), np.tile(1 / avlength, [Niters - startav, 1])])
    
    outorderlist = outorderlist.astype(int)
    outorder = outorderlist[:, int(np.ceil(np.random.rand() * Nparticles)), chosenone]
    outorderpos = outorder
    outorderpos[outorder] = np.arange(Nc, dtype=int)

    # We've now finished, so restore the random generator if necessary.
    # TODO setting analysis seed 
    # if analysisseed: 
    #     randstates(oldstate)

    return outorder, outorderpos, outorderlist, outorderprobs



if __name__=='__main__': 
    outorder, outorderpos, outorderlist, outorderprobs = sim2(5, 200, 6)

    # TODO issues still : 
    # errors and rmserrors look large comparitively 
    # sometimes outorder contains duplicate entries 