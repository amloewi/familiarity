import itertools
import copy
import sys

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


# The idea here -- people pass messages, based on their understanding of
# the people they're passing them TOO. However, that understanding changes
# with the strength of the relationship. SO, see if you can ESTIMATE the
# MEANING of the strength of the relationship (how much you've learned,
# when you say you know somebody "pretty well") when you're observing
# the accuracy with which people passed the messages.

# In THIS MODEL: You have a random graph, with random edge weights between
# 1 and 10, for relationship strength. These correspond to how much of a
# bit vector you can observe of your friends. This isn't observed -- that
# relationship is what you're trying to estimate. Each person has k (10)
# bits, each of which has exponentially rarer frequency in the population
# as a whole. (Because -- I assume everything in a population is heavy
# tailed.)
#
# There's a target, T, randomly chosen from the population. A random sample of
# seeds is chosen, and asked to get to the target. They choose the person they
# pass to as p(choose j) = exp(R[:F]^T(x_j == x_T)[:F])/k
#where k is a normalizing constant, R is the inverse
# of the population frequencies of each trait, x_j/T are the bit vectors
# of the choice j, and the target T, and F is the number of bits that the
# sender is able to see.
#
# Estimating the model -- that's gonna be messy. Now, I just want to
# see if even brute-force methods work.



#####################################
# Functions for CREATING the data!  #
# (Some also necessory for fitting) #
#####################################

def single_choice_weight(g, s, n, F):
	'''Calculates "the UNNORMALIZED probability of s choosing n, given F observed bits."

	Used both to set the proper probabilities when the paths are being
	created, and also to estimate the true number of observed bits, by
	calculating hypothetical values given hypothesized F.'''
	# This is the core of the complexity: first,
	# see how many bits are VISIBLE to the sender, based on their
	# familiarity with the receiver. Then, compare how many
	# of THOSE bits are the same as the targets.
	#       And MAYbe ...
	# Then weight those similarities with the rareness of the corresponding
	# traits,
	# and finally, exponentiate the score.
	R = [1 for i in range(F)]
	print 'R: ',R
	print "(g.node[s]['features']==g.node[n]['features'])+0):",(g.node[s]['features']==g.node[n]['features'])+0
	x = np.inner(R[:F],
	 			((g.node[s]['features']==g.node[n]['features'])+0)[:F])
	# Exponentiating seems like a good idea -- somebody even a little ahead
	# will be a much more obvious choice, due to their being ahead --
	# but it is, of course, a design decision subject to critique.
	return np.exp(x)


def every_choice_weight(g, s, neighbors, theta):
	'''Calculates single_choice_weight for neighbors(s) -> probability.

	The other weights are necessary for calculating the normalizing constant.
	However, given that not every neighbor is necessarily in the running
	(having maybe already been visited), the neighbors desired needs to be
	passed explicitly.

	The theta dictionary is necessary to calculate how much of another
	person each decision can take into account.
	'''
	w = [single_choice_weight(g, s, n, theta[g[s][n]['weight']])
			for n in neighbors]
	return np.array(w)


def single_choice_probability(g, s, n, neighbors, theta):
	'''Gives the probability of choosing a particular receiver, given F.'''
	w = every_choice_weight(g, s, neighbors, theta)
	return w[neighbors.index(n)]/sum(w)


# Tested on small graphs successfully -- fixed cases where the path
# ran out of next steps.
def find_target(g, s, t, path, true_mapping, max_depth=11):
	'''Takes a graph, seed node, and target node. Returns the search path.

	A recursive function, that finds the next person in the chain at each
	step, and then passes the job on to that node to find the next again.
	Passes the list of people-thus-far as it goes.

	Should be called with "find_target(G, 0, T, [-1])" as the initial call,
	because there is a backwards look to "who sent me this?" and there needs
	to be a flag for "nobody!" (the '-1') as opposed to an error. In other
	words, every path begins with -1, which signals that the sender to the
	first person was whoever organized the study/ they were self-motivated.
	'''

	path.append(s)
	# Maximum recursion depth exceeded
	# (Also, people ... almost NEVER held on past 10 passes.)
	if len(path) > max_depth: return path

	if t in g.neighbors(s):
		path.append(t) #signals it completed successfully
		return(path)
	else:
		# Whether or not you can revisit someone is an important
		# design decision. I'm going to go with: you can't pass
		# BACK (because that would be ridiculous) BUT you also
		# can't see all the people who had it BEFORE it was passed to you.
		# This is both behaviorally realistic, AND ~maintains Markovity.
		# In addition, it allows you to pass to the same person AGAIN
		# (which is kind of like saying, you don't realize it's THE SAME
		# message -- you just say "hey, I'm hearing a LOT about this!")

		# path[-1] is s (added above). path[-2] is s's sender.
		neighbors = [n for n in g.neighbors(s) if n != path[-2]]
		# If you've hit a dead-end, return what you've got.
		if len(neighbors) < 1:
			return path

		# Provide the CORRECT mapping for F: strength 1 => 1 bit visible.
		w = every_choice_weight(g, s, neighbors, true_mapping)
		next = np.random.choice(neighbors, p=w/sum(w))

		# Keep going from the choice. (They'll be appended to the path
		# at the next level of recursion.)
		return(find_target(g, next, t, path, true_mapping))


# Tested and looks good. Small tests.
def parse_into_observations(path):
	'''Take a path, and turn it into from-to-tuples: nodes (i-1, i, i+1).

	These 3-tuples allow you to know the CONTEXT in which the decision was
	made, in a behaviorally meaningful manner. I got it FROM i, so I can't
	pass it BACK to them, but don't know who THEY got it from. And I gave it
	to i+1. Computationally, the i-1 sender is excluded from the i+1 decision
	so it's important to know who they were.
	'''

	tuples = [(path[i-1], path[i], path[i+1]) for i in range(1, len(path)-1)]
	return tuples


####################################
# Functions for FITTING the data!  #
####################################

# Performs correctly on the two simple cases shown above.
def likelihood(g, obs, theta):
	'''The log of the product of the probability of every decision.

	Each choice to pass the message to a particular person, conditional
	on getting it from another person, is associated with a probability.
	Each of these can be calculated using an from-to-tuple, given an
	assumed familiarity score for each observed weight (theta). The
	likelihood of the model is the product of the probability of these
	probabilities, to the log-lik is the sum of the log of those probs.
	'''

	# If the observations
	# REALLY AREN'T INDEPENDENT, then how does that affect the
	# calculation of the likelihood?

	# single_choice_probability takes: g, s, n, neighbors, theta
	total = 0
	for o in obs:
		args = (g, o[1], o[2],
				[j for j in g.neighbors(o[1]) if j != o[0]],
				theta)
		total += np.log(single_choice_probability(*args))
	return total


# Two spot-tests for up, and down -- looks good.
def make_monotone(theta, ix):
	'''Forces the parameters to satisfy the monotone increasing constraint.

	'ix' is the index for where the change was made in the parameter
	vector -- this is important because everything needs to be change
	to accomodate IT. Anything to the left, but higher, is truncated.
	Anything to the right, but lower, is dragged up to the same value.
	'''
	new = copy.copy(theta)
	for key in theta:
		if key < ix and theta[key] > theta[ix]:
			new[key] = new[ix]
		if key > ix and theta[key] < theta[ix]:
			new[key] = new[ix]
	return new


# Looks good after testing -- DO BE CAREFUL though, it's got
# a global variable (nL)
def metropolis_step(theta, ix):
	'''Shifts a given index's value up or down randomly by one.

	In addition, makes sure the dictionary obeys the constraint,
	by calling make_monotone.
	'''
	new = copy.copy(theta)
	step = np.random.choice([-1, 1])
	# Do a bounds check -- are you changing to a legal value?
	if theta[ix] + step in range(1, nL+1):
		new[ix] += step
	else:
		# If not, change the direction of the step.
		step *= -1
		new[ix] += step

	return make_monotone(new, ix)


# Preliminary testing -- finally looks okay.
def fit_by_metropolis(g, obs, BACKTRACK, n_steps = 100):
	'''Sequentially proposes unit changes to randomly selected parameters,
	and accepts those changes via a Metropolis step on the improved likelihood.

	Runs for a specified number of iterations at the moment, rather than
	until a convergence criterion in met -- and, returns both the mle,
	and the final estimate encountered. (So these can be compared.)
	'''

	current_theta = {i:1 for i in range(1, nL+1)}
	current_likelihood = likelihood(g, obs, current_theta)
	likelihood_path = [current_likelihood]
	max_likelihood = current_likelihood
	max_theta = current_theta

	for iteration in range(n_steps):

		sys.stdout.write('iteration: %d \r' %iteration)
		sys.stdout.flush()

		j = np.random.choice(current_theta.keys())
		proposed_theta = metropolis_step(current_theta, j)
		proposed_likelihood = likelihood(g, obs, proposed_theta)
		ratio = proposed_likelihood / current_likelihood
		choice = min([1, ratio])

		if choice == 1: # i.e. ratio > 1 i.e. new likelihood is larger
			current_theta = proposed_theta
			current_likelihood = proposed_likelihood
		################################################
		# Rigging it to NEVER accept the backtrack.    #
		# Does this work? (Is the function ~convex?)   #
		################################################
		# It's not better, so only MAYBE accept        #
		elif BACKTRACK and ratio > np.random.uniform():#
			current_theta = proposed_theta             #
			current_likelihood = proposed_likelihood   #
		################################################
		#else: Nothing changes.

		likelihood_path.append(current_likelihood)

		# Still trying to figure out how, properly, to maximize:
		# Properly, I think, do I NEED to backtrack?
		# In other words, sort of, is this convex?
		if current_likelihood > max_likelihood:
			max_likelihood = current_likelihood
			max_theta = current_theta

	return current_theta, max_theta, likelihood_path


def gibbs_step(g, obs, theta, ix):
	'''Written past 4 in the morning, full of fried tofu and thoughts.

	'''

	lkhds = []
	thetas = []
	for i in range(1, nD+1):
		# Gotta copy each time, because it's harder to undo
		# the change once you run make_monotone
		new_theta = copy.copy(theta)
		new_theta[ix] = i
		new_theta = make_monotone(new_theta, ix)
		lkhds.append(likelihood(g, obs, new_theta))
		# Save them all, versus recreate the best -- both inexpensive.
		thetas.append(new_theta)

	# THE OTHER POSSIBILITY -- choose them via weighted draw
	# Without them, it took 20 iterations.
	return thetas[lkhds.index(max(lkhds))]


# After a serious fix for the 4am gibbs-step function, this method
# works, and possibly much better than the metropolis version.
# Still being played with,
def fit_by_gibbs(g, obs):

	theta = {i:1 for i in range(1, nL+1)}
	old_likelihood = likelihood(g, obs, theta)
	new_likelihood = 1 # old_l is < 1, => won't trip the break immediately
	likelihood_path = [old_likelihood]
	max_likelihood = old_likelihood
	max_theta = theta
	epsilon = 1e-3

	k = 1
	while k < 20: #True:

		sys.stdout.write('iteration: %d \r' %k)
		sys.stdout.flush()
		k += 1

		# I could cycle, or go randomly -- but because it's maximization,
		# hitting the same param twice in a row is a waste. 10% chance each
		# time, and that's with such high nD -- so, go sequentially.
		# j = np.random.choice(theta.keys())

		# Just an experiment -- is backwards faster? Maybe slightly.
		# Two cycles seems to do it though. Thoroughly. Even one
		# is pretty good.
		j = [key for key in reversed(theta.keys())][k % nD]
		#j = theta.keys()[k % nD]
		theta = gibbs_step(g, obs, theta, j)
		print theta.values()
		new_likelihood = likelihood(g, obs, theta)
		likelihood_path.append(new_likelihood)

		# Still trying to figure out how, properly, to maximize:
		# Convex => no backtracking.
		if new_likelihood > max_likelihood:
			max_likelihood = new_likelihood
			max_theta = theta

		# This might not be a good stopping criterion --
		# in particular, if the function is non-convex.
		#if abs(new_likelihood - old_likelihood) < epsilon:
		#	break

		old_likelihood = new_likelihood

	return theta, max_theta, likelihood_path



################
#G = nx.star_graph(3)
#for i in [1,2,3]:
#	G[0][i]['weight'] = 1
#for i in [0,1,2,3]:
#	G.node[i]['features'] = np.array([1,1,0])
#single_choice_weight(G, 0, 1, 1)
#single_choice_weight(G, 0, 1, 2)
#single_choice_weight(G, 0, 1, 3)
#obs = [(-1, 1, 0), (-1, 2, 0), (-1, 3, 0)] # P: 1, 1, 1
#print likelihood(G, obs, {1:1, 2:0})# => 0 = 3*log(1)
#obs = [(-1, 0, 1), (-1, 0, 2), (-1, 0, 3)] # P: 1/3, 1/3, 1/3
#print likelihood(G, obs, {1:1, 2:1})# => 1/27 = 3*log(1/3)
################
# The above two look good too
#every_choice_weight(G, 0, [1,2,3], {1:2})
# (This is with features 110,110,100,100 for 0-4)
# array([ 7.3890561 ,  2.71828183,  2.71828183])
#single_choice_probability(G, 0, 1, [1,2,3], {1:2})
# => 0.576 = 7.39/(7.4+2.7+2.7) so, that's good too.


if __name__ == '__main__':

	#######################
	# THE SIMULATION      #
	#######################

	# This is the number of dimensions of the parameter space
	nD = 10
	# This is the number of levels of each parameter
	nL = 10
	# 1/2 the people have the first trait, 1/4 the second, 1/8 the third ...
	# NO -- that scheme was quite difficult to learn. Next -- ALL 50/50.
	feature_frequencies = np.array([1.0/(2**1) for k in range(1, nD+1)])
	R = 1/feature_frequencies #"rarity" score
	# The parameters of how many bits you observe, for a given closeness.
	# I NEED TO RANDOMIZE THIS so that I don't just fit 1:10.
	theta_true = {i:i for i in range(1, nL+1)}

	#######################
	# DEFINING THE GRAPH  #
	#######################

	n = 1000 # so as to have an expected 10 people with the rarest feature
	p = 10.0/n #=> on average, you have 100 friends.
	G = nx.erdos_renyi_graph(n, p) #as a simple first step
	# Give every edge a weight (familiarity)
	for e in G.edges():
		G[e[0]][e[1]]['weight'] = int(np.random.uniform(1,nL+1))
	# Give every node its features
	for n in G.nodes():
		# +0 casts from booleans to integers -- because NP.BOOLEANS DON'T ADD
		G.node[n]['features'] =\
			(np.random.uniform(size=nD) < feature_frequencies) + 0

	# What I'd LIKE to do: starting with the target,
	# (who's all ones) have other people's features fade slowly
	# away, layer of remove by layer -- to simulate homophily.

	# Also,for the PHILO part of homophily, have edge weights
	# be stronger when ... you have more in common?




	# Get the largest connected component (the list is sorted)
	G = nx.connected_component_subgraphs(G)[0]

	# Where the sampling paths start:
	seeds = np.random.choice(G.nodes(), size=100 , replace=False)
	# The target can't be chosen as both seed and target this way.
	target = seeds[-1]
	seeds = seeds[:-1]

	# In this way, all the binary features can be thought of as
	# "has this in common with the target"
	G.node[target]['features'] = np.ones(nD)

	# THINGS STILL TO DO:
	# Have homophily. I.e., if you have a feature,
	# you're more likely to be connected to somebody else with it.
	# THIS, of course, is the REASON THIS SEARCH STRATEGY IS USED.

	####################
	# MAKING THE DATA  #
	####################

	path_dict = {seed:[] for seed in seeds}
	for seed in seeds:
		#print "seed: ", seed
		path_dict[seed] = find_target(G, seed, target, [-1], theta_true)
	# GOD this line is ugly. Does the job though. Good function to know.
	# http://stackoverflow.com/questions/18642428/
	# concatenate-an-arbitrary-number-of-lists-in-a-function-in-python
	observations = list(itertools.chain.from_iterable(
					[parse_into_observations(path_dict[k])
		 				for k in path_dict
					]
				))

	# The fitting itself
	#theta_last, theta_mle, likelihoods = fit_by_metropolis(G, observations,
	#														1, 100)
	#tlast, tmle, lks                   = fit_by_metropolis(G, observations,
	# 														0, 100)

	# And the accuracy:
	#print "theta_true: ", theta_true.values()
	#print "theta_last: ", theta_last.values()
	#print "theta_mle:  ", theta_mle.values()
	#print "tlast: ", tlast.values()
	#print "tmle:  ", tmle.values()

	theta_gibbs, theta_gibbs_mle, likelihoods = fit_by_gibbs(G, observations)
	print "theta_true:       ", theta_true.values()
	print "theta_gibbs:      ", theta_gibbs.values()
	print "theta_gibbs_mle:  ", theta_gibbs_mle.values()
	print len(observations)
	print np.median([len(path_dict[p]) for p in path_dict])
	plt.plot(likelihoods)
	#plt.plot(lks)
	plt.show()
