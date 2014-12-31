import numpy as np
from matplotlib import pyplot as plt

# Make a set of parameters to be recovered: random variables, uniform
# between -1 and 1.
# Only be able to see the ORDERINGS of SUMMED SUBSETS of these variables,
# I.e. a < b+c, c < a+b+c, etc. TAKING that ... what can you learn?
# To see what vectors satisfy these constraints, sample and reject.

K = 3
N = 30
ntrials = 1000
theta = np.random.uniform(-1, 1, K)
# This matrix represents the presence or absence of the variables:
# 1,1,0 = a,b, not c.
lhs = np.random.randint(2, size=(N,K))
# Each row will be compared with the row below it
# This simply shifts all the rows down one
rhs = np.vstack((lhs[-1],lhs[:-1]))
lhs_rhs = lhs - rhs
# Finds the indices of the comparisons where lhs <= rhs
less_than_pairs = [i for i in range(N) if np.inner(theta, lhs_rhs[i]) <= 0]
# Ah, fuck -- theta's supposed to be positive. So ... ? Well -- for LP.
# Let's see what happens in this case.
trials = np.random.uniform(-1, 1, (ntrials,K))
good_trials = [i for i in range(ntrials) if
                np.all(np.dot(lhs_rhs[less_than_pairs], trials[i]) <=0)]

print "theta: ", theta
print "hat: ", np.mean(trials[good_trials], axis=0)
