# make a graph with k types of nodes
# have transition probabilities for all of them, stay, transition, complete.
# mumble mumble actual implementation ...
# compare the true shortest path distribution with the Markov-estimated one.
# details will be on -- degree distribution, which I'll have to make explicit
# because I have to creat the links that go ... whatsit, to each plcae. So,
# make it regular? To begin with -- i.e. 5 edges per node, and ... as where
# they go, let's say it's random, within the groups in question. Right.
# FUCK, ah, but ... okay, everybody has an OUT degree of five -- rest is
# determined by the process. Also -- I'll need to actually observe the
# true target distance distribution, AND ... um ... see how other kinds of
# graphs are represented by this? OH -- um ... does the similarity
# structure (more similar you are, closer you are) part actually apply?

# question that will be important at some point -- how to actually wire a
# network that actually FITS the single-choice-model conception on which the
# similarity search is based? The coefficients could also be put into a
# logistic regression, and then edges could be drawn based on the probability
# induced by the similarity. There. I've just got to choose --

import random

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy import cluster as cl

# Needed to obtain data once the graph has been created.
# Has functions such as 'find_target'
import familiarity_models as fm



def logit(x, b):
	# Requires that both be arrays.
	return 1.0/(1 + np.exp(-np.inner(x, b)))


# This function is large. Should it have helper functions?
def create_homophily_graph(n, k, density, theta=None):
	"""

	Notes: edges are UNdirected, familiarity is ORDERED and 1:10,


	"""

	G = nx.Graph()
	G.add_nodes_from(range(n))
	# Features are stored separately for easy generation,
	# and an easy manipulation later on.
	X = np.int64(np.random.random_sample((n,k)) > 0.5)
	for i in G.nodes():
		# Associating them this way -- creates copies or pointers?
		# Either way, we're saving big, or not doing worse.
		G.node[i]['features'] = X[i] # The i-th row

	# A Familiarity score for every POSSIBLE edge
	F = np.random.randint(1,11, n*(n-1)/2)

	# Make the parameters, if none have been passed
	if not theta:
		theta = np.array([random.choice([-1,1])*random.random()
							for i in range(k)])

	# The indexing for values would be triangular, so ... fuck that. No?
	values = []
	for i in G.nodes():
		for j in range(i+1, n):
			# This familiarity is being treated symmetrically -- should it be?
			values.append(
				logit(
				np.int64(G.node[i]['features']==G.node[j]['features'])[:F[i]],
				theta[:F[i]]
				)
			)

	# Get the ... intercept, to set the desired density
	# This is what what will make the average p equal to 'density'
	intercept = -np.log(1.0/density - 1) - np.mean(values)

	# Another bummer: [intercept] + np.array() doesn't do SHIT
	theta1 = np.insert(theta, 0, intercept)

	# Stick on the values for the intercept, to simplify multiplication
	X1 = np.hstack((np.ones((n,1)), X))

	# WIRING
	for i in G.nodes():
		for j in range(i+1, n):
			if logit(np.int64(X1[i]==X1[j]), theta1) > random.random():
				G.add_edge(i, j)
				G.edge[i][j]['weight'] = F[i]

	# Get the largest connected component (the list is sorted)
	G = nx.connected_component_subgraphs(G)[0]

	return G, theta1, X1


def get_data_from_graph(g, theta, num_seeds=None):
	"""Chooses seeds and a target, traces paths, returns transition tuples.
	"""

	if not num_seeds:
		num_seeds = round(g.number_of_nodes()/10)

	# Where the sampling paths start:
	seeds = np.random.choice(g.nodes(),
							size=num_seeds,
							replace=False)
	# The target can't be chosen as both seed and target this way.
	target = seeds[-1]
	seeds = seeds[:-1]

	####################
	# MAKING THE DATA  #
	####################

	path_dict = {seed:[] for seed in seeds}
	for seed in seeds:
		#print "seed: ", seed
		path_dict[seed] = fm.find_target(g, seed, target, [-1], theta)

	observations = list(itertools.chain.from_iterable(
					[fm.parse_into_observations(path_dict[k])
		 				for k in path_dict
					]
				))

	return observations, seeds, target

def markov_distance_estimation(g, t, num_iterations=100):
	"""Takes a transition graph 'g', returns dist. of times to target 't'

	"""

	nodes = g.nodes()
	distances = np.zeroes((len(nodes), num_iterations))
	for iteration in range(num_iterations):
		for n in nodes:
			distance = 0
			state = n
			while state != t:
				# WILL THESE ALWAYS BE IN A CONSISTENT ORDER?
				weights = [g.edge[state][i]['count'] for i in g[state]]
				state = np.random.choice([i for i in g[state]], p=weights)
				distance += 1
			distances[n,iteration] = distance

	return distances #{k, np.mean(v) for k, v in distances.items()}


def find_target(g, seeds, target, theta):
	""""""
	choices = {}
	optimal_choices = {}
	for seed in seeds:
		at = seed
		previous = -1
		while at != target and at not in choices:
			options = [n for n in g.neighbors(at) if n != previous]
			obs_values = [np.inner(theta,
				((g[j]['x'] and g[at][j]['known'])+0)==g[target]['x'])
				for j in options]
			true_values = [np.inner(theta,
				(g[j]['x']+0)==g[target]['x'])
				for j in options]
			choices[at] = [j for i,j in sorted(zip(obs_values, options))]
			optimal_choices = [j for i,j in sorted(zip(true_values, options))]
			previous = at
			at = choices[at][0]
	return choices, optimal choices


if __name__ == '__main__':



	# What kind of -- what would I LIKE the structure of this to be?
	# Slash, should I be cleaning up? I DO find things a little
	# confusing, already. So. How SHOULD this look?
	# generate_graph
	# get_data_from_graph
	# analyze_data. And that's ... kinda it.
	# make comparisons (and plot them)

	num_nodes = 20
	num_features = 5
	num_clusters = 2

	g, theta, x = create_homophily_graph(num_nodes, num_features, 0.3)
	#nx.draw(g)
	#plt.show()


	# There IS kmeans2, but it balks. We're TOLD to 'whiten' in the docs,
	# but that returns garbage. So, this call.
	centroids, distortion = cl.vq.kmeans(x, num_clusters)
	# Fuck scipy, but this is the call.
	labels, distances = cl.vq.vq(x, centroids)

	#t = {i:i for i in theta}
	f = {i:i for i in range(1,11)}
	observations, seeds, target = get_data_from_graph(g, f, 5)

	# Change the ... transition tuples from node-node to cluster-cluster,
	# then count them.

	# NOW -- we've got data. Now we -- 1) cluster, 2) estimate transition
	# probabilities, 3) use those probabilities to estimate distance
	# distributions, and 4) compare those distributions with the truth.

	model = nx.DiGraph()
	num_states = num_clusters+1
	model.add_nodes_from(range(num_states))
	edges = [(i,j) for i, j in np.ndindex(num_states, num_states) if i!=j]
	model.add_edges_from(edges, count=0)

	for o in observations:
		# The labels (cluster assignments) of the current (1) and next(2)
		# nodes for a particular transition (o)
		k1, k2 = labels[o[1]], labels[o[2]]
		model.edge[k1][k2]['count'] += 1

	# NOW -- ... eh ... ah, estimating expected path lengths
	# given different starting points. Right. Easy enough ...
	distances = markov_distance_estimation(model, target)

	# Now PLOT ... 95% confidence intervals based on the returned
	# distance distributions against the TRUE shortest paths, AND ...
	# The empirically discovered shortest paths.
	#distances = {k:sorted(v) for k, v in distances.items()}
	#intervals = {k: np.mean(v[96] - v[3]) for k, v in distances.items()}
	# plt.errorbar()
	plt.boxplot(distances.T)
	# Also, but the real values on top!
	true_distances = nx.all_pairs_dijkstra_path_length(g)

	plt.show()



	#pos = nx.layout.spring_layout(g)
	#nx.draw_networkx_nodes(g, pos, nodelist=[], node_color="red")
	#nx.draw_networkx_nodes(g, pos, nodelist=[], node_color="blue")
	#plt.show()
