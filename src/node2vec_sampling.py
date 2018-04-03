import numpy as np
import scipy as sp
import networkx as nx
import random

class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		# self.A = nx.adjacency_matrix(nx_G).astype(np.float32)
		# self.A_with_self_links = self.A + sp.sparse.identity(self.A.shape[0])
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			# print cur
			cur_nbrs = sorted(G.neighbors(cur))
			# cur_nbrs = sorted(A[cur].nonzero()[1])
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
					# probs = self.compute_node_probs(cur)
					# walk.append(alias_draw(probs))
				else:
					prev = walk[-2]
					next_ = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					# probs = self.compute_edge_probs(prev, cur)
					# next = alias_draw(probs)
					walk.append(next_)
			else:
				break

		return walk

	# def compute_node_probs(self, node):

	# 	A = self.A 
	# 	unnormalized_probs = A[node].toarray().flatten()
	# 	normalized_probs = unnormalized_probs / unnormalized_probs.sum()
	# 	return normalized_probs


	# def compute_edge_probs(self, prev, cur):

	# 	A = self.A
	# 	A_with_self_links = self.A_with_self_links
	# 	p = self.p
	# 	q = self.q

	# 	unnormalized_probs = A[cur].toarray().flatten()
	# 	unnormalized_probs[prev] /= p

	# 	not_connected_to_prev = A_with_self_links[prev].toarray().flatten() == 0#.toarray().flatten()
	# 	unnormalized_probs[not_connected_to_prev] /= q

	# 	normalized_probs = unnormalized_probs / unnormalized_probs.sum()

	# 	return normalized_probs

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		i = 0
		for walk_iter in range(num_walks):
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
				if i % 1000 == 0:
					print ("peformed walk {}/{}".format(i, num_walks*len(G)))
				i += 1

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		# A = self.A
		# A_with_self_links = self.A_with_self_links
		p = self.p
		q = self.q

		# unnormalized_probs = A[dst].toarray().flatten()
		# unnormalized_probs[src] /= p

		# not_connected_to_src = A_with_self_links[src].toarray().flatten() == 0#.toarray().flatten()
		# unnormalized_probs[not_connected_to_src] /= q

		# normalized_probs = unnormalized_probs / unnormalized_probs.sum()
		# return normalized_probs

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)


	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		print ("preprocessing transition probs")
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		i = 0
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

			# unnormalized_probs = self.A[node].toarray().flatten()
			# normalized_probs = unnormalized_probs / unnormalized_probs.sum()
			# alias_nodes[node] = normalized_probs
			alias_nodes[node] = alias_setup(normalized_probs)
			if i % 1000 == 0:
				print ("completed node {}/{}".format(i, len(G)))
			i += 1

		print ("completed node {}/{}".format(i, len(G)))

		alias_edges = {}
		# triads = {}

		print ("DONE nodes")

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			i = 0
			for edge in G.edges():
				if i % 1000 == 0:
					print ("completed edge {}/{}".format(i, 2*len(G.edges())))
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				i += 1
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
				i += 1
			print ("completed edge {}/{}".format(i, 2*len(G.edges())))

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		print ("DONE edges")

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]

# def alias_draw(probs):
# 	N = len(probs)
# 	return np.random.choice(N, p=probs)