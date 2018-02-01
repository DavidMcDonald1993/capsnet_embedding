import numpy as np
import networkx as nx
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from node2vec_sampling import Graph

def load_karate():

	G = nx.karate_club_graph()

	nx.set_edge_attributes(G, name="weight", values=1)

	X = sp.sparse.identity(len(G))

	map_ = {"Mr. Hi" : 0, "Officer" : 1}

	Y = np.zeros((len(G), 2))
	assignments = dict(nx.get_node_attributes(G, "club")).values()
	assignments =[map_[x] for x in assignments]
	Y[np.arange(len(G)), assignments] = 1
	Y = sp.sparse.csr_matrix(Y)

	return G, X, Y

def load_cora():

	G = nx.read_edgelist("../data/cora/cites.tsv", delimiter="\t", )
	
	nx.set_edge_attributes(G, name="weight", values=1)

	X = sp.sparse.load_npz("../data/cora/cited_words.npz")
	Y = sp.sparse.load_npz("../data/cora/paper_labels.npz")

	return G, X, Y

def compute_negative_sampling_mask(batch_size, n, spacing):
	mask = np.zeros((batch_size, n*spacing))
	mask[:, np.arange(n) * spacing] = 1. / n
	return mask

def compute_label_mask(Y, p=20):

	assignments = Y.argmax(axis=1)
	patterns_to_remove = np.concatenate([np.random.permutation(np.where(assignments==i)[0])[:-p] 
										 for i in range(Y.shape[1])])
	mask = np.ones(Y.shape, dtype=np.float32)
	mask[patterns_to_remove] = 0

	return mask

def neighbourhood_sample_generator(G, X, Y, sample_sizes, num_positive_samples, num_negative_samples, batch_size):
	
	'''
	performs node2vec style neighbourhood sampling for positive samples.
	negative samples are selected according to degree
	uniform sampling of neighbours for aggregation

	'''
	'''
	PRECOMPUTATION
	
	'''

	X = X.toarray()
	Y = Y.toarray()

	# print X.shape, X.mean(axis=0).shape, X.std(axis=0).shape, X.std(axis=0).min()

	X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
	# print X.mean(axis=0)
	# print X.std(axis=0)
	# print X[0]
	# raise SystemExit

	G = nx.convert_node_labels_to_integers(G)
	
	N = len(G)
	
	node2vec_graph = Graph(nx_G=G, is_directed=False, p=1, q=1)
	node2vec_graph.preprocess_transition_probs()
	walks = node2vec_graph.simulate_walks(num_walks=1, walk_length=15)
	
	neighbours = [list(G.neighbors(n)) for n in list(G.nodes())]
	
	frequencies = np.array(dict(G.degree()).values()) ** 0.75

	num_aggregation_layers = sample_sizes.shape[0]
	
	'''
	END OF PRECOMPUTATION
	'''
	
	i = 0
	nodes = np.random.permutation(N)
	output_dimension = 1 + num_positive_samples + num_negative_samples

	# negative_sample_target = np.empty((batch_size, num_positive_samples+num_negative_samples))

	label_mask = compute_label_mask(Y)
	
	while True:
		
		batch_nodes = np.zeros((batch_size, output_dimension), dtype=int)
		
		for j in range(batch_size):
			
			if i == N:
				nodes = np.random.permutation(N)
				i = 0
				
			n = nodes[i]
			
			positive_samples = np.random.choice(walks[n], replace=True, size=num_positive_samples)
			possible_negative_samples = np.setdiff1d(range(N), walks[n])
			possible_negative_sample_frequencies = frequencies[possible_negative_samples]
			negative_samples = np.random.choice(possible_negative_samples, replace=True, size=num_negative_samples, 
							p=possible_negative_sample_frequencies / possible_negative_sample_frequencies.sum())
													 
			batch_nodes[j, 0] = n
			batch_nodes[j, 1 : 1 + num_positive_samples] = positive_samples
			batch_nodes[j, 1 + num_positive_samples : ] = negative_samples
		
			i += 1
	
		neighbour_list = [batch_nodes]
		for sample_size in sample_sizes:
			neighbour_list.append(np.array([
				np.concatenate([ np.append(n, np.random.choice(neighbours[n], replace=True, size=sample_size)) 
								for n in batch]) for batch in neighbour_list[-1]]))

		# for nl in neighbour_list:
		# 	print nl.shape

		# raise SystemExit

		# shape is [batch_size, output_shape*prod(sample_sizes), D]
		x = X[neighbour_list[-1]]
		x = np.expand_dims(x, 2)
		# shape is now [batch_nodes, output_shape*prod(sample_sizes), 1, D]

		y_label_mask = label_mask[neighbour_list[1]]
		if (y_label_mask>0).any():
			y_label = np.append(y_label_mask, Y[neighbour_list[1]], axis=-1)

			negative_sample_targets = [Y[nl].argmax(axis=-1) for nl in neighbour_list[-2::-1]]

			yield x, [y_label] + negative_sample_targets
			# yield x, negative_sample_targets

def draw_embedding(embedder, generator, dim=2):

	x, yl = generator.next()
	# x, [_, y, _, _] = generator.next()

	y = yl[-1]

	embedding = embedder.predict(x)
	embedding = embedding.reshape(-1, dim)

	# num_classes = y.shape[-1] / 2
	# assignments = y[:,:,num_classes:].argmax(axis=-1).flatten()

	fig = plt.figure(figsize=(5, 5))
	if dim == 3:
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=y.flatten())
	else:
		plt.scatter(embedding[:,0], embedding[:,1], c=y.flatten())
	plt.show()
	

