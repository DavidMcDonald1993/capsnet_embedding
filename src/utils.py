import random
import numpy as np
import networkx as nx
import scipy as sp
import pandas as pd 

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import pairwise_distances

from itertools import izip_longest

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from node2vec_sampling import Graph

def load_karate():

	G = nx.karate_club_graph()
	G = nx.convert_node_labels_to_integers(G)

	X = sp.sparse.identity(len(G))

	map_ = {"Mr. Hi" : 0, "Officer" : 1}

	Y = np.zeros((len(G), 2))
	assignments = dict(nx.get_node_attributes(G, "club")).values()
	assignments =[map_[x] for x in assignments]
	Y[np.arange(len(G)), assignments] = 1
	Y = sp.sparse.csr_matrix(Y)


	X = X.toarray()
	Y = Y.toarray()

	return G, X, Y, None

def load_cora():

	G = nx.read_edgelist("../data/cora/cites.tsv", delimiter="\t", )
	G = nx.convert_node_labels_to_integers(G)

	X = sp.sparse.load_npz("../data/cora/cited_words.npz")
	Y = sp.sparse.load_npz("../data/cora/paper_labels.npz")

	X = X.toarray()
	Y = Y.toarray()

	label_name_df = pd.read_csv("../data/cora/paper.tsv", sep="\t", index_col=0)
	label_names = label_name_df["class_label"].unique()
	label_name_map = {i: label_name for i, label_name in enumerate(label_names) }

	return G, X, Y, label_name_map

def load_facebook():

	G = nx.read_gml("../data/facebook/facebook_graph.gml",  )
	G = nx.convert_node_labels_to_integers(G)

	X = sp.sparse.load_npz("../data/facebook/features.npz")
	Y = sp.sparse.load_npz("../data/facebook/circle_labels.npz")

	X = X.toarray()
	Y = Y.toarray()

	return G, X, Y, None

def preprocess_data(X):
	# X = VarianceThreshold().fit_transform(X)
	X = StandardScaler().fit_transform(X)
	return X

def remove_edges(G, number_of_edges_to_remove):

	# print number_of_edges_to_remove
	# print nx.is_connected(G), nx.number_connected_components(G), len(G), len(G.edges)
	# raise SystemExit

	N = len(G)
	removed_edges = set()
	edges = list(G.edges)
	random.shuffle(edges)

	for u, v in edges:

		if len(removed_edges) == number_of_edges_to_remove:
			# print "BREAKING"
			break

		if G.degree(u) > 1 and G.degree(v) > 1:
			G.remove_edge(u, v)
			i = min(u, v)
			j = max(u, v)
			removed_edges.add((i, j))
			print "removed edge {}: {}".format(len(removed_edges), (i, j))

	return G, removed_edges


def connect_layers(layer_tuples, x):
	
	y = x

	for layer_tuple in layer_tuples:
		for layer in layer_tuple:
			y = layer(y)

	return y

def compute_label_mask(Y, num_patterns_to_keep=20):

	assignments = Y.argmax(axis=1)
	patterns_to_keep = np.concatenate([np.random.choice(np.where(assignments==i)[0], replace=False, size=num_patterns_to_keep)  
										 for i in range(Y.shape[1])])
	mask = np.zeros(Y.shape, dtype=np.float32)
	mask[patterns_to_keep] = 1

	return mask

def generate_samples_node2vec(G, num_positive_samples, num_negative_samples, context_size,
	p, q, num_walks, walk_length):

	nx.set_edge_attributes(G, 1, "weight")
	
	N = nx.number_of_nodes(G)

	frequencies = np.array(dict(G.degree()).values()) ** 0.75

	node2vec_graph = Graph(nx_G=G, is_directed=False, p=p, q=q)
	node2vec_graph.preprocess_transition_probs()

	while True:
		walks = node2vec_graph.simulate_walks(num_walks=num_walks, walk_length=walk_length)
		for walk in walks:

			possible_negative_samples = np.setdiff1d(np.arange(N), walk)
			possible_negative_sample_frequencies = frequencies[possible_negative_samples]
			possible_negative_sample_frequencies /= possible_negative_sample_frequencies.sum()
			# negative_samples = np.random.choice(possible_negative_samples, replace=True, size=num_negative_samples, 
			# 				p=possible_negative_sample_frequencies / possible_negative_sample_frequencies.sum())

			for i in range(len(walk)):
				for j in range(i+1, min(len(walk), i+1+context_size)):
					pair = np.array([walk[i], walk[j]])
					for negative_samples in np.random.choice(possible_negative_samples, replace=True, 
						size=(1+num_positive_samples, num_negative_samples),
						p=possible_negative_sample_frequencies):
						yield np.append(pair, negative_samples)
						pair = pair[::-1]

def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def create_neighbourhood_sample_list(nodes, neighbourhood_sample_sizes, neighbours):

	neighbourhood_sample_list = [nodes]

	for neighbourhood_sample_size in neighbourhood_sample_sizes[::-1]:
			neighbourhood_sample_list.append(
				np.array([
				np.concatenate([ 
					np.append(n, np.random.choice(neighbours[n], replace=True, size=neighbourhood_sample_size)) 
								for n in batch]) 
				for batch in neighbourhood_sample_list[-1]]))

	# flip neighbour list
	neighbourhood_sample_list = neighbourhood_sample_list[::-1]


	return neighbourhood_sample_list



def neighbourhood_sample_generator(G, X, Y, neighbourhood_sample_sizes, num_capsules_per_layer,
	num_positive_samples, num_negative_samples, context_size, batch_size, p, q, num_walks, walk_length,
	num_samples_per_class=20):
	
	'''
	performs node2vec style neighbourhood sampling for positive samples.
	negative samples are selected according to degree
	uniform sampling of neighbours for aggregation

	'''
	'''
	PRECOMPUTATION
	
	'''

	num_classes = Y.shape[1]
	if num_samples_per_class is not None:
		label_mask = compute_label_mask(Y, num_patterns_to_keep=num_samples_per_class)
	else:
		label_mask = np.ones(Y.shape)

	label_prediction_layers = np.where(num_capsules_per_layer==num_classes)[0] + 1
	
	neighbours = [list(G.neighbors(n)) for n in list(G.nodes())]

	num_layers = neighbourhood_sample_sizes.shape[0]

	node2vec_sampler = generate_samples_node2vec(G, num_positive_samples, num_negative_samples, context_size, 
		p, q, num_walks, walk_length)
	batch_sampler = grouper(batch_size, node2vec_sampler)
	
	'''
	END OF PRECOMPUTATION
	'''
	
	while True:

		batch_nodes = batch_sampler.next()
		batch_nodes = np.array(batch_nodes)

		neighbour_list = create_neighbourhood_sample_list(batch_nodes, neighbourhood_sample_sizes, neighbours)

		# shape is [batch_size, output_shape*prod(sample_sizes), D]
		x = X[neighbour_list[0]]
		x = np.expand_dims(x, 2)
		# shape is now [batch_nodes, output_shape*prod(sample_sizes), 1, D]


		negative_sample_targets = [Y[nl].argmax(axis=-1) for nl in neighbour_list[1:]]

		labels = []
		for layer in label_prediction_layers:
			y = Y[neighbour_list[layer]]
			mask = label_mask[neighbour_list[layer]]
			y_masked = np.append(mask, y, axis=-1)
			labels.append(y_masked)

		if all([(y_masked[:,:,:num_classes] > 0).any() for y_masked in labels]):
			yield x, labels + negative_sample_targets

def perform_embedding(G, X, neighbourhood_sample_sizes, embedder):

	print "Performing embedding"

	nodes = np.arange(len(G)).reshape(-1, 1)
	neighbours = [list(G.neighbors(n)) for n in list(G.nodes())]
	neighbour_list = create_neighbourhood_sample_list(nodes, neighbourhood_sample_sizes, neighbours)

	x = X[neighbour_list[0]]
	# print x.shape
	x = np.expand_dims(x, 2)
	# print x.shape

	embedding = embedder.predict(x)
	# print embedding.shape
	dim = embedding.shape[-1]
	embedding = embedding.reshape(-1, dim)

	return embedding

def hyperbolic_distance(u, v):
	return np.arccosh(1 + 2 * np.linalg.norm(u - v, axis=-1)**2 / (1 - np.linalg.norm(u, axis=-1)**2) * (1 - np.linalg.norm(v, axis=-1)**2))

def evaluate_link_prediction(G, embedding, removed_edges):

	N = len(G)

	print "computing hyperbolic distance between all points"
	candidate_edges = np.array([(u, v)for u in range(N) for v in range(u+1, N) if (u, v) not in G.edges and (v, u) not in G.edges])
	# print candidate_edges
	hyperbolic_distances = hyperbolic_distance(embedding[candidate_edges[:,0]], embedding[candidate_edges[:,1]])
	# print hyperbolic_distances[:10]
	# print "DONE"
	# raise SystemExit
	num_candidates = candidate_edges.shape[0]

	candidates = {(candidate_edges[i,0], candidate_edges[i,1]) : hyperbolic_distances[i] for i in range(num_candidates)}
	sorted_candidates = sorted(candidates, key=candidates.get)

	precisions = np.zeros(num_candidates)
	recalls = np.zeros(num_candidates)

	print "computing precision and recalls"

	for i in range(num_candidates):
		
		if i > 0:
			number_of_removed_edges_above_threshold = float(len(set(sorted_candidates[:i]) & removed_edges))
			precisions[i] = number_of_removed_edges_above_threshold / i

		if i < num_candidates-1:	
			number_of_removed_edges_below_threshold = float(len(set(sorted_candidates[i:]) & removed_edges))
			recalls[i] = number_of_removed_edges_below_threshold / (num_candidates - i)

	return precisions, recalls

def plot_ROC(precisions, recalls):
	plt.figure(figsize=(10, 10))
	plt.scatter(recalls, precisions)
	plt.xlabel("recall")
	plt.ylabel("precision")
	plt.savefig("../plots/ROC.png")

def plot_embedding(embedding, Y, label_map,annotate=False, path=None):

	print "Plotting network and saving to {}...".format(path)

	y = Y.argmax(axis=1)
	embedding_dim = embedding.shape[-1]

	fig = plt.figure(figsize=(10, 10))
	if embedding_dim == 3:
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=y)
	else:
		plt.scatter(embedding[:,0], embedding[:,1], c=y)
		if annotate:
			for label, p in zip(list(G), embedding[:,:2]):
				plt.annotate(label, p)
		if label_map is not None:
			present_classes = np.unique(y)
			representatives = np.array([np.where(y==c)[0][0] for c in present_classes])
			present_classes = [label_map[c] for c in present_classes]
			for label, p in zip(present_classes, embedding[representatives, :2]):
				plt.annotate(label, p)
	plt.show()
	if path is not None:
		plt.savefig(path)

	print "Done"
	
def make_and_evaluate_label_predictions(G, X, Y, predictor, num_capsules_per_layer, neighbourhood_sample_sizes, batch_size):

	_, num_classes = Y.shape
	label_prediction_layers = np.where(num_capsules_per_layer==num_classes)[0] + 1

	nodes = np.arange(len(G)).reshape(-1, 1)
	neighbours = [list(G.neighbors(n)) for n in list(G.nodes())]
	neighbour_list = create_neighbourhood_sample_list(nodes, neighbourhood_sample_sizes[:label_prediction_layers[-1]], neighbours)

	x = X[neighbour_list[0]]
	# print x.shape
	x = np.expand_dims(x, 2)
	# print x.shape

	predictions = predictor.predict(x, batch_size=batch_size)
	predictions = predictions.reshape(-1, predictions.shape[-1])

	true_labels = Y.argmax(axis=-1)
	predicted_labels = predictions.argmax(axis=-1)

	print "NMI of predictions: {}".format(normalized_mutual_info_score(true_labels, predicted_labels))
	print "Classification accuracy: {}".format((true_labels==predicted_labels).sum() / float(true_labels.shape[0]))


