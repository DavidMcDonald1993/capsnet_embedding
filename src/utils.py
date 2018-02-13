import random
import numpy as np
import networkx as nx
import scipy as sp
import pandas as pd 

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, precision_recall_curve, average_precision_score
from sklearn.metrics.pairwise import pairwise_distances

from itertools import izip_longest

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.callbacks import Callback

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
	G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")

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

def split_data(G, X, Y, split=0.2):

	num_samples = X.shape[0]
	training_size = int(num_samples * (1-split))
	validation_size = num_samples - training_size

	# training_samples = np.random.choice(np.arange(num_samples), replace=False, size=training_size)
	# validation_samples = np.setdiff1d(np.arange(num_samples), training_samples)
	n = np.random.choice(np.arange(num_samples))
	training_samples = [n]
	while len(training_samples) < training_size:
		n = np.random.choice(list(G.neighbors(n)))
		if n not in training_samples:
			training_samples.append(n)

	training_samples = np.array(training_samples)
	validation_samples= np.setdiff1d(np.arange(num_samples), training_samples)

	training_samples = np.append(training_samples, list(nx.isolates(G.subgraph(validation_samples))))
	validation_samples= np.setdiff1d(np.arange(num_samples), training_samples)

	training_samples = sorted(training_samples)
	validation_samples = sorted(validation_samples)

	X_train = X[training_samples]
	Y_train = Y[training_samples]
	G_train = nx.Graph(G.subgraph(training_samples))


	X_val = X[validation_samples]
	Y_val = Y[validation_samples]
	G_val = nx.Graph(G.subgraph(validation_samples))

	return (X_train, Y_train, G_train), (X_val, Y_val, G_val)

def remove_edges(G, number_of_edges_to_remove):

	# print number_of_edges_to_remove
	# print nx.is_connected(G), nx.number_connected_components(G), len(G), len(G.edges)
	# raise SystemExit

	N = len(G)
	removed_edges = []
	edges = list(G.edges())
	random.shuffle(edges)

	for u, v in edges:

		if len(removed_edges) == number_of_edges_to_remove:
			# print "BREAKING"
			break

		if G.degree(u) > 1 and G.degree(v) > 1:
			G.remove_edge(u, v)
			i = min(u, v)
			j = max(u, v)
			removed_edges.append((i, j))
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
					if walk[i] == walk[j]:
						continue
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

		neighbourhood_sample_list.append(np.array([np.concatenate([np.append(n, np.random.choice(neighbours[n], 
			replace=True, size=neighbourhood_sample_size)) for n in batch]) for batch in neighbourhood_sample_list[-1]]))

	# flip neighbour list
	neighbourhood_sample_list = neighbourhood_sample_list[::-1]


	return neighbourhood_sample_list



# def neighbourhood_sample_generator(G, X, Y, neighbourhood_sample_sizes, num_capsules_per_layer,
# 	num_positive_samples, num_negative_samples, context_size, batch_size, p, q, num_walks, walk_length,
# 	num_samples_per_class=20):
	
# 	'''
# 	performs node2vec style neighbourhood sampling for positive samples.
# 	negative samples are selected according to degree
# 	uniform sampling of neighbours for aggregation

# 	'''
# 	'''
# 	PRECOMPUTATION
	
# 	'''
# 	# print "OLD", max(G.nodes)
# 	G = nx.convert_node_labels_to_integers(G)
# 	# print "NEW", max(G.nodes)
# 	# return

# 	num_classes = Y.shape[1]
# 	if num_samples_per_class is not None:
# 		label_mask = compute_label_mask(Y, num_patterns_to_keep=num_samples_per_class)
# 	else:
# 		label_mask = np.ones(Y.shape)

# 	label_prediction_layers = np.where(num_capsules_per_layer==num_classes)[0] + 1
	
# 	neighbours = {n: list(G.neighbors(n)) for n in list(G.nodes())}

# 	num_layers = neighbourhood_sample_sizes.shape[0]

# 	node2vec_sampler = generate_samples_node2vec(G, num_positive_samples, num_negative_samples, context_size, 
# 		p, q, num_walks, walk_length)
# 	batch_sampler = grouper(batch_size, node2vec_sampler)
	
# 	'''
# 	END OF PRECOMPUTATION
# 	'''
	
# 	while True:

# 		batch_nodes = batch_sampler.next()
# 		batch_nodes = np.array(batch_nodes)
# 		neighbour_list = create_neighbourhood_sample_list(batch_nodes, neighbourhood_sample_sizes, neighbours)

# 		# shape is [batch_size, output_shape*prod(sample_sizes), D]
# 		x = X[neighbour_list[0]]
# 		x = np.expand_dims(x, 2)
# 		# shape is now [batch_nodes, output_shape*prod(sample_sizes), 1, D]


# 		negative_sample_targets = [Y[nl].argmax(axis=-1) for nl in neighbour_list[1:]]

# 		labels = []
# 		for layer in label_prediction_layers:
# 			y = Y[neighbour_list[layer]]
# 			mask = label_mask[neighbour_list[layer]]
# 			y_masked = np.append(mask, y, axis=-1)
# 			labels.append(y_masked)

# 		if all([(y_masked[:,:,:num_classes] > 0).any() for y_masked in labels]):
# 			yield x, labels + negative_sample_targets

def perform_embedding(G, X, neighbourhood_sample_sizes, embedder):

	# print "Performing embedding"

	nodes = np.arange(len(G)).reshape(-1, 1)
	neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
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

# def compute_precision_and_recall(i, sorted_candidates, removed_edges):
	# num_candidates = len(sorted_candidates)
	# number_of_removed_edges_above_threshold = float(len(set(sorted_candidates[:i]) & removed_edges))
	# number_of_removed_edges_below_threshold = len(removed_edges) - number_of_removed_edges_above_threshold
	# return number_of_removed_edges_above_threshold / (i + 1e-8), number_of_removed_edges_below_threshold / (num_candidates - i)

def evaluate_link_prediction(G, embedding, removed_edges):

	def check_edge_in_edgelist((u, v), edgelist):
		for u_prime, v_prime in edgelist:
			if u_prime > u:
				return False
			if u_prime == u and v_prime == v:
				edgelist.remove((u, v))
				return True

	def hyperbolic_distance(u, v):
		return np.arccosh(1 + 2 * np.linalg.norm(u - v, axis=-1)**2 / ((1 - np.linalg.norm(u, axis=-1)**2) * (1 - np.linalg.norm(v, axis=-1)**2)))

	def sigmoid(x):
		return 1. / (1 + np.exp(-x))

	N = len(G)

	print  "determining candidate edges"
	removed_edges.sort(key=lambda (u, v): u)
	candidate_edges = [(u, v)for u in range(N) for v in range(u+1, N) if (u, v) not in G.edges() and (v, u) not in G.edges()]
	# candidate_edges.extend(removed_edges)
	zipped_candidate_edges = zip(*candidate_edges)

	print "computing hyperbolic distance between all points"
	hyperbolic_distances = hyperbolic_distance(embedding[zipped_candidate_edges[0],:], 
		embedding[zipped_candidate_edges[1], :])
	r = 1
	t = 1
	print "converting distances into probabilities"
	y_pred = sigmoid((r - hyperbolic_distances) / t)
	print "determining labels"
	y_true = np.array([1. if check_edge_in_edgelist(edge, removed_edges) else 0 for edge in candidate_edges])

	print "computing precision and recalls"
	average_precision = average_precision_score(y_true, y_pred)
	precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
	f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)

	plt.figure(figsize=(10, 10))
	plt.step(recalls[:-1], precisions[:-1], c="b", where="post")
	plt.fill_between(recalls[:-1], precisions[:-1], step='post', alpha=0.2,
					 color='b')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	# plt.ylim([0.0, 1.05])
	# plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
	plt.savefig("../plots/recall-precision.png")
	plt.close()

	plt.figure(figsize=(10, 10))
	plt.plot(thresholds, f1_scores[:-1], c="r")
	plt.xlabel("threshold")
	plt.ylabel("F1 score")
	plt.savefig("../plots/F1.png")
	plt.close()

	return precisions, recalls, f1_scores

# def plot_ROC(precisions, recalls):
# 	plt.figure(figsize=(10, 10))
# 	plt.scatter(recalls, precisions)
# 	plt.xlabel("recall")
# 	plt.ylabel("precision")
# 	plt.savefig("../plots/ROC.png")

class PlotCallback(Callback):

	def __init__(self, G, X, Y, neighbourhood_sample_sizes, embedder, label_map, annotate, path):
		self.G = G 
		self.X = X
		self.Y = Y
		self.neighbourhood_sample_sizes = neighbourhood_sample_sizes
		self.embedder = embedder
		self.label_map = label_map
		self.annotate = annotate
		self.path = path

	def on_epoch_end(self, epoch, logs={}):
		embedding = perform_embedding(self.G, self.X, self.neighbourhood_sample_sizes, self.embedder)
		plot_embedding(embedding, self.Y, self.label_map, self.annotate, path="{}/embedding_epoch_{:04}".format(self.path, epoch))

def plot_embedding(embedding, Y, label_map,annotate=False, path=None):

	# print "Plotting network and saving to {}...".format(path)

	y = Y.argmax(axis=1)
	embedding_dim = embedding.shape[-1]

	fig = plt.figure(figsize=(10, 10))
	if embedding_dim == 3:
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=y)
	else:
		plt.scatter(embedding[:,0], embedding[:,1], c=y)
		if annotate:
			for label, p in zip(range(embedding.shape[0]), embedding[:,:2]):
				plt.annotate(label, p)
		if label_map is not None:
			present_classes = np.unique(y)
			representatives = np.array([np.where(y==c)[0][0] for c in present_classes])
			present_class_names = [label_map[c] for c in present_classes]
			for c, label, p in zip(present_classes, present_class_names, embedding[representatives, :2]):
				# print present_classes
				plt.annotate(label, p)#, c=c)
	# plt.show()
	if path is not None:
		plt.savefig(path)
	plt.close()

	# print "Done"
	
def make_and_evaluate_label_predictions(G, X, Y, predictor, num_capsules_per_layer, neighbourhood_sample_sizes, batch_size):

	_, num_classes = Y.shape
	label_prediction_layers = np.where(num_capsules_per_layer==num_classes)[0] + 1

	nodes = np.arange(len(G)).reshape(-1, 1)
	neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
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


