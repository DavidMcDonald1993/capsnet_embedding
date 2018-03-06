import sys
import os
import random
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx

import pickle as pkl

from sklearn.preprocessing import StandardScaler



def load_data_gcn(dataset_str):
	"""Load data."""

	def parse_index_file(filename):
		"""Parse index file."""
		index = []
		for line in open(filename):
			index.append(int(line.strip()))
		return index

	def sample_mask(idx, l):
		"""Create mask."""
		mask = np.zeros(l)
		mask[idx] = 1
		return np.array(mask, dtype=np.bool)

	names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
	objects = []
	for i in range(len(names)):
		with open("../data/labelled_attributed_networks/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
			if sys.version_info > (3, 0):
				objects.append(pkl.load(f, encoding='latin1'))
			else:
				objects.append(pkl.load(f))

	x, y, tx, ty, allx, ally, graph = tuple(objects)
	test_idx_reorder = parse_index_file("../data/labelled_attributed_networks/ind.{}.test.index".format(dataset_str))
	test_idx_range = np.sort(test_idx_reorder)

	if dataset_str == 'citeseer':
		# Fix citeseer dataset (there are some isolated nodes in the graph)
		# Find isolated nodes, add them as zero-vecs into the right position
		test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
		tx_extended = sp.sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
		tx_extended[test_idx_range-min(test_idx_range), :] = tx
		tx = tx_extended
		ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
		ty_extended[test_idx_range-min(test_idx_range), :] = ty
		ty = ty_extended

	features = sp.sparse.vstack((allx, tx)).tolil()
	features[test_idx_reorder, :] = features[test_idx_range, :]
	adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

	labels = np.vstack((ally, ty))
	labels[test_idx_reorder, :] = labels[test_idx_range, :]

	idx_test = test_idx_range.tolist()
	idx_train = range(len(y))
	idx_val = range(len(y), len(y)+500)

	train_mask = sample_mask(idx_train, labels.shape[0])
	val_mask = sample_mask(idx_val, labels.shape[0])
	test_mask = sample_mask(idx_test, labels.shape[0])

	# y_train = np.zeros(labels.shape)
	# y_val = np.zeros(labels.shape)
	# y_test = np.zeros(labels.shape)
	# y_train[train_mask, :] = labels[train_mask, :]
	# y_val[val_mask, :] = labels[val_mask, :]
	# y_test[test_mask, :] = labels[test_mask, :]

	# return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels

	G = nx.from_numpy_array(adj.toarray())
	G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")
	nx.set_edge_attributes(G=G, name="weight", values=1)

	X = features.toarray()
	Y = labels

	X = preprocess_data(X)

	val_file = "../data/labelled_attributed_networks/{}_val_edges.pkl".format(dataset_str)
	test_file = "../data/labelled_attributed_networks/{}_test_edges.pkl".format(dataset_str)

	if not os.path.exists(val_file):

		number_of_edges_to_remove = int(len(G.edges) * 0.2)

		G, val_edges = remove_edges(G, number_of_edges_to_remove)
		G, test_edges = remove_edges(G, number_of_edges_to_remove)

		with open(val_file, "wb") as f:
			pkl.dump(val_edges, f)
		with open(test_file, "wb") as f:
			pkl.dump(test_edges, f)
	else:
		with open(val_file, "rb") as f:
			val_edges = pkl.load(f) 
		with open(test_file, "rb") as f:
			test_edges = pkl.load(f)

		G.remove_edges_from(val_edges)
		G.remove_edges_from(test_edges)

	train_mask = train_mask.reshape(-1, 1).astype(np.float32)
	val_mask = train_mask.reshape(-1, 1).astype(np.float32)
	test_mask = train_mask.reshape(-1, 1).astype(np.float32)

	return G, X, Y, val_edges, test_edges, train_mask, val_mask, test_mask

	# if not os.path.exists("../data/labelled_attributed_networks/{}_training_idx".format(dataset_str)):
	# 	training_idx, val_idx = split_data(G, X, Y, split=0.3)
	# 	np.savetxt(X=training_idx, fname="../data/labelled_attributed_networks/{}_training_idx".format(dataset_str), delimiter=" ", fmt="%i")
	# 	np.savetxt(X=val_idx, fname="../data/labelled_attributed_networks/{}_val_idx".format(dataset_str), delimiter=" ", fmt="%i")
	# else:
	# 	training_idx = np.genfromtxt("../data/labelled_attributed_networks/{}_training_idx".format(dataset_str), delimiter=" ", dtype=np.int)
	# 	val_idx = np.genfromtxt("../data/labelled_attributed_networks/{}_val_idx".format(dataset_str), delimiter=" ", dtype=np.int)

	# X_train = X[training_idx]
	# Y_train = Y[training_idx]
	# G_train = nx.Graph(G.subgraph(training_idx))
	# G_train = nx.convert_node_labels_to_integers(G_train, ordering="sorted")

	# X_val = X[val_idx]
	# Y_val = Y[val_idx]
	# G_val = nx.Graph(G.subgraph(val_idx))
	# G_val = nx.convert_node_labels_to_integers(G_val, ordering="sorted")

	# return (G, X, Y), (G_train, X_train, Y_train), (G_val, X_val, Y_val)




def load_citation_network(dataset_str):
	assert dataset_str in ["AstroPh", "CondMat", "GrQc", "HepPh"], "dataset string is not valid"

	G = nx.read_edgelist("../data/collaboration_networks/ca-{}.txt.gz".format(dataset_str))
	G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")
	nx.set_edge_attributes(G=G, name="weight", values=1)

	N = len(G)

	X = sp.sparse.identity(N, format="csr")
	Y = np.ones((N, 1))

	# X = preprocess_data(X)

	val_file = "../data/collaboration_networks/{}_val_edges.pkl".format(dataset_str)
	test_file = "../data/collaboration_networks/{}_test_edges.pkl".format(dataset_str)

	if not os.path.exists(val_file):

		number_of_edges_to_remove = int(len(G.edges) * 0.2)

		G, val_edges = remove_edges(G, number_of_edges_to_remove)
		G, test_edges = remove_edges(G, number_of_edges_to_remove)

		with open(val_file, "wb") as f:
			pkl.dump(val_edges, f)
		with open(test_file, "wb") as f:
			pkl.dump(test_edges, f)
	else:
		with open(val_file, "rb") as f:
			val_edges = pkl.load(f) 
		with open(test_file, "rb") as f:
			test_edges = pkl.load(f)

		G.remove_edges_from(val_edges)
		G.remove_edges_from(test_edges)

	# not relevant for citation networks
	train_mask = np.zeros((N, 1))
	val_mask = np.zeros((N, 1))
	test_mask = np.zeros((N, 1))

	return G, X, Y, val_edges, test_edges, train_mask, val_mask, test_mask

	# if not os.path.exists("../data/collaboration_networks/{}_training_idx".format(dataset_str)):
	# 	training_idx, val_idx = split_data(G, X, Y, split=0.3)
	# 	np.savetxt(X=training_idx, fname="../data/collaboration_networks/{}_training_idx".format(dataset_str), delimiter=" ", fmt="%i")
	# 	np.savetxt(X=val_idx, fname="../data/collaboration_networks/{}_val_idx".format(dataset_str), delimiter=" ", fmt="%i")
	# else:
	# 	training_idx = np.genfromtxt("../data/collaboration_networks/{}_training_idx".format(dataset_str), delimiter=" ", dtype=np.int)
	# 	val_idx = np.genfromtxt("../data/collaboration_networks/{}_val_idx".format(dataset_str), delimiter=" ", dtype=np.int)

	# X_train = X[training_idx]
	# Y_train = Y[training_idx]
	# G_train = nx.Graph(G.subgraph(training_idx))
	# G_train = nx.convert_node_labels_to_integers(G_train, ordering="sorted")

	# X_val = X[val_idx]
	# Y_val = Y[val_idx]
	# G_val = nx.Graph(G.subgraph(val_idx))
	# G_val = nx.convert_node_labels_to_integers(G_val, ordering="sorted")

	# return (G, X, Y), (G_train, X_train, Y_train), (G_val, X_val, Y_val)

def load_karate():

	G = nx.karate_club_graph()
	G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")
	nx.set_edge_attributes(G=G, name="weight", values=1)

	N = len(G)

	X = sp.sparse.identity(N, format="csr")

	map_ = {"Mr. Hi" : 0, "Officer" : 1}

	Y = np.zeros((len(G), 2))
	assignments = dict(nx.get_node_attributes(G, "club")).values()
	assignments =[map_[x] for x in assignments]
	Y[np.arange(len(G)), assignments] = 1
	Y = sp.sparse.csr_matrix(Y)

	X = X.toarray()
	Y = Y.toarray()

	X = preprocess_data(X)

	val_file = "../data/karate/val_edges.pkl"
	test_file = "../data/karate/test_edges.pkl"

	if not os.path.exists(val_file):

		number_of_edges_to_remove = int(len(G.edges) * 0.2)

		G, val_edges = remove_edges(G, number_of_edges_to_remove)
		G, test_edges = remove_edges(G, number_of_edges_to_remove)

		with open(val_file, "wb") as f:
			pkl.dump(val_edges, f)
		with open(test_file, "wb") as f:
			pkl.dump(test_edges, f)
	else:
		with open(val_file, "rb") as f:
			val_edges = pkl.load(f) 
		with open(test_file, "rb") as f:
			test_edges = pkl.load(f)

		G.remove_edges_from(val_edges)
		G.remove_edges_from(test_edges)

	return G, X, Y, val_edges, test_edges

	# if not os.path.exists("../data/karate/training_idx"):
	# 	training_idx, val_idx = split_data(G, X, Y, split=0.3)
	# 	np.savetxt(X=training_idx, fname="../data/karate/training_idx", delimiter=" ", fmt="%i")
	# 	np.savetxt(X=val_idx, fname="../data/karate/val_idx", delimiter=" ", fmt="%i")
	# else:
	# 	training_idx = np.genfromtxt("../data/karate/training_idx", delimiter=" ", dtype=np.int)
	# 	val_idx = np.genfromtxt("../data/karate/val_idx", delimiter=" ", dtype=np.int)

	# X_train = X[training_idx]
	# Y_train = Y[training_idx]
	# G_train = nx.Graph(G.subgraph(training_idx))
	# G_train = nx.convert_node_labels_to_integers(G_train, ordering="sorted")

	# X_val = X[val_idx]
	# Y_val = Y[val_idx]
	# G_val = nx.Graph(G.subgraph(val_idx))
	# G_val = nx.convert_node_labels_to_integers(G_val, ordering="sorted")

	# return (G, X, Y), (G_train, X_train, Y_train), (G_val, X_val, Y_val)

# def load_cora():

# 	G = nx.read_edgelist("../data/cora/cites.tsv", delimiter="\t", )
# 	G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")
# 	nx.set_edge_attributes(G=G, name="weight", values=1)

# 	X = sp.sparse.load_npz("../data/cora/cited_words.npz")
# 	Y = sp.sparse.load_npz("../data/cora/paper_labels.npz")

# 	X = X.toarray()
# 	Y = Y.toarray()

# 	X = preprocess_data(X)

# 	if not os.path.exists("../data/cora/training_idx"):
# 		training_idx, val_idx = split_data(G, X, Y, split=0.3)
# 		np.savetxt(X=training_idx, fname="../data/cora/training_idx", delimiter=" ", fmt="%i")
# 		np.savetxt(X=val_idx, fname="../data/cora/val_idx", delimiter=" ", fmt="%i")
# 	else:
# 		training_idx = np.genfromtxt("../data/cora/training_idx", delimiter=" ", dtype=np.int)
# 		val_idx = np.genfromtxt("../data/cora/val_idx", delimiter=" ", dtype=np.int)

# 	X_train = X[training_idx]
# 	Y_train = Y[training_idx]
# 	G_train = nx.Graph(G.subgraph(training_idx))
# 	G_train = nx.convert_node_labels_to_integers(G_train, ordering="sorted")

# 	X_val = X[val_idx]
# 	Y_val = Y[val_idx]
# 	G_val = nx.Graph(G.subgraph(val_idx))
# 	G_val = nx.convert_node_labels_to_integers(G_val, ordering="sorted")

# 	return (G, X, Y), (G_train, X_train, Y_train), (G_val, X_val, Y_val)

# def load_facebook():

	# G = nx.read_gml("../data/facebook/facebook_graph.gml",  )
	# G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")

	# X = sp.sparse.load_npz("../data/facebook/features.npz")
	# Y = sp.sparse.load_npz("../data/facebook/circle_labels.npz")

	# # X = X.toarray()
	# # Y = Y.toarray()

	# return G, X, Y

def load_wordnet():

	G = nx.read_edgelist("../data/wordnet/wordnet_filtered.edg", )
	G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")
	nx.set_edge_attributes(G=G, name="weight", values=1)

	# fasttext vectors?
	N = len(G)
	X = np.genfromtxt("../data/wordnet/feats.txt", delimiter=" ")
	Y = sp.sparse.csr_matrix(np.ones((N, 1)))

	Y = Y.toarray()

	X = preprocess_data(X)

	val_file = "../data/wordnet/val_edges.pkl"
	test_file = "../data/wordnet/test_edges.pkl"

	if not os.path.exists(val_file):

		number_of_edges_to_remove = int(len(G.edges) * 0.2)

		G, val_edges = remove_edges(G, number_of_edges_to_remove)
		G, test_edges = remove_edges(G, number_of_edges_to_remove)

		with open(val_file, "wb") as f:
			pkl.dump(val_edges, f)
		with open(test_file, "wb") as f:
			pkl.dump(test_edges, f)
	else:
		with open(val_file, "rb") as f:
			val_edges = pkl.load(f) 
		with open(test_file, "rb") as f:
			test_edges = pkl.load(f)

		G.remove_edges_from(val_edges)
		G.remove_edges_from(test_edges)

	return G, X, Y, val_edges, test_edges

	# if not os.path.exists("../data/wordnet/training_idx"):
	# 	training_idx, val_idx = split_data(G, X, Y, split=0.3)
	# 	np.savetxt(X=training_idx, fname="../data/wordnet/training_idx", delimiter=" ", fmt="%i")
	# 	np.savetxt(X=val_idx, fname="../data/wordnet/val_idx", delimiter=" ", fmt="%i")
	# else:
	# 	training_idx = np.genfromtxt("../data/wordnet/training_idx", delimiter=" ", dtype=np.int)
	# 	val_idx = np.genfromtxt("../data/wordnet/val_idx", delimiter=" ", dtype=np.int)

	# X_train = X[training_idx]
	# Y_train = Y[training_idx]
	# G_train = nx.Graph(G.subgraph(training_idx))
	# G_train = nx.convert_node_labels_to_integers(G_train, ordering="sorted")

	# X_val = X[val_idx]
	# Y_val = Y[val_idx]
	# G_val = nx.Graph(G.subgraph(val_idx))
	# G_val = nx.convert_node_labels_to_integers(G_val, ordering="sorted")

	# return (G, X, Y), (G_train, X_train, Y_train), (G_val, X_val, Y_val)

def preprocess_data(X):
	# X = VarianceThreshold().fit_transform(X)
	X = StandardScaler().fit_transform(X)
	return X

def split_data(G, X, Y, split=0.2):

	num_samples = X.shape[0]
	num_samples_training = int(num_samples * (1-split))
	num_samples_validation = num_samples - num_samples_training

	# training_samples = np.random.choice(np.arange(num_samples), replace=False, size=training_size)
	# validation_samples = np.setdiff1d(np.arange(num_samples), training_samples)
	H = max(nx.connected_component_subgraphs(G), key=len)
	n = np.random.choice(H.nodes())
	training_samples = [n]
	while len(training_samples) < num_samples_training:
		n = np.random.choice(list(H.neighbors(n)))
		if n not in training_samples:
			training_samples.append(n)

	training_samples = np.array(training_samples)
	validation_samples= np.setdiff1d(np.arange(num_samples), training_samples)

	training_samples = np.append(training_samples, list(nx.isolates(G.subgraph(validation_samples))))
	validation_samples= np.setdiff1d(np.arange(num_samples), training_samples)

	training_samples = sorted(training_samples)
	validation_samples = sorted(validation_samples)

	return training_samples, validation_samples

	# X_train = X[training_samples]
	# Y_train = Y[training_samples]
	# G_train = nx.Graph(G.subgraph(training_samples))


	# X_val = X[validation_samples]
	# Y_val = Y[validation_samples]
	# G_val = nx.Graph(G.subgraph(validation_samples))

	# return (X_train, Y_train, G_train), (X_val, Y_val, G_val)

def remove_edges(G, number_of_edges_to_remove):

	# N = len(G)
	removed_edges = []
	edges = list(G.edges())

	while len(removed_edges) < number_of_edges_to_remove:
		
		random.shuffle(edges)
		for u, v in edges:

			if len(removed_edges) == number_of_edges_to_remove:
				break

			# do not remove edges connecting leaf nodes
			if G.degree(u) > 1 and G.degree(v) > 1:
				G.remove_edge(u, v)
				i = min(u, v)
				j = max(u, v)
				removed_edges.append((i, j))
				print "removed edge {}: {}".format(len(removed_edges), (i, j))

	return G, removed_edges