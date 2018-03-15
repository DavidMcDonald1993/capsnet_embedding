import sys
import os
import json
import random
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
import pandas as pd
import networkx as nx

import pickle as pkl

from sklearn.preprocessing import StandardScaler

def select_label_split(Y, num_train=20, num_val=50):

	num_samples, num_classes = Y.shape
	assignments = Y.argmax(axis=1)

	nodes_in_classes = [np.random.permutation(np.where(assignments == c)[0]) for c in range(num_classes)]

	train_idx = np.concatenate([nodes_in_class[:num_train] for nodes_in_class in nodes_in_classes])
	val_idx =  np.concatenate([nodes_in_class[num_train:num_train+num_val] for nodes_in_class in nodes_in_classes])
	test_idx = np.concatenate([nodes_in_class[num_train+num_val:] for nodes_in_class in nodes_in_classes])

	# train_mask = np.zeros((num_samples, 1), dtype=np.float32)
	# val_mask = np.zeros((num_samples, 1), dtype=np.float32)
	# test_mask = np.zeros((num_samples, 1), dtype=np.float32)

	# train_mask[train_idx] = 1
	# val_mask[val_idx] = 1
	# test_mask[test_idx] = 1

	return train_idx, val_idx, test_idx

def load_data(dataset):

	if dataset == "wordnet":
		reconstruction_adj, G_train, G_val, G_test,\
		X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx = load_wordnet()
	elif dataset == "karate":
		reconstruction_adj, G_train, G_val, G_test,\
		X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx = load_karate()
	elif dataset in ["citeseer", "cora", "pubmed"]:
		reconstruction_adj, G_train, G_val, G_test,\
		X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx = load_labelled_attributed_network(dataset)
	elif dataset in ["AstroPh", "CondMat", "HepPh", "GrQc"]:
		reconstruction_adj, G_train, G_val, G_test,\
		X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx = load_collaboration_network(dataset)
	elif dataset == "reddit":
		reconstruction_adj, G_train, G_val, G_test,\
		X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx = load_reddit()
	else:
		raise Exception

	return reconstruction_adj, G_train, G_val, G_test, X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx

def load_labelled_attributed_network(dataset_str):
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
	reconstruction_adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

	labels = np.vstack((ally, ty))
	labels[test_idx_reorder, :] = labels[test_idx_range, :]

	test_label_idx = test_idx_range.tolist()
	train_label_idx = range(len(y))
	val_label_idx = range(len(y), len(y)+500)

	train_label_mask = sample_mask(train_label_idx, labels.shape[0])
	# val_mask = sample_mask(idx_val, labels.shape[0])
	# test_mask = sample_mask(idx_test, labels.shape[0])

	# y_train = np.zeros(labels.shape)
	# y_val = np.zeros(labels.shape)
	# y_test = np.zeros(labels.shape)
	# y_train[train_mask, :] = labels[train_mask, :]
	# y_val[val_mask, :] = labels[val_mask, :]
	# y_test[test_mask, :] = labels[test_mask, :]

	# return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels

	G = nx.from_numpy_matrix(reconstruction_adj.toarray())
	G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")
	nx.set_edge_attributes(G=G, name="weight", values=1)

	X = features#.toarray()
	Y = labels

	# X = preprocess_data(X)

	# val_file = "../data/labelled_attributed_networks/{}_val_edges.pkl".format(dataset_str)
	# test_file = "../data/labelled_attributed_networks/{}_test_edges.pkl".format(dataset_str)

	# if not os.path.exists(val_file):

	# 	number_of_edges_to_remove = int(len(G.edges) * 0.2)

	# 	G, val_edges = remove_edges(G, number_of_edges_to_remove)
	# 	G, test_edges = remove_edges(G, number_of_edges_to_remove)

	# 	with open(val_file, "wb") as f:
	# 		pkl.dump(val_edges, f)
	# 	with open(test_file, "wb") as f:
	# 		pkl.dump(test_edges, f)
	# else:
	# 	with open(val_file, "rb") as f:
	# 		val_edges = pkl.load(f) 
	# 	with open(test_file, "rb") as f:
	# 		test_edges = pkl.load(f)

	# 	G.remove_edges_from(val_edges)
	# 	G.remove_edges_from(test_edges)

	# same networks for train, test, val
	G_train = G
	G_val = G 
	G_test = G

	# no removed edges for GCN networks
	val_edges = None
	test_edges = None

	train_label_mask = train_label_mask.reshape(-1, 1).astype(np.float32)
	# val_mask = train_mask.reshape(-1, 1).astype(np.float32)
	# test_mask = train_mask.reshape(-1, 1).astype(np.float32)

	# return adj, G, X, Y, val_edges, test_edges, train_mask, val_mask, test_mask
	return reconstruction_adj, G_train, G_val, G_test, X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx



def load_collaboration_network(dataset_str):
	'''
	FOR LINK PREDICTION
	removing random 20% edges for validation
	removing random 20% edges for testing
	'''
	assert dataset_str in ["AstroPh", "CondMat", "GrQc", "HepPh"], "dataset string is not valid"

	G = nx.read_edgelist("../data/collaboration_networks/ca-{}.txt.gz".format(dataset_str))
	G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")
	nx.set_edge_attributes(G=G, name="weight", values=1)

	# reconstruct original network
	reconstruction_adj = nx.adjacency_matrix(G)

	N = len(G)

	# sparse identity features
	X = sp.sparse.identity(N, format="csr")
	Y = np.ones((N, 1))

	# X = preprocess_data(X)

	# remove validation and testing edges
	val_file = "../data/collaboration_networks/{}_val_edges.pkl".format(dataset_str)
	test_file = "../data/collaboration_networks/{}_test_edges.pkl".format(dataset_str)

	if not os.path.exists(val_file):

		number_of_edges_to_remove = int(len(G.edges) * 0.2)

		G, val_edges = remove_edges(G, number_of_edges_to_remove)
		G, test_edges = remove_edges(G, number_of_edges_to_remove)

		with open(val_file, "wb") as f:
			pkl.dump(val_edges, f, pkl.HIGHEST_PROTOCOL)
		with open(test_file, "wb") as f:
			pkl.dump(test_edges, f, pkl.HIGHEST_PROTOCOL)
	else:
		with open(val_file, "rb") as f:
			val_edges = pkl.load(f) 
		with open(test_file, "rb") as f:
			test_edges = pkl.load(f)

		G.remove_edges_from(val_edges)
		G.remove_edges_from(test_edges)


	# not relevant for collaboration networks -- not testing label predicition capacity
	train_label_mask = np.zeros((N, 1))
	# val_mask = np.zeros((N, 1))
	# test_mask = np.zeros((N, 1))
	val_label_idx = None
	test_label_idx = None

	# same network for all phases
	G_train = G 
	G_val = G 
	G_test = G

	return reconstruction_adj, G_train, G_val, G_test, X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx
	# return original_adj, G, X, Y, val_edges, test_edges, train_mask, val_mask, test_mask

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

	G = nx.read_edgelist("../data/karate/karate.edg")

	# reconstruct original adjacency matrix
	reconstruction_adj = nx.adjacency_matrix(G)

	# identity features
	N = len(G)
	X = sp.sparse.identity(N, format="csr")

	label_df = pd.read_csv("../data/karate/mod-based-clusters.txt", sep=" ", index_col=0, header=None,)
	label_df.index = [str(idx) for idx in label_df.index]
	label_df = label_df.reindex(G.nodes())

	assignments = label_df.values

	# sparse label matrix
	Y = sp.sparse.csr_matrix(([1] * N, (range(N), assignments)), shape=(N, 4))

	# Y = np.zeros((N, 4))
	# for i, assignment in enumerate(assignments):
	# 	Y[i, assignment] = 1

	G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")
	nx.set_edge_attributes(G=G, name="weight", values=1)
	# map_ = {"Mr. Hi" : 0, "Officer" : 1}

	# Y = np.zeros((len(G), 2))
	# assignments = dict(nx.get_node_attributes(G, "club")).values()
	# assignments =[map_[x] for x in assignments]
	# Y[np.arange(len(G)), assignments] = 1
	# Y = sp.sparse.csr_matrix(Y)

	X = X.toarray()
	# Y = Y.toarray()

	X = preprocess_data(X)

	val_file = "../data/karate/val_edges.pkl"
	test_file = "../data/karate/test_edges.pkl"

	if not os.path.exists(val_file):

		number_of_edges_to_remove = int(len(G.edges) * 0.2)

		G, val_edges = remove_edges(G, number_of_edges_to_remove)
		G, test_edges = remove_edges(G, number_of_edges_to_remove)

		with open(val_file, "wb") as f:
			pkl.dump(val_edges, f, pkl.HIGHEST_PROTOCOL)
		with open(test_file, "wb") as f:
			pkl.dump(test_edges, f, pkl.HIGHEST_PROTOCOL)
	else:
		with open(val_file, "rb") as f:
			val_edges = pkl.load(f) 
		with open(test_file, "rb") as f:
			test_edges = pkl.load(f)

		G.remove_edges_from(val_edges)
		G.remove_edges_from(test_edges)

	# train, val and test on same network
	G_train = G
	G_val = G
	G_test = G

	train_label_idx_file = "../data/karate/train_label_idx"
	val_label_idx_file = "../data/karate/val_label_idx"
	test_label_idx_file = "../data/karate/test_label_idx"

	if not os.path.exists(train_label_idx_file):
		train_label_idx, val_label_idx, test_label_idx = select_label_split(Y, num_train=1, num_val=3)
		np.savetxt(train_label_idx_file, train_label_idx, delimiter=",", fmt="%i")
		np.savetxt(val_label_idx_file, val_label_idx, delimiter=",", fmt="%i")
		np.savetxt(test_label_idx_file, test_label_idx, delimiter=",", fmt="%i")
	else:
		train_label_idx = np.genfromtxt(train_label_idx_file, delimiter=",")
		val_label_idx = np.genfromtxt(val_label_idx_file, delimiter=",")
		test_label_idx = np.genfromtxt(test_label_idx_file, delimiter=",")
	# use selected train idx to mask all other labels
	train_label_mask = np.zeros((Y.shape[0], 1))
	train_label_mask[train_label_idx] = 1


	# return original_adj, G, X, Y, val_edges, test_edges, train_mask, val_mask, test_mask
	return reconstruction_adj, G_train, G_val, G_test, X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx


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

def load_reddit():
	
	G = nx.read_edgelist("../data/reddit/reddit_G.edg", delimiter=" ", data=True)
	nx.set_edge_attributes(G=G, name="weight", values=1)

	feats = np.load("../data/reddit/reddit-feats.npy")
	id_map = json.load(open("../data/reddit/reddit-id_map.json"))
	conversion = lambda n : n
	id_map = {conversion(k):int(v) for k,v in id_map.items()}
	
	G = nx.relabel_nodes(G, id_map, )

	with open("../data/reddit/train_nodes", "rb") as f:
		train_nodes = pkl.load(f)
	with open("../data/reddit/val_nodes", "rb") as f:
		val_nodes = pkl.load(f)
	with open("../data/reddit/test_nodes", "rb") as f:
		test_nodes = pkl.load(f)
		
	train_label_idx = [id_map[n] for n in train_nodes]
	val_label_idx = [id_map[n] for n in val_nodes]
	test_label_idx = [id_map[n] for n in test_nodes]
	
	# normalize by training data
	train_feats = feats[train_label_idx]
	scaler = StandardScaler()
	scaler.fit(train_feats)
	feats = scaler.transform(feats)
	
	X = feats
	
	class_map = json.load(open("../data/reddit/reddit-class_map.json"))
	if isinstance(list(class_map.values())[0], list):
		lab_conversion = lambda n : n
	else:
		lab_conversion = lambda n : int(n)
	class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}
	
	num_classes = max(class_map.values()) + 1
	class_map = {id_map[k]: v for k, v in class_map.items()}
	# print (class_map.keys(), type(class_map.keys()))
	# print (class_map.values(), type(class_map.values()))
	Y = csr_matrix(([1] * len(class_map), (list(class_map.keys()), list(class_map.values()))), 
		shape=(len(class_map), num_classes))
		
	G_train = G.subgraph(train_label_idx)
	G_val = G.subgraph(train_label_idx + val_label_idx)
	G_test = G

	# measure reconstruction capacity for trained network
	reconstruction_adj = nx.adjacency_matrix(G_train)

	# no link prediction
	val_edges = None
	test_edges = None

	# no label masking -- train on all training labels
	train_label_mask = np.ones(Y.shape[0])

	# return train_G, val_G, test_G, X, Y, train_idx, val_idx, test_idx, train_mask
	return reconstruction_adj, G_train, G_val, G_test, X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx

def load_wordnet():

	'''
	testing link prediciton / reconstruction / lexical entailment
	'''

	G = nx.read_edgelist("../data/wordnet/wordnet_filtered.edg", )
	G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")
	nx.set_edge_attributes(G=G, name="weight", values=1)

	# mesaure capaity for reconstructing original network
	reconstruction_adj = nx.adjacency_matrix(G)

	# fasttext vectors?
	N = len(G)
	X = sp.sparse.identity(N, format="csr")
	# X = np.genfromtxt("../data/wordnet/feats.txt", delimiter=" ")
	Y = np.ones((N, 1))


	# X = preprocess_data(X)

	val_file = "../data/wordnet/val_edges.pkl"
	test_file = "../data/wordnet/test_edges.pkl"

	if not os.path.exists(val_file):

		number_of_edges_to_remove = int(len(G.edges) * 0.2)

		G, val_edges = remove_edges(G, number_of_edges_to_remove)
		G, test_edges = remove_edges(G, number_of_edges_to_remove)

		with open(val_file, "wb") as f:
			pkl.dump(val_edges, f, pkl.HIGHEST_PROTOCOL)
		with open(test_file, "wb") as f:
			pkl.dump(test_edges, f, pkl.HIGHEST_PROTOCOL)
	else:
		with open(val_file, "rb") as f:
			val_edges = pkl.load(f) 
		with open(test_file, "rb") as f:
			test_edges = pkl.load(f)

		G.remove_edges_from(val_edges)
		G.remove_edges_from(test_edges)

	# no label prediction

	# same network for all label prediction phases
	G_train = G
	G_val = G
	G_test = G

	val_label_idx = None
	test_label_idx = None

	# no masking
	train_label_mask = np.zeros((N, 1))
	# val_mask = np.zeros((N, 1))
	# test_mask = np.zeros((N, 1))

	return reconstruction_adj, G_train, G_val, G_test, X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx

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

			if u == v:
				continue

			# do not remove edges connecting leaf nodes
			if G.degree(u) > 1 and G.degree(v) > 1:
				G.remove_edge(u, v)
				i = min(u, v)
				j = max(u, v)
				removed_edges.append((i, j))
				print ("removed edge {}: {}".format(len(removed_edges), (i, j)))

	return G, removed_edges