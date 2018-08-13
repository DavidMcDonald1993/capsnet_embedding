import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import re
import argparse

import random

import numpy as np
from scipy.sparse import identity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from sklearn.metrics.pairwise import pairwise_distances

import sys
import pandas as pd
import scipy as sp
import pickle as pkl
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, roc_curve, precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# from node2vec_sampling import Graph
from utils import load_walks, determine_positive_and_negative_samples

from keras.layers import Input, Layer, Dense
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback, TerminateOnNaN, TensorBoard, ModelCheckpoint, CSVLogger, EarlyStopping

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from tensorflow.python.training import optimizer

# K.clear_session() # ?
K.set_floatx("float64")
K.set_epsilon(1e-15)

# eps = 1e-6
np.set_printoptions(suppress=True)


# Set random seed
# seed = 0
# np.random.seed(seed)
# tf.set_random_seed(seed)

# TensorFlow wizardry
config = tf.ConfigProto()


# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5

# config.allow_soft_placement = True
config.log_device_placement=False

config.allow_soft_placement=True

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

def create_second_order_topology_graph(topology_graph, args):

	adj = nx.adjacency_matrix(topology_graph).A
	adj_sim = cosine_similarity(adj)
	adj_sim -= np.identity(len(topology_graph))
	adj_sim [adj_sim  < args.rho] = 0
	second_order_topology_graph = nx.from_numpy_matrix(adj_sim)

	print ("Created second order topology graph graph with {} edges".format(len(second_order_topology_graph.edges())))

	return second_order_topology_graph


def create_feature_graph(features, args):

	features_sim = cosine_similarity(features)
	features_sim -= np.identity(len(features))
	features_sim [features_sim  < args.rho] = 0
	feature_graph = nx.from_numpy_matrix(features_sim)

	print ("Created feature correlation graph with {} edges".format(len(feature_graph.edges())))

	return feature_graph

def split_edges(edges, val_split=0.05, test_split=0.1):
	num_val_edges = int(len(edges) * val_split)
	num_test_edges = int(len(edges) * test_split)


	random.shuffle(edges)

	val_edges = edges[:num_val_edges]
	test_edges = edges[num_val_edges:num_val_edges+num_test_edges]
	train_edges = edges[num_val_edges+num_test_edges:]

	return train_edges, val_edges, test_edges


def load_karate():

	topology_graph = nx.read_edgelist("../data/karate/karate.edg")

	label_df = pd.read_csv("../data/karate/mod-based-clusters.txt", sep=" ", index_col=0, header=None,)
	label_df.index = [str(idx) for idx in label_df.index]
	label_df = label_df.reindex(topology_graph.nodes())

	labels = label_df.iloc[:,0].values

	topology_graph = nx.convert_node_labels_to_integers(topology_graph, label_attribute="original_name")
	nx.set_edge_attributes(G=topology_graph, name="weight", values=1.)


	# node2vec_graph = Graph(nx_G=G, is_directed=False, p=1, q=1)
	# node2vec_graph.preprocess_transition_probs()
	# walks = node2vec_graph.simulate_walks(num_walks=10, walk_length=5)

	features = np.genfromtxt("../data/karate/feats.csv", delimiter=",")
	# feature_graph = create_feature_graph(features)

	return topology_graph, features, labels

	# node2vec_graph = Graph(nx_G=G_att, is_directed=False, p=1, q=1)
	# node2vec_graph.preprocess_transition_probs()
	# walks_att = node2vec_graph.simulate_walks(num_walks=1, walk_length=5)


	# walks_att = []


	# positive_samples, ground_truth_negative_samples =\
	# # determine_positive_and_ground_truth_negative_samples(G, walks+walks_att, context_size=1)
	# return (G, assignments, positive_samples, ground_truth_negative_samples)



def load_labelled_attributed_network(dataset_str, args, scale=False):
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
		test_idx_range_full = list(range(min(test_idx_reorder), max(test_idx_reorder)+1))
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
	labels = labels.argmax(axis=-1)

	# test_label_idx = test_idx_range.tolist()
	# train_label_idx = list(range(len(y)))
	# val_label_idx = list(range(len(y), len(y)+500))

	topology_graph = nx.from_numpy_matrix(adj.toarray())
	topology_graph = nx.convert_node_labels_to_integers(topology_graph, label_attribute="original_name")
	nx.set_edge_attributes(G=topology_graph, name="weight", values=1.)

	if args.only_lcc:
		topology_graph = max(nx.connected_component_subgraphs(topology_graph), key=len)
		features = features[topology_graph.nodes()]
		labels = labels[topology_graph.nodes()]
		topology_graph = nx.convert_node_labels_to_integers(topology_graph, label_attribute="original_name")
		nx.set_edge_attributes(G=topology_graph, name="weight", values=1.)

	# node2vec_graph = Graph(nx_G=G, is_directed=False, p=1, q=1)
	# node2vec_graph.preprocess_transition_probs()
	# walks = node2vec_graph.simulate_walks(num_walks=10, walk_length=5)

	# all_edges = G.edges()
	features = features.A
	if scale:
		scaler = StandardScaler()
		features = scaler.fit_transform(features)

	# feature_graph = create_feature_graph(features)

	return topology_graph, features, labels

	# # node2vec_graph = Graph(nx_G=G_att, is_directed=False, p=1, q=1)
	# # node2vec_graph.preprocess_transition_probs()
	# # walks_att = node2vec_graph.simulate_walks(num_walks=10, walk_length=5)
	
	# walks_att = []

	# Y = labels

	# positive_samples, ground_truth_negative_samples =\
	# determine_positive_and_ground_truth_negative_samples(G, walks+walks_att, context_size=1 )

	# return (G, Y.argmax(axis=-1), positive_samples, ground_truth_negative_samples,)


# def pad_negative_samples(negative_samples):
# 	def pad_l(l, len_):
# 		i = 0
# 		while len(l) < len_:
# 			l.append(l[i])
# 			i += 1 

# 	max_len = 0
# 	for l in negative_samples.values():
# 		if len(l) > max_len:
# 			max_len = len(l)
	
# 	for l in negative_samples.values():
# 		pad_l(l, max_len)

# 	return np.array(negative_samples.values(), dtype=int)

# def get_training_sample(batch_positive_samples, negative_samples, num_negative_samples, probs):
# 	input_nodes = batch_positive_samples[:,0]
# 	negative_samples = negative_samples[input_nodes]
# 	a = np.random.random(negative_samples.shape)
# 	idx = a.argsort(axis=-1)[:,:num_negative_samples]
	
# 	batch_negative_samples = negative_samples[np.arange(negative_samples.shape[0])[:,None],idx]

# 	batch_nodes = np.append(batch_positive_samples, batch_negative_samples, axis=1)
# 	return batch_nodes


def get_training_sample(batch_positive_samples, negative_samples, num_negative_samples, probs):

	input_nodes = batch_positive_samples[:,0]

	batch_negative_samples = np.array([
		np.random.choice(negative_samples[u], 
		replace=True, size=(num_negative_samples,), 
		p=probs[u] if probs is not None else probs
		)
		for u in input_nodes
	], dtype=np.int64)
	batch_nodes = np.append(batch_positive_samples, batch_negative_samples, axis=1)
	return batch_nodes


def training_generator(positive_samples, negative_samples, probs,
	num_negative_samples, batch_size=10):
	
	n = len(negative_samples)
	num_steps = int((len(positive_samples) + batch_size - 1 )/ batch_size)
	# I = sp.sparse.csr_matrix(sp.sparse.identity(n))
	# I = np.identity(n)

	while True:

		random.shuffle(positive_samples)

		for step in range(num_steps):

			batch_positive_samples = np.array(
				positive_samples[step * batch_size : (step + 1) * batch_size]).astype(np.int64)
			training_sample = get_training_sample(batch_positive_samples, 
												  negative_samples, num_negative_samples, probs)
			# training_sample = I[training_sample.flatten()].reshape(list(training_sample.shape) + [-1])
			yield training_sample, np.zeros(list(training_sample.shape)+[1], dtype=np.int64)

def convert_edgelist_to_dict(edgelist, undirected=True, self_edges=False):
	if edgelist is None:
		return None
	sorts = [lambda x: sorted(x)]
	if undirected:
		sorts.append(lambda x: sorted(x, reverse=True))
	edges = (sort(edge) for edge in edgelist for sort in sorts)
	edge_dict = {}
	for u, v in edges:
		if self_edges:
			default = [u]#set(u)
		else:
			default = []#set()
		edge_dict.setdefault(u, default).append(v)
	for u, v in edgelist:
		assert v in edge_dict[u]
		if undirected:
			assert u in edge_dict[v]
	return edge_dict

# def evaluate_rank_and_MAP(dists, edgelist, non_edgelist):
# 	assert isinstance(edgelist, list)

# 	if not isinstance(edgelist, np.ndarray):
# 		edgelist = np.array(edgelist)

# 	if not isinstance(non_edgelist, np.ndarray):
# 		non_edgelist = np.array(non_edgelist)

# 	edge_dists = dists[edgelist[:,0], edgelist[:,1]]
# 	non_edge_dists = dists[non_edgelist[:,0], non_edgelist[:,1]]

# 	targets = np.append(np.ones_like(edge_dists), np.zeros_like(non_edge_dists))
# 	ap_score = average_precision_score(targets, -np.append(edge_dists, non_edge_dists))
# 	auc_score = roc_auc_score(targets, -np.append(edge_dists, non_edge_dists))


# 	idx = non_edge_dists.argsort()
# 	ranks = np.searchsorted(non_edge_dists, edge_dists, sorter=idx).mean()

# 	print ("MEAN RANK=", ranks, "MEAN AP=", ap_score, 
# 		"MEAN ROC AUC=", auc_score)

# 	return ranks, ap_score, auc_score

def evaluate_rank_and_MAP(dists, edge_dict, non_edge_dict):

	ranks = []
	ap_scores = []
	roc_auc_scores = []
	
	for u, neighbours in edge_dict.items():
		_dists = dists[u, neighbours + non_edge_dict[u]]
		_labels = np.append(np.ones(len(neighbours)), np.zeros(len(non_edge_dict[u])))
		# _dists = dists[u]
		# _dists[u] = 1e+12
		# _labels = np.zeros(embedding.shape[0])
		# _dists_masked = _dists.copy()
		# _ranks = []
		# for v in v_set:
		# 	_labels[v] = 1
		# 	_dists_masked[v] = np.Inf
		ap_scores.append(average_precision_score(_labels, -_dists))
		roc_auc_scores.append(roc_auc_score(_labels, -_dists))

		neighbour_dists = dists[u, neighbours]
		non_neighbour_dists = dists[u, non_edge_dict[u]]
		idx = non_neighbour_dists.argsort()
		_ranks = np.searchsorted(non_neighbour_dists, neighbour_dists, sorter=idx) + 1

		# _ranks = []
		# _dists_masked = _dists.copy()
		# _dists_masked[:len(neighbours)] = np.inf

		# for v in neighbours:
		# 	d = _dists_masked.copy()
		# 	d[v] = _dists[v]
		# 	r = np.argsort(d)
		# 	raise Exception
		# 	_ranks.append(np.where(r==v)[0][0] + 1)

		ranks.append(np.mean(_ranks))
	print ("MEAN RANK=", np.mean(ranks), "MEAN AP=", np.mean(ap_scores), 
		"MEAN ROC AUC=", np.mean(roc_auc_scores))
	return np.mean(ranks), np.mean(ap_scores), np.mean(roc_auc_scores)

def evaluate_classification(klein_embedding, labels, 
	label_percentages=np.arange(0.02, 0.11, 0.01),):

	def idx_shuffle(labels):
		class_memberships = [list(np.random.permutation(np.where(labels==c)[0])) for c in sorted(set(labels))]
		idx = []
		while len(class_memberships) > 0:
			for _class in class_memberships:
				idx.append(_class.pop(0))
				if len(_class) == 0:
					class_memberships.remove(_class)
		return idx

	num_nodes, dim = klein_embedding.shape

	f1_micros = []
	f1_macros = []

	classes = sorted(set(labels))

	# print len(classes)
	idx = idx_shuffle(labels)

	
	for label_percentage in label_percentages:
		num_labels = int(max(num_nodes * label_percentage, len(classes)))
		# idx = np.random.permutation(num_nodes)
		model = LogisticRegression(multi_class="multinomial", solver="newton-cg", random_state=0)
		model.fit(klein_embedding[idx[:num_labels]], labels[idx[:num_labels]])
		predictions = model.predict(klein_embedding[idx[num_labels:]])
		f1_micro = f1_score(labels[idx[num_labels:]], predictions, average="micro")
		f1_macro = f1_score(labels[idx[num_labels:]], predictions, average="macro")
		f1_micros.append(f1_micro)
		f1_macros.append(f1_macro)

	# print label_percentages, f1_micros, f1_macros
	# raise SystemExit

	return label_percentages, f1_micros, f1_macros



	

def minkowski_dot_np(x, y):
	assert len(x.shape) == 2
	rank = x.shape[1] - 1
	return np.sum(x[:,:rank] * y[:,:rank], axis=-1, keepdims=True) - x[:,rank:] * y[:,rank:]

def minkowski_dot_pairwise(u, v):
	"""
	`u` and `v` are vectors in Minkowski space.
	"""
	rank = u.shape[-1] - 1
	euc_dp = u[:,:rank].dot(v[:,:rank].T)
	return euc_dp - u[:,rank, None] * v[:,rank]

def hyperbolic_distance_hyperboloid_pairwise(X, Y):
	inner_product = minkowski_dot_pairwise(X, Y)
	inner_product = np.clip(inner_product, a_max=-1, a_min=-np.inf)
	return np.arccosh(-inner_product)

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]


def minkowski_dot(x, y):
	assert len(x.shape) == 2
	rank = x.shape[1] - 1
	if len(y.shape) == 2:
		return K.sum(x[:,:rank] * y[:,:rank], axis=-1, keepdims=True) - x[:,rank:] * y[:,rank:]
	else:
		# print x.shape, y.shape
		# x = K.expand_dims(x, 1)
		# x = K.tile(x, [1, y.shape[1], 1])
		# print x.shape
		# print K.squeeze(K.sum(x[:,:,:rank] * y[:,:,:rank], axis=-1, keepdims=True) - x[:,:,rank:] * y[:,:,rank:], -1).shape
		# raise SystemExit
		# return K.squeeze( K.sum(x[:,:,:rank] * y[:,:,:rank], axis=-1, keepdims=True) - x[:,:,rank:] * y[:,:,rank:], -1)
		return K.batch_dot( x[:,:rank], y[:,:,:rank], axes=[1,2]) - K.batch_dot(x[:,rank:], y[:,:,rank:], axes=[1, 2])
		# return K.batch_dot(x, y, axes=[1,2])

def hyperbolic_negative_sampling_loss(r, t):

	def loss(y_true, y_pred, r=r, t=t):

		r = K.cast(r, K.floatx())
		t = K.cast(t, K.floatx())

		u_emb = y_pred[:,0]
		samples_emb = y_pred[:,1:]
		
		inner_uv = minkowski_dot(u_emb, samples_emb)
		inner_uv = K.clip(inner_uv, min_value=-np.inf, max_value=-(1+K.epsilon()))
		d_uv = tf.acosh(-inner_uv)
		out_uv = (K.square(r) - K.square(d_uv)) / t
		# out_uv = (r - d_uv) / t

		pos_out_uv = out_uv[:,0]
		neg_out_uv = out_uv[:,1:]
		
		pos_p_uv = tf.nn.sigmoid(pos_out_uv)
		neg_p_uv = 1. - tf.nn.sigmoid(neg_out_uv)

		pos_p_uv = K.clip(pos_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())
		neg_p_uv = K.clip(neg_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())

		
		return -K.mean(K.log(pos_p_uv) + K.sum(K.log(neg_p_uv), axis=-1))

	return loss

def hyperbolic_sigmoid_loss(y_true, y_pred,):

	u_emb = y_pred[:,0]
	samples_emb = y_pred[:,1:]
	
	inner_uv = minkowski_dot(u_emb, samples_emb)

	pos_inner_uv = inner_uv[:,0]
	neg_inner_uv = inner_uv[:,1:]
	
	pos_p_uv = tf.nn.sigmoid(pos_inner_uv)
	neg_p_uv = 1. - tf.nn.sigmoid(neg_inner_uv)

	pos_p_uv = K.clip(pos_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())
	neg_p_uv = K.clip(neg_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())

	return - K.mean( K.log( pos_p_uv ) + K.sum( K.log(neg_p_uv), axis=-1) )

def hyperbolic_softmax_loss(y_true, y_pred,):

	u_emb = y_pred[:,0]
	samples_emb = y_pred[:,1:]
	
	inner_uv = minkowski_dot(u_emb, samples_emb)

	return - K.mean(K.log(tf.nn.softmax(inner_uv, axis=-1,)[:,0], ))

# def expectation_loss(y_true, y_pred):

# 	u_emb = y_pred[:,0]
# 	samples_emb = y_pred[:,1:]
	
# 	# inner_uv = K.concatenate([minkowski_dot(u_emb, samples_emb[:,j]) for j in range(samples_emb.shape[1])], axis=-1)
# 	inner_uv = minkowski_dot(u_emb, samples_emb)

# 	# minimuse expected difference of 

# 	return - tf.reduce_mean( inner_uv[:,0]  -  tf.reduce_mean( inner_uv[:,1:], axis=-1)  )


def hyperboloid_initializer(shape, r_max=1e-3):

	def poincare_ball_to_hyperboloid(X, append_t=True):
		x = 2 * X
		t = 1. + K.sum(K.square(X), axis=-1, keepdims=True)
		if append_t:
			x = K.concatenate([x, t], axis=-1)
		return 1 / (1. - K.sum(K.square(X), axis=-1, keepdims=True)) * x

	def sphere_uniform_sample(shape, r_max):
		num_samples, dim = shape
		X = tf.random_normal(shape=shape, dtype=K.floatx())
		X_norm = K.sqrt(K.sum(K.square(X), axis=-1, keepdims=True))
		U = tf.random_uniform(shape=(num_samples, 1), dtype=K.floatx())
		return r_max * U ** (1./dim) * X / X_norm

	w = sphere_uniform_sample(shape, r_max=r_max)
	return poincare_ball_to_hyperboloid(w)
	# return np.genfromtxt("../data/labelled_attributed_networks/cora-lcc-warmstart.weights")

class EmbeddingLayer(Layer):
	
	def __init__(self, num_nodes, embedding_dim, **kwargs):
		super(EmbeddingLayer, self).__init__(**kwargs)
		self.num_nodes = num_nodes
		self.embedding_dim = embedding_dim

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.embedding = self.add_weight(name='embedding', 
									  shape=(self.num_nodes, self.embedding_dim),
									  initializer=hyperboloid_initializer,
									  trainable=True)


		super(EmbeddingLayer, self).build(input_shape)

	def call(self, x):
		x = K.cast(x, dtype=tf.int64)
		embedding = tf.gather(self.embedding, x, name="embedding_gather")
		
		# embedding = K.dot(x, self.embedding, )

		return embedding

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1], self.embedding_dim+1)
	
	def get_config(self):
		base_config = super(EmbeddingLayer, self).get_config()
		return base_config.update({"num_nodes": self.num_nodes, "embedding_dim": self.embedding_dim})

class ExponentialMappingOptimizer(optimizer.Optimizer):
	
	def __init__(self, learning_rate=0.001, use_locking=False, name="ExponentialMappingOptimizer"):
		super(ExponentialMappingOptimizer, self).__init__(use_locking, name)
		self._lr = learning_rate
		# print type(self._lr)
		# raise SystemExit
		
		# Tensor versions of the constructor arguments, created in _prepare().
		# self._lr_t = None

	def _prepare(self):
		self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate", dtype=K.floatx())

	def _apply_dense(self, grad, var):
#         print "dense"
		assert False
		lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
			# K.floatx())
		spacial_grad = grad[:,:-1]
		t_grad = -grad[:,-1:]
		
		ambient_grad = tf.concat([spacial_grad, t_grad], axis=-1)
		tangent_grad = self.project_onto_tangent_space(var, ambient_grad)
		
		exp_map = self.exponential_mapping(var, - lr_t * tangent_grad)
		
		return tf.assign(var, exp_map)

	# def minkowski_dot(self, x, y):
	# 	assert len(x.shape) == 2
	# 	rank = x.shape[1] - 1
	# 	if len(y.shape) == 2:
	# 		return K.sum(x[:,:rank] * y[:,:rank], axis=-1, keepdims=True) - x[:,rank:] * y[:,rank:]
	# 	else:
	# 		return K.batch_dot( x[:,:rank], y[:,:,:rank], axes=[1,2]) - K.batch_dot(x[:,rank:], y[:,:,rank:], axes=[1, 2])
		
	def project_onto_tangent_space(self, hyperboloid_point, minkowski_tangent):
		tang = minkowski_tangent + minkowski_dot(hyperboloid_point, minkowski_tangent) * hyperboloid_point
		return tang
   
	def exponential_mapping( self, p, x, ):

		def adjust_to_hyperboloid(x):
			x = x[:,:-1]
			t = K.sqrt(1. + K.sum(K.square(x), axis=-1, keepdims=True))
			return tf.concat([x,t], axis=-1)

		norm_x = tf.sqrt( tf.maximum(K.cast(0., K.floatx()), minkowski_dot(x, x), name="maximum") )

		# norm_x = tf.minimum(norm_x, 1.)
		#####################################################
		# exp_map_p = tf.cosh(norm_x) * p
		
		# idx = tf.cast( tf.where(norm_x > K.cast(0., K.floatx()), )[:,0], tf.int32)
		# non_zero_norm = tf.gather(norm_x, idx)
		# z = tf.gather(x, idx) / non_zero_norm

		# updates = tf.sinh(non_zero_norm) * z
		# dense_shape = tf.cast( tf.shape(p), tf.int32)
		# exp_map_x = tf.scatter_nd(indices=idx, updates=updates, shape=dense_shape)

		
		# exp_map = exp_map_p + exp_map_x    
		###################################################
		y = p
		# z = x / norm_x
		z = x / tf.clip_by_value(norm_x, clip_value_min=K.epsilon(), clip_value_max=np.inf)

		exp_map = tf.cosh(norm_x) * y + tf.sinh(norm_x) * z
		#####################################################
		exp_map = adjust_to_hyperboloid(exp_map)
		# idx = tf.where(tf.abs(exp_map + 1) < K.epsilon())[:,0]
		# params = tf.gather(exp_map, idx)

		# params = adjust_to_hyperboloid(params)
		# exp_map = tf.scatter_update(ref=exp_map, updates=params, indices=idx)

		# exp_map = K.minimum(exp_map, 10000)


		return exp_map

	def _apply_sparse(self, grad, var):
		# assert False
		indices = grad.indices
		values = grad.values
		# dense_shape = grad.dense_shape
		# p = tf.nn.embedding_lookup(var, indices)
		p = tf.gather(var, indices, name="gather_apply_sparse")

		lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
		spacial_grad = values[:,:-1]
		t_grad = -values[:,-1:]

		ambient_grad = tf.concat([spacial_grad, t_grad], axis=-1, name="optimizer_concat")
		tangent_grad = self.project_onto_tangent_space(p, ambient_grad)
		# exp_map = ambient_grad
		exp_map = self.exponential_mapping(p, - lr_t * tangent_grad)

		out = tf.scatter_update(ref=var, updates=exp_map, indices=indices, name="scatter_update")

		# return control_flow_ops.group(out, name="grouping")
		return out
		# return tf.assign(var, var)


class PeriodicStdoutLogger(Callback):

	def __init__(self, reconstruction_edges, val_edges, non_edges, non_edge_dict, labels, 
	epoch, n, args):
		self.reconstruction_edges = reconstruction_edges
		self.reconstruction_edge_dict = convert_edgelist_to_dict(reconstruction_edges)
		self.val_edges = val_edges
		self.val_edge_dict = convert_edgelist_to_dict(val_edges)
		self.non_edges = non_edges
		# self.non_edge_dict = convert_edgelist_to_dict(non_edges)
		self.non_edge_dict = non_edge_dict
		self.labels = labels
		self.epoch = epoch
		self.n = n
		self.args = args

	def on_epoch_end(self, batch, logs={}):
	
		self.epoch += 1


		s = "Completed epoch {}, loss={}".format(self.epoch, logs["loss"])
		if "val_loss" in logs.keys():
			s += ", val_loss={}".format(logs["val_loss"])
		print (s)

		hyperboloid_embedding = self.model.layers[-1].get_weights()[0]
		# print (hyperboloid_embedding)

		dists = hyperbolic_distance_hyperboloid_pairwise(hyperboloid_embedding, hyperboloid_embedding)

		# print minkowski_dot_np(hyperboloid_embedding, hyperboloid_embedding)

		print ("reconstruction")
		(mean_rank_reconstruction, map_reconstruction, 
			mean_roc_reconstruction) = evaluate_rank_and_MAP(dists, 
			self.reconstruction_edge_dict, self.non_edge_dict)
		# (mean_rank_reconstruction, map_reconstruction, 
		# 	mean_roc_reconstruction) = evaluate_rank_and_MAP(dists, 
		# 	self.reconstruction_edges, self.non_edges)

		logs.update({"mean_rank_reconstruction": mean_rank_reconstruction, 
			"map_reconstruction": map_reconstruction,
			"mean_roc_reconstruction": mean_roc_reconstruction})


		if self.args.evaluate_link_prediction:
			print ("link prediction")
			(mean_rank_lp, map_lp, 
			mean_roc_lp) = evaluate_rank_and_MAP(dists, 
			self.val_edge_dict, self.non_edge_dict)

			# (mean_rank_lp, map_lp, 
			# mean_roc_lp) = evaluate_rank_and_MAP(dists, 
			# self.val_edges, self.non_edges)

			logs.update({"mean_rank_lp": mean_rank_lp, 
				"map_lp": map_lp,
				"mean_roc_lp": mean_roc_lp})
		else:

			mean_rank_lp, map_lp, mean_roc_lp = None, None, None

		poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
		klein_embedding = hyperboloid_to_klein(hyperboloid_embedding)

		if self.args.evaluate_class_prediction:
			label_percentages, f1_micros, f1_macros = evaluate_classification(klein_embedding, self.labels)

			for label_percentage, f1_micro, f1_macro in zip(label_percentages, f1_micros, f1_macros):
				logs.update({"{}_micro".format(label_percentage): f1_micro})
				logs.update({"{}_macro".format(label_percentage): f1_macro})



		if self.epoch % self.n == 0:

			plot_path = os.path.join(self.args.plot_path, "epoch_{:05d}_plot.png".format(self.epoch) )
			plot_disk_embeddings(self.epoch, self.reconstruction_edges, 
				poincare_embedding, klein_embedding,
				self.labels, 
				mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
				mean_rank_lp, map_lp, mean_roc_lp,
				plot_path)

			roc_path = os.path.join(self.args.plot_path, "epoch_{:05d}_roc_curve.png".format(self.epoch) )
			plot_roc(dists, self.reconstruction_edges, self.val_edges, self.non_edges, roc_path)

			precision_recall_path = os.path.join(self.args.plot_path, "epoch_{:05d}_precision_recall_curve.png".format(self.epoch) )
			plot_precisions_recalls(dists, self.reconstruction_edges, 
				self.val_edges, self.non_edges, precision_recall_path)

			if self.args.evaluate_class_prediction:
				f1_path = os.path.join(self.args.plot_path, "epoch_{:05d}_class_prediction_f1.png".format(self.epoch))
				plot_classification(label_percentages, f1_micros, f1_macros, f1_path)

def build_model(num_nodes, args):

	x = Input(shape=(1+args.num_positive_samples+args.num_negative_samples,), name="model_input")
	y = EmbeddingLayer(num_nodes, args.embedding_dim, name="embedding_layer")(x)
	# y = Dense(args.embedding_dim, use_bias=False, activation=None, 
	# 	kernel_initializer=hyperboloid_initializer, name="embedding_layer")(x)

	model = Model(x, y)

	saved_models = sorted([f for f in os.listdir(args.model_path) 
		if re.match(r"^[0-9][0-9][0-9][0-9]*", f)])
	initial_epoch = len(saved_models)

	print (model.layers[-1].get_weights()[0])

	if initial_epoch > 0:
		model_file = os.path.join(args.model_path, saved_models[-1])
		print ("Loading model from file: {}".format(model_file))
		model.load_weights(model_file)

		print (model.layers[-1].get_weights()[0])

	return model, initial_epoch

def plot_disk_embeddings(epoch, edges, poincare_embedding, klein_embedding, labels, 
	mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
	mean_rank_lp, map_lp, mean_roc_lp, path):

	print ("saving plot to {}".format(path))

	fig = plt.figure(figsize=[14, 7])
	title = "Epoch={:05d}, Mean_rank_recon={}, MAP_recon={}, Mean_AUC_recon={}".format(epoch, 
		mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction)
	if mean_rank_lp is not None:
		title += "\nMean_rank_lp={}, MAP_lp={}, Mean_AUC_lp={}".format(mean_rank_lp,
			map_lp, mean_roc_lp)
	plt.suptitle(title)
	
	ax = fig.add_subplot(121)
	plt.title("Poincare")
	ax.add_artist(plt.Circle([0,0], 1, fill=False))
	for u, v in edges:
		u_emb = poincare_embedding[u]
		v_emb = poincare_embedding[v]
		plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.05, zorder=0)
	plt.scatter(poincare_embedding[:,0], poincare_embedding[:,1], s=10, c=labels, zorder=1)
	plt.xlim([-1,1])
	plt.ylim([-1,1])

	ax = fig.add_subplot(122)
	plt.title("Klein")
	ax.add_artist(plt.Circle([0,0], 1, fill=False))
	for u, v in edges:
		u_emb = klein_embedding[u]
		v_emb = klein_embedding[v]
		plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.05, zorder=0)
	plt.scatter(klein_embedding[:,0], klein_embedding[:,1], s=10, c=labels, zorder=1)
	plt.xlim([-1,1])
	plt.ylim([-1,1])

	
	plt.savefig(path)
	plt.close()

def plot_precisions_recalls(dists, reconstruction_edges, removed_edges, non_edges, path):

	print ("saving precision recall curves to {}".format(path))

	fig = plt.figure(figsize=[7, 7])
	title = "Embedding quality precision-recall curve"
	plt.suptitle(title)

	reconstruction_edges = np.array(reconstruction_edges)
	non_edges = np.array(non_edges) 

	edge_dists = dists[reconstruction_edges[:,0], reconstruction_edges[:,1]]
	non_edge_dists = dists[non_edges[:,0], non_edges[:,1]]

	targets = np.append(np.ones_like(edge_dists), np.zeros_like(non_edge_dists))
	_dists = np.append(edge_dists, non_edge_dists)

	precisions, recalls, _ = precision_recall_curve(targets, -_dists)

	plt.plot(recalls, precisions, c="r")

	legend = ["reconstruction"]

	if removed_edges is not None:
		removed_edges = np.array(removed_edges)
		removed_edge_dists = dists[removed_edges[:,0], removed_edges[:,1]]

		targets = np.append(np.ones_like(removed_edge_dists), np.zeros_like(non_edge_dists))
		_dists = np.append(removed_edge_dists, non_edge_dists)

		precisions, recalls, _ = precision_recall_curve(targets, -_dists)

		plt.plot(recalls, precisions, c="b")

		legend += ["link prediction"]


	plt.xlabel("recall")
	plt.ylabel("precision")
	plt.legend(legend)
	plt.savefig(path)
	plt.close()


def plot_roc(dists, reconstruction_edges, removed_edges, non_edges, path):

	print ("saving roc plot to {}".format(path))

	fig = plt.figure(figsize=[7, 7])
	title = "Embedding quality ROC curve"
	plt.suptitle(title)

	reconstruction_edges = np.array(reconstruction_edges)
	non_edges = np.array(non_edges) 

	edge_dists = dists[reconstruction_edges[:,0], reconstruction_edges[:,1]]
	non_edge_dists = dists[non_edges[:,0], non_edges[:,1]]

	targets = np.append(np.ones_like(edge_dists), np.zeros_like(non_edge_dists))
	_dists = np.append(edge_dists, non_edge_dists)

	fpr, tpr, _ = roc_curve(targets, -_dists)
	auc = roc_auc_score(targets, -_dists)
	precisions, recalls, _ = precision_recall_curve(targets, -_dists)

	plt.plot(fpr, tpr, c="r")

	legend = ["reconstruction AUC={}".format(auc)]

	if removed_edges is not None:
		removed_edges = np.array(removed_edges)
		removed_edge_dists = dists[removed_edges[:,0], removed_edges[:,1]]

		targets = np.append(np.ones_like(removed_edge_dists), np.zeros_like(non_edge_dists))
		_dists = np.append(removed_edge_dists, non_edge_dists)

		fpr, tpr, _ = roc_curve(targets, -_dists)
		auc = roc_auc_score(targets, -_dists)

		plt.plot(fpr, tpr, c="b")

		legend += ["link prediction AUC={}".format(auc)]

	plt.plot([0,1], [0,1], c="k")

	plt.xlabel("fpr")
	plt.ylabel("tpr")
	plt.legend(legend)
	plt.savefig(path)
	plt.close()

def plot_classification(label_percentages, f1_micros, f1_macros, path):

	print ("saving classification plot to {}".format(path))


	fig = plt.figure(figsize=[7, 7])
	title = "Node classification"
	plt.suptitle(title)
	
	plt.plot(label_percentages, f1_micros, c="r")
	plt.plot(label_percentages, f1_macros, c="b")
	plt.legend(["f1_micros", "f1_macros"])
	plt.xlabel("label_percentages")
	plt.ylabel("f1 score")
	plt.ylim([0,1])
	plt.savefig(path)
	plt.close()

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Hyperbolic Skipgram for feature learning on complex networks")

	parser.add_argument("--dataset", dest="dataset", type=str, default="karate",
		help="The dataset to load. Must be one of [wordnet, cora, citeseer, pubmed,\
		AstroPh, CondMat, GrQc, HepPh, karate]. (Default is karate)")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")

	parser.add_argument("-r", dest="r", type=float, default=3.,
		help="Radius of hypercircle (default is 3).")
	parser.add_argument("-t", dest="t", type=float, default=1.,
		help="Steepness of logistic function (defaut is 1).")


	parser.add_argument("--lr", dest="lr", type=float, default=1e-2,
		help="Learning rate (default is 1e-2).")

	parser.add_argument("--rho", dest="rho", type=float, default=0,
		help="Minimum feature correlation (default is 0).")

	parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=50000,
		help="The number of epochs to train for (default is 50000).")
	parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=32, 
		help="Batch size for training (default is 32).")
	parser.add_argument("--nneg", dest="num_negative_samples", type=int, default=10, 
		help="Number of negative samples for training (default is 10).")
	parser.add_argument("--context-size", dest="context_size", type=int, default=1,
		help="Context size for generating positive samples (default is 1).")
	parser.add_argument("--patience", dest="patience", type=int, default=25,
		help="The number of epochs of no improvement in validation loss before training is stopped. (Default is 25)")

	parser.add_argument("--plot-freq", dest="plot_freq", type=int, default=1000, 
		help="Frequency for plotting (default is 1000).")

	parser.add_argument("-d", "--dim", dest="embedding_dim", type=int,
		help="Dimension of embeddings for each layer (default is 2).", default=2)

	parser.add_argument("-p", dest="p", type=float, default=1.,
		help="node2vec return parameter (default is 1.).")
	parser.add_argument("-q", dest="q", type=float, default=1.,
		help="node2vec in-out parameter (default is 1.).")
	parser.add_argument('--num-walks', dest="num_walks", type=int, default=10, 
		help="Number of walks per source (default is 10).")
	parser.add_argument('--walk-length', dest="walk_length", type=int, default=15, 
		help="Length of random walk from source (default is 15).")

	# parser.add_argument("--alpha", dest="alpha", type=float, default=.5,
	# 	help="weighting of attributes (default is 0.5).")


	# parser.add_argument("--second-order", action="store_true", 
	# 	help="Use this flag to use second order topological similarity information.")
	parser.add_argument("--add-attributes", action="store_true", 
		help="Use this flag to add attribute sim to adj.")
	parser.add_argument("--multiply-attributes", action="store_true", 
		help="Use this flag to multiply attribute sim to adj.")
	parser.add_argument("--jump-prob", dest="jump_prob", type=float, default=0, 
		help="Probability of randomly jumping to a similar node when walking.")

	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", 
		help="Use this flag to set verbosity of training.")

	parser.add_argument("--sigmoid", dest="sigmoid", action="store_true", 
		help="Use this flag to use sigmoid loss.")
	parser.add_argument("--softmax", dest="softmax", action="store_true", 
		help="Use this flag to use softmax loss.")

	
	
	parser.add_argument("--plot", dest="plot_path", default="../plots/", 
		help="path to save plots (default is '../plots/)'.")
	parser.add_argument("--embeddings", dest="embedding_path", default="../embeddings/", 
		help="path to save embeddings (default is '../embeddings/)'.")
	parser.add_argument("--logs", dest="log_path", default="../logs/", 
		help="path to save logs (default is '../logs/)'.")
	parser.add_argument("--boards", dest="board_path", default="../tensorboards/", 
		help="path to save tensorboards (default is '../tensorboards/)'.")
	parser.add_argument("--walks", dest="walk_path", default="../walks/", 
		help="path to save random walks (default is '../walks/)'.")
	parser.add_argument("--model", dest="model_path", default="../models/", 
		help="path to save model after each epoch (default is '../models/)'.")

	parser.add_argument('--no-gpu', action="store_true", help='flag to train on cpu')

	parser.add_argument('--only-lcc', action="store_true", help='flag to train on only lcc')

	parser.add_argument('--evaluate-class-prediction', action="store_true", help='flag to evaluate class prediction')
	parser.add_argument('--evaluate-link-prediction', action="store_true", help='flag to evaluate link prediction')


	args = parser.parse_args()
	return args

def configure_paths(args):
	'''
	build directories on local system for output of model after each epoch
	'''

	dataset = args.dataset
	directory = "dim={}/seed={}/".format(args.embedding_dim, args.seed)

	if args.only_lcc:
		directory += "lcc/"
	else:
		directory += "all_components/"

	if args.evaluate_link_prediction:
		directory += "eval_lp/"
	elif args.evaluate_class_prediction:
		directory += "eval_class_pred/"
	else: 
		directory += "no_lp/"


	if args.softmax:
		directory += "softmax_loss/"
	elif args.sigmoid:
		directory += "sigmoid_loss/"
	else:
		directory += "hyperbolic_distance_loss/r={}_t={}/".format(args.r, args.t)


	
	if args.multiply_attributes:
		directory += "multiply_attributes/"
	elif args.add_attributes:
		directory += "add_attributes/"
	elif args.jump_prob > 0:
		directory += "jump_prob={}/".format(args.jump_prob)
	else:
		directory += "no_attributes/"


	# if args.second_order:
	# 	directory += "second_order_sim/"


	args.plot_path = os.path.join(args.plot_path, dataset)
	if not os.path.exists(args.plot_path):
		os.makedirs(args.plot_path)
	args.plot_path = os.path.join(args.plot_path, directory)
	if not os.path.exists(args.plot_path):
		os.makedirs(args.plot_path)

	args.embedding_path = os.path.join(args.embedding_path, dataset)
	if not os.path.exists(args.embedding_path):
		os.makedirs(args.embedding_path)
	args.embedding_path = os.path.join(args.embedding_path, directory)
	if not os.path.exists(args.embedding_path):
		os.makedirs(args.embedding_path)

	args.log_path = os.path.join(args.log_path, dataset)
	if not os.path.exists(args.log_path):
		os.makedirs(args.log_path)
	args.log_path = os.path.join(args.log_path, directory)
	if not os.path.exists(args.log_path):
		os.makedirs(args.log_path)
	args.log_path += "{}.log".format(dataset)

	args.board_path = os.path.join(args.board_path, dataset)
	if not os.path.exists(args.board_path):
		os.makedirs(args.board_path)
	args.board_path = os.path.join(args.board_path, directory)
	if not os.path.exists(args.board_path):
		os.makedirs(args.board_path)

	args.walk_path = os.path.join(args.walk_path, dataset)
	if not os.path.exists(args.walk_path):
		os.makedirs(args.walk_path)
	if args.only_lcc:
		args.walk_path += "/lcc/"
	else:
		args.walk_path += "/all_components/"
	if args.evaluate_link_prediction:
		args.walk_path += "eval_lp/"
	# elif args.evaluate_class_prediction:
	# 	args.walk_path += "/eval_class_pred/"
	else:
		args.walk_path += "no_lp/"
	if not os.path.exists(args.walk_path):
		os.makedirs(args.walk_path)

	args.model_path = os.path.join(args.model_path, dataset)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	args.model_path = os.path.join(args.model_path, directory)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

def make_validation_data(edges, non_edge_dict, probs, args):

	edges = np.array(edges)
	idx = np.random.choice(len(edges), size=args.batch_size, replace=False,)
	positive_samples = edges[idx]#
	# non_edge_dict = convert_edgelist_to_dict(non_edges)

	x = get_training_sample(positive_samples, 
		non_edge_dict, args.num_negative_samples, probs=None)
	y = np.zeros(list(x.shape)+[1], dtype=np.int64)

	return x, y

def main():

	args = parse_args()
	args.num_positive_samples = 1
	args.only_lcc = True
	if not args.evaluate_link_prediction:
		args.evaluate_class_prediction = True

	assert not sum([args.multiply_attributes, args.add_attributes, args.jump_prob>0]) > 1

	random.seed(args.seed)
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)

	# if args.no_gpu:
	# 	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	# if args.combine_attributes:
	# 	args.use_attributes = True
	configure_paths(args)

	dataset = args.dataset
	if dataset == "karate":
		topology_graph, features, labels = load_karate()
	elif dataset in ["cora", "pubmed", "citeseer"]:
		topology_graph, features, labels = load_labelled_attributed_network(dataset, args)
	else:
		raise Exception

	# original edges for reconstruction
	reconstruction_edges = topology_graph.edges()
	print ("determined reconstruction edges")
	non_edges = list(nx.non_edges(topology_graph))
	non_edge_dict = convert_edgelist_to_dict(non_edges)
	print ("determined non edges")

	# edge_dict = convert_edgelist_to_dict(reconstruction_edges)

	if features is not None:
		feature_sim = cosine_similarity(features)
		feature_sim -= np.identity(len(features))
		feature_sim [feature_sim  < args.rho] = 0
	else:
		feature_sim = None

	if args.evaluate_link_prediction:
		train_edges, val_edges, test_edges = split_edges(reconstruction_edges)
		topology_graph.remove_edges_from(val_edges + test_edges)

		# train_edges = convert_edgelist_to_dict(train_edges)
		# val_edges = convert_edgelist_to_dict(val_edges)
		# test_edges = convert_edgelist_to_dict(test_edges)

	else:
		train_edges = reconstruction_edges
		# train_edges = convert_edgelist_to_dict(reconstruction_edges)
		val_edges = None
		test_edges = None



	if args.add_attributes:
		walk_file = os.path.join(args.walk_path, "add_attributes")
		g = nx.from_numpy_matrix(nx.adjacency_matrix(topology_graph).A + feature_sim)
	elif args.multiply_attributes:
		walk_file = os.path.join(args.walk_path, "multiply_attributes")
		g = nx.from_numpy_matrix(nx.adjacency_matrix(topology_graph).A * feature_sim)
	elif args.jump_prob > 0:
		walk_file = os.path.join(args.walk_path, "jump_prob={}".format(args.jump_prob))
		g = topology_graph
	else:
		walk_file = os.path.join(args.walk_path, "no_attributes")
		g = topology_graph
	walk_file += "_num_walks={}-walk_len={}-p={}-q={}.walk".format(args.num_walks, 
				args.walk_length, args.p, args.q)

	walks = load_walks(g, walk_file, feature_sim, args)

	

	positive_samples, negative_samples, probs =\
		determine_positive_and_negative_samples(nodes=topology_graph.nodes(), 
		walks=walks, context_size=args.context_size)

	# negative_samples = pad_negative_samples(negative_samples)
	# print negative_samples.shape
	# raise SystemExit

	# for e in topology_graph.edges():
	# 	assert e in positive_samples, "edge {} is not in positive_samples".format(e)
	# print "passed"
	# print "missing {} edges".format(sum([e not in positive_samples for e in topology_graph.edges()]))
	# raise SystemExit

	num_nodes = len(topology_graph)
	num_steps = int((len(positive_samples) + args.batch_size - 1) / args.batch_size)
	# num_steps=10000

	# training_gen = training_generator(positive_samples, negative_samples, probs,
	# 								  num_negative_samples=args.num_negative_samples, 
	# 								  batch_size=args.batch_size)

	model, initial_epoch = build_model(num_nodes, args)
	optimizer = ExponentialMappingOptimizer(learning_rate=args.lr)
	loss = (
		hyperbolic_softmax_loss 
		if args.softmax 
		else hyperbolic_sigmoid_loss 
		if args.sigmoid 
		else hyperbolic_negative_sampling_loss(r=args.r, t=args.t)
	)
	model.compile(optimizer=optimizer, loss=loss)
	model.summary()

	# if sys.version_info[0] == 2:
	# 	val_in, val_out = training_gen.next()
	# else:
	# 	val_in, val_out = training_gen.__next__()
	val_in, val_target = make_validation_data(reconstruction_edges, non_edge_dict, probs, args)
	# val_in = get_training_sample(np.array(reconstruction_edges[:args.batch_size]), 
	# 	negative_samples, args.num_negative_samples, probs)
	# val_target = np.zeros(list(val_in.shape)+[1], dtype=np.int64)
	print ("determined validation data")
	# model.fit(_in, _out, epochs=1, verbose=args.verbose)
	# raise SystemExit

	early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, verbose=1)
	logger = PeriodicStdoutLogger(reconstruction_edges, val_edges, non_edges, non_edge_dict, labels, 
				n=args.plot_freq, epoch=initial_epoch, args=args) 
	print ("created logger")
	# for epoch in range(initial_epoch, args.num_epochs):
	# 	for step in range(num_steps):
	# 		x, y = training_gen.next()
	# 		model.train_on_batch(x, y)
	# 	print "COMPLETED EPOCH {}/{}".format(epoch, args.num_epochs)
	x = get_training_sample(np.array(positive_samples), negative_samples, args.num_negative_samples, probs)
	y = np.zeros(list(x.shape) + [1])
	# I = np.identity(len(topology_graph))
	# x = I[x]
	# print x.shape
	# raise SystemExit
	print ("determined training samples")
	# model.fit_generator(training_gen, epochs=args.num_epochs, 
	# 	workers=1, max_queue_size=100, use_multiprocessing=True,
	# 	steps_per_epoch=num_steps, verbose=args.verbose, initial_epoch=initial_epoch,
	# print len(positive_samples), len(x), args.batch_size
	# raise SystemExit
	model.fit(x, y, batch_size=args.batch_size, epochs=args.num_epochs, 
		initial_epoch=initial_epoch, verbose=args.verbose,
		validation_data=[val_in, val_target],
		callbacks=[
			TerminateOnNaN(), 
			logger,
			# PeriodicStdoutLogger(reconstruction_edges, labels, 
			# 	n=args.plot_freq, epoch=initial_epoch, edge_dict=edge_dict, args=args), 
			# TensorBoard(log_dir=args.board_path, histogram_freq=1, 
			# 	batch_size=args.batch_size, write_graph=True, write_grads=True, write_images=True, 
			# 	embeddings_freq=1, embeddings_layer_names="embedding_layer", 
			# 	embeddings_metadata="/home/david/Documents/capsnet_embedding/data/karate/labels.tsv"
			# 	),
			ModelCheckpoint(os.path.join(args.model_path, 
				"{epoch:05d}.h5"), save_weights_only=True),
			CSVLogger(args.log_path, append=True), 
			early_stopping
		]
		)

	hyperboloid_embedding = model.layers[-1].get_weights()[0]
	dists = hyperbolic_distance_hyperboloid_pairwise(hyperboloid_embedding, hyperboloid_embedding)
	print (hyperboloid_embedding)
	# print minkowski_dot_np(hyperboloid_embedding, hyperboloid_embedding)

	reconstruction_edge_dict = convert_edgelist_to_dict(reconstruction_edges)
	non_edge_dict = convert_edgelist_to_dict(non_edges)
	(mean_rank_reconstruction, map_reconstruction, 
		mean_roc_reconstruction) = evaluate_rank_and_MAP(dists, 
		reconstruction_edge_dict, non_edge_dict)

	if args.evaluate_link_prediction:
		test_edge_dict = convert_edgelist_to_dict(test_edges)	
		(mean_rank_lp, map_lp, 
		mean_roc_lp) = evaluate_rank_and_MAP(dists, test_edge_dict, non_edge_dict)
	else:
		mean_rank_lp, map_lp, mean_roc_lp = None, None, None 

	poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
	klein_embedding = hyperboloid_to_klein(hyperboloid_embedding)

	# epoch = early_stopping.stopped_epoch#
	epoch = logger.epoch

	plot_path = os.path.join(args.plot_path, "epoch_{:05d}_plot_test.png".format(epoch) )
	plot_disk_embeddings(epoch, reconstruction_edges, 
		poincare_embedding, klein_embedding,
		labels, 
		mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
		mean_rank_lp, map_lp, mean_roc_lp,
		plot_path)

	roc_path = os.path.join(args.plot_path, "epoch_{:05d}_roc_curve_test.png".format(epoch) )
	plot_roc(dists, reconstruction_edges, test_edges, non_edges, roc_path)

	precision_recall_path = os.path.join(args.plot_path, 
		"epoch_{:05d}_precision_recall_curve_test.png".format(epoch) )
	plot_precisions_recalls(dists, reconstruction_edges, 
		test_edges, non_edges, precision_recall_path)

	if args.evaluate_class_prediction:
		label_percentages, f1_micros, f1_macros = evaluate_classification(klein_embedding, labels)

		f1_path = os.path.join(args.plot_path, "epoch_{:05d}_class_prediction_f1_test.png".format(epoch))
		plot_classification(label_percentages, f1_micros, f1_macros, f1_path)



if __name__ == "__main__":
	main()