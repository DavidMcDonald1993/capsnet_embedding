import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

# from node2vec_sampling import Graph
from utils import load_walks, determine_positive_and_negative_samples

from keras.layers import Input, Layer 
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback, TerminateOnNaN, TensorBoard, ModelCheckpoint, CSVLogger, EarlyStopping


import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from tensorflow.python.training import optimizer


eps = 1e-6
np.set_printoptions(suppress=True)


# Set random seed
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

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



def load_labelled_attributed_network(dataset_str, scale=True):
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




def get_training_sample(batch_positive_samples, negative_samples, num_negative_samples, probs):

	input_nodes = batch_positive_samples[:,0]

	batch_negative_samples = np.array([
		np.random.choice(negative_samples[u], 
		replace=True, size=(num_negative_samples,), p=probs[u])
		for u in input_nodes
	], dtype=K.floatx())
	batch_nodes = np.append(batch_positive_samples, batch_negative_samples, axis=1)
	return batch_nodes, np.zeros(list(batch_nodes.shape) + [1], dtype=K.floatx())


def training_generator(positive_samples, negative_samples, probs,
	num_negative_samples, batch_size=10):
	
	num_steps = int((len(positive_samples) + batch_size - 1 )/ batch_size)

	while True:

		random.shuffle(positive_samples)

		for step in range(num_steps):

			batch_positive_samples = np.array(
				positive_samples[step * batch_size : (step + 1) * batch_size])
			training_sample = get_training_sample(batch_positive_samples, 
												  negative_samples, num_negative_samples, probs)
			yield training_sample

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
			default = set(u)
		else:
			default = set()
		edge_dict.setdefault(u, default).add(v)
	for u, v in edgelist:
		assert v in edge_dict[u]
		if undirected:
			assert u in edge_dict[v]
	return edge_dict

def evaluate_rank_and_MAP(embedding, edge_dict):

	ranks = []
	ap_scores = []
	
	dists = hyperbolic_distance_hyperboloid_pairwise(embedding, embedding)

	for u, v_list in edge_dict.items():
		_dists = dists[u]
		_dists[u] = 1e+12
		_labels = np.zeros(embedding.shape[0])
		_dists_masked = _dists.copy()
		_ranks = []
		for v in v_list:
			_dists_masked[v] = np.Inf
			_labels[v] = 1
		ap_scores.append(average_precision_score(_labels, -_dists))
		for v in v_list:
			d = _dists_masked.copy()
			d[v] = _dists[v]
			r = np.argsort(d)
			_ranks.append(np.where(r==v)[0][0] + 1)
		ranks.append(np.mean(_ranks))
	print ("MEAN RANK=", np.mean(ranks), "MEAN AP=", np.mean(ap_scores))
	return np.mean(ranks), np.mean(ap_scores)

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
	# print rank, len(y.shape)
	# print K.batch_dot( x[:,:rank], y[:,:,:rank], axes=[1,2]) - K.batch_dot(x[:,rank:], y[:,:,rank:], axes=[1, 2])
	# raise SystemExit
	if len(y.shape) == 2:
		return K.sum(x[:,:rank] * y[:,:rank], axis=-1, keepdims=True) - x[:,rank:] * y[:,rank:]
	else:
		return K.batch_dot( x[:,:rank], y[:,:,:rank], axes=[1,2]) - K.batch_dot(x[:,rank:], y[:,:,rank:], axes=[1, 2])

def hyperbolic_negative_sampling_loss(r, t):

	def loss(y_true, y_pred, ):
		u_emb = y_pred[:,0]
		samples_emb = y_pred[:,1:]
		
		# inner_uv = K.concatenate([minkowski_dot(u_emb, samples_emb[:,j]) for j in range(samples_emb.shape[1])], axis=-1)
		inner_uv = minkowski_dot(u_emb, samples_emb)
		inner_uv = K.clip(inner_uv, min_value=-np.inf, max_value=-(1+K.epsilon()))
		d_uv = tf.acosh(-inner_uv)
		out_uv = (r - d_uv ** 2) / t
		pos_out_uv = out_uv[:,0]
		neg_out_uv = out_uv[:,1:]
		
		pos_p_uv = tf.nn.sigmoid(pos_out_uv)
		neg_p_uv = tf.nn.sigmoid(neg_out_uv)
		
		return -tf.reduce_mean(tf.log(pos_p_uv) + tf.reduce_sum(tf.log(1 - neg_p_uv), axis=-1))

	return loss

def hyperbolic_sigmoid_loss(y_true, y_pred,):

	u_emb = y_pred[:,0]
	samples_emb = y_pred[:,1:]
	
	# inner_uv = K.concatenate([minkowski_dot(u_emb, samples_emb[:,j]) for j in range(samples_emb.shape[1])], axis=-1)
	inner_uv = minkowski_dot(u_emb, samples_emb)

	pos_inner_uv = inner_uv[:,0]
	neg_inner_uv = -inner_uv[:,1:]
	
	pos_p_uv = tf.nn.sigmoid(pos_inner_uv)
	neg_p_uv = tf.nn.sigmoid(neg_inner_uv)

	return - tf.reduce_mean( tf.log( pos_p_uv ) + tf.reduce_sum( tf.log(neg_p_uv), axis=-1) )

def hyperbolic_softmax_loss(y_true, y_pred,):

	u_emb = y_pred[:,0]
	samples_emb = y_pred[:,1:]
	
	# inner_uv = K.concatenate([minkowski_dot(u_emb, samples_emb[:,j]) for j in range(samples_emb.shape[1])], axis=-1)
	inner_uv = minkowski_dot(u_emb, samples_emb)

	return - tf.reduce_mean(tf.log(tf.nn.softmax(inner_uv, axis=-1,)[:,0], ))

def expectation_loss(y_true, y_pred):

	u_emb = y_pred[:,0]
	samples_emb = y_pred[:,1:]
	
	# inner_uv = K.concatenate([minkowski_dot(u_emb, samples_emb[:,j]) for j in range(samples_emb.shape[1])], axis=-1)
	inner_uv = minkowski_dot(u_emb, samples_emb)

	# minimuse expected difference of 

	return - tf.reduce_mean( inner_uv[:,0]  -  tf.reduce_mean( inner_uv[:,1:], axis=-1)  )


def hyperboloid_initializer(shape, r_max=1e-3):

	def poincare_ball_to_hyperboloid(X, append_t=True):
		x = 2 * X
		t = 1 + K.sum(K.square(X), axis=-1, keepdims=True)
		if append_t:
			x = K.concatenate([x, t], axis=-1)
		return 1 / (1 - K.sum(K.square(X), axis=-1, keepdims=True)) * x

	def sphere_uniform_sample(shape, r_max=1e-3):
		num_samples, dim = shape
		X = tf.random_normal(shape=shape, dtype=K.floatx())
		X_norm = K.sqrt(K.sum(K.square(X), axis=-1, keepdims=True))
		U = tf.random_uniform(shape=(num_samples, 1), dtype=K.floatx())
		return r_max * U ** (1./dim) * X / X_norm

	w = sphere_uniform_sample(shape, r_max=r_max)
	return poincare_ball_to_hyperboloid(w)

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
		x = K.cast(x, dtype=tf.int32)
		# embedding = tf.nn.embedding_lookup(params=self.embedding, ids=x)
		embedding = tf.gather(self.embedding, x, name="embedding_gather")

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
	        t = tf.sqrt(1. + tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
	        return tf.concat([x,t], axis=-1)

	    norm_x = tf.sqrt( tf.maximum(K.cast(0., K.floatx()), minkowski_dot(x, x), name="maximum") )
	    #####################################################
	    exp_map_p = tf.cosh(norm_x) * p
	    
	    idx = tf.where(norm_x > K.cast(0., K.floatx()), )[:,0]
	    non_zero_norm = tf.gather(norm_x, idx)
	    z = tf.gather(x, idx) / non_zero_norm


	    updates = tf.sinh(non_zero_norm) * z
	    # updates = tf.reshape( updates , [-1,])
	    dense_shape = tf.cast( tf.shape(p), tf.int64)
	    exp_map_x = tf.scatter_nd(indices=idx, updates=updates, shape=dense_shape)

	    
	    # _, num_cols = p.shape
	    
	    # a = idx
	    # b = K.arange(num_cols, dtype=tf.int64)
	    
	    # tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0]])  
	    # tile_a = tf.expand_dims(tile_a, 2) 
	    # tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1]) 
	    # tile_b = tf.expand_dims(tile_b, 2) 
	    # indices = tf.concat([tile_a, tile_b], axis=2) 
	    # indices = tf.reshape(indices,[-1,2])
	    
	    # 
	    
	    # sparse_update = tf.SparseTensor(indices=indices, values=updates, dense_shape=dense_shape)
	    # exp_map_x = tf.sparse_tensor_to_dense(sparse_update)

	    exp_map = exp_map_p + exp_map_x    
	    ###################################################
	    # y = p
	    # z = x / tf.clip_by_value(norm_x, clip_value_min=K.epsilon(), clip_value_max=np.inf)

	    # exp_map = tf.cosh(norm_x) * y + tf.sinh(norm_x) * z
	    #####################################################
	    exp_map = adjust_to_hyperboloid(exp_map)
	    # idx = tf.where(tf.abs(exp_map + 1) < K.epsilon())[:,0]
	    # params = tf.gather(exp_map, idx)

	    # params = adjust_to_hyperboloid(params)
	    # exp_map = tf.scatter_update(ref=exp_map, updates=params, indices=idx)


	    return exp_map

	def _apply_sparse(self, grad, var):
	# def _apply_sparse_duplicate_indices(self, grad, var):
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
		exp_map = self.exponential_mapping(p, - lr_t * tangent_grad)

		out =  tf.scatter_update(ref=var, updates=exp_map, indices=indices, name="scatter_update")

		# return control_flow_ops.group(out, name="grouping")
		return out
		# return tf.assign(var, var)


class PeriodicStdoutLogger(Callback):

	def __init__(self, reconstruction_edges, labels,
	epoch, n, edge_dict, args):
		self.reconstruction_edges = reconstruction_edges
		self.labels = labels
		self.epoch = epoch
		self.n = n
		self.edge_dict = edge_dict
		self.args = args

	def on_epoch_end(self, batch, logs={}):
	
		self.epoch += 1
		if self.epoch % self.n == 0:
			print ("Completed epoch {}, loss={}, val_loss={}".format(self.epoch, logs["loss"], logs["val_loss"]))

			hyperboloid_embedding = self.model.layers[-1].get_weights()[0]
			print (hyperboloid_embedding)
			# print minkowski_dot_np(hyperboloid_embedding, hyperboloid_embedding)

			mean_rank, mean_average_precision = evaluate_rank_and_MAP(hyperboloid_embedding, self.edge_dict)

			poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
			klein_embedding = hyperboloid_to_klein(hyperboloid_embedding)

			plot_disk_embeddings(self.epoch, self.reconstruction_edges, 
				poincare_embedding, klein_embedding,
				self.labels, 
				mean_rank, mean_average_precision,
				self.args)

def build_model(num_nodes, args):

	x = Input(shape=(1+args.num_positive_samples+args.num_negative_samples,), name="model_input")
	y = EmbeddingLayer(num_nodes, args.embedding_dim, name="embedding_layer")(x)

	model = Model(x, y)

	saved_models = sorted([f for f in os.listdir(args.model_path) if re.match(r"^[0-9][0-9][0-9][0-9]*", f)])
	initial_epoch = len(saved_models)

	print (model.layers[-1].get_weights()[0])

	if initial_epoch > 0:
		model_file = os.path.join(args.model_path, saved_models[-1])
		print ("Loading model from file: {}".format(model_file))
		model.load_weights(model_file)

		print (model.layers[-1].get_weights()[0])

	return model, initial_epoch

def plot_disk_embeddings(epoch, edges, poincare_embedding, klein_embedding, labels, 
	mean_rank, mean_average_precision, args):

	fig = plt.figure(figsize=[14, 7])
	plt.suptitle("Epoch={:05d}, Mean Rank={}, MAP={}".format(epoch, mean_rank, mean_average_precision))
	
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
	
	plt.savefig(os.path.join(args.plot_path, "epoch_{:05d}.png".format(epoch) ))
	plt.close()


def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Hyperbolic Skipgram for feature learning on complex networks")

	parser.add_argument("--dataset", dest="dataset", type=str, default="karate",
		help="The dataset to load. Must be one of [wordnet, cora, citeseer, pubmed,\
		AstroPh, CondMat, GrQc, HepPh, karate]. (Default is karate)")


	parser.add_argument("-r", dest="r", type=float, default=10.,
		help="Radius of hypercircle (defaut is 10).")
	parser.add_argument("-t", dest="t", type=float, default=1.,
		help="Steepness of logistic function (defaut is 1).")


	parser.add_argument("--lr", dest="lr", type=float, default=1e-1,
		help="Learning rate (default is 1e-1).")

	parser.add_argument("--rho", dest="rho", type=float, default=0,
		help="Minimum feature correlation (default is 0).")

	parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=50000,
		help="The number of epochs to train for (default is 50000).")
	parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=128, 
		help="Batch size for training (default is 128).")
	parser.add_argument("--nneg", dest="num_negative_samples", type=int, default=10, 
		help="Number of negative samples for training (default is 10).")
	parser.add_argument("--context-size", dest="context_size", type=int, default=1,
		help="Context size for generating positive samples (default is 1).")
	parser.add_argument("--patience", dest="patience", type=int, default=1000,
		help="The number of epochs of no improvement before training is stopped. (Default is 1000)")

	parser.add_argument("--plot-freq", dest="plot_freq", type=int, default=1000, 
		help="Frequency for plotting (default is 1000).")

	parser.add_argument("--dim", dest="embedding_dim", type=int,
		help="Dimension of embeddings for each layer (default is 2).", default=2)

	parser.add_argument("-p", dest="p", type=float, default=1.,
		help="node2vec return parameter (default is 1.).")
	parser.add_argument("-q", dest="q", type=float, default=1.,
		help="node2vec in-out parameter (default is 1.).")
	parser.add_argument('--num-walks', dest="num_walks", type=int, default=10, 
		help="Number of walks per source (default is 10).")
	parser.add_argument('--walk-length', dest="walk_length", type=int, default=5, 
		help="Length of random walk from source (default is 15).")


	parser.add_argument("--second-order", action="store_true", 
		help="Use this flag to use second order topological similarity information.")
	parser.add_argument("--use-attributes", action="store_true", 
		help="Use this flag to use attributes in the embedding.")
	parser.add_argument("--combine-attributes", action="store_true", 
		help="Use this flag to combine attributes with topology in the embedding.")

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


	args = parser.parse_args()
	return args

def configure_paths(args):
	'''
	build directories on local system for output of model after each epoch
	'''

	dataset = args.dataset
	directory = "embedding_dim={}_r={}_t={}_rho={}".format(args.embedding_dim, args.r, args.t, args.rho)
	if args.second_order:
		directory += "_second_order"
	if args.combine_attributes:
		directory += "_combine_attributes"
	elif args.use_attributes:
		directory += "_with_attributes"

	if args.softmax:
		directory += "_softmax"
	elif args.sigmoid:
		directory += "_sigmoid"

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
	args.log_path = os.path.join(args.log_path, directory + ".log")

	args.board_path = os.path.join(args.board_path, dataset)
	if not os.path.exists(args.board_path):
		os.makedirs(args.board_path)
	args.board_path = os.path.join(args.board_path, directory)
	if not os.path.exists(args.board_path):
		os.makedirs(args.board_path)


	args.walk_path = os.path.join(args.walk_path, dataset)
	if not os.path.exists(args.walk_path):
		os.makedirs(args.walk_path)

	args.model_path = os.path.join(args.model_path, dataset)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	args.model_path = os.path.join(args.model_path, directory)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

def main():

	args = parse_args()
	args.num_positive_samples = 1
	if args.combine_attributes:
		args.use_attributes = True
	configure_paths(args)

	dataset = args.dataset
	if dataset == "karate":
		topology_graph, features, labels = load_karate()
	elif dataset in ["cora", "pubmed", "citeseer"]:
		topology_graph, features, labels = load_labelled_attributed_network(dataset)
	else:
		raise Exception

	# original edges for reconstruction
	reconstruction_edges = topology_graph.edges()
	edge_dict = convert_edgelist_to_dict(reconstruction_edges)


	topology_walk_file = os.path.join(args.walk_path, "top_walks-{}-{}-{}-{}".format(args.num_walks, 
		args.walk_length, args.p, args.q))
	if args.second_order:
		topology_walk_file += "_second_order"
		print ("weighting edges by second order similarity")
		second_order_topology_graph = create_second_order_topology_graph(topology_graph, args)
		top_adj = nx.adjacency_matrix(topology_graph).A
		second_order_top_adj = nx.adjacency_matrix(second_order_topology_graph).A
		combined_adjacency = top_adj + second_order_top_adj 
		topology_graph = nx.from_numpy_matrix(combined_adjacency)

	walks = load_walks(topology_graph, topology_walk_file, args)


	if args.use_attributes:
		feature_graph = create_feature_graph(features, args)

		if args.combine_attributes:
			top_adj = nx.adjacency_matrix(topology_graph).A
			feat_adj = nx.adjacency_matrix(feature_graph).A

			combined_adjacency = top_adj + feat_adj
			combined_graph = nx.from_numpy_matrix(combined_adjacency)
			combined_walk_file = os.path.join(args.walk_path, "combined_walks-{}-{}-{}-{}".format(args.num_walks, 
				args.walk_length, args.p, args.q))
			if args.second_order:
				combined_walk_file += "_second_order"
			walks = load_walks(combined_graph, combined_walk_file, args)

		else:

			feature_walk_file = os.path.join(args.walk_path, "feat_walks-{}-{}-{}-{}".format(args.num_walks, 
				args.walk_length, args.p, args.q))
			walks += load_walks(feature_graph, feature_walk_file, args)


	positive_samples, negative_samples, probs =\
		determine_positive_and_negative_samples(nodes=topology_graph.nodes(), 
		walks=walks, context_size=args.context_size)



	num_nodes = len(topology_graph)
	num_steps = int((len(positive_samples) + args.batch_size - 1) / args.batch_size)
	# num_steps=1

	training_gen = training_generator(positive_samples, negative_samples, probs,
									  num_negative_samples=args.num_negative_samples, batch_size=args.batch_size)
	with tf.device("/gpu:0"):

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

		val_in, val_out = training_gen.next()

		# model.fit(_in, _out, epochs=1, verbose=args.verbose)
		# raise SystemExit

		model.fit_generator(training_gen, epochs=args.num_epochs, 
			steps_per_epoch=num_steps, verbose=args.verbose, initial_epoch=initial_epoch,
			validation_data=[val_in, val_out],
			callbacks=[
				TerminateOnNaN(), 
				PeriodicStdoutLogger(reconstruction_edges, labels, 
					n=args.plot_freq, epoch=initial_epoch, edge_dict=edge_dict, args=args), 
				TensorBoard(log_dir=args.board_path, histogram_freq=1, 
					batch_size=args.batch_size, write_graph=True, write_grads=True, write_images=True, 
					embeddings_freq=1, embeddings_layer_names="embedding_layer", 
					embeddings_metadata="/home/david/Documents/capsnet_embedding/data/karate/labels.tsv"
					),
				ModelCheckpoint(os.path.join(args.model_path, 
					"{epoch:05d}.h5"), save_weights_only=True),
				CSVLogger(args.log_path, append=True), 
				EarlyStopping(monitor="val_loss", patience=args.patience, verbose=1)
			])



if __name__ == "__main__":
	main()