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

from node2vec_sampling import Graph

eps = 1e-6
np.set_printoptions(suppress=True)

def load_karate():

	G = nx.read_edgelist("../data/karate/karate.edg")

	label_df = pd.read_csv("../data/karate/mod-based-clusters.txt", sep=" ", index_col=0, header=None,)
	label_df.index = [str(idx) for idx in label_df.index]
	label_df = label_df.reindex(G.nodes())

	assignments = label_df.iloc[:,0].values


	G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")
	nx.set_edge_attributes(G=G, name="weight", values=1)


	node2vec_graph = Graph(nx_G=G, is_directed=False, p=1, q=1)
	node2vec_graph.preprocess_transition_probs()
	walks = node2vec_graph.simulate_walks(num_walks=10, walk_length=5)



	X = np.genfromtxt("../data/karate/feats.csv", delimiter=",")
	X_sim = cosine_similarity(X)
	X_sim -= np.identity(len(G))
	X_sim[X_sim < 0] = 0
	G_att = nx.from_numpy_matrix(X_sim)

	node2vec_graph = Graph(nx_G=G_att, is_directed=False, p=1, q=1)
	node2vec_graph.preprocess_transition_probs()
	walks_att = node2vec_graph.simulate_walks(num_walks=10, walk_length=5)


	# walks_att = []


	positive_samples, ground_truth_negative_samples =\
	determine_positive_and_ground_truth_negative_samples(G, walks+walks_att, context_size=1)
	return (G, assignments, positive_samples, ground_truth_negative_samples)

def load_labelled_attributed_network(dataset_str, ):
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

	test_label_idx = test_idx_range.tolist()
	train_label_idx = list(range(len(y)))
	val_label_idx = list(range(len(y), len(y)+500))

	G = nx.from_numpy_matrix(adj.toarray())
	G = nx.convert_node_labels_to_integers(G, label_attribute="original_name")
	nx.set_edge_attributes(G=G, name="weight", values=1)

	# all_edges = G.edges()
	features = features.A
	scaler = StandardScaler()
# 	features = scaler.fit_transform(features)
	X_train = features
	X_val = features
	X_test = features

	Y = labels

	# same networks for train, test, val
	G_train = G
	G_val = G 
	G_test = G
	# all_edges = G_train.edges()
# 	G_train_neighbours = pad_neighbours(G_train, sample_sizes)
# 	G_val_neighbours = G_train_neighbours
# 	G_test_neighbours = G_train_neighbours

	# no removed edges for GCN networks
	val_edges = None
	test_edges = None

	positive_samples, ground_truth_negative_samples = determine_positive_and_ground_truth_negative_samples(G_train)

	return (G, Y.argmax(axis=-1), positive_samples, ground_truth_negative_samples,)

def euclidean_to_hyperboloid(X):
	r = np.linalg.norm(X, axis=-1, keepdims=True) + 1e-7
	d = X / r
	return np.append(np.sinh(r) * d, np.cosh(r), axis=-1)

def minkowski_dot(X, Y):
	J = np.array([1] * (X.shape[-1] - 1) + [-1.])
	# print J.dtype
	# raise SystemExit
	B =  (X * J * Y).sum(axis=-1, keepdims=True) 
	return B

def minkowski_dot_pairwise(u, v):
	"""
	`u` and `v` are vectors in Minkowski space.
	"""
	rank = u.shape[-1] - 1
	euc_dp = u[:,:rank].dot(v[:,:rank].T)
	return euc_dp - u[:,rank, None] * v[:,rank]

def hyperbolic_distance_hyperboloid(X, Y):
	inner_product = minkowski_dot(X, Y)
	inner_product = np.clip(inner_product, a_max=-1, a_min=-np.inf)
	return np.arccosh(-inner_product)

def hyperbolic_distance_hyperboloid_pairwise(X, Y):
	inner_product = minkowski_dot_pairwise(X, Y)
	inner_product = np.clip(inner_product, a_max=-1, a_min=-np.inf)
	return np.arccosh(-inner_product)

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def poincare_ball_to_hyperboloid(X):
	x = 2 * X
	t = 1 + np.sum(np.square(X), axis=-1, keepdims=True)
	return 1 / (1 - np.sum(np.square(X), axis=-1, keepdims=True)) * np.append(x, t, axis=-1)

# def ambient_to_tangent(p, x):
# 	tang = x + minkowski_dot(p, x) * p
# 	return tang

def project_onto_tangent_space(hyperboloid_point, minkowski_tangent):
	# assert np.allclose(minkowski_dot(hyperboloid_point, hyperboloid_point), -1, rtol=0, atol=eps)
	tang = minkowski_tangent + minkowski_dot(hyperboloid_point, minkowski_tangent) * hyperboloid_point
	# assert np.allclose(minkowski_dot(hyperboloid_point, tang), 0, rtol=0, atol=eps), (
	# 	minkowski_dot(hyperboloid_point, tang),
	# 	hyperboloid_point, 
	# 	tang
	# 	)
	return tang

def exponential_mapping(p, x, max_norm=1.):

	def adjust_to_manifold(x):
		t = np.sqrt(1 + np.linalg.norm(x[:,:-1], axis=-1, keepdims=False)**2)
		x [:,-1] = t
		return x

	norm_x = np.sqrt( np.maximum(0, minkowski_dot(x, x)) )
	norm_idx = np.where(norm_x > max_norm)[0]
	if len(norm_idx) > 0:
		# print norm_x
		# print norm_idx
		x[norm_idx] /= (norm_x[norm_idx] / max_norm)
		# norm_x[norm_idx] = max_norm
		# print "clipped max norm"
		norm_x = np.sqrt( np.maximum(0, minkowski_dot(x, x)) )
		# print norm_x
		# raise SystemExit
	exp_map = p.copy()
	idx = np.where(norm_x > 0)[0]
	# print norm_x[:,0]
	# print 0 <= norm_x[:,0] <= max_norm
	# raise SystemExit
	# assert (0 <= norm_x[:,0]).all() and (norm_x[:,0] <= max_norm+1e-8).all(), (
	# 	norm_x, max_norm, norm_x <= max_norm#, len( norm_idx)
	# )
	# assert not np.isnan(norm_x).any()
	exp_x = np.exp(norm_x[idx])
	y = p[idx]
	z = x[idx] / norm_x[idx]
	exp_map[idx] = adjust_to_manifold(.5 * (exp_x * (y + z) + (y - z) / exp_x))
	# exp_map[idx] = np.cosh(norm_x[idx]) * p[idx] + np.sinh(norm_x[idx]) * x[idx] / norm_x[idx]
	# assert np.allclose(minkowski_dot(exp_map, exp_map), -1, rtol=0, atol=eps), (minkowski_dot(exp_map, exp_map), p, x)
	# idx = np.where(exp_map[:,-1] < 1)[0]
	# assert np.allclose(minkowski_dot(p, p), -1, rtol=0, atol=eps)
	# assert np.allclose(minkowski_dot(exp_map, exp_map), -1, rtol=0, atol=eps), (
	# 	y,
	# 	z,
	# 	exp_x,
	# 	minkowski_dot(p, p),
	# 	minkowski_dot(exp_map, exp_map),
	# # 	# (norm_x>max_norm), 
	# # 	# np.where(norm_x>max_norm),
	# # 	norm_x[idx],
	# # 	# np.cosh(norm_x[idx]),
	# # 	# np.sinh(norm_x[idx])
	# # np.sqrt( np.maximum(0, minkowski_dot(z, z)) )
	# )
	# # if len(idx) > 0:
	# 	print exp_map
	# 	exp_map[idx] = np.array([0,0,1])
		# print "clipped neg 1"
		# raise SystemExit
	return exp_map

def sphere_uniform_sample(shape, r_max=1):
	num_samples, dim = shape
	X = np.random.normal(size=shape)
	X_norm = np.linalg.norm(X, axis=-1, keepdims=True)
	U = np.random.uniform(size=(num_samples, 1))
	return r_max * U ** (1./dim) * X / X_norm

def random_initialization(shape, r_max=1e-2):
	X = sphere_uniform_sample(shape, r_max=r_max)
	return poincare_ball_to_hyperboloid(X)


def determine_positive_and_ground_truth_negative_samples(G, walks, context_size):

	print ("determining positive and negative samples")

	nodes = set(G.nodes())

	# positive_samples = G.edges() + [(v, u) for u, v in G.edges()]
	# all_positive_samples = {n : set(G.neighbors(n)) for n in G.nodes()}
	# print positive_samples
	# print all_positive_samples
	# raise SystemExit
	
	all_positive_samples = {n: set() for n in G.nodes()}
	positive_samples = []
	for num_walk, walk in enumerate(walks):
		for i in range(len(walk)):
			for j in range(i+1, min(len(walk), i+1+context_size)):
				u = walk[i]
				v = walk[j]
				if u == v:
					continue

				positive_samples.append((u, v))
				positive_samples.append((v, u))
				
				all_positive_samples[u].add(v)
				all_positive_samples[v].add(u)
 
		if num_walk % 1000 == 0:  
			print ("processed walk {}/{}".format(num_walk, len(walks)))
			
	ground_truth_negative_samples = {n: sorted(list(nodes.difference(all_positive_samples[n]))) for n in G.nodes()}
	for u in ground_truth_negative_samples:
		# print len(ground_truth_negative_samples[u])
		assert len(ground_truth_negative_samples[u]) > 0, "node {} does not have any negative samples".format(u)

	# print ground_truth_negative_samples
	# raise SystemExit

	print ("DETERMINED POSITIVE AND NEGATIVE SAMPLES")
	print ("found {} positive sample pairs".format(len(positive_samples)))

	# # print len(walks)
	# print G.neighbors(1)
	# print nodes
	# print all_positive_samples[1]
	# print len(nodes.difference(all_positive_samples[1]))
	# # print positive_samples[:10]
	# # print all_positive_samples
	# # print ground_truth_negative_samples
	# raise SystemExit
	
	return positive_samples, ground_truth_negative_samples

# def determine_positive_and_ground_truth_negative_samples(G, undirected=True):

# 	nodes = set(G.nodes())
# 	positive_samples = G.edges()
# 	# positive_samples += [(u,u) for u in G.nodes()]
# 	if undirected:
# 		positive_samples += [(v, u) for u, v in G.edges()]
# 	all_positive_samples = {n : set(G.neighbors(n) + [n]) for n in G.nodes()}

# 	ground_truth_negative_samples = {n: sorted(list(nodes.difference(all_positive_samples[n]))) for n in G.nodes()}

# 	for u, neg_samples in ground_truth_negative_samples.items():
# 		assert len(ground_truth_negative_samples[u]) > 0, "node {} does not have any negative samples".format(u)
# 		# for v in neg_samples:
# 		# 	assert not (u, v) in G.edges() and (v, u) not in G.edges()

# 	return positive_samples, ground_truth_negative_samples

def get_training_sample(batch_positive_samples, ground_truth_negative_samples, num_negative_samples):


	batch_positive_samples = np.array(batch_positive_samples)	
	batch_negative_samples = np.array([
		np.random.choice(ground_truth_negative_samples[u], 
		replace=True, size=(num_negative_samples,))
		for u in batch_positive_samples[:,0]
	], dtype=np.int32)
	batch_nodes = np.append(batch_positive_samples, batch_negative_samples, axis=1)
	return batch_nodes


def training_generator(positive_samples, ground_truth_negative_samples, num_negative_samples,
					  batch_size=10):
	num_steps = int((len(positive_samples) + batch_size - 1 )/ batch_size)
	random.shuffle(positive_samples)

	for step in range(num_steps):

		batch_positive_samples = positive_samples[step * batch_size : (step + 1) * batch_size]
		training_sample = get_training_sample(batch_positive_samples, 
											  ground_truth_negative_samples, num_negative_samples)
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

def update(W1, 
	# W2, 
	nodes, alpha=0.01, max_norm=1., r=1., sigma=1., batch_size=10):
	
	def sigmoid(x):
		sigma = 1. / (1 + np.exp(-x))
		# assert not np.isnan(sigma).any(), sigma
		return sigma

	def loss(outputs):
		return - np.mean(np.sum(np.log(outputs), axis=-1))
	
	u = nodes[:,0]
	samples = nodes[:,1:]
	
	N, dim = W1.shape
	num_nodes_in_batch, num_samples = samples.shape

	num_negative_samples = num_samples - 1
	
	u_emb = W1[u]
	# samples_emb = W2[samples]
	samples_emb = W1[samples]

	#########################################################

	# out_j = np.concatenate([minkowski_dot(u_emb, samples_emb[:,j]) for j in range(num_samples)], axis=-1)
	
	# for i in range(batch_size):
	# 	for j in range(num_samples):
	# 		assert np.allclose(out_j[i, j], B(u_emb[i], samples_emb[i,j])) 
	
	# outputs = sigmoid(np.array([[1] + [-1] * (num_samples-1)]) * out_j)
	# print "loss={}".format(loss(outputs))

	# partial_E_partial_out_j = sigmoid(out_j)
	# partial_E_partial_out_j[:,0] -= 1.

	# nabla_E_vj_ambient = np.array([
		# partial_E_partial_out_j[i,:,None] * u_emb[i] for i in range(batch_size)
	# ])


	#########################

	inner_uv = np.concatenate([minkowski_dot(u_emb, samples_emb[:,j]) for j in range(num_samples)], axis=-1)
	inner_uv = np.clip(inner_uv, a_min=-np.inf, a_max=-(1+1e-10))
	d_uv = np.arccosh(-inner_uv)
	out_uv = (r - d_uv ** 2) / sigma
	out_uv[:,1:] *= -1
	p_uv = sigmoid(out_uv)

	# print "loss=", - np.mean(np.sum(np.log(p_uv), axis=-1))

	# partial_E_partial_d = 2 * d_uv * (1 - p_uv) / sigma 
	# partial_E_partial_d[:,1:] *= -1

	# partial_d_partial_inner = - 1. / np.sqrt(inner_uv ** 2 - 1 )

	# partial_E_partial_inner = partial_E_partial_d * partial_d_partial_inner

	partial_E_partial_inner = -2 * (1 - p_uv) * d_uv  / ( np.sqrt(inner_uv ** 2 - 1 ) * sigma )
	partial_E_partial_inner [:,1:] *= -1


	nabla_E_vj_ambient = np.array([
		partial_E_partial_inner[i,:,None] * u_emb[i] for i in range(num_nodes_in_batch)
	])


	#########################

	# mean over batch
	nabla_E_vj_ambient /= batch_size

	# mean over negative samples
	nabla_E_vj_ambient[:,1:] /= num_negative_samples 

	################################

	W1_update_dict = {}

	for vj, grad in zip(samples.reshape(-1,), nabla_E_vj_ambient.reshape(-1, W1.shape[1])):
		if vj in W1_update_dict:
			W1_update_dict[vj] += grad
		else:
			W1_update_dict[vj] = grad


	# nabla_E_vj_tangent = ambient_to_tangent(p=samples_emb, x=nabla_E_vj_ambient)
	

	##############################
	# CONSTRUCT W2 update dict
	# print "W2 dict"
	# W2_update_dict = {}
	# W1_update_dict = {}

	# for vj, grad in zip(samples.reshape(-1,), nabla_E_vj_tangent.reshape(-1, W1.shape[1])):
	# 	if vj in W1_update_dict:
	# 		W1_update_dict[vj] += grad
	# 	else:
	# 		W1_update_dict[vj] = grad

	###############################



	nabla_E_ui_ambient = partial_E_partial_inner[:,:,None] * samples_emb



	################################################

	# mean over batch
	nabla_E_ui_ambient /= batch_size

	# mean over negative samples
	nabla_E_ui_ambient[:,1:] /= num_negative_samples 

	# sum over all samples for each u
	nabla_E_ui_ambient = nabla_E_ui_ambient.sum(axis=1)

	for ui, grad in zip(u.reshape(-1,), nabla_E_ui_ambient.reshape(-1, W1.shape[1])):
		if ui in W1_update_dict:
			W1_update_dict[ui] += grad
		else:
			W1_update_dict[ui] = grad

	
	# nabla_E_ui_tangent = ambient_to_tangent(p=u_emb, x=nabla_E_ui_ambient)

	##############################
	# CONSTRUCT W1 update dict

	# print "W1 dict"
	# W1_update_dict = {}
	# for ui, grad in zip(u.reshape(-1,), nabla_E_ui_tangent.reshape(-1, W1.shape[1])):
	# 	if ui in W1_update_dict:
	# 		W1_update_dict[ui] += grad
	# 	else:
	# 		W1_update_dict[ui] = grad
	# # print 
			
	# for k, v in W1_update_dict.items():
		# if len(v.shape) > 1:
			# v = v.mean(axis=0)
			# W1_update_dict[k] = v
	# 	# assert abs(B(W1[k], v) < eps), (k, W1[k], v, B(W1[k], v))

	###############################

	# MAKE THE UPDATES TO W1 and W2

	W1_update_nodes = np.array(W1_update_dict.keys())
	W1_ambient_updates = np.array(W1_update_dict.values())

	W1_tangent_updates = project_onto_tangent_space(hyperboloid_point=W1[W1_update_nodes], minkowski_tangent=W1_ambient_updates)
	W1[W1_update_nodes] = exponential_mapping(p=W1[W1_update_nodes], x=-alpha * W1_tangent_updates, max_norm=max_norm)

	# W2_update_nodes = np.array(W2_update_dict.keys())
	# W2_tangent_updates = np.array(W2_update_dict.values())
	# W2[W2_update_nodes] = exponential_mapping(p=W2[W2_update_nodes], x=-alpha * W2_tangent_updates)

	# assert (np.allclose(minkowski_dot(W1, W1), -1)), (minkowski_dot(W1, W1).max(), minkowski_dot(W1, W1).min())
	# assert (np.allclose(minkowski_dot(W2, W2), -1)), (minkowski_dot(W2, W2).max(), minkowski_dot(W2, W2).min())

	# assert (W1[:,-1] > 0).all(), (
	# 	minkowski_dot(W1, W1),
	# 	# minkowski_dot_pairwise(W1, W1)
	# )
	return W1#, W2 


def main():
	# G, assignments, positive_samples, ground_truth_negative_samples = load_labelled_attributed_network("cora")
	G, assignments, positive_samples, ground_truth_negative_samples = load_karate()
	edge_dict = convert_edgelist_to_dict(G.edges())


	N = len(G)
	dim = 2

	W1 = random_initialization(shape=(N, dim),r_max=1e-10)

	print W1
	# print minkowski_dot(W1, W1)
	# print minkowski_dot_pairwise(W1, W1)
	# raise SystemExit
	# W2 = random_initialization(shape=(N, dim), r_max=1e-3)
	# print W1
	# print W2
	# evaluate_rank_and_MAP(edge_dict=edge_dict, embedding=W1)
	# evaluate_rank_and_MAP(edge_dict=edge_dict, embedding=W2)
	# print 


	alpha = 1e-3
	r = 10
	sigma = 1
	max_epochs = 100000
	batch_size = 100
	num_negative_samples = 10
	max_norm = 1e-0

	for epoch in range(1, max_epochs+1):
		training_gen = training_generator(positive_samples, ground_truth_negative_samples, 
									  num_negative_samples=num_negative_samples, batch_size=batch_size)
		for batch_nodes in training_gen:
			W1 = update(W1, 
				# W2, 
				nodes=batch_nodes, alpha=alpha, max_norm=max_norm, r=r, sigma=sigma, batch_size=batch_size)

		# print "Completed epoch {} / {}".format(epoch, max_epochs) 

		if epoch % 1000 == 0:
			print W1
			# print minkowski_dot_pairwise(W1, W1).diagonal()
			# print hyperboloid_to_poincare_ball(W2)
			# print hyperbolic_distance_hyperboloid_pairwise(W1, W1) ** 2
			print "Completed epoch {} / {}".format(epoch, max_epochs) 
			evaluate_rank_and_MAP(edge_dict=edge_dict, embedding=W1)
			# evaluate_rank_and_MAP(edge_dict=edge_dict, embedding=W2)
			# print
			
	evaluate_rank_and_MAP(edge_dict=edge_dict, embedding=W1)
	# evaluate_rank_and_MAP(edge_dict=edge_dict, embedding=W2)

	# print minkowski_dot(W1, W1)

	W_disk = hyperboloid_to_poincare_ball(W1)
	assert (np.linalg.norm(W_disk, axis=-1) < 1).all()
	plt.figure(figsize=[7, 7])
	c = assignments
	for u, v in G.edges():
		u_emb = W_disk[u]
		v_emb = W_disk[v]
		plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.05, zorder=0)
	plt.scatter(W_disk[:,0], W_disk[:,1], c=c, zorder=1)
	plt.xlim([-1,1])
	plt.ylim([-1,1])
	plt.show()

	# print W2

	# W_disk = hyperboloid_to_poincare_ball(W2)
	# assert (np.linalg.norm(W_disk, axis=-1) < 1).all()
	# plt.figure(figsize=[7, 7])
	# c = assignments
	# for u, v in G.edges():
	# 	u_emb = W_disk[u]
	# 	v_emb = W_disk[v]
	# 	plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.05, zorder=0)
	# plt.scatter(W_disk[:,0], W_disk[:,1], c=c, zorder=1)
	# plt.xlim([-1,1])
	# plt.ylim([-1,1])
	# plt.show()


if __name__ == "__main__":
	main()