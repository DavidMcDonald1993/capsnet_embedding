import os
import gzip
import random
import numpy as np
import networkx as nx
import scipy as sp
import pandas as pd 

import pickle as pkl

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, normalized_mutual_info_score

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.callbacks import Callback

# from data_utils import preprocess_data
from node2vec_sampling import Graph 
from metrics import evaluate_link_prediction

def load_walks(G, walk_file, args):

	if not os.path.exists(walk_file):
		node2vec_graph = Graph(nx_G=G, is_directed=False, p=args.p, q=args.q)
		node2vec_graph.preprocess_transition_probs()
		walks = node2vec_graph.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length)
		with open(walk_file, "wb") as f:
			pkl.dump(walks, f)
		print "saved walks to {}".format(walk_file)
	else:
		print "loading walks from {}".format(walk_file)
		with open(walk_file, "rb") as f:
			walks = pkl.load(f)
	return walks


def compute_label_mask(Y, num_patterns_to_keep=20):

	assignments = Y.argmax(axis=1)
	patterns_to_keep = np.concatenate([np.random.choice(np.where(assignments==i)[0], replace=False, size=num_patterns_to_keep)  
										 for i in range(Y.shape[1])])
	mask = np.zeros(Y.shape, dtype=np.float32)
	mask[patterns_to_keep] = 1

	return mask


def create_neighbourhood_sample_list(nodes, neighbourhood_sample_sizes, neighbours):

	neighbourhood_sample_list = [nodes]

	for neighbourhood_sample_size in neighbourhood_sample_sizes[::-1]:

		neighbourhood_sample_list.append(np.array([np.concatenate([np.append(n, np.random.choice(neighbours[n], 
			replace=True, size=neighbourhood_sample_size)) for n in batch]) for batch in neighbourhood_sample_list[-1]]))

	# flip neighbour list
	neighbourhood_sample_list = neighbourhood_sample_list[::-1]


	return neighbourhood_sample_list



class EmbeddingCallback(Callback):

	def __init__(self, G, X, Y, removed_edges, neighbourhood_sample_sizes, num_capsules_per_layer,
		embedder, predictor, batch_size,
	 # annotate_idx, 
	 embedding_path, plot_path,):
		self.G = G 
		self.X = X
		self.Y = Y
		self.removed_edges = sorted(removed_edges, key=lambda(u, v): (u, v))
		self.neighbourhood_sample_sizes = neighbourhood_sample_sizes
		self.num_capsules_per_layer = num_capsules_per_layer
		self.embedder = embedder
		self.predictor = predictor
		self.batch_size = batch_size
		# self.annotate_idx = annotate_idx
		self.embedding_path = embedding_path
		self.plot_path = plot_path
		self.removed_edges_dict = self.convert_edgelist_to_dict(removed_edges)
		self.G_edge_dict = self.convert_edgelist_to_dict(G.edges(), self_edges=True)
		# self.candidate_edges_path = candidate_edges_path
		# if not os.path.exists(candidate_edges_path):
		# 	print "writing candidate edges to {}".format(candidate_edges_path)
		# 	self.write_candidate_edges()

	def convert_edgelist_to_dict(self, edgelist, undirected=True, self_edges=False):
		sorts = [lambda x: sorted(x)]
		if undirected:
			sorts.append(lambda x: sorted(x, reverse=True))
		edges = (sort(edge) for edge in edgelist for sort in sorts)
		edge_dict = {}
		for u, v in edges:
			if self_edges:
				default = {u}
			else:
				default = set()
			edge_dict.setdefault(u, default).add(v)
		return edge_dict

	def evaluate_rank_and_MAP(self, embedding, ):

		def hyperbolic_distance(u, v):
			return np.arccosh(1 + 2 * np.linalg.norm(u - v, axis=-1)**2 / ((1 - np.linalg.norm(u, axis=-1)**2) * (1 - np.linalg.norm(v, axis=-1)**2)))

		def sigmoid(x):
			return 1. / (1 + np.exp(-x))

		# def check_edge_in_edgelist((u, v), edgelist):
		# 	for u_prime, v_prime in edgelist:
		# 		if u_prime > u:
		# 			return False
		# 		if u_prime == u and v_prime == v:
		# 			edgelist.remove((u, v))
		# 			return True
		
		print "evaluating rank and MAP"

		G = self.G
		G_edge_dict = self.G_edge_dict
		removed_edges_dict = self.removed_edges_dict
		N = len(G)

		r = 1.
		t = 1.

		ranks = np.zeros(len(removed_edges_dict))
		MAPs = np.zeros(len(removed_edges_dict))

		for i, u in enumerate(sorted(removed_edges_dict.keys())):
			u_neighbors_in_G = G_edge_dict[u]
			removed_u_neighbours = removed_edges_dict[u]
			removed_u_neighbours_dist = hyperbolic_distance(embedding[u], embedding[list(removed_u_neighbours)])
			removed_u_neighbours_P = sigmoid((r - removed_u_neighbours_dist) / t)
			all_neighbours = u_neighbors_in_G.union(removed_u_neighbours)
			non_neighbours = list(set(range(N)).difference(all_neighbours))
			non_neighbour_dists = hyperbolic_distance(embedding[u], embedding[non_neighbours])
			non_neighbour_P = sigmoid((r - non_neighbour_dists) / t)
			
			y_true = np.append(np.ones(len(removed_u_neighbours_P)), np.zeros(len(non_neighbour_P)))
			y_pred = np.append(removed_u_neighbours_P, non_neighbour_P)
			
			MAPs[i] = average_precision_score(y_true, y_pred)
			y_pred[::-1].sort()
			ranks[i] = np.array([np.searchsorted(-y_pred, -p) for p in removed_u_neighbours_P]).mean()

			if i % 1000 == 0:
				print "completed node {}/{}".format(i, N)

		return ranks.mean(), MAPs.mean()

			


		# removed_edges = self.removed_edges_val[:]
		# removed_edges.sort(key=lambda (u, v): (u, v))

	# def write_candidate_edges(self):
	# 	G = self.G
	# 	N = len(G)
	# 	candidate_edges = ((u, v)for u in range(N) for v in range(u+1, N) if (u, v) not in G.edges() and (v, u) not in G.edges())
	# 	with gzip.open(self.candidate_edges_path, "w") as f:
	# 		for u, v in candidate_edges:
	# 			f.write("{} {}\n".format(u, v))


	def on_epoch_end(self, epoch, logs={}):
		embedding = self.perform_embedding()
		# average_precision = evaluate_link_prediction(self.G, embedding, 
		# 	self.removed_edges_val, epoch=epoch, path=self.plot_path, candidate_edges_path=self.candidate_edges_path)
		mean_rank, mean_precision = self.evaluate_rank_and_MAP(embedding)
		logs.update({"mean_rank" : mean_rank, "mean_precision" : mean_precision})

		self.plot_embedding(embedding, path="{}/embedding_epoch_{:04}".format(self.embedding_path, epoch))


	def perform_embedding(self):

		def embedding_generator(X, input_nodes, batch_size=100):
			num_steps = (input_nodes.shape[0] + batch_size - 1) / batch_size
			for step in range(num_steps):
				batch_nodes = input_nodes[batch_size*step : batch_size*(step+1)]
				x = X[batch_nodes]
				yield x.reshape([-1, input_nodes.shape[1], 1, X.shape[-1]])

		G = self.G
		X = self.X
		neighbourhood_sample_sizes = self.neighbourhood_sample_sizes
		embedder = self.embedder
		batch_size = self.batch_size

		nodes = np.arange(len(G)).reshape(-1, 1)
		# nodes = nodes.reshape(-1, 1)
		neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
		neighbour_list = create_neighbourhood_sample_list(nodes, neighbourhood_sample_sizes, neighbours)

		input_nodes = neighbour_list[0]
		# original_shape = list(input_nodes.shape)
		# print original_shape, X.shape
		# raise SystemExit
		# print input_nodes.flatten().shape
		# x = X[input_nodes.flatten()]#.toarray()
		# x = preprocess_data(x)
		# add atrifical bacth dimension and capsule dimension 
		# x = x.reshape(original_shape + [1, -1])

		# x = X[neighbour_list[0]]
		# x = np.expand_dims(x, 2)

		# embedding = embedder.predict(x)
		num_steps = (input_nodes.shape[0] + batch_size - 1) / batch_size
		embedding_gen = embedding_generator(X, input_nodes, batch_size=batch_size)
		embedding = embedder.predict_generator(embedding_gen, steps=num_steps)
		dim = embedding.shape[-1]
		embedding = embedding.reshape(-1, dim)

		return embedding

	def plot_embedding(self, embedding, path):

		y = self.Y.argmax(axis=1)#.A1
		embedding_dim = embedding.shape[-1]

		fig = plt.figure(figsize=(10, 10))
		if embedding_dim == 3:
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=y)
		else:
			# if self.annotate_idx and embedding.shape[0] == self.annotate_idx.shape[0]:
			# 	y = y[annotate_idx]
			plt.scatter(embedding[:,0], embedding[:,1], c=y)
			# if self.annotate_idx is not None:
			# 	H = self.G.subgraph(annotate_idx)
			# 	original_names = nx.get_node_attributes(H, "original_name").values()
			# 	if embedding.shape[0] == self.annotate_idx.shape[0]:
			# 		annotation_points = embedding
			# 	else:
			# 		annotation_points = embedding[annotate_idx]			
			# 	# for label, p in zip(original_names, annotation_points):
			# 	# 	plt.annotate(label, p)
		if path is not None:
			plt.savefig(path)
		plt.close()

	
	def make_and_evaluate_label_predictions(self):

		print "evaluating label predictions"


		def prediction_generator(X, input_nodes, batch_size=100):
			num_steps = (input_nodes.shape[0] + batch_size - 1) / batch_size
			for step in range(num_steps):
				batch_nodes = input_nodes[batch_size*step : batch_size*(step+1)]
				x = X[batch_nodes]
				yield x.reshape([-1, input_nodes.shape[1], 1, X.shape[-1]])

		G = self.G
		X = self.X
		Y = self.Y
		predictor = self.predictor
		num_capsules_per_layer = self.num_capsules_per_layer
		neighbourhood_sample_sizes = self.neighbourhood_sample_sizes
		batch_size = self.batch_size


		_, num_classes = Y.shape
		label_prediction_layers = np.where(num_capsules_per_layer==num_classes)[0] + 1

		nodes = np.arange(len(G)).reshape(-1, 1)
		neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
		neighbour_list = create_neighbourhood_sample_list(nodes, neighbourhood_sample_sizes[:label_prediction_layers[-1]], neighbours)

		input_nodes = neighbour_list[0]

		num_steps = (input_nodes.shape[0] + batch_size - 1) / batch_size
		prediction_gen = prediction_generator(X, input_nodes, batch_size=batch_size)

		predictions = predictor.predict_generator(prediction_gen, steps=num_steps)
		predictions = predictions.reshape(-1, predictions.shape[-1])

		true_labels = Y.argmax(axis=-1)
		predicted_labels = predictions.argmax(axis=-1)

		print "NMI of predictions: {}".format(normalized_mutual_info_score(true_labels, predicted_labels))
		print "Classification accuracy: {}".format((true_labels==predicted_labels).sum() / float(true_labels.shape[0]))


