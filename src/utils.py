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
from sklearn.metrics import average_precision_score, normalized_mutual_info_score, accuracy_score

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.callbacks import Callback

from data_utils import preprocess_data
from node2vec_sampling import Graph 
from metrics import evaluate_link_prediction

def load_positive_samples_and_ground_truth_negative_samples(G, walks, args, 
	walk_file, positive_samples_filename, negative_samples_filename):

	
	if os.path.exists(positive_samples_filename):

		with open(positive_samples_filename, "rb") as f:
			positive_samples = pkl.load(f)
		with open(negative_samples_filename, "rb") as f:
			ground_truth_negative_samples = pkl.load(f)

	else:

		walks = load_walks(G, walk_file, args)
		positive_samples, ground_truth_negative_samples = determine_positive_and_groud_truth_negative_samples(G, walks, context_size)

		with open(positive_samples_filename, "wb") as f:
			pkl.dump(positive_samples, f)
		with open(negative_samples_filename, "wb") as f:
			pkl.dump(ground_truth_negative_samples, f)


	return positive_samples, ground_truth_negative_samples

def determine_positive_and_groud_truth_negative_samples(G, walks, context_size):

	print "determining positive and negative samples"
	
	N = len(G)
	nodes = set(G.nodes())
	
	all_positive_samples = {n: set() for n in G.nodes()}
	positive_samples = []
	for num_walk, walk in enumerate(walks):
		for i in range(len(walk)):
			for j in range(i+1, min(len(walk), i+1+context_size)):
				u = walk[i]
				v = walk[j]

				positive_samples.append((u, v))
				positive_samples.append((v, u))
				
				all_positive_samples[u].add(v)
				all_positive_samples[v].add(u)
 
		if num_walk % 1000 == 0:  
			print "processed walk {}/{}".format(num_walk, len(walks))
			
	ground_truth_negative_samples = {n: sorted(list(nodes.difference(all_positive_samples[n]))) for n in G.nodes()}
	
	return positive_samples, ground_truth_negative_samples

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


def create_neighbourhood_sample_list(nodes, neighbourhood_sample_sizes, neighbours):

	neighbourhood_sample_list = [nodes]

	for neighbourhood_sample_size in neighbourhood_sample_sizes[::-1]:

		for batch in neighbourhood_sample_list[-1]:
			for n in batch:
				if len(neighbours[n]) == 0:
					print "ZERO", n

		neighbourhood_sample_list.append(np.array([np.concatenate([np.append(n, np.random.choice(neighbours[n], 
			replace=True, size=neighbourhood_sample_size)) for n in batch]) for batch in neighbourhood_sample_list[-1]]))

	# flip neighbour list
	neighbourhood_sample_list = neighbourhood_sample_list[::-1]


	return neighbourhood_sample_list



class ValidationCallback(Callback):

	def __init__(self, G, X, Y, original_adj, 
		val_mask, removed_edges_val, ground_truth_negative_samples,
		neighbourhood_sample_sizes, num_capsules_per_layer,
		embedder, predictor, batch_size,
		embedding_path, plot_path,):
		self.G = G 
		self.X = X
		self.Y = Y
		self.original_adj = original_adj
		self.val_mask = val_mask
		self.ground_truth_negative_samples = ground_truth_negative_samples
		# self.removed_edges_val = sorted(removed_edges_val, key=lambda(u, v): (u, v))
		self.neighbourhood_sample_sizes = neighbourhood_sample_sizes
		self.num_capsules_per_layer = num_capsules_per_layer
		self.embedder = embedder
		self.predictor = predictor
		self.batch_size = batch_size
		# self.annotate_idx = annotate_idx
		self.embedding_path = embedding_path
		self.plot_path = plot_path
		self.removed_edges_dict = self.convert_edgelist_to_dict(removed_edges_val)
		# self.G_edge_dict = self.convert_edgelist_to_dict(G.edges(), self_edges=True)
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
				default = [u]
			else:
				default = []
			edge_dict.setdefault(u, default).append(v)
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

		original_adj = self.original_adj
		G = self.G
		removed_edges_dict = self.removed_edges_dict
		ground_truth_negative_samples = self.ground_truth_negative_samples

		N = len(G)

		r = 1.
		t = 1.

		MAPs_reconstruction = np.zeros(N)
		ranks_reconstruction = np.zeros(N)
		MAPs_link_prediction = np.zeros(N)
		ranks_link_prediction = np.zeros(N)

		for u in range(N):
		    
		    dists_u = hyperbolic_distance(embedding[u], embedding)
		    y_pred = sigmoid((r - dists_u) / t) 
		   
		    y_pred_reconstruction = y_pred.copy()
		    y_true_reconstruction = original_adj[u].toarray().flatten()
		    MAPs_reconstruction[u] = average_precision_score(y_true=y_true_reconstruction, 
		    	y_score=y_pred_reconstruction)
		    
		    y_pred_reconstruction[::-1].sort()
		    ranks_reconstruction[u] = np.array([np.searchsorted(-y_pred_reconstruction, -p) 
		                                        for p in y_pred[y_true_reconstruction.astype(np.bool)]]).mean()
		    
		    if removed_edges_dict.has_key(u):
		    
		        removed_neighbours = removed_edges_dict[u]
		        all_negative_samples = ground_truth_negative_samples[u]
		        y_true_link_prediction = np.append(np.ones(len(removed_neighbours)), 
		                                           np.zeros(len(all_negative_samples)))
		        y_pred_link_prediction = np.append(y_pred[removed_neighbours], y_pred[all_negative_samples])
		        MAPs_link_prediction[u] = average_precision_score(y_true=y_true_link_prediction, y_score=y_pred_link_prediction)

		        y_pred_link_prediction[::-1].sort()
		        ranks_link_prediction[u] = np.array([np.searchsorted(-y_pred_link_prediction, -p) 
		                                            for p in y_pred[removed_neighbours]]).mean()


		# ranks = np.zeros(len(removed_edges_dict))
		# MAPs_val = np.zeros(len(removed_edges_dict))
		# MAPs_reconstruction = np.zeros(len(removed_edges_dict))


		# for i, u in enumerate(sorted(removed_edges_dict.keys())):
		# 	u_neighbors_in_G = G_edge_dict[u]
		# 	removed_u_neighbours = removed_edges_dict[u]
		# 	removed_u_neighbours_dist = hyperbolic_distance(embedding[u], embedding[list(removed_u_neighbours)])
		# 	removed_u_neighbours_P = sigmoid((r - removed_u_neighbours_dist) / t)
		# 	all_neighbours = u_neighbors_in_G.union(removed_u_neighbours)
		# 	non_neighbours = list(set(range(N)).difference(all_neighbours))
		# 	non_neighbour_dists = hyperbolic_distance(embedding[u], embedding[non_neighbours])
		# 	non_neighbour_P = sigmoid((r - non_neighbour_dists) / t)
			
		# 	y_true = np.append(np.ones(len(removed_u_neighbours_P)), np.zeros(len(non_neighbour_P)))
		# 	y_pred = np.append(removed_u_neighbours_P, non_neighbour_P)
			
		# 	MAPs_val[i] = average_precision_score(y_true, y_pred)
		# 	y_pred[::-1].sort()
		# 	ranks[i] = np.array([np.searchsorted(-y_pred, -p) for p in removed_u_neighbours_P]).mean()

		# 	if i % 1000 == 0:
		# 		print "completed node {}/{}".format(i, N)

		return ranks_reconstruction.mean(), MAPs_reconstruction.mean(), ranks_link_prediction.mean(), MAPs_link_prediction.mean(),


			


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

		if self.predictor is not None and self.Y.shape[1] > 1:
			NMI, classification_accuracy = self.make_and_evaluate_label_predictions()
			logs.update({"NMI": NMI, "classification_accuracy": classification_accuracy})

		self.plot_embedding(embedding, path="{}/embedding_epoch_{:04}".format(self.embedding_path, epoch))


	def perform_embedding(self):

		print "performing embedding"

		def embedding_generator(X, input_nodes, batch_size=100):
			num_steps = (input_nodes.shape[0] + batch_size - 1) / batch_size
			for step in range(num_steps):
				batch_nodes = input_nodes[batch_size*step : batch_size*(step+1)]
				if sp.sparse.issparse(X):
					x = X[batch_nodes.flatten()].toarray()
					x = preprocess_data(x)
				else:
					x = X[batch_nodes]
				yield x.reshape([-1, input_nodes.shape[1], 1, X.shape[-1]])

		G = self.G
		X = self.X
		neighbourhood_sample_sizes = self.neighbourhood_sample_sizes
		embedder = self.embedder
		batch_size = self.batch_size * 12

		nodes = np.array(sorted(list(G.nodes()))).reshape(-1, 1)

		# nodes = nodes.reshape(-1, 1)
		neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
		neighbour_list = create_neighbourhood_sample_list(nodes, neighbourhood_sample_sizes, neighbours)

		input_nodes = neighbour_list[0]
		# print input_nodes
		# print input_nodes.shape
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

		print "plotting embedding and saving to {}".format(path)

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

		print "evaluating label predictions on validation set"


		def prediction_generator(X, input_nodes, batch_size=100):
			num_steps = (input_nodes.shape[0] + batch_size - 1) / batch_size
			for step in range(num_steps):
				batch_nodes = input_nodes[batch_size*step : batch_size*(step+1)]
				if sp.sparse.issparse(X):
					x = X[batch_nodes.flatten()].toarray()
					x = preprocess_data(x)
				else:
					x = X[batch_nodes]
				yield x.reshape([-1, input_nodes.shape[1], 1, X.shape[-1]])

		G = self.G
		X = self.X
		Y = self.Y
		predictor = self.predictor
		num_capsules_per_layer = self.num_capsules_per_layer
		neighbourhood_sample_sizes = self.neighbourhood_sample_sizes
		batch_size = self.batch_size
		val_mask = self.val_mask.flatten().astype(np.bool)

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

		# only consider validation labels
		true_labels = Y[val_mask].argmax(axis=-1)
		predicted_labels = predictions[val_mask].argmax(axis=-1)

		NMI = normalized_mutual_info_score(true_labels, predicted_labels)
		classification_accuracy = accuracy_score(true_labels, predicted_labels, normalize=True)

		print "NMI of predictions: {}".format(NMI)
		print "Classification accuracy: {}".format(classification_accuracy)
		
		return NMI, classification_accuracy

