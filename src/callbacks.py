import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx

from scipy.stats import spearmanr

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, normalized_mutual_info_score, accuracy_score, f1_score
from sklearn.metrics.pairwise import pairwise_distances

from keras.callbacks import Callback

from data_utils import preprocess_data
from generators import get_neighbourhood_samples

def hyperbolic_distance(u, v):
	assert (np.linalg.norm(u, axis=-1) < 1).all(), "u norm: {}, {}".format(np.linalg.norm(u, axis=-1), u)
	assert (np.linalg.norm(v, axis=-1) < 1).all(), "v norm: {}, {}".format(np.linalg.norm(v, axis=-1), v)
	return np.arccosh(1. + 2. * np.linalg.norm(u - v, axis=-1)**2 /\
	((1. - np.linalg.norm(u, axis=-1)**2) * (1. - np.linalg.norm(v, axis=-1)**2)))
	# return np.linalg.norm(u - v, axis=-1)

def perform_prediction(G_neighbours, X, idx, predictor, neighbourhood_sample_sizes, batch_size, scale_data):
	# neighbours = {n: sorted(list(G.neighbors(n))) for n in G.nodes()}
	nodes_to_embed = np.array(idx).reshape(-1, 1)
	assert nodes_to_embed.shape == (len(idx), 1)
	neighbourhood_sample_list = get_neighbourhood_samples(nodes_to_embed, neighbourhood_sample_sizes, G_neighbours)
	input_nodes = neighbourhood_sample_list[0]
	num_steps = int((len(idx) + batch_size - 1) // batch_size)
	input_x = []
	for step in range(num_steps):
		nodes = input_nodes[step * batch_size : (step + 1) * batch_size]
		if sp.sparse.issparse(X):
			x = X[nodes.flatten()].toarray()
			assert not np.isnan(x).any()
			if scale_data:
				# print "scaling input data"
				x = preprocess_data(x)
				assert not np.isnan(x).any()
			# x = x.reshape(-1, nodes.shape[1], 1, X.shape[-1])
		else:
			x = X[nodes]
		x = x.reshape(-1, nodes.shape[1], 1, X.shape[-1])
		input_x.append(x)
	input_x = np.concatenate(input_x)
	assert input_x.shape == (len(idx), input_nodes.shape[1], 1, X.shape[-1])

	predictions = predictor.predict(input_x, batch_size=batch_size)
	dim = predictions.shape[-1]
	predictions = predictions.reshape(-1, dim)
	assert predictions.shape == (len(idx), dim)

	predictions = predictions.astype(np.float32)

	return predictions



class ReconstructionLinkPredictionCallback(Callback):

	def __init__(self, G, X, Y, embedder,
		all_edges, removed_edges, ground_truth_negative_samples, 
		embedding_path, plot_path, args, G_neighbours):
		self.G = G 
		self.X = X
		self.Y = Y
		self.embedder = embedder
		self.all_edges_dict = self.convert_edgelist_to_dict(all_edges)
		self.removed_edges_dict = self.convert_edgelist_to_dict(removed_edges)
		self.ground_truth_negative_samples = ground_truth_negative_samples
		self.args = args
		self.embedding_path = embedding_path
		self.plot_path = plot_path
		self.embedding_gen = None
		# self.nodes_to_val = None
		self.idx = G.nodes()
		self.shortest_path_length = None
		self.G_neighbours = G_neighbours

	def on_epoch_end(self, epoch, logs={}):
		embedding = self.perform_embedding()

		mean_rank_reconstruction, mean_precision_reconstruction =\
			self.evaluate_rank_and_MAP(embedding, self.all_edges_dict)
		logs.update({"mean_rank_reconstruction" : mean_rank_reconstruction, 
			"mean_precision_reconstruction" : mean_precision_reconstruction,})
		if self.removed_edges_dict is not None:
			mean_rank_link_prediction, mean_precision_link_prediction =\
				self.evaluate_rank_and_MAP(embedding, self.removed_edges_dict)
			logs.update({"mean_rank_link_prediction" : mean_rank_reconstruction, 
				"mean_precision_link_prediction" : mean_precision_reconstruction,})

		if self.args.dataset in ["wordnet", "wordnet_attributed"]:
			r, p = self.evaluate_lexical_entailment(embedding, self.args.dataset)
			logs.update({"lex_r" : r, "lex_p" : p})
		
		# distortion = self.evaluate_distortion(embedding)
		distortion = "none"
		logs.update({"distortion": distortion})

		self.save_embedding(embedding, path="{}/embedding_epoch_{:04}.npy".format(self.embedding_path, epoch))
		self.plot_embedding(embedding, mean_rank_reconstruction, mean_precision_reconstruction, distortion,
			path="{}/embedding_epoch_{:04}.png".format(self.plot_path, epoch))

	def perform_embedding(self):

		# def argsort(seq):
		#     # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
		#     return sorted(range(len(seq)), key=seq.__getitem__)

		print ("\nperforming embedding")
		batch_size = self.args.batch_size * (1 + self.args.num_positive_samples + self.args.num_negative_samples)
		embedding = perform_prediction(self.G_neighbours, self.X, self.idx, self.embedder,
			self.args.neighbourhood_sample_sizes, batch_size, self.args.scale_data)

		# G = self.G
		# neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
		# X = self.X
		# idx = self.idx
		# neighbourhood_sample_sizes = self.args.neighbourhood_sample_sizes
		# batch_size = self.args.batch_size * (1 + self.args.num_positive_samples + self.args.num_negative_samples)
		# num_steps = int((len(idx) + batch_size - 1) // batch_size)

		# nodes_to_embed = np.array(idx).reshape(-1, 1)
		# assert nodes_to_embed.shape == (len(G), 1)
		# neighbourhood_sample_list = get_neighbourhood_samples(nodes_to_embed, neighbourhood_sample_sizes, neighbours)
		# input_nodes = neighbourhood_sample_list[0]
		# input_x = []
		# for step in range(num_steps):
		# 	nodes = input_nodes[step * batch_size : (step + 1) * batch_size]
		# 	if sp.sparse.issparse(X):
		# 		x = X[nodes.flatten()].toarray()
		# 		x = preprocess_data(x)
		# 		# x = x.reshape(-1, nodes.shape[1], 1, X.shape[-1])
		# 	else:
		# 		x = X[nodes]
		# 	x = x.reshape(-1, nodes.shape[1], 1, X.shape[-1])
		# 	input_x.append(x)
		# input_x = np.concatenate(input_x)
		# assert input_x.shape == (len(idx), input_nodes.shape[1], 1, X.shape[-1])

		# embedder = self.embedder
		# embedding = embedder.predict(input_x, batch_size=batch_size)
		# dim = embedding.shape[-1]
		# embedding = embedding.reshape(-1, dim)
		# assert embedding.shape == (len(idx), dim)
		print (embedding.shape)
		print (embedding)
		return embedding

	def save_embedding(self, embedding, path):
		print ("saving embedding to", path)
		np.save(path, embedding)

	def plot_embedding(self, embedding, mean_rank_reconstruction, mean_precision_reconstruction, distortion, path):

		print ("plotting embedding and saving to {}".format(path))

		_min, _max = -1, 1

		Y = self.Y
		y = Y.argmax(axis=1)
		if sp.sparse.issparse(Y):
			y = y.A1

		# pred = np.exp(-embedding)
		# pred = pred.argmax(axis=-1)

		fig = plt.figure(figsize=(10, 10))
		plt.suptitle("Mean Rank Reconstruction={} MAP Reconstruction={} Distortion={}".format(mean_rank_reconstruction, mean_precision_reconstruction, distortion))
		
		# plt.subplot("221")
		for u, v in self.G.edges():
			u_emb = embedding[u]
			v_emb = embedding[v]
			plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.05, zorder=0)
		plt.scatter(embedding[:,0], embedding[:,1], s=10, c=y, zorder=1)
		plt.xlim([_min, _max])
		plt.ylim([_min, _max])
		plt.xlabel("dimension 1")
		plt.ylabel("dimension 2")
		# plt.subplot("222")
		# for u, v in self.G.edges():
		# 	u_emb = embedding[u]
		# 	v_emb = embedding[v]
		# 	plt.plot([u_emb[2], v_emb[2]], [u_emb[3], v_emb[3]], c="k", linewidth=0.2, zorder=0)
		# plt.scatter(embedding[:,2], embedding[:,3], c=y, zorder=1)
		# plt.xlim([0, 1])
		# plt.ylim([0, 1])
		# plt.xlabel("dimension 3")
		# plt.ylabel("dimension 4")
		# plt.subplot("223")
		# for u, v in self.G.edges():
		# 	u_emb = embedding[u]
		# 	v_emb = embedding[v]
		# 	plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.2, zorder=0)
		# plt.scatter(embedding[:,0], embedding[:,1], c=pred, zorder=1)
		# plt.xlim([0, 1])
		# plt.ylim([0, 1])
		# plt.xlabel("dimension 1")
		# plt.ylabel("dimension 2")
		# plt.subplot("224")
		# for u, v in self.G.edges():
		# 	u_emb = embedding[u]
		# 	v_emb = embedding[v]
		# 	plt.plot([u_emb[2], v_emb[2]], [u_emb[3], v_emb[3]], c="k", linewidth=0.2, zorder=0)
		# plt.scatter(embedding[:,2], embedding[:,3], c=pred, zorder=1)
		# plt.xlim([0, 1])
		# plt.ylim([0, 1])
		# plt.xlabel("dimension 3")
		# plt.ylabel("dimension 4")
		if path is not None:
			plt.savefig(path)
		plt.close()
	
	def convert_edgelist_to_dict(self, edgelist, undirected=True, self_edges=False):
		if edgelist is None:
			return None
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
		for u, v in edgelist:
			assert v in edge_dict[u]
			if undirected:
				assert u in edge_dict[v]
		return edge_dict

	def evaluate_rank_and_MAP(self, embedding, edge_dict):

		# 	lt = th.from_numpy(model.embedding())
		#     embedding = Variable(lt, volatile=True)
		#     ranks = []
		#     ap_scores = []
		#     for s, s_types in types.items():
		#         s_e = Variable(lt[s].expand_as(embedding), volatile=True)
		#         _dists = model.dist()(s_e, embedding).data.cpu().numpy().flatten()
		#         _dists[s] = 1e+12
		#         _labels = np.zeros(embedding.size(0))
		#         _dists_masked = _dists.copy()
		#         _ranks = []
		#         for o in s_types:
		#             _dists_masked[o] = np.Inf
		#             _labels[o] = 1
		#         ap_scores.append(average_precision_score(_labels, -_dists))
		#         for o in s_types:
		#             d = _dists_masked.copy()
		#             d[o] = _dists[o]
		#             r = np.argsort(d)
		#             _ranks.append(np.where(r == o)[0][0] + 1)
		#         ranks += _ranks
		# return np.mean(ranks), np.mean(ap_scores)

		ranks = []
		ap_scores = []

		for u, v_list in edge_dict.items():
			_dists = hyperbolic_distance(embedding[u], embedding)
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

	def evaluate_distortion(self, embedding):
		print ("evaluating distortion")
		mask = ~ np.eye(len(self.G), dtype=bool)
		if self.shortest_path_length is None:
			self.shortest_path_length = nx.floyd_warshall_numpy(self.G).A
		shortest_path_length = self.shortest_path_length[np.where(mask)]
		hyperbolic_distances = pairwise_distances(embedding, metric=hyperbolic_distance)
		hyperbolic_distances = hyperbolic_distances[np.where(mask)]

		distortion = (np.abs(hyperbolic_distances - shortest_path_length) / shortest_path_length).mean()
		return distortion

	def evaluate_lexical_entailment(self, embedding, dataset):

		def is_a_score(u, v, alpha=1e3):
			return -(1 + alpha * (np.linalg.norm(v, axis=-1) - np.linalg.norm(u, axis=-1))) * hyperbolic_distance(u, v)

		print ("evaluating lexical entailment")

		if dataset == "wordnet":
			f = "../data/wordnet/hyperlex_idx_ranks.txt"
		else:
			f = "../data/wordnet/hyperlex_idx_ranks_filtered.txt"
		hyperlex_noun_idx_df = pd.read_csv(f, index_col=0, sep=" ")

		U = np.array(hyperlex_noun_idx_df["WORD1"], dtype=int)
		V = np.array(hyperlex_noun_idx_df["WORD2"], dtype=int)

		true_is_a_score = np.array(hyperlex_noun_idx_df["AVG_SCORE_0_10"])
		predicted_is_a_score = is_a_score(embedding[U], embedding[V])

		r, p = spearmanr(true_is_a_score, predicted_is_a_score)

		print ("r=", r, "p=", p)

		return r, p

class LabelPredictionCallback(Callback):

	def __init__(self, val_G, X, Y, predictor, val_idx, args, G_neighbours):
		self.val_G = val_G
		self.X = X
		self.Y = Y
		self.predictor = predictor
		self.val_idx = val_idx
		self.args = args
		self.val_prediction_gen = None
		self.nodes_to_val = None
		self.G_neighbours = G_neighbours

	def on_epoch_end(self, epoch, logs={}):

		if (self.args.number_of_capsules_per_layer == self.Y.shape[1]).any():
			margin_loss, f1_micro, f1_macro, NMI, classification_accuracy = self.make_and_evaluate_label_predictions()
			logs.update({"margin_loss": margin_loss, "f1_micro": f1_micro, "f1_macro": f1_macro, 
				"NMI": NMI, "classification_accuracy": classification_accuracy})

	def make_and_evaluate_label_predictions(self, ):

		# X = self.X
		# Y = self.Y
		# predictor = self.predictor
		
		# idx = self.val_idx
		# G = self.val_G
		print ("evaluating label predictions on validation set")

		number_of_capsules_per_layer = self.args.number_of_capsules_per_layer
		num_classes = self.Y.shape[1]


		# neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
		neighbourhood_sample_sizes = self.args.neighbourhood_sample_sizes
		label_prediction_layers = np.where(self.args.number_of_capsules_per_layer==num_classes)[0] + 1
		prediction_layer = label_prediction_layers[-1]
		neighbourhood_sample_sizes = neighbourhood_sample_sizes[:prediction_layer]
		batch_size = self.args.batch_size * (1 + self.args.num_positive_samples + self.args.num_negative_samples)


		predictions = perform_prediction(self.G_neighbours, self.X, self.val_idx, self.predictor,
			neighbourhood_sample_sizes, batch_size, self.args.scale_data)



		# num_steps = int((len(idx) + batch_size - 1) // batch_size)

		# batch_nodes = np.array(idx).reshape(-1, 1)
		# assert batch_nodes.shape == (len(idx), 1)
		# label_prediction_layers = np.where(number_of_capsules_per_layer==num_classes)[0] + 1
		# prediction_layer = label_prediction_layers[-1]
		# neighbourhood_sample_sizes = neighbourhood_sample_sizes[:prediction_layer]
		# neighbourhood_sample_list = get_neighbourhood_samples(batch_nodes, neighbourhood_sample_sizes, neighbours)
		# input_nodes = neighbourhood_sample_list[0]
		# input_x = []
		# for step in range(num_steps):
		# 	nodes = input_nodes[step * batch_size : (step + 1) * batch_size]
		# 	if sp.sparse.issparse(X):
		# 		x = X[nodes.flatten()].toarray()
		# 		x = preprocess_data(x)
		# 		# x = x.reshape(-1, nodes.shape[1], 1,  X.shape[-1])
		# 	else:
		# 		x = X[nodes]
		# 	x = x.reshape(-1, nodes.shape[1], 1,  X.shape[-1])
		# 	input_x.append(x)
		# input_x = np.concatenate(input_x)
		# assert input_x.shape == (len(idx), input_nodes.shape[1], 1, X.shape[-1])


		# predictions = predictor.predict_generator(self.val_prediction_gen, steps=self.num_steps, )
		# predictions = predictor.predict(input_x, batch_size=batch_size)


		# predictions = predictions.reshape(-1, predictions.shape[-1])
		print (predictions)
		val_Y = self.Y[self.val_idx]
		if sp.sparse.issparse(val_Y):
			val_Y = val_Y.toarray()
		
		# print (self.nodes_to_val)

		# print(val_Y)
		# only consider validation labels (shuffled in generator)
		true_labels = val_Y.argmax(axis=-1)
		print ("TRUE LABELS")
		print (true_labels)	

		predicted_labels = predictions.argmax(axis=-1)
		# print (predictions)
		print ("PREDICTED LABELS")
		print(predicted_labels)

		margin_loss = np.mean(np.sum(val_Y * np.square(np.maximum(0., 0.9 - predictions)) + \
        0.5 * (1 - val_Y) * np.square(np.maximum(0., predictions - 0.1)), axis=-1))

		f1_micro = f1_score(true_labels, predicted_labels, average="micro")
		f1_macro = f1_score(true_labels, predicted_labels, average="macro")
		NMI = normalized_mutual_info_score(true_labels, predicted_labels)
		classification_accuracy = accuracy_score(true_labels, predicted_labels, normalize=True)

		print ("margin_loss={}".format(margin_loss))
		print ("f1_micro={}, f1_macro={}".format(f1_micro, f1_macro))
		print ("NMI of predictions: {}".format(NMI))
		print ("Classification accuracy: {}".format(classification_accuracy))
		
		return margin_loss, f1_micro, f1_macro, NMI, classification_accuracy