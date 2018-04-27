import numpy as np
import scipy as sp
import pandas as pd

# import random

from scipy.stats import spearmanr

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import average_precision_score, normalized_mutual_info_score, accuracy_score, f1_score
from sklearn.metrics.pairwise import pairwise_distances

from keras.callbacks import Callback

# from utils import create_neighbourhood_sample_list
from data_utils import preprocess_data
from generators import get_neighbourhood_samples, validation_generator #embedding_generator, prediction_generator

def hyperbolic_distance(u, v):
	assert (np.linalg.norm(u, axis=-1) < 1).all(), "u norm: {}, {}".format(np.linalg.norm(u, axis=-1), u)
	assert (np.linalg.norm(v, axis=-1) < 1).all(), "v norm: {}, {}".format(np.linalg.norm(v, axis=-1), v)
	return np.arccosh(1. + 2. * np.linalg.norm(u - v, axis=-1)**2 /\
	((1. - np.linalg.norm(u, axis=-1)**2) * (1. - np.linalg.norm(v, axis=-1)**2)))
	# return np.linalg.norm(u - v, axis=-1)

class ReconstructionLinkPredictionCallback(Callback):

	def __init__(self, G, X, Y, embedder,
		all_edges, removed_edges, ground_truth_negative_samples, 
		embedding_path, plot_path, args,):
		self.G = G 
		self.X = X
		self.Y = Y
		self.embedder = embedder
		self.all_edges_dict = self.convert_edgelist_to_dict(all_edges)
		self.removed_edges_dict = self.convert_edgelist_to_dict(removed_edges)
		# if removed_edges is not None:
		# 	# self.removed_edges_dict = self.convert_edgelist_to_dict(removed_edges_val)
		# 	self.removed_edges = np.array(removed_edges_val)
		# else:
		# 	# self.removed_edges_dict = None
		# 	self.removed_edges = None
		self.ground_truth_negative_samples = ground_truth_negative_samples
		self.args = args
		self.embedding_path = embedding_path
		self.plot_path = plot_path
		self.embedding_gen = None
		self.nodes_to_val = None
		self.idx = G.nodes()

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

		# metrics = self.evaluate_rank_and_MAP(embedding)
		# mean_rank_reconstruction, mean_precision_reconstruction = metrics[:2]
		# logs.update({"mean_rank_reconstruction" : mean_rank_reconstruction, 
		# 	"mean_precision_reconstruction" : mean_precision_reconstruction,})
		# if self.removed_edges is not None:
		# 	mean_rank_link_prediction, mean_precision_link_prediction = metrics[2:4]
		# 	logs.update({"mean_rank_link_prediction": mean_rank_link_prediction,
		# 	"mean_precision_link_prediction": mean_precision_link_prediction})

		if self.args.dataset in ["wordnet", "wordnet_attributed"]:
			r, p = self.evaluate_lexical_entailment(embedding, self.args.dataset)
			logs.update({"lex_r" : r, "lex_p" : p})
			
		self.save_embedding(embedding, path="{}/embedding_epoch_{:04}.npy".format(self.embedding_path, epoch))
		self.plot_embedding(embedding, path="{}/embedding_epoch_{:04}.png".format(self.plot_path, epoch))

	def perform_embedding(self):

		def argsort(seq):
		    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
		    return sorted(range(len(seq)), key=seq.__getitem__)

		print ("performing embedding")

		G = self.G
		neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
		X = self.X
		idx = self.idx
		neighbourhood_sample_sizes = self.args.neighbourhood_sample_sizes
		batch_size = self.args.batch_size * (1 + self.args.num_positive_samples + self.args.num_negative_samples)
		# nodes_to_embed = np.array(sorted(G.nodes())).reshape(-1, 1)
		num_steps = int((len(idx) + batch_size - 1) // batch_size)

		batch_nodes = np.array(idx).reshape(-1, 1)
		assert batch_nodes.shape == (len(G), 1)
		neighbourhood_sample_list = get_neighbourhood_samples(batch_nodes, neighbourhood_sample_sizes, neighbours)
		input_nodes = neighbourhood_sample_list[0]
		input_x = []
		for step in range(num_steps):
			nodes = input_nodes[step * batch_size : (step + 1) * batch_size]
			if sp.sparse.issparse(X):
				x = X[nodes.flatten()].toarray()
				x = preprocess_data(x)
				x = x.reshape(-1, nodes.shape[1], 1, X.shape[-1])
			else:
				x = X[nodes]
			input_x.append(x)
		input_x = np.concatenate(input_x)
		assert input_x.shape == (len(idx), input_nodes.shape[1], 1, X.shape[-1])

		# def embedding_generator(G, X, nodes_to_embed, num_steps, neighbourhood_sample_sizes, batch_size=100):
		# 	# step = 0
		# 	neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
		# 	while True:
		# 		np.random.shuffle(nodes_to_embed)
		# 		for step in range(num_steps):			
		# 			batch_nodes = nodes_to_embed[batch_size*step : batch_size*(step+1)]
		# 			batch_nodes = create_neighbourhood_sample_list(batch_nodes, neighbourhood_sample_sizes, neighbours)
		# 			batch_nodes = batch_nodes[0]
		# 			if sp.sparse.issparse(X):
		# 				x = X[batch_nodes.flatten()].toarray()
		# 				x = preprocess_data(x)
		# 			else:
		# 				x = X[batch_nodes]
		# 			yield x.reshape([-1, batch_nodes.shape[1], 1, X.shape[-1]])
					# step = (step + 1) % num_steps


		# embedding_gen = embedding_generator(X, input_nodes, num_steps=num_steps, batch_size=batch_size)
		# if self.embedding_gen is None:
		# 	G = self.G
		# 	X = self.X
		# 	idx = self.idx
		# 	neighbourhood_sample_sizes = self.args.neighbourhood_sample_sizes
		# 	batch_size = self.args.batch_size * (1 + self.args.num_positive_samples + self.args.num_negative_samples)
		# 	# nodes_to_embed = np.array(sorted(G.nodes())).reshape(-1, 1)
		# 	self.num_steps = int((len(idx) + batch_size - 1) // batch_size)
		# 	self.embedding_gen = validation_generator(self, G, X, idx, neighbourhood_sample_sizes, 
		# 		self.num_steps, batch_size)
		

		embedder = self.embedder
		# embedding = embedder.predict_generator(self.embedding_gen, steps=self.num_steps, )
		embedding = embedder.predict(input_x, batch_size=batch_size)
		dim = embedding.shape[-1]
		embedding = embedding.reshape(-1, dim)

		# sort back into numerical order -- generator shuffles idx
		# for i, j, k in zip(self.idx, self.nodes_to_val, argsort(self.nodes_to_val)):
		# 	assert i == j == k
		# print self.nodes_to_val
		# embedding = embedding[argsort(self.nodes_to_val)]
		assert embedding.shape == (len(self.G), dim)

		# print "EMBEDDING"
		# print embedding

		# raise SystemExit

		return embedding

	def save_embedding(self, embedding, path):
		print ("saving embedding to", path)
		np.save(path, embedding)

	def plot_embedding(self, embedding, path):

		print ("plotting embedding and saving to {}".format(path))

		Y = self.Y
		y = Y.argmax(axis=1)
		if sp.sparse.issparse(Y):
			y = y.A1
		
		# embedding_dim = embedding.shape[-1]

		fig = plt.figure(figsize=(10, 10))
		for u, v in self.G.edges():
			u_emb = embedding[u]
			v_emb = embedding[v]
			plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.2, zorder=0)
		plt.scatter(embedding[:,0], embedding[:,1], c=y, zorder=1)
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
			# print u, v_list
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
			# print _ranks, np.mean(_ranks)
			# print 
			ranks.append(np.mean(_ranks))
			# print ranks
			# print 
		print (np.mean(ranks), np.mean(ap_scores))
		return np.mean(ranks), np.mean(ap_scores)


	# def evaluate_rank_and_MAP(self, embedding,):

		
	# 	def sigmoid(x):
	# 		return 1. / (1 + np.exp(-x))
		
	# 	print ("evaluating rank and MAP")

	# 	original_adj = self.original_adj
	# 	# if test_edges is None:
	# 	# removed_edges_dict = self.removed_edges_dict
	# 	removed_edges = self.removed_edges
	# 	print ("evaluating on validation edges")
	# 	# else:
	# 	# 	removed_edges_dict = self.convert_edgelist_to_dict(test_edges)
	# 	# 	print ("evaluating on test edges")

	# 	ground_truth_negative_samples = self.ground_truth_negative_samples

	# 	r = 5.
	# 	t = 1.
		
	# 	G = self.G
	# 	N = len(G)


	# 	d = pairwise_distances(embedding, metric=hyperbolic_distance)
	# 	assert d.shape == (N, N)
	# 	y_pred = sigmoid((r - d) / t)
	# 	assert y_pred.shape == (N, N)
	# 	print d.shape
	# 	print d.flatten()
	# 	print y_pred.flatten()
	# 	print original_adj.toarray().flatten().sum()


	# 	sorted_d = d.copy().flatten()
	# 	sorted_d.sort()
	# 	sorted_d = sorted_d[N:] # the first N distances are zero (the diagonal of the distance matrix)
	# 	mean_rank_reconstruction = np.searchsorted(sorted_d, d[np.nonzero(original_adj)]).mean()

	# 	MAP_reconstruction = average_precision_score(y_true=original_adj.toarray().flatten(),
	# 		y_score=y_pred.flatten())

	# 	metrics = [mean_rank_reconstruction, MAP_reconstruction]


	# 	if removed_edges is not None:

	# 		mean_rank_link_prediction = np.searchsorted(sorted_d, d[removed_edges[:,0], removed_edges[:,1]]).mean()

	# 		y_pred_removed = y_pred[removed_edges[:,0], removed_edges[:,1]]
	# 		neg_samples = np.array([(u, v) for u in removed_edges[:,0] for v in ground_truth_negative_samples[u]])
	# 		y_pred_neg_samples = y_pred[neg_samples[:,0], neg_samples[:,1]]

	# 		y_true_link_prediction = np.append(np.ones_like(y_pred_removed), np.zeros_like(y_pred_neg_samples))
	# 		y_pred_link_prediction = np.append(y_pred_removed, y_pred_neg_samples)


	# 		MAP_link_prediction = average_precision_score(y_true=y_true_link_prediction, 
	# 			y_score=y_pred_link_prediction)


	# 		metrics.extend([mean_rank_link_prediction, MAP_link_prediction])



	# 	print metrics
	# 	# raise SystemExit
	# 	return metrics


		# MAPs_reconstruction = np.zeros(N)
		# ranks_reconstruction = np.zeros(N)
		# if removed_edges_dict is not None:
		# 	MAPs_link_prediction = np.zeros(N)
		# 	ranks_link_prediction = np.zeros(N)



		# for u in range(N):
			
		# 	dists_u = hyperbolic_distance(embedding[u], embedding)

		# 	_, all_neighbours = np.nonzero(original_adj[u])
		# 	all_node_dists = dists_u.copy()
		# 	all_node_dists.sort()

		# 	ranks_reconstruction[u] = np.array([np.searchsorted(all_node_dists, d) 
		# 										for d in dists_u[all_neighbours]]).mean()
		# 	y_pred = sigmoid((r - dists_u) / t) 
		   
		# 	# y_pred_reconstruction = y_pred.copy()
		# 	y_true_reconstruction = original_adj[u].todense().A1
		# 	# print embedding[u]
		# 	# print dists_u
		# 	MAPs_reconstruction[u] = average_precision_score(y_true=y_true_reconstruction, 
		# 		y_score=y_pred)
			
		# 	# y_pred_reconstruction[::-1].sort()
		# 	# ranks_reconstruction[u] = np.array([np.searchsorted(-y_pred_reconstruction, -p) 
		# 	# 									for p in y_pred[y_true_reconstruction.astype(np.bool)]]).mean()
			
		# 	if removed_edges_dict is not None and removed_edges_dict.has_key(u):
			
		# 		removed_neighbours = removed_edges_dict[u]
		# 		all_negative_samples = ground_truth_negative_samples[u]

		# 		dist_negative_samples = dists_u[all_negative_samples]
		# 		dist_negative_samples.sort()

		# 		ranks_link_prediction[u] = np.array([np.searchsorted(dist_negative_samples, d) 
		# 											for d in dists_u[removed_neighbours]]).mean()

		# 		y_true_link_prediction = np.append(np.ones(len(removed_neighbours)), 
		# 										   np.zeros(len(all_negative_samples)))
		# 		y_pred_link_prediction = np.append(y_pred[removed_neighbours], y_pred[all_negative_samples])
		# 		MAPs_link_prediction[u] = average_precision_score(y_true=y_true_link_prediction, 
		# 			y_score=y_pred_link_prediction)

		# 		# y_pred_link_prediction[::-1].sort()
		# 		# ranks_link_prediction[u] = np.array([np.searchsorted(-y_pred_link_prediction, -p) 
		# 		# 									for p in y_pred[removed_neighbours]]).mean()

		# 	if u % 1000 == 0:
		# 		print ("completed node {}/{}".format(u, N))

		# print ("completed node {}/{}".format(u, N))

		# metrics = [ranks_reconstruction.mean(), MAPs_reconstruction.mean()]
		# if removed_edges_dict is not None:
		# 	metrics += [ranks_link_prediction.mean(), MAPs_link_prediction.mean()]
		# return metrics


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

	def __init__(self, val_G, X, Y, predictor, val_idx, args):
		self.val_G = val_G
		self.X = X
		self.Y = Y
		self.predictor = predictor
		self.val_idx = val_idx
		self.args = args
		self.val_prediction_gen = None
		self.nodes_to_val = None

	def on_epoch_end(self, epoch, logs={}):

		if self.val_idx is not None:
			margin_loss, f1_micro, f1_macro, NMI, classification_accuracy = self.make_and_evaluate_label_predictions()
			logs.update({"margin_loss": margin_loss, "f1_micro": f1_micro, "f1_macro": f1_macro, 
				"NMI": NMI, "classification_accuracy": classification_accuracy})

	def make_and_evaluate_label_predictions(self, ):

		X = self.X
		Y = self.Y
		predictor = self.predictor
		
		# if test_G is None:
		idx = self.val_idx
		G = self.val_G
		print ("evaluating label predictions on validation set")

		number_of_capsules_per_layer = self.args.number_of_capsules_per_layer
		num_classes = Y.shape[1]


		neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
		neighbourhood_sample_sizes = self.args.neighbourhood_sample_sizes
		batch_size = self.args.batch_size * (1 + self.args.num_positive_samples + self.args.num_negative_samples)
		# nodes_to_embed = np.array(sorted(G.nodes())).reshape(-1, 1)
		num_steps = int((len(idx) + batch_size - 1) // batch_size)

		batch_nodes = np.array(idx).reshape(-1, 1)
		assert batch_nodes.shape == (len(idx), 1)
		label_prediction_layers = np.where(number_of_capsules_per_layer==num_classes)[0] + 1
		prediction_layer = label_prediction_layers[-1]
		neighbourhood_sample_sizes = neighbourhood_sample_sizes[:prediction_layer]
		neighbourhood_sample_list = get_neighbourhood_samples(batch_nodes, neighbourhood_sample_sizes, neighbours)
		input_nodes = neighbourhood_sample_list[0]
		input_x = []
		for step in range(num_steps):
			nodes = input_nodes[step * batch_size : (step + 1) * batch_size]
			if sp.sparse.issparse(X):
				x = X[nodes.flatten()].toarray()
				x = preprocess_data(x)
				# x = x.reshape(-1, nodes.shape[1], 1,  X.shape[-1])
			else:
				x = X[nodes]
			x = x.reshape(-1, nodes.shape[1], 1,  X.shape[-1])
			input_x.append(x)
		input_x = np.concatenate(input_x)
		assert input_x.shape == (len(idx), input_nodes.shape[1], 1, X.shape[-1])

		# if self.val_prediction_gen is None:

		# 	print ("creating new label validation generator")

		# 	_, num_classes = Y.shape
		# 	number_of_capsules_per_layer = self.args.number_of_capsules_per_layer
		# 	label_prediction_layers = np.where(number_of_capsules_per_layer==num_classes)[0] + 1
		# 	prediction_layer = label_prediction_layers[-1]
		# 	neighbourhood_sample_sizes = self.args.neighbourhood_sample_sizes
		# 	self.neighbourhood_sample_sizes = neighbourhood_sample_sizes[:prediction_layer]

		# 	# nodes_to_predict = np.array(idx).reshape(-1, 1)

		# 	self.batch_size = self.args.batch_size * (1 + self.args.num_positive_samples + self.args.num_negative_samples)
		# 	self.num_steps = int((len(idx) + self.batch_size - 1) // self.batch_size)
		# 	self.val_prediction_gen = validation_generator(self, G, X, idx, self.neighbourhood_sample_sizes, 
		# 		num_steps=self.num_steps, batch_size=self.batch_size)


		# predictions = predictor.predict_generator(self.val_prediction_gen, steps=self.num_steps, )
		predictions = predictor.predict(input_x, batch_size=batch_size)


		# else:
		# 	idx = test_idx
		# 	G = test_G
		# 	print ("evaluating label predictions on test set")
		# 	# nodes_to_predict = np.array(idx).reshape(-1, 1)
		# 	num_steps = int((len(idx) + self.batch_size - 1) // self.batch_size)
		# 	test_prediction_gen = validation_generator(self, G, X, idx, self.neighbourhood_sample_sizes, 
		# 			num_steps=num_steps, batch_size=self.batch_size)

		# 	predictions = predictor.predict_generator(test_prediction_gen, steps=num_steps)


		predictions = predictions.reshape(-1, predictions.shape[-1])
		print (predictions)
		val_Y = Y[idx]
		if sp.sparse.issparse(val_Y):
			val_Y = val_Y.toarray()
		
		# print (self.nodes_to_val)

		# print(val_Y)
		# only consider validation labels (shuffled in generator)
		true_labels = val_Y.argmax(axis=-1)
		# print (true_labels)	

		predicted_labels = predictions.argmax(axis=-1)
		# print (predictions)
		# print (predicted_labels)

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