import numpy as np
import scipy as sp
import pandas as pd

# import random

from scipy.stats import spearmanr

# import matplotlib
# matplotlib.use('agg')

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import average_precision_score, normalized_mutual_info_score, accuracy_score, f1_score

from keras.callbacks import Callback

# from utils import create_neighbourhood_sample_list
# from data_utils import preprocess_data
from generators import validation_generator #embedding_generator, prediction_generator

def hyperbolic_distance(u, v):
	return np.arccosh(1. + 2. * np.linalg.norm(u - v, axis=-1)**2 / ((1. - np.linalg.norm(u, axis=-1)**2) * (1. - np.linalg.norm(v, axis=-1)**2)))

class ReconstructionLinkPredictionCallback(Callback):

	def __init__(self, G, X, Y, original_adj, embedder,
		removed_edges_val, ground_truth_negative_samples, embedding_path, plot_path, args,):
		self.G = G 
		self.X = X
		self.Y = Y
		self.original_adj = original_adj
		self.embedder = embedder
		if removed_edges_val is not None:
			self.removed_edges_dict = self.convert_edgelist_to_dict(removed_edges_val)
		else:
			self.removed_edges_dict = None
		self.ground_truth_negative_samples = ground_truth_negative_samples
		self.args = args
		self.embedding_path = embedding_path
		self.plot_path = plot_path
		self.embedding_gen = None
		self.nodes_to_val = None
		self.idx = G.nodes()

	def on_epoch_end(self, epoch, logs={}):
		embedding = self.perform_embedding()

		metrics = self.evaluate_rank_and_MAP(embedding)
		mean_rank_reconstruction, mean_precision_reconstruction = metrics[:2]
		logs.update({"mean_rank_reconstruction" : mean_rank_reconstruction, 
			"mean_precision_reconstruction" : mean_precision_reconstruction,})
		if self.removed_edges_dict is not None:
			mean_rank_link_prediction, mean_precision_link_prediction = metrics[2:4]
			logs.update({"mean_rank_link_prediction": mean_rank_link_prediction,
			"mean_precision_link_prediction": mean_precision_link_prediction})

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
		if self.embedding_gen is None:
			G = self.G
			X = self.X
			idx = self.idx
			neighbourhood_sample_sizes = self.args.neighbourhood_sample_sizes
			batch_size = self.args.batch_size * (1 + self.args.num_positive_samples + self.args.num_negative_samples)
			# nodes_to_embed = np.array(sorted(G.nodes())).reshape(-1, 1)
			self.num_steps = int((len(idx) + batch_size - 1) // batch_size)
			self.embedding_gen = validation_generator(self, G, X, idx, neighbourhood_sample_sizes, 
				self.num_steps, batch_size)
		

		embedder = self.embedder
		embedding = embedder.predict_generator(self.embedding_gen, steps=self.num_steps, )
		dim = embedding.shape[-1]
		embedding = embedding.reshape(-1, dim)

		# sort back into numerical order -- generator shuffles idx
		embedding = embedding[argsort(self.nodes_to_val)]

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
		
		embedding_dim = embedding.shape[-1]

		fig = plt.figure(figsize=(10, 10))
		plt.scatter(embedding[:,0], embedding[:,1], c=y)
		if path is not None:
			plt.savefig(path)
		plt.close()
	
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

	def evaluate_rank_and_MAP(self, embedding,):

		
		def sigmoid(x):
			return 1. / (1 + np.exp(-x))
		
		print ("evaluating rank and MAP")

		original_adj = self.original_adj
		# if test_edges is None:
		removed_edges_dict = self.removed_edges_dict
		print ("evaluating on validation edges")
		# else:
		# 	removed_edges_dict = self.convert_edgelist_to_dict(test_edges)
		# 	print ("evaluating on test edges")

		ground_truth_negative_samples = self.ground_truth_negative_samples

		r = 1.
		t = 1.
		
		G = self.G
		N = len(G)

		MAPs_reconstruction = np.zeros(N)
		ranks_reconstruction = np.zeros(N)
		if removed_edges_dict is not None:
			MAPs_link_prediction = np.zeros(N)
			ranks_link_prediction = np.zeros(N)

		for u in range(N):
			
			dists_u = hyperbolic_distance(embedding[u], embedding)

			_, all_neighbours = np.nonzero(original_adj[u])
			all_node_dists = dists_u.copy()
			all_node_dists.sort()

			ranks_reconstruction[u] = np.array([np.searchsorted(all_node_dists, d) 
												for d in dists_u[all_neighbours]]).mean()
			y_pred = sigmoid((r - dists_u) / t) 
		   
			# y_pred_reconstruction = y_pred.copy()
			y_true_reconstruction = original_adj[u].todense().A1
			MAPs_reconstruction[u] = average_precision_score(y_true=y_true_reconstruction, 
				y_score=y_pred)
			
			# y_pred_reconstruction[::-1].sort()
			# ranks_reconstruction[u] = np.array([np.searchsorted(-y_pred_reconstruction, -p) 
			# 									for p in y_pred[y_true_reconstruction.astype(np.bool)]]).mean()
			
			if removed_edges_dict is not None and removed_edges_dict.has_key(u):
			
				removed_neighbours = removed_edges_dict[u]
				all_negative_samples = ground_truth_negative_samples[u]

				dist_negative_samples = dists_u[all_negative_samples]
				dist_negative_samples.sort()

				ranks_link_prediction[u] = np.array([np.searchsorted(dist_negative_samples, d) 
													for d in dists_u[removed_neighbours]]).mean()

				y_true_link_prediction = np.append(np.ones(len(removed_neighbours)), 
												   np.zeros(len(all_negative_samples)))
				y_pred_link_prediction = np.append(y_pred[removed_neighbours], y_pred[all_negative_samples])
				MAPs_link_prediction[u] = average_precision_score(y_true=y_true_link_prediction, 
					y_score=y_pred_link_prediction)

				# y_pred_link_prediction[::-1].sort()
				# ranks_link_prediction[u] = np.array([np.searchsorted(-y_pred_link_prediction, -p) 
				# 									for p in y_pred[removed_neighbours]]).mean()

			if u % 1000 == 0:
				print ("completed node {}/{}".format(u, N))

		print ("completed node {}/{}".format(u, N))

		metrics = [ranks_reconstruction.mean(), MAPs_reconstruction.mean()]
		if removed_edges_dict is not None:
			metrics += [ranks_link_prediction.mean(), MAPs_link_prediction.mean()]
		return metrics


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

		if self.val_prediction_gen is None:

			print ("creating new label validation generator")

			_, num_classes = Y.shape
			number_of_capsules_per_layer = self.args.number_of_capsules_per_layer
			label_prediction_layers = np.where(number_of_capsules_per_layer==num_classes)[0] + 1
			prediction_layer = label_prediction_layers[-1]
			neighbourhood_sample_sizes = self.args.neighbourhood_sample_sizes
			self.neighbourhood_sample_sizes = neighbourhood_sample_sizes[:prediction_layer]

			# nodes_to_predict = np.array(idx).reshape(-1, 1)

			self.batch_size = self.args.batch_size * (1 + self.args.num_positive_samples + self.args.num_negative_samples)
			self.num_steps = int((len(idx) + self.batch_size - 1) // self.batch_size)
			self.val_prediction_gen = validation_generator(self, G, X, idx, self.neighbourhood_sample_sizes, 
				num_steps=self.num_steps, batch_size=self.batch_size)


		predictions = predictor.predict_generator(self.val_prediction_gen, steps=self.num_steps, )


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

		val_Y = Y[self.nodes_to_val]
		if sp.sparse.issparse(val_Y):
			val_Y = val_Y.toarray()
		
		print (self.nodes_to_val)

		print(val_Y)
		# only consider validation labels (shuffled in generator)
		true_labels = val_Y.argmax(axis=-1)
		print (true_labels)	

		predicted_labels = predictions.argmax(axis=-1)
		print (predictions)
		print (predicted_labels)

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