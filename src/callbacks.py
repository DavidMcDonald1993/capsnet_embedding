import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import average_precision_score, normalized_mutual_info_score, accuracy_score, f1_score

from keras.callbacks import Callback

from utils import create_neighbourhood_sample_list
from data_utils import preprocess_data


class ReconstructionLinkPredictionCallback(Callback):

	def __init__(self, G, X, Y, original_adj, embedder,
		removed_edges_val, ground_truth_negative_samples, embedding_path, plot_path, args,):
		# neighbourhood_sample_sizes, num_capsules_per_layer,
		# embedder, predictor, batch_size,
		self.G = G 
		self.X = X
		self.Y = Y
		self.original_adj = original_adj
		self.embedder = embedder
		if removed_edges_val is not None:
			self.removed_edges_dict = self.convert_edgelist_to_dict(removed_edges_val)
			# self.compute_link_prediction = True
		else:
			self.removed_edges_dict = None
			# self.compute_link_prediction = False
		# self.val_mask = val_mask
		self.ground_truth_negative_samples = ground_truth_negative_samples
		self.args = args
		# self.removed_edges_val = sorted(removed_edges_val, key=lambda(u, v): (u, v))
		# self.neighbourhood_sample_sizes = neighbourhood_sample_sizes
		# self.num_capsules_per_layer = num_capsules_per_layer
		# self.embedder = embedder
		# self.predictor = predictor
		# self.batch_size = batch_size
		# self.annotate_idx = annotate_idx
		self.embedding_path = embedding_path
		self.plot_path = plot_path
		# self.G_edge_dict = self.convert_edgelist_to_dict(G.edges(), self_edges=True)
		# self.candidate_edges_path = candidate_edges_path
		# if not os.path.exists(candidate_edges_path):
		# 	print "writing candidate edges to {}".format(candidate_edges_path)
		# 	self.write_candidate_edges()

	def on_epoch_end(self, epoch, logs={}):
		embedding = self.perform_embedding()
		# average_precision = evaluate_link_prediction(self.G, embedding, 
		# 	self.removed_edges_val, epoch=epoch, path=self.plot_path, candidate_edges_path=self.candidate_edges_path)
		metrics = self.evaluate_rank_and_MAP(embedding)
		mean_rank_reconstruction, mean_precision_reconstruction = metrics[:2]
		logs.update({"mean_rank_reconstruction" : mean_rank_reconstruction, 
			"mean_precision_reconstruction" : mean_precision_reconstruction,})
		if self.removed_edges_dict is not None:
			mean_rank_link_prediction, mean_precision_link_prediction = metrics[2:4]
			logs.update({"mean_rank_link_prediction": mean_rank_link_prediction,
			"mean_precision_link_prediction": mean_precision_link_prediction})
			

		self.plot_embedding(embedding, path="{}/embedding_epoch_{:04}".format(self.embedding_path, epoch))
	
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

	def evaluate_rank_and_MAP(self, embedding, test_edges=None):

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
		
		print ("evaluating rank and MAP")

		original_adj = self.original_adj
		if test_edges is None:
			removed_edges_dict = self.removed_edges_dict
			print ("evaluating on validation edges")
		else:
			removed_edges_dict = self.convert_edgelist_to_dict(test_edges)
			print ("evaluating on test edges")

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
		    y_pred = sigmoid((r - dists_u) / t) 
		   
		    y_pred_reconstruction = y_pred.copy()
		    y_true_reconstruction = original_adj[u].toarray().flatten()
		    MAPs_reconstruction[u] = average_precision_score(y_true=y_true_reconstruction, 
		    	y_score=y_pred_reconstruction)
		    
		    y_pred_reconstruction[::-1].sort()
		    ranks_reconstruction[u] = np.array([np.searchsorted(-y_pred_reconstruction, -p) 
		                                        for p in y_pred[y_true_reconstruction.astype(np.bool)]]).mean()
		    
		    if removed_edges_dict is not None and removed_edges_dict.has_key(u):
		    
		        removed_neighbours = removed_edges_dict[u]
		        all_negative_samples = ground_truth_negative_samples[u]
		        y_true_link_prediction = np.append(np.ones(len(removed_neighbours)), 
		                                           np.zeros(len(all_negative_samples)))
		        y_pred_link_prediction = np.append(y_pred[removed_neighbours], y_pred[all_negative_samples])
		        MAPs_link_prediction[u] = average_precision_score(y_true=y_true_link_prediction, y_score=y_pred_link_prediction)

		        y_pred_link_prediction[::-1].sort()
		        ranks_link_prediction[u] = np.array([np.searchsorted(-y_pred_link_prediction, -p) 
		                                            for p in y_pred[removed_neighbours]]).mean()

		    if u % 1000 == 0:
				print( "completed node {}/{}".format(u, N))


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
		metrics = [ranks_reconstruction.mean(), MAPs_reconstruction.mean()]
		if removed_edges_dict is not None:
			metrics += [ranks_link_prediction.mean(), MAPs_link_prediction.mean()]
		return metrics


		# removed_edges = self.removed_edges_val[:]
		# removed_edges.sort(key=lambda (u, v): (u, v))

	# def write_candidate_edges(self):
	# 	G = self.G
	# 	N = len(G)
	# 	candidate_edges = ((u, v)for u in range(N) for v in range(u+1, N) if (u, v) not in G.edges() and (v, u) not in G.edges())
	# 	with gzip.open(self.candidate_edges_path, "w") as f:
	# 		for u, v in candidate_edges:
	# 			f.write("{} {}\n".format(u, v))


	


	def perform_embedding(self):

		print ("performing embedding")

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
		neighbourhood_sample_sizes = self.args.neighbourhood_sample_sizes
		embedder = self.embedder
		batch_size = self.args.batch_size

		nodes_to_embed = np.array(sorted(list(G.nodes()))).reshape(-1, 1)

		# nodes = nodes.reshape(-1, 1)
		neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
		neighbour_list = create_neighbourhood_sample_list(nodes_to_embed, neighbourhood_sample_sizes, neighbours)

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

		Y = self.Y
		y = Y.argmax(axis=1)
		if sp.sparse.issparse(Y):
			y = y.A1
		
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

	
class LabelPredicitonCallback(Callback):

	def __init__(self, val_G, X, Y, predictor, val_idx, args):
		self.val_G = val_G
		self.X = X
		self.Y = Y
		self.predictor = predictor
		self.val_idx = val_idx
		self.args = args

	def on_epoch_end(self, epoch, logs={}):

		if self.val_idx is not None:
			f1_micro, f1_macro, NMI, classification_accuracy = self.make_and_evaluate_label_predictions()
			logs.update({"f1_micro": f1_micro, "f1_macro": f1_macro, 
				"NMI": NMI, "classification_accuracy": classification_accuracy})

	def make_and_evaluate_label_predictions(self, test_G=None, test_idx=None):



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

		X = self.X
		Y = self.Y
		predictor = self.predictor
		number_of_capsules_per_layer = self.args.number_of_capsules_per_layer
		neighbourhood_sample_sizes = self.args.neighbourhood_sample_sizes
		batch_size = self.args.batch_size
		
		if test_G is None:
			idx = self.val_idx
			G = self.val_G
			print ("evaluating label predictions on validation set")
		else:
			idx = test_idx
			G = test_G
			print ("evaluating label predictions on test set")

		if idx is None:
			print ("no labels to predict")
			return None

		_, num_classes = Y.shape
		label_prediction_layers = np.where(number_of_capsules_per_layer==num_classes)[0] + 1

		nodes_to_predict = np.array(idx).reshape(-1, 1)
		neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
		neighbour_list = create_neighbourhood_sample_list(nodes_to_predict, 
			neighbourhood_sample_sizes[:label_prediction_layers[-1]], neighbours)

		input_nodes = neighbour_list[0]

		num_steps = (input_nodes.shape[0] + batch_size - 1) / batch_size
		prediction_gen = prediction_generator(X, input_nodes, batch_size=batch_size)

		predictions = predictor.predict_generator(prediction_gen, steps=num_steps)
		predictions = predictions.reshape(-1, predictions.shape[-1])

		# only consider validation labels
		true_labels = Y[idx].argmax(axis=-1)
		if sp.sparse.issparse(Y):
			true_labels = true_labels.A1			

		predicted_labels = predictions.argmax(axis=-1)

		f1_micro = f1_score(true_labels, predicted_labels, average="micro")
		f1_macro = f1_score(true_labels, predicted_labels, average="macro")
		NMI = normalized_mutual_info_score(true_labels, predicted_labels)
		classification_accuracy = accuracy_score(true_labels, predicted_labels, normalize=True)

		print ("f1_micro={}, f1_macro={}".format(f1_micro, f1_macro))
		print ("NMI of predictions: {}".format(NMI))
		print ("Classification accuracy: {}".format(classification_accuracy))
		
		return f1_micro, f1_macro, NMI, classification_accuracy