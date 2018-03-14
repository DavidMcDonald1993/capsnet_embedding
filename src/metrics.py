import os
import gzip
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, average_precision_score
# from sklearn.metrics.pairwise import pairwise_distances

# import matplotlib
# matplotlib.use('agg')

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# from utils import create_neighbourhood_sample_list

def hyperbolic_distance(u, v):
		return np.arccosh(1 + 2 * np.linalg.norm(u - v, axis=-1)**2 / ((1 - np.linalg.norm(u, axis=-1)**2) * (1 - np.linalg.norm(v, axis=-1)**2)))

def sigmoid(x):
	return 1. / (1 + np.exp(-x))


def evaluate_link_prediction(G, embedding, removed_edges, epoch, path, candidate_edges_path):

	def check_edge_in_edgelist((u, v), edgelist):
		for u_prime, v_prime in edgelist:
			if u_prime > u:
				return False
			if u_prime == u and v_prime == v:
				edgelist.remove((u, v))
				return True


	N = len(G)

	removed_edges = removed_edges[:]
	removed_edges.sort(key=lambda (u, v): (u, v))
	print ("loading candidate edges")
	candidate_edges = np.genfromtxt(candidate_edges_path, dtype=np.int)
	# print  "determining candidate edges"
	# candidate_edges = [(u, v)for u in range(N) for v in range(u+1, N) if (u, v) not in G.edges() and (v, u) not in G.edges()]
	# candidate_edges.extend(removed_edges)
	# zipped_candidate_edges = zip(*candidate_edges)

	print ("computing hyperbolic distance between all points")
	hyperbolic_distances = hyperbolic_distance(embedding[candidate_edges[:,0]], 
		embedding[candidate_edges[:,1]])
	r = 1
	t = 1
	print ("converting distances into probabilities")
	y_pred = sigmoid((r - hyperbolic_distances) / t)
	print ("determining labels")
	y_true = np.array([1. if check_edge_in_edgelist(edge, removed_edges) else 0 for edge in candidate_edges])

	print ("computing precision and recalls")
	average_precision = average_precision_score(y_true, y_pred)
	print ("MAP", average_precision)
	precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
	f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)

	plt.figure(figsize=(10, 10))
	plt.step(recalls[:-1], precisions[:-1], c="b", where="post")
	plt.fill_between(recalls[:-1], precisions[:-1], step='post', alpha=0.2,
					 color='b')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	# plt.ylim([0.0, 1.05])
	# plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
	plt.savefig(os.path.join(path, "recall-precision-epoch-{:04}.png".format(epoch)))
	plt.close()

	plt.figure(figsize=(10, 10))
	plt.plot(thresholds, f1_scores[:-1], c="r")
	plt.xlabel("threshold")
	plt.ylabel("F1 score")
	plt.savefig(os.path.join(path, "F1-epoch-{:04}.png".format(epoch)))
	plt.close()

	return average_precision

def evaluate_lexical_entailment(embedding):

	def is_a_score(u, v, alpha=1e3):
		return -(1 + alpha * (np.linalg.norm(v, axis=-1) - np.linalg.norm(u, axis=-1))) * hyperbolic_distance(u, v)

	print ("evaluating lexical entailment")

	hyperlex_noun_idx_df = pd.read_csv("../data/wordnet/hyperlex_idx_ranks.txt", index_col=0, sep=" ")

	U = np.array(hyperlex_noun_idx_df["WORD1"], dtype=int)
	V = np.array(hyperlex_noun_idx_df["WORD2"], dtype=int)

	true_is_a_score = np.array(hyperlex_noun_idx_df["AVG_SCORE_0_10"])
	predicted_is_a_score = is_a_score(embedding[U], embedding[V])

	r, p = spearmanr(true_is_a_score, predicted_is_a_score)

	print (r, p)

	return r, p


def evaluate_reconstruction(G, embedding):

	pass


	
# def make_and_evaluate_label_predictions(G, X, Y, predictor, num_capsules_per_layer, neighbourhood_sample_sizes, batch_size):

# 	print ("evaluating label predictions")


# 	def prediction_generator(X, input_nodes, batch_size=batch_size):
# 		num_steps = (input_nodes.shape[0] + batch_size - 1) / batch_size
# 		for step in range(num_steps):
# 			batch_nodes = input_nodes[batch_size*step : batch_size*(step+1)]
# 			x = X[batch_nodes]
# 			yield x.reshape([-1, input_nodes.shape[1], 1, X.shape[-1]])


# 	_, num_classes = Y.shape
# 	label_prediction_layers = np.where(num_capsules_per_layer==num_classes)[0] + 1

# 	nodes = np.arange(len(G)).reshape(-1, 1)
# 	neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
# 	neighbour_list = create_neighbourhood_sample_list(nodes, neighbourhood_sample_sizes[:label_prediction_layers[-1]], neighbours)

# 	input_nodes = neighbour_list[0]

# 	batch_size = 10
# 	num_steps = (input_nodes.shape[0] + batch_size - 1) / batch_size
# 	prediction_gen = prediction_generator(X, input_nodes, batch_size=batch_size)

# 	predictions = predictor.predict_generator(prediction_gen, steps=num_steps)
# 	predictions = predictions.reshape(-1, predictions.shape[-1])

# 	true_labels = Y.argmax(axis=-1)
# 	predicted_labels = predictions.argmax(axis=-1)

# 	print ("NMI of predictions: {}".format(normalized_mutual_info_score(true_labels, predicted_labels)))
# 	print ("Classification accuracy: {}".format((true_labels==predicted_labels).sum() / float(true_labels.shape[0])))