import numpy as np
import networkx as nx

# from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, TensorBoard

import argparse
import os
import pickle as pkl

from models import load_models, generate_graphcaps_model
from generators import neighbourhood_sample_generator
from data_utils import load_karate, load_wordnet, load_citation_network, load_data_gcn, preprocess_data, remove_edges, split_data 
from utils import load_walks, ValidationCallback
from metrics import evaluate_link_prediction, make_and_evaluate_label_predictions, evaluate_lexical_entailment



# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.75

# config.allow_soft_placement = True
# config.log_device_placement=True

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

def parse_args():
	parser = argparse.ArgumentParser(description="GraphCaps for feature learning of complex networks")

	parser.add_argument("--dataset", dest="dataset", type=str, default="cora",
		help="The dataset to load. Must be one of [wordnet, cora, citeseer, pubmed,\
		AstroPh, CondMat, GrQc, HepPh, karate]. (Default is cora)")

	parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=1000,
		help="The number of epochs to train for (default is 1000).")
	parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=100, 
		help="Batch size for training (default is 100).")
	# parser.add_argument("--npos", dest="num_pos", type=int, default=1, 
	# 	help="Number of positive samples for training (default is 1).")
	parser.add_argument("--nneg", dest="num_neg", type=int, default=10, 
		help="Number of negative samples for training (default is 10).")
	parser.add_argument("--context_size", dest="context_size", type=int, default=5,
		help="Context size for generating positive samples (default is 5).")

	parser.add_argument("-s", "--sample_sizes", dest="neighbourhood_sample_sizes", type=int, nargs="+",
		help="Number of neighbourhood node samples for each layer separated by a space (default is [25,5]).", default=[25,5])
	parser.add_argument("-c", "--num_primary_caps", dest="num_primary_caps_per_layer", type=int, nargs="+",
		help="Number of primary capsules for each layer separated by space (default is [32, 32]).", default=[32, 32])
	parser.add_argument("-f", "--num_filters", dest="num_filters_per_layer", type=int, nargs="+",
		help="Number of filters for each layer separated by space (default is [32, 32]).", default=[32, 32])
	parser.add_argument("-a", "--agg_dim", dest="agg_dim_per_layer", type=int, nargs="+",
		help="Dimension of agg output for each layer separated by a space (default is [8, 8]).", default=[8, 8])
	parser.add_argument("-n", "--num_caps", dest="num_capsules_per_layer", type=int, nargs="+",
		help="Number of capsules for each layer separated by space (default is [7, 1]).", default=[7, 1])
	parser.add_argument("-d", "--capsule_dim", dest="capsule_dim_per_layer", type=int, nargs="+",
		help="Dimension of capule output for each layer separated by a space (default is [8, 2]).", default=[8, 2])


	parser.add_argument("-p", dest="p", type=float, default=1.,
		help="node2vec return parameter (default is 1.).")
	parser.add_argument("-q", dest="q", type=float, default=1.,
		help="node2vec in-out parameter (default is 1.).")
	parser.add_argument('--num-walks', dest="num_walks", type=int, default=10, 
		help="Number of walks per source (default is 10).")
	parser.add_argument('--walk-length', dest="walk_length", type=int, default=15, 
		help="Length of random walk from source (default is 15).")

	

	parser.add_argument("--plot", dest="plot_path", default="../plots/", 
		help="path to save plots (default is '../plots/)'.")
	parser.add_argument("--embeddings", dest="embedding_path", default="../embeddings/", 
		help="path to save embeddings (default is '../embeddings/)'.")
	parser.add_argument("--logs", dest="log_path", default="../logs/", 
		help="path to save logs (default is '../logs/)'.")
	parser.add_argument("--walks", dest="walk_path", default="../walks/", 
		help="path to save random walks (default is '../walks/)'.")
	parser.add_argument("--model", dest="model_path", default="../models/", 
		help="path to save model after each epoch (default is '../models/)'.")


	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	assert len(args.neighbourhood_sample_sizes) == len(args.num_primary_caps_per_layer) ==\
	len(args.num_filters_per_layer) == len(args.agg_dim_per_layer) == \
	len(args.num_capsules_per_layer) == len(args.capsule_dim_per_layer), "lengths of all input lists must be the same"

	dataset = args.dataset

	plot_path = os.path.join(args.plot_path, dataset)
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)
	embedding_path = os.path.join(args.embedding_path, dataset)
	if not os.path.exists(embedding_path):
		os.makedirs(embedding_path)
	embedding_path = os.path.join(embedding_path, 
			"neighbourhood_sample_sizes={}_num_primary_caps={}_num_filters={}_agg_dim={}_num_caps={}_caps_dim={}".format(args.neighbourhood_sample_sizes, 
				args.num_primary_caps_per_layer, args.num_filters_per_layer, 
				args.agg_dim_per_layer, args.num_capsules_per_layer, args.capsule_dim_per_layer))
	if not os.path.exists(embedding_path):
		os.makedirs(embedding_path)
	log_path = os.path.join(args.log_path, dataset)
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	walk_path = os.path.join(args.walk_path, dataset)
	if not os.path.exists(walk_path):
		os.makedirs(walk_path)
	model_path = os.path.join(args.model_path, dataset)
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	model_path = os.path.join(model_path, 
		"neighbourhood_sample_sizes={}_num_primary_caps={}_num_filters={}_agg_dim={}_num_caps={}_caps_dim={}".format(args.neighbourhood_sample_sizes, 
			args.num_primary_caps_per_layer, args.num_filters_per_layer, 
			args.agg_dim_per_layer, args.num_capsules_per_layer, args.capsule_dim_per_layer))
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	if dataset == "wordnet":
		G, X, Y, removed_edges_val, removed_edges_test = load_wordnet()
	elif dataset == "karate":
		G, X, Y, removed_edges_val, removed_edges_test = load_karate()
	elif dataset in ["citeseer", "cora", "pubmed"]:
		G, X, Y, removed_edges_val, removed_edges_test, train_mask, val_mask, test_mask = load_data_gcn(dataset)
	else:
		G, X, Y, removed_edges_val, removed_edges_test, train_mask, val_mask, test_mask = load_citation_network(dataset)


	walk_file = os.path.join(walk_path, "walks.pkl")
	# walk_train_file = os.path.join(walk_path, "walks_train.pkl")
	# walk_val_file = os.path.join(walk_path, "walks_val.pkl")



	# walks_train = load_walks(G_train, walk_train_file, args)
	# walks_val = load_walks(G_val, walk_val_file, args)
	walks = load_walks(G, walk_file, args)


	data_dim = X.shape[1]
	num_classes = Y.shape[1]

	batch_size = args.batch_size
	num_positive_samples = 1
	num_negative_samples = args.num_neg

	neighbourhood_sample_sizes = np.array(args.neighbourhood_sample_sizes[::-1])
	num_primary_caps_per_layer = np.array(args.num_primary_caps_per_layer)
	num_filters_per_layer = np.array(args.num_filters_per_layer)
	agg_dim_per_layer = np.array(args.agg_dim_per_layer)
	num_capsules_per_layer = np.array(args.num_capsules_per_layer)
	capsule_dim_per_layer = np.array(args.capsule_dim_per_layer)



	training_generator = neighbourhood_sample_generator(G, X, Y, train_mask, walks,
		neighbourhood_sample_sizes, num_capsules_per_layer, 
		num_positive_samples, num_negative_samples, args.context_size, args.batch_size,)

	model, embedder, label_prediction_model, initial_epoch = load_models(X, Y, model_path, 
		neighbourhood_sample_sizes, num_primary_caps_per_layer, num_filters_per_layer, agg_dim_per_layer,
		num_capsules_per_layer, capsule_dim_per_layer, args)


	validation_callback = ValidationCallback(G, X, Y, val_mask, 
		removed_edges_val, neighbourhood_sample_sizes, num_capsules_per_layer,
		embedder, label_prediction_model, batch_size,
		# annotate_idx=nodes_to_annotate, 
		embedding_path=embedding_path, plot_path=plot_path, )
	
	print "BEGIN TRAINING"
	# raise SystemExit

	model.fit_generator(training_generator, 
		steps_per_epoch=1,#len(G) / batch_size,
		epochs=args.num_epochs, 
		initial_epoch=initial_epoch,
		# validation_data=validation_generator, validation_steps=1,
		verbose=1, callbacks=[validation_callback, TerminateOnNaN(), 
		EarlyStopping(monitor="mean_rank", patience=10, mode="min", verbose=1),
		ModelCheckpoint(os.path.join(model_path, "{epoch:04d}-{mean_precision:.4f}-{mean_rank:.2f}.h5"), 
			monitor="mean_precision", save_weights_only=False),])
		# TensorBoard(log_dir=log_path, batch_size=batch_size)])

	if label_prediction_model is not None and Y.shape[1] > 1:
		validation_callback.make_and_evaluate_label_predictions()

	embedding = validation_callback.perform_embedding()

	mean_rank, MAP = validation_callback.evaluate_rank_and_MAP(embedding, )
	print "Mean rank:", mean_rank, "MAP:", MAP

	if dataset == "wordnet":
		evaluate_lexical_entailment(embedding)

if __name__  == "__main__":
	main()