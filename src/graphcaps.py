import numpy as np
# import networkx as nx

import tensorflow as tf
from keras import backend as K
from keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, CSVLogger

import argparse
import os
# import pickle as pkl

from models import load_models#, generate_graphcaps_model
from generators import neighbourhood_sample_generator
# from data_utils import load_karate, load_wordnet, load_collaboration_network, load_data_gcn, load_reddit
from data_utils import load_data
from utils import load_positive_samples_and_ground_truth_negative_samples#, load_walks#, ValidationCallback
from metrics import evaluate_lexical_entailment#evaluate_link_prediction, make_and_evaluate_label_predictions, evaluate_lexical_entailment
from callbacks import ReconstructionLinkPredictionCallback, LabelPredictionCallback


# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5

# config.allow_soft_placement = True
# config.log_device_placement=True

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

def parse_args():
	parser = argparse.ArgumentParser(description="GraphCaps for feature learning of complex networks")

	parser.add_argument("--dataset", dest="dataset", type=str, default="cora",
		help="The dataset to load. Must be one of [wordnet, cora, citeseer, pubmed,\
		AstroPh, CondMat, GrQc, HepPh, karate]. (Default is cora)")

	parser.add_argument("--use-labels", action="store_true",
		help="Use this flag to include label prediction cross entropy in the final loss function.")
	parser.add_argument("--no-intermediary-loss", action="store_true", 
		help="Use this flag to not include loss from intermediary hyperbolic embeddings in final loss function.")
	parser.add_argument("--no-embedding-loss", action="store_true", 
		help="Use this flag to not include loss from all hyperbolic embeddings in final loss function.")

	parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=1000,
		help="The number of epochs to train for (default is 1000).")
	parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=100, 
		help="Batch size for training (default is 100).")
	# parser.add_argument("--npos", dest="num_pos", type=int, default=1, 
	# 	help="Number of positive samples for training (default is 1).")
	parser.add_argument("--nneg", dest="num_negative_samples", type=int, default=10, 
		help="Number of negative samples for training (default is 10).")
	parser.add_argument("--context-size", dest="context_size", type=int, default=5,
		help="Context size for generating positive samples (default is 5).")

	parser.add_argument("-s", "--sample_sizes", dest="neighbourhood_sample_sizes", type=int, nargs="+",
		help="Number of neighbourhood node samples for each layer separated by a space (default is [25,5]).", default=[25,5])
	parser.add_argument("-c", "--num_primary_caps", dest="num_primary_caps_per_layer", type=int, nargs="+",
		help="Number of primary capsules for each layer separated by space (default is [32, 32]).", default=[32, 32])
	parser.add_argument("-f", "--num_filters", dest="num_filters_per_layer", type=int, nargs="+",
		help="Number of filters for each layer separated by space (default is [32, 32]).", default=[32, 32])
	parser.add_argument("-a", "--agg_dim", dest="agg_dim_per_layer", type=int, nargs="+",
		help="Dimension of agg output for each layer separated by a space (default is [8, 8]).", default=[8, 8])
	parser.add_argument("-n", "--num_caps", dest="number_of_capsules_per_layer", type=int, nargs="+",
		help="Number of capsules for each layer separated by space (default is [7, 1]).", default=[7, 1])
	parser.add_argument("-d", "--capsule_dim", dest="capsule_dim_per_layer", type=int, nargs="+",
		help="Dimension of capule output for each layer separated by a space (default is [8, 2]).", default=[8, 2])
	parser.add_argument("--dim", dest="embedding_dim", type=int,
		help="Dimension of embedding capsule (default is 10).", default=10)

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
	# parser.add_argument("--pos_samples_path", dest="pos_samples_path", default="../positive_samples/", 
	# 	help="path to save positive sample list (default is '../positive_samples/)'.")
	# parser.add_argument("--neg_samples_path", dest="neg_samples_path", default="../negative_samples/", 
	# 	help="path to save ground truth negative sample for each node (default is '../negative_samples/)'.")
	parser.add_argument("--model", dest="model_path", default="../models/", 
		help="path to save model after each epoch (default is '../models/)'.")


	args = parser.parse_args()
	return args

def fix_parameters(args):


	args.neighbourhood_sample_sizes = [25, 5]
	args.num_primary_caps_per_layer = [16, 16]
	args.num_filters_per_layer = [16, 16]
	args.agg_dim_per_layer = [8, 8]
	# args.neighbourhood_sample_sizes = [5, 5 ]
	# args.num_primary_caps_per_layer = [8, 8]
	# args.num_filters_per_layer = [8, 8]
	# args.agg_dim_per_layer = [8, 8]
	args.batch_size = 1


	dataset = args.dataset
	if dataset in ["AstroPh", "CondMat", "HepPh", "GrQc", "wordnet"]:

		args.number_of_capsules_per_layer = [8, 1]
		args.capsule_dim_per_layer = [8, args.embedding_dim]

	elif dataset in ["citeseer", "cora", "pubmed", "reddit"]:

		if dataset == "citeseer":
			num_classes = 6
		elif dataset == "cora":
			num_classes = 7
		elif dataset == "pubmed":
			num_classes = 3
		else:
			num_classes = 41

		args.number_of_capsules_per_layer = [num_classes, 1]
		args.capsule_dim_per_layer = [8, 10]

def configure_paths(args):

	dataset = args.dataset
	directory = "neighbourhood_sample_sizes={}_num_primary_caps={}_num_filters={}_agg_dim={}_num_caps={}_caps_dim={}".format(args.neighbourhood_sample_sizes, 
				args.num_primary_caps_per_layer, args.num_filters_per_layer, 
				args.agg_dim_per_layer, args.number_of_capsules_per_layer, args.capsule_dim_per_layer)
	if args.no_embedding_loss:
		directory = "no_embedding_loss_" + directory
	elif args.no_intermediary_loss:
		directory = "no_intermediary_loss_" + directory

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
	# if not os.path.exists(log_path):
	# 	os.makedirs(log_path)
	args.walk_path = os.path.join(args.walk_path, dataset)
	if not os.path.exists(args.walk_path):
		os.makedirs(args.walk_path)
	# positive_samples_path = os.path.join(args.pos_samples_path, dataset)
	# if not os.path.exists(positive_samples_path):
	# 	os.makedirs(positive_samples_path)
	# negative_samples_path = os.path.join(args.neg_samples_path, dataset)
	# if not os.path.exists(negative_samples_path):
		# os.makedirs(negative_samples_path)
	args.model_path = os.path.join(args.model_path, dataset)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	args.model_path = os.path.join(args.model_path, directory)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)


def main():

	args = parse_args()
	args.num_positive_samples = 1

	fix_parameters(args)

	assert len(args.neighbourhood_sample_sizes) == len(args.num_primary_caps_per_layer) ==\
	len(args.num_filters_per_layer) == len(args.agg_dim_per_layer) == \
	len(args.number_of_capsules_per_layer) == len(args.capsule_dim_per_layer), "lengths of all input lists must be the same"

	dataset = args.dataset

	configure_paths(args)

	reconstruction_adj, G_train, G_val, G_test,\
	X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx = load_data(dataset)

	if dataset in ["citeseer", "cora", "pubmed", "reddit"]:
		assert Y.shape[1] in args.number_of_capsules_per_layer, "You must have a layer with {} capsules".format(Y.shape[1])
		args.use_labels = True
		monitor = "f1_micro"
		mode = "max"
		print("using labels in training")
	else:
		monitor = "mean_rank_reconstruction"
		mode = "min"

	walk_file = os.path.join(args.walk_path, "walks-{}-{}".format(args.num_walks, args.walk_length))
	# walk_train_file = os.path.join(walk_path, "walks_train.pkl")
	# walk_val_file = os.path.join(walk_path, "walks_val.pkl")

	# positive_samples_filename = os.path.join(positive_samples_path, "positive_samples")
	# negative_samples_filename = os.path.join(negative_samples_path, "negative_samples")


	# walks_train = load_walks(G_train, walk_train_file, args)
	# walks_val = load_walks(G_val, walk_val_file, args)
	# walks = load_walks(G, walk_file, args)
	positive_samples, ground_truth_negative_samples =\
	load_positive_samples_and_ground_truth_negative_samples(G_train, args, walk_file,)# positive_samples_filename, negative_samples_filename)

	# data_dim = X.shape[1]
	# num_classes = Y.shape[1]

	# batch_size = args.batch_size
	# num_positive_samples = 1
	# num_negative_samples = args.num_neg

	neighbourhood_sample_sizes = np.array(args.neighbourhood_sample_sizes[::-1])
	num_primary_caps_per_layer = np.array(args.num_primary_caps_per_layer)
	num_filters_per_layer = np.array(args.num_filters_per_layer)
	agg_dim_per_layer = np.array(args.agg_dim_per_layer)
	number_of_capsules_per_layer = np.array(args.number_of_capsules_per_layer)
	capsule_dim_per_layer = np.array(args.capsule_dim_per_layer)

	args.neighbourhood_sample_sizes = neighbourhood_sample_sizes
	args.num_primary_caps_per_layer = num_primary_caps_per_layer
	args.num_filters_per_layer = num_filters_per_layer
	args.agg_dim_per_layer = agg_dim_per_layer
	args.number_of_capsules_per_layer = number_of_capsules_per_layer
	args.capsule_dim_per_layer = capsule_dim_per_layer


	training_generator = neighbourhood_sample_generator(G_train, X, Y, train_label_mask, 
		positive_samples, ground_truth_negative_samples, args)
		# neighbourhood_sample_sizes, num_capsules_per_layer, 
		# args.num_positive_samples, args.num_negative_samples, args.batch_size,)

	model, embedder, label_prediction_model, initial_epoch = load_models(X, Y, args.model_path, args)
		# neighbourhood_sample_sizes, num_primary_caps_per_layer, num_filters_per_layer, agg_dim_per_layer,
		# num_capsules_per_layer, capsule_dim_per_layer, args)

	# validation_callback = ValidationCallback(G, X, Y, original_adj, 
	# 	val_mask, removed_edges_val, ground_truth_negative_samples, 
	# 	neighbourhood_sample_sizes, num_capsules_per_layer,
	# 	embedder, label_prediction_model, args.batch_size,
	# 	embedding_path=embedding_path, plot_path=plot_path, )

	nan_terminate_callback = TerminateOnNaN()
	reconstruction_callback = ReconstructionLinkPredictionCallback(G_train, X, Y, reconstruction_adj, embedder,
		val_edges, ground_truth_negative_samples, args.embedding_path, args.plot_path, args)
	label_prediction_callback = LabelPredictionCallback(G_val, X, Y, label_prediction_model, val_label_idx, args)
	early_stopping_callback = EarlyStopping(monitor=monitor, patience=10, mode=mode, verbose=1)
	checkpoint_callback = ModelCheckpoint(os.path.join(args.model_path, "{epoch:04d}-{mean_precision_reconstruction:.4f}-{mean_rank_reconstruction:.2f}.h5"),
		monitor=monitor, save_weights_only=False)
	logger_callback = CSVLogger(args.log_path, append=True)

	callbacks = [nan_terminate_callback, 
	reconstruction_callback, 
	label_prediction_callback, 
	early_stopping_callback, checkpoint_callback, logger_callback]

	
	print ("BEGIN TRAINING")

	# num_steps = int((len(positive_samples) // args.num_walks + args.batch_size - 1) // args.batch_size)
	num_steps = 1000
	model.fit_generator(training_generator, 
		steps_per_epoch=num_steps,
		epochs=args.num_epochs, 
		initial_epoch=initial_epoch,
		# validation_data=validation_generator, validation_steps=1,
		verbose=1, #)
		callbacks=callbacks)

	if test_label_idx is not None:
		label_prediction_callback.make_and_evaluate_label_predictions(G_test, test_label_idx)

	embedding = reconstruction_callback.perform_embedding()
	metrics = reconstruction_callback.evaluate_rank_and_MAP(embedding, test_edges)
	print ("Mean rank reconstruction:", metrics[0], "MAP reconstruction:", metrics[1])
	if test_edges is not None:
		print ("Mean rank link predicion:", metrics[2], "MAP link prediction:", metrics[3])

	if dataset == "wordnet":
		evaluate_lexical_entailment(embedding)

if __name__  == "__main__":
	main()