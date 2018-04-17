import numpy as np
import pandas as pd
from pandas import Index
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
# from metrics import evaluate_lexical_entailment#evaluate_link_prediction, make_and_evaluate_label_predictions, evaluate_lexical_entailment
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
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="GraphCaps for feature learning of complex networks")

	parser.add_argument("--dataset", dest="dataset", type=str, default="cora",
		help="The dataset to load. Must be one of [wordnet, cora, citeseer, pubmed,\
		AstroPh, CondMat, GrQc, HepPh, karate]. (Default is cora)")

	parser.add_argument("--just-walks", action="store_true",
		help="Use this flag to precompute walks and not train the model.")

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
	parser.add_argument("--model", dest="model_path", default="../models/", 
		help="path to save model after each epoch (default is '../models/)'.")


	args = parser.parse_args()
	return args

def fix_parameters(args):
	'''
	fix parameters for experiments
	'''


	args.neighbourhood_sample_sizes = [5, 5]
	args.num_primary_caps_per_layer = [16, 16]
	args.num_filters_per_layer = [16, 16]
	args.agg_dim_per_layer = [8, 8]
	args.batch_size = 100


	dataset = args.dataset
	if dataset in ["AstroPh", "CondMat", "HepPh", "GrQc", "wordnet", "wordnet_attributed"]:

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
	'''
	build directories on local system for output of model after each epoch
	'''

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

	args.walk_path = os.path.join(args.walk_path, dataset)
	if not os.path.exists(args.walk_path):
		os.makedirs(args.walk_path)

	args.model_path = os.path.join(args.model_path, dataset)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	args.model_path = os.path.join(args.model_path, directory)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

def record_initial_losses(model, gen, val_label_idx, val_edges, args, 
	reconstruction_callback, label_prediction_callback):

	'''
	record the loss of model in its untrained state -- with purely random weights
	'''
	
	print ("recording losses before training begins")

	initial_losses = {}

	if val_label_idx is not None:
		f1_micro, f1_macro, NMI, classification_accuracy =\
		label_prediction_callback.make_and_evaluate_label_predictions()
		initial_losses.update({"f1_micro": f1_micro, "f1_macro": f1_macro, 
				"NMI": NMI, "classification_accuracy": classification_accuracy})

	embedding = reconstruction_callback.perform_embedding()
	metrics = reconstruction_callback.evaluate_rank_and_MAP(embedding)
	mean_rank_reconstruction, mean_precision_reconstruction = metrics[:2]
	initial_losses.update({"mean_rank_reconstruction" : mean_rank_reconstruction, 
		"mean_precision_reconstruction" : mean_precision_reconstruction,})
	if val_edges is not None:
		mean_rank_link_prediction, mean_precision_link_prediction = metrics[2:4]
		initial_losses.update({"mean_rank_link_prediction": mean_rank_link_prediction,
		"mean_precision_link_prediction": mean_precision_link_prediction})

	if args.dataset in ["wordnet", "wordnet_attributed"]:
		r, p = reconstruction_callback.evaluate_lexical_entailment(embedding)
		initial_losses.update({"lex_r" : r, "lex_p" : p})
	
	# for l in model.output_layers:
	# 	initial_losses.update({"{}_loss".format(l.name) : np.NaN})
	# initial_losses.update({"loss" : np.NaN})
	print ("evaluating model on one step of training generator")
	evaluations = model.evaluate_generator(gen, steps=1)
	for metric, loss in zip(model.metrics_names, evaluations):
	# for i, l in enumerate(model.output_layers):
		print (metric, loss)
		initial_losses.update({metric : loss})
	# print ("loss", evaluations[-1])
	# initial_losses.update({"loss": evaluations[-1]})

	loss_df = pd.DataFrame(initial_losses, index=Index([0], name="epoch"))
	loss_df.to_csv(args.log_path)


def main():
	'''
	main function
	'''

	args = parse_args()
	args.num_positive_samples = 1

	# fix args for evaluation purposes
	fix_parameters(args)

	assert len(args.neighbourhood_sample_sizes) == len(args.num_primary_caps_per_layer) ==\
	len(args.num_filters_per_layer) == len(args.agg_dim_per_layer) == \
	len(args.number_of_capsules_per_layer) == len(args.capsule_dim_per_layer), "lengths of all input lists must be the same"

	dataset = args.dataset

	configure_paths(args)

	# load the dataset -- written for many types of exeriments so some returned objects are None
	reconstruction_adj, G_train, G_val, G_test,\
	X, Y, val_edges, test_edges, train_label_mask, val_label_idx, test_label_idx = load_data(dataset)

	# use labels for labelled networks
	if dataset in ["citeseer", "cora", "pubmed", "reddit"]:
		assert Y.shape[1] in args.number_of_capsules_per_layer, "You must have a layer with {} capsules".format(Y.shape[1])
		args.use_labels = True
		monitor = "f1_micro"
		mode = "max"
		print("using labels in training")
	else:
		monitor = "mean_precision_link_prediction"
		mode = "min"

	# the path of the file that contains the random walks for this network
	walk_file = os.path.join(args.walk_path, "walks-{}-{}".format(args.num_walks, args.walk_length))
	
	# will perform random walks if the walk file does not exist
	# uses these walks to build the set of posive and negative samples to train upon
	positive_samples, ground_truth_negative_samples =\
	load_positive_samples_and_ground_truth_negative_samples(G_train, args, walk_file,)# positive_samples_filename, negative_samples_filename)

	# use this flag to generator walks and not train the model -- for blue bear purposes (to save GPU requests)
	if args.just_walks:
		print ("Only precomputing walks -- terminating")
		return

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

	# create training generator object to produce samples from the precomputed postive and negative samples
	# also masks labels if dataset is labelled
	training_generator = neighbourhood_sample_generator(G_train, X, Y, train_label_mask, 
		positive_samples, ground_truth_negative_samples, args)

	# generates / loads a graph caps model according the args passed in from the command line
	# will load a model if an existing model exists on ther system with the same specifications
	model, embedder, label_prediction_model, initial_epoch = load_models(X, Y, args.model_path, args)

	# callbacks
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

	if initial_epoch == 1:
		record_initial_losses(model, training_generator,
		val_label_idx, val_edges, args, reconstruction_callback, label_prediction_callback)

	
	print ("BEGIN TRAINING")

	num_steps = int((len(positive_samples) // args.num_walks + args.batch_size - 1) // args.batch_size)
	# num_steps = 100
	model.fit_generator(training_generator, 
		steps_per_epoch=num_steps,
		epochs=args.num_epochs, 
		initial_epoch=initial_epoch,
		verbose=1, #)
		callbacks=callbacks)

	print ("TRAINING COMPLETE -- TESTING MODEL")

	if test_label_idx is not None:
		f1_micro, f1_macro, NMI, classification_accuracy =\
		label_prediction_callback.make_and_evaluate_label_predictions(G_test, test_label_idx)

	embedding = reconstruction_callback.perform_embedding()
	metrics = reconstruction_callback.evaluate_rank_and_MAP(embedding, test_edges)
	print ("Mean rank reconstruction:", metrics[0], "MAP reconstruction:", metrics[1])
	if test_edges is not None:
		print ("Mean rank link predicion:", metrics[2], "MAP link prediction:", metrics[3])

	if dataset in ["wordnet", "wordnet_attributed"]:
		r, p = reconstruction_callback.evaluate_lexical_entailment(embedding, dataset)

if __name__  == "__main__":
	main()