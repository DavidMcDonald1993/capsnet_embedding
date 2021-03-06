import numpy as np
import scipy as sp
import pandas as pd
from pandas import Index
# import networkx as nx

import tensorflow as tf
from keras import backend as K
from keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard

import argparse
import os
# import pickle as pkl

from models import load_graphcaps#, generate_graphcaps_model
from generators import training_generator, get_training_sample#neighbourhood_sample_generator
# from data_utils import load_karate, load_wordnet, load_collaboration_network, load_data_gcn, load_reddit
from data_utils import load_data#, preprocess_data
# from utils import load_positive_samples_and_ground_truth_negative_samples#, load_walks#, ValidationCallback
# from metrics import evaluate_lexical_entailment#evaluate_link_prediction, make_and_evaluate_label_predictions, evaluate_lexical_entailment
# from callbacks import ReconstructionLinkPredictionCallback, LabelPredictionCallback
from callbacks import ValidationCallback

np.set_printoptions(suppress=True)

# Set random seed
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.75

# config.allow_soft_placement = True
config.log_device_placement=True

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
	parser.add_argument("--no-reconstruction", action="store_true", 
		help="Use this flag to not include loss from reconstruction in loss function.")
	
	parser.add_argument('--feature_prob_loss_weight', dest="feature_prob_loss_weight", 
		type=float, default=1e-0, 
		help="Weighting of feature prob loss (default is 1.).")
	parser.add_argument('--embedding_loss_weight', dest="embedding_loss_weight", 
		type=float, default=1e-2, 
		help="Weighting of embedding loss (default is 1.).")
	parser.add_argument('--reconstruction_loss_weight', dest="reconstruction_loss_weight", 
			type=float, default=0.0005, 
			help="Weighting of reconstruction loss (default is 0.0005).")

	parser.add_argument("--scale_data", action="store_true", 
		help="Use this flag to standard scale input data.")


	parser.add_argument("--test-only", action="store_true", 
		help="Use this flag to only test the model.")



	parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=10000,
		help="The number of epochs to train for (default is 10000).")
	parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=100, 
		help="Batch size for training (default is 100).")
	parser.add_argument("--num_steps", dest="num_steps", type=int, default=1000, 
		help="Number of batch steps per epoch for training (default is 1000).")
	parser.add_argument("--nneg", dest="num_negative_samples", type=int, default=10, 
		help="Number of negative samples for training (default is 10).")
	parser.add_argument("--context-size", dest="context_size", type=int, default=5,
		help="Context size for generating positive samples (default is 5).")
	parser.add_argument("--patience", dest="patience", type=int, default=10,
		help="The number of epochs of no improvement before training is stopped. (Default is 10)")

	parser.add_argument("-s", "--sample_sizes", dest="neighbourhood_sample_sizes", type=int, nargs="+",
		help="Number of neighbourhood node samples for each layer separated by a space (default is [25,5]).", default=[25,5])
	parser.add_argument("-c", "--num_primary_caps", dest="num_primary_caps", type=int, 
		help="Number of primary capsules (default is 128).", default=128)
	parser.add_argument("-cd", "--primary_cap_dim", dest="primary_cap_dim", type=int, 
		help="Dimension of primary capsule (default is 8).", default=8)
	# parser.add_argument("-f", "--num_filters", dest="num_filters_per_layer", type=int, nargs="+",
	# 	help="Number of filters for each layer separated by space (default is [32, 32]).", default=[32, 32])
	# parser.add_argument("-a", "--agg_dim", dest="agg_dim_per_layer", type=int, nargs="+",
	# 	help="Dimension of agg output for each layer separated by a space (default is [8, 8]).", default=[8, 8])
	parser.add_argument("-n", "--num_caps", dest="number_of_capsules_per_layer", type=int, nargs="+",
		help="Number of capsules for each layer separated by space (default is [7, 1]).", default=[7, 1])
	parser.add_argument("-d", "--capsule_dim", dest="capsule_dim_per_layer", type=int, nargs="+",
		help="Dimension of capule output for each layer separated by a space (default is [8, 2]).", default=[8, 2])
	parser.add_argument("--num_routing", dest="num_routing", type=int,
		help="Number of iterations of routing algorithm (default is 3).", default=3)
	parser.add_argument("--dim", dest="embedding_dims", type=int, nargs="+",
		help="Dimension of embeddings for each layer (default is [10]).", default=[10])

	parser.add_argument("-p", dest="p", type=float, default=1,
		help="node2vec return parameter (default is 1.).")
	parser.add_argument("-q", dest="q", type=float, default=1.,
		help="node2vec in-out parameter (default is 1.).")
	parser.add_argument('--num-walks', dest="num_walks", type=int, default=10, 
		help="Number of walks per source (default is 10).")
	parser.add_argument('--walk-length', dest="walk_length", type=int, default=15, 
		help="Length of random walk from source (default is 15).")
	
	
	parser.add_argument('--reconstruction_dim', dest="reconstruction_dim", 
		type=int, default=512, 
		help="Number of hidden units for reconstrction model (default is 512).")
	

	parser.add_argument("--plot", dest="plot_path", default="../plots/", 
		help="path to save plots (default is '../plots/)'.")
	parser.add_argument("--embeddings", dest="embedding_path", default="../embeddings/", 
		help="path to save embeddings (default is '../embeddings/)'.")
	parser.add_argument("--logs", dest="log_path", default="../logs/", 
		help="path to save logs (default is '../logs/)'.")
	parser.add_argument("--boards", dest="board_path", default="../tensorboards/", 
		help="path to save tensorboards (default is '..tensorboards/)'.")
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
	dataset = args.dataset

	if dataset in ["cora", "karate", "wordnet_attributed"]:

		# args.num_walks = 25
		# args.walk_length = 2
		args.context_size = 3



		# args.num_primary_caps_per_layer = [16, ]
		# args.num_filters_per_layer = [ 1, ]
		# args.agg_dim_per_layer = [8, ]
		args.batch_size = 10



		if dataset == "cora":

			args.neighbourhood_sample_sizes = [0 , ]
			args.num_negative_samples = 10

			num_classes = 7
			args.scale_data = True
			# args.use_labels = True
			args.num_primary_caps = 64
			args.primary_cap_dim = 8


			args.embedding_dims = [2,  ]



			args.number_of_capsules_per_layer = [8, ]
			args.capsule_dim_per_layer = [ 16, ]

		else:
			args.neighbourhood_sample_sizes = [0, ]
			args.num_negative_samples = 10

			num_classes = 4
			args.scale_data = True
			# args.use_labels = True
			args.num_primary_caps = 32
			args.primary_cap_dim = 8

			args.reconstruction_dim = 32

			args.embedding_dims = [2,  ]


			args.number_of_capsules_per_layer = [4, ]
			args.capsule_dim_per_layer = [ 16, ]

		return 


	args.neighbourhood_sample_sizes = [5, 5]
	args.batch_size = 100


	if dataset in ["AstroPh", "CondMat", "HepPh", "GrQc", "wordnet", "wordnet_attributed"]:

		args.number_of_capsules_per_layer = [16, 1]
		args.capsule_dim_per_layer = [16, args.embedding_dim]

	elif dataset in ["citeseer", "cora", "pubmed", "reddit"]:

		args.use_labels = True

		if dataset == "citeseer":
			num_classes = 6
		elif dataset == "cora":
			num_classes = 7
		elif dataset == "pubmed":
			num_classes = 3
		else:
			num_classes = 41

		args.number_of_capsules_per_layer = [num_classes, 1]
		args.capsule_dim_per_layer = [16, 32]

def configure_paths(args):
	'''
	build directories on local system for output of model after each epoch
	'''

	dataset = args.dataset
	directory = "neighbourhood_sample_sizes={}_num_primary_caps={}_primary_cap_dim={}_num_caps={}_caps_dim={}_embedding_dims={}".format(args.neighbourhood_sample_sizes, 
				args.num_primary_caps, args.primary_cap_dim, 
				args.number_of_capsules_per_layer, args.capsule_dim_per_layer, 
				args.embedding_dims)
	if args.no_embedding_loss:
		directory = "no_embedding_loss_" + directory
	elif args.no_intermediary_loss:
		directory = "no_intermediary_loss_" + directory

	if args.use_labels:
		directory = "use_labels_" + directory

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

	args.board_path = os.path.join(args.board_path, dataset)
	if not os.path.exists(args.board_path):
		os.makedirs(args.board_path)
	args.board_path = os.path.join(args.board_path, directory)
	if not os.path.exists(args.board_path):
		os.makedirs(args.board_path)
	

	# args.log_path = os.path.join(args.log_path, directory)
	# print args.log_path, args.logger_path
	# raise SystemExit


	args.walk_path = os.path.join(args.walk_path, dataset)
	if not os.path.exists(args.walk_path):
		os.makedirs(args.walk_path)

	args.model_path = os.path.join(args.model_path, dataset)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	args.model_path = os.path.join(args.model_path, directory)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

def record_initial_losses(model, validation_callback, training_gen, args):

	'''
	record the loss of model in its untrained state -- with purely random weights
	'''
	
	print ("recording losses before training begins")

	initial_losses = validation_callback.evaluate(None)

	print ("evaluating model on one step of training generator")
	evaluations = model.evaluate_generator(training_gen, steps=1)
	if not isinstance(evaluations, list):
		evaluations = [evaluations]
	for metric, loss in zip(model.metrics_names, evaluations):
		print (metric, loss)
		initial_losses.update({metric : loss})

	loss_df = pd.DataFrame(initial_losses, index=Index(["initial"], name="epoch"))
	loss_df.to_csv(args.log_path)

	print ("COMPLETED RECORDING OF INITIAL LOSSES")

def main():
	'''
	main function
	'''

	args = parse_args()
	args.num_positive_samples = 1

	# fix args for evaluation purposes
	fix_parameters(args)

	assert len(args.neighbourhood_sample_sizes) ==\
	len(args.number_of_capsules_per_layer) == len(args.capsule_dim_per_layer),\
	"lengths of all input lists must be the same"

	dataset = args.dataset

	configure_paths(args)

	# load the dataset -- written for many types of exeriments so some returned objects are None
	G_train_neighbours, G_val_neighbours, G_test_neighbours,\
	X_train, X_val, X_test, Y, positive_samples, ground_truth_negative_samples,\
	val_edges, test_edges,\
	train_idx, val_idx, test_idx = load_data(dataset, args.neighbourhood_sample_sizes)
	
	if sp.sparse.issparse(Y):
		Y = Y.A


	neighbourhood_sample_sizes = np.array(args.neighbourhood_sample_sizes[::-1])
	number_of_capsules_per_layer = np.array(args.number_of_capsules_per_layer)
	capsule_dim_per_layer = np.array(args.capsule_dim_per_layer)

	args.neighbourhood_sample_sizes = neighbourhood_sample_sizes
	args.number_of_capsules_per_layer = number_of_capsules_per_layer
	args.capsule_dim_per_layer = capsule_dim_per_layer

	# use labels for labelled networks
	if dataset in ["citeseer", "pubmed", "reddit", ]:#"cora", "karate"]:
		assert Y.shape[1] in args.number_of_capsules_per_layer, "You must have a layer with {} capsules".format(Y.shape[1])
		prediction_layer = np.where(args.number_of_capsules_per_layer==Y.shape[1])[0][-1]
		# print prediction_layer, args.number_of_capsules_per_layer, Y.shape[1]
		# raise SystemExit
		monitor = "margin_loss_layer_{}".format(prediction_layer)
		mode = "min"
		print("using labels in training")
	else:
		embedding_layer = len(args.number_of_capsules_per_layer) - 1
		monitor = "mean_rank_reconstruction_layer_{}".format(embedding_layer)
		mode = "min"

	# the path of the file that contains the random walks for this network
	walk_file = os.path.join(args.walk_path, "walks-{}-{}".format(args.num_walks, args.walk_length))
	
	# will perform random walks if the walk file does not exist
	# uses these walks to build the set of posive and negative samples to train upon
	# positive_samples, ground_truth_negative_samples =\
	# load_positive_samples_and_ground_truth_negative_samples(G_train, args, walk_file,)# positive_samples_filename, negative_samples_filename)

	# use this flag to generate walks and not train the model -- for blue bear purposes (to save GPU requests)
	if args.just_walks:
		print ("Only precomputing walks -- terminating")
		return

	if not args.test_only:

		# train_input, train_output = get_training_sample(X_train, Y, G_train_neighbours, train_idx,
		# 	positive_samples[:len(X_train)], ground_truth_negative_samples, args)
		training_gen = training_generator(X_train, Y, G_train_neighbours, train_idx, 
			positive_samples, ground_truth_negative_samples,
			args)

		# generates / loads a graph caps model according the args passed in from the command line
		# will load a model if an existing model exists on ther system with the same specifications
		model, initial_epoch = load_graphcaps(X_train, Y, args.model_path, args)

		patience = args.patience

		# validation data -- for tensorboard
		val_input, val_output = get_training_sample(X_val, Y, G_val_neighbours, val_idx,
			positive_samples[:len(X_val)], ground_truth_negative_samples, args)

		# callbacks
		nan_terminate_callback = TerminateOnNaN()
		validation_callback = ValidationCallback(G_val_neighbours, X_val, Y, model,
		positive_samples, val_edges, ground_truth_negative_samples, 
		train_idx, val_idx, args,)
		early_stopping_callback = EarlyStopping(monitor=monitor, patience=patience, mode=mode, verbose=1)
		reload_callback = ModelCheckpoint(os.path.join(args.model_path, 
			"{epoch:04d}.h5"), save_weights_only=False)
		best_model_callback = ModelCheckpoint(os.path.join(args.model_path, "best_model.h5"),
			monitor=monitor, mode=mode, save_weights_only=False, save_best_only=True, verbose=1)

		# print args.log_path
		# raise SystemExit

		tensorboard_callback = TensorBoard(log_dir=args.board_path, 
			histogram_freq=1, write_grads=True, write_images=True, batch_size=X_val.shape[0])
		logger_callback = CSVLogger(args.log_path, append=True)

		callbacks = [
			nan_terminate_callback, 
			validation_callback,
			# early_stopping_callback, 
			reload_callback, 
			best_model_callback, 
			logger_callback,
			tensorboard_callback
		]

		if initial_epoch == 0:
			record_initial_losses(model, validation_callback, training_gen, args)
		raise SystemExit
		
		print ("BEGIN TRAINING")

		# num_steps = int((len(positive_samples) // args.num_walks + args.batch_size - 1) // args.batch_size)
		# num_steps = int((len(positive_samples) + args.batch_size - 1) // args.batch_size)
		# num_steps = args.num_steps
		model.fit_generator(training_gen, 
			steps_per_epoch=args.num_steps,
		# model.fit(train_input, train_output, batch_size=len(X_train),
			epochs=args.num_epochs, 
			initial_epoch=initial_epoch,
			verbose=1,
			validation_data=[val_input, val_output],
			callbacks=callbacks)

		print ("TRAINING COMPLETE")


	print ("TESTING MODEL")
	print ("LOADING BEST MODEL ACCORDING TO: {}".format(monitor))

	model, _ = load_graphcaps(X_test, Y, args.model_path, args, load_best=True)
	testing_callback = ValidationCallback(G_test_neighbours, X_test, Y, model,
		positive_samples, test_edges, ground_truth_negative_samples, 
		train_idx, test_idx, args, )

	testing_results = testing_callback.evaluate(None)
	for k, v in testing_results.items():
		print (k, v)

if __name__  == "__main__":
	main()