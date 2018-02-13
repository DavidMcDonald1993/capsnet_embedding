import numpy as np

from keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, TensorBoard

import argparse

from models import build_graphcaps, generate_graphcaps_model
from generators import neighbourhood_sample_generator
from utils import load_karate, load_cora, load_facebook, preprocess_data, remove_edges, split_data, perform_embedding, evaluate_link_prediction, plot_embedding, make_and_evaluate_label_predictions, PlotCallback


def parse_args():
	parser = argparse.ArgumentParser(description="GraphCaps for feature learning of complex networks")

	parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=1000,
		help="The number of epochs to train for (default is 1000).")
	parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=100, 
		help="Batch size for training (default is 100).")
	# parser.add_argument("--npos", dest="num_pos", type=int, default=1, 
	# 	help="Number of positive samples for training (default is 1).")
	parser.add_argument("--nneg", dest="num_neg", type=int, default=5, 
		help="Number of negative samples for training (default is 5).")
	parser.add_argument("--context_size", dest="context_size", type=int, default=5,
		help="Context size for generating positive samples (default is 5).")

	parser.add_argument("-s", "--sample_sizes", dest="neighbourhood_sample_sizes", type=int, nargs="+",
		help="Number of neighbourhood node samples for each layer separated by a space (default is [5,5,5]).", default=[5,5,5])
	parser.add_argument("-f", "--num_filters", dest="num_filters_per_layer", type=int, nargs="+",
		help="Number of capsules for each layer separated by space (default is [16, 16, 8]).", default=[16, 16, 8])
	parser.add_argument("-a", "--agg_dim", dest="agg_dim_per_layer", type=int, nargs="+",
		help="Dimension of agg output for each layer separated by a space (default is [128, 32, 16]).", default=[128,32,16])
	parser.add_argument("-n", "--num_caps", dest="num_capsules_per_layer", type=int, nargs="+",
		help="Number of capsules for each layer separated by space (default is [16, 7, 1]).", default=[16, 7, 1])
	parser.add_argument("-d", "--capsule_dim", dest="capsule_dim_per_layer", type=int, nargs="+",
		help="Dimension of capule output for each layer separated by a space (default is [8, 4, 2]).", default=[8,4,2])


	parser.add_argument("-p", dest="p", type=float, default=1.,
		help="node2vec return parameter (default is 1.).")
	parser.add_argument("-q", dest="q", type=float, default=1.,
		help="node2vec in-out parameter (default is 1.).")
	parser.add_argument('--num-walks', dest="num_walks", type=int, default=10, 
		help="Number of walks per source (default is 10).")
	parser.add_argument('--walk_length', dest="walk_length", type=int, default=15, 
		help="Length of random walk from source (default is 15).")

	

	parser.add_argument("--plot", dest="plot_path", default="../plots/", 
		help="path to save plot (default is '../plots/.'")


	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	G, X, Y, label_map = load_cora()

	X = preprocess_data(X)
	number_of_edges_to_remove = int(len(G.edges())*0.0)
	G, removed_edges = remove_edges(G, number_of_edges_to_remove=number_of_edges_to_remove)

	(X_train, Y_train, G_train), (X_val, Y_val, G_val) = split_data(G, X, Y, split=0.3)
	print X_train.shape, Y_train.shape, len(G_train)
	print X_val.shape, Y_val.shape, len(G_val)

	data_dim = X.shape[1]
	num_classes = Y.shape[1]

	batch_size = args.batch_size
	num_positive_samples = 1
	num_negative_samples = args.num_neg

	neighbourhood_sample_sizes = np.array(args.neighbourhood_sample_sizes[::-1])
	num_filters_per_layer = np.array(args.num_filters_per_layer)
	agg_dim_per_layer = np.array(args.agg_dim_per_layer)
	num_capsules_per_layer = np.array(args.num_capsules_per_layer)
	capsule_dim_per_layer = np.array(args.capsule_dim_per_layer)

	assert len(neighbourhood_sample_sizes) == len(num_filters_per_layer) == len(agg_dim_per_layer) == \
	len(num_capsules_per_layer) == len(capsule_dim_per_layer), "lengths of all input lists must be the same"


	context_size = args.context_size
	p = args.p
	q = args.q
	num_walks = args.num_walks
	walk_length = args.walk_length

	# generator = neighbourhood_sample_generator(G, X, Y,
	# 	neighbourhood_sample_sizes, num_capsules_per_layer, 
	# 	num_positive_samples, num_negative_samples, context_size, batch_size,
	# 	p, q, num_walks, walk_length, num_samples_per_class=None)

	training_generator = neighbourhood_sample_generator(G_train, X_train, Y_train,
		neighbourhood_sample_sizes, num_capsules_per_layer, 
		num_positive_samples, num_negative_samples, context_size, batch_size,
		p, q, num_walks, walk_length, num_samples_per_class=None)
	validation_generator = neighbourhood_sample_generator(G_val, X_val, Y_val,
		neighbourhood_sample_sizes, num_capsules_per_layer, 
		num_positive_samples, num_negative_samples, context_size, batch_size,
		p, q, num_walks, walk_length, num_samples_per_class=None)

	capsnet, embedder, label_prediction_model = generate_graphcaps_model(X, Y, batch_size, 
	# capsnet = generate_graphcaps_model(X, Y, batch_size, 
		num_positive_samples, num_negative_samples,
		neighbourhood_sample_sizes, num_filters_per_layer, agg_dim_per_layer,
		num_capsules_per_layer, capsule_dim_per_layer)

	print "GRAPHCAPS SUMMARY"
	capsnet.summary()
	print "EMBEDDER SUMMARY"
	embedder.summary()
	if label_prediction_model is not None:
		print "LABEL PREDICTOR SUMMARY"
		label_prediction_model.summary()

	plot_callback = PlotCallback(G, X, Y, neighbourhood_sample_sizes, embedder, label_map, annotate=False, path=args.plot_path)

	capsnet.fit_generator(training_generator, 
		# steps_per_epoch=float(len(G))*context_size/args.batch_size, 
		steps_per_epoch=100,
		epochs=args.num_epochs, 
		validation_data=validation_generator, validation_steps=1,
		verbose=1, callbacks=[plot_callback, TerminateOnNaN(), 
		EarlyStopping(monitor="loss", patience=100),
		ModelCheckpoint("../models/{epoch:04d}-{loss:.2f}.h5", monitor="loss"), ])
		# TensorBoard(log_dir="../logs", batch_size=batch_size)])

	if label_prediction_model is not None:
		make_and_evaluate_label_predictions(G, X, Y, label_prediction_model, num_capsules_per_layer, 
			neighbourhood_sample_sizes, batch_size)

	embedding = perform_embedding(G, X, neighbourhood_sample_sizes, embedder)

	if number_of_edges_to_remove > 0:
		precisions, recalls, f1_scores = evaluate_link_prediction(G, embedding, removed_edges)

	# plot_embedding(embedding, Y, label_map, annotate=False, 
	# 	path=args.plot_path+)

if __name__  == "__main__":
	main()