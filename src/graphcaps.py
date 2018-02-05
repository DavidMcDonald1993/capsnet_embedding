import numpy as np

from keras.callbacks import TerminateOnNaN

import argparse

from utils import load_karate, load_cora, neighbourhood_sample_generator, draw_embedding
from models import build_graphcaps, generate_graphcaps_model


def parse_args():
	parser = argparse.ArgumentParser(description="GraphCaps for feature learning of complex networks")

	parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=100,
		help="The number of epochs to train for (default is 1000).")
	parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=50, 
		help="Batch size for training (default is 50).")
	parser.add_argument("--npos", dest="num_pos", type=int, default=1, 
		help="Number of positive samples for training (default is 1).")
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
		help="Dimension of capule output for each layer separated by a space (default is [8, 4, 1]).", default=[8,4,2])


	parser.add_argument("-p", dest="p", type=float, default=1.,
		help="node2vec return parameter (default is 1.).")
	parser.add_argument("-q", dest="q", type=float, default=1.,
		help="node2vec in-out parameter (default is 1.).")
	parser.add_argument('--num-walks', dest="num_walks", type=int, default=10, 
		help="Number of walks per source (default is 10).")
	parser.add_argument('--walk_length', dest="walk_length", type=int, default=15, 
		help="Length of random walk from source (default is 15).")

	

	parser.add_argument("--plot", dest="plot_path", default="../plots/embedding.png", 
		help="path to save plot (default is '../plots/embedding.png.'")


	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	G, X, Y = load_cora()

	data_dim = X.shape[1]
	num_classes = Y.shape[1]

	batch_size = args.batch_size
	num_positive_samples = args.num_pos
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

	generator = neighbourhood_sample_generator(G, X, Y,
		neighbourhood_sample_sizes, num_capsules_per_layer, 
		num_positive_samples, num_negative_samples, context_size, batch_size,
		p, q, num_walks, walk_length)


	embedding_dim = num_capsules_per_layer[-1] * capsule_dim_per_layer[-1]
	# print embedding_dim
	# raise SystemExit

	# capsnet, embedder = build_graphcaps(data_dim, num_classes, embedding_dim,
	# 	num_positive_samples, num_negative_samples, neighbourhood_sample_sizes)
	capsnet, embedder = generate_graphcaps_model(data_dim, num_classes, num_positive_samples, num_negative_samples,
	neighbourhood_sample_sizes, num_filters_per_layer, agg_dim_per_layer,
	num_capsules_per_layer, capsule_dim_per_layer)

	capsnet.summary()
	# raise SystemExit

	capsnet.fit_generator(generator, steps_per_epoch=1, epochs=args.num_epochs, 
		verbose=1, callbacks=[TerminateOnNaN()])

	draw_embedding(embedder, generator, dim=embedding_dim, path=args.plot_path)

if __name__  == "__main__":
	main()