import numpy as np

from keras.callbacks import TerminateOnNaN

import argparse

from utils import load_karate, load_cora, neighbourhood_sample_generator, draw_embedding
from models import build_graphcaps


def parse_args():
	parser = argparse.ArgumentParser(description="GraphCaps for feature learning of complex networks")

	parser.add_argument("-s", dest="sample_sizes", type=int, nargs="+",
		help="number of neighbourhood node samples for each layer REQUIRED", required=True)

	
	parser.add_argument("-b", dest="batch_size", type=int, default=50, help="batch size for training (default is 50).")
	parser.add_argument("-npos", dest="num_pos", type=int, default=1, help="number of positive samples for training (default is 1).")
	parser.add_argument("-nneg", dest="num_neg", type=int, default=5, help="number of negative samples for training (default is 5).")

	parser.add_argument("-d", dest="embedding_dim", type=int, default=2, help="embedding dimension of hyperbolic space (default is 2).")


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

	sample_sizes = np.array(args.sample_sizes)

	embedding_dim = args.embedding_dim

	generator = neighbourhood_sample_generator(G, X, Y,
		sample_sizes, num_positive_samples, num_negative_samples, batch_size)

	# x, [y, m1, m2] = generator.next()

	# print x.shape, y.shape, m1.shape, m2.shape
	# print m2

	# raise SystemExit


	# output_shapes = [[32, 8], [num_classes, 4], [1, embedding_dim]]

	model, embedder = build_graphcaps(data_dim, num_classes, embedding_dim,
		num_positive_samples, num_negative_samples, sample_sizes)

	model.summary()
	# raise SystemExit

	# model.fit_generator(generator, steps_per_epoch=1, epochs=1000, verbose=1, callbacks=[TerminateOnNaN()])

	# draw_embedding(embedder, generator, dim=embedding_dim)

if __name__  == "__main__":
	main()