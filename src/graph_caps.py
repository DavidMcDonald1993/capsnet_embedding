import numpy as np

from keras.callbacks import TerminateOnNaN

from utils import load_karate, load_cora, neighbourhood_sample_generator, draw_embedding
from models import build_graphcaps


def main():

	G, X, Y = load_cora()

	data_dim = X.shape[1]
	num_classes = Y.shape[1]

	batch_size = 50
	num_positive_samples = 1
	num_negative_samples = 5

	sample_sizes = np.array([3, 3, 3])

	embedding_dim = 2

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

	model.fit_generator(generator, steps_per_epoch=1, epochs=1000, verbose=1, callbacks=[TerminateOnNaN()])

	draw_embedding(embedder, generator, dim=embedding_dim)

if __name__  == "__main__":
	main()