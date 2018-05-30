import random
import numpy as np
import scipy as sp

def get_training_sample(X, Y, G_neighbours, idx,
	batch_positive_samples, ground_truth_negative_samples, args):
	
	num_layers = len(args.embedding_dims)
	num_classes = Y.shape[1]
	# num_label_prediction_layers = len(np.where(args.number_of_capsules_per_layer==num_classes)[0])

	mask = np.zeros((Y.shape[0], 1))
	if idx is not None:
		mask[idx] = 1
	masked_Y = np.append(mask, Y, axis=-1)
	batch_positive_samples = np.array(batch_positive_samples)	
	batch_negative_samples =\
					np.array([np.random.choice(ground_truth_negative_samples[u], 
						replace=True, size=(args.num_negative_samples,))\
						for u in batch_positive_samples[:,0]], dtype=np.int32)
	batch_nodes = np.append(batch_positive_samples, batch_negative_samples, axis=1)
	return ([X, G_neighbours],
		[batch_nodes] * (2 * num_layers) + [X])
		# [masked_Y] * num_layers + [batch_nodes] * num_layers + [X])
	

def training_generator(X, Y, G_neighbours, train_idx, 
	positive_samples, ground_truth_negative_samples,
	args):

	batch_size = X.shape[0]
	# num_steps = int((len(positive_samples) + batch_size - 1) // batch_size)
	# print num_steps
	num_steps = int(len(positive_samples) / batch_size)
	# train_mask = np.zeros((Y.shape[0], 1))
	# if train_idx is not None:
	# 	train_mask[train_idx] =1
	# masked_Y = np.append(train_mask, Y, axis=-1)

	# num_layers = len(args.embedding_dims)


	# num_classes = Y.shape[1]
	# num_label_prediction_layers = len(np.where(args.number_of_capsules_per_layer==num_classes)[0])

	while True:

		random.shuffle(positive_samples)

		# skip = 0
		for step in range(num_steps):

			batch_positive_samples = positive_samples[step * batch_size : (step + 1) * batch_size]
			yield get_training_sample(X, Y, G_neighbours, train_idx,
				batch_positive_samples, ground_truth_negative_samples, args)



			# batch_positive_samples = np.array(positive_samples[step * batch_size : (step + 1) * batch_size], dtype=np.int32)
			# batch_negative_samples =\
			# 	np.array([np.random.choice(ground_truth_negative_samples[u], replace=True, size=(args.num_negative_samples,))\
			# 		for u in batch_positive_samples[:,0]], dtype=np.int32)
			# batch_nodes = np.append(batch_positive_samples, batch_negative_samples, axis=1)
			# # print [masked_Y]*num_label_prediction_layers + [batch_nodes] * num_layers
			# # raise SystemExit



			# yield ([X, G_neighbours], 
			# [masked_Y]*num_label_prediction_layers + [batch_nodes] * num_layers + [X])

