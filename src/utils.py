import os
import gzip
import random
import numpy as np
import networkx as nx
import scipy as sp
import pandas as pd 

import pickle as pkl

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import average_precision_score, normalized_mutual_info_score, accuracy_score

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.callbacks import Callback

from data_utils import preprocess_data
from node2vec_sampling import Graph 
from metrics import evaluate_link_prediction

def load_positive_samples_and_ground_truth_negative_samples(G, args, 
	walk_file, positive_samples_filename, negative_samples_filename):

	
	if os.path.exists(positive_samples_filename):

		print "loading positive and negative samples from file"

		with open(positive_samples_filename, "rb") as f:
			positive_samples = pkl.load(f)
		with open(negative_samples_filename, "rb") as f:
			ground_truth_negative_samples = pkl.load(f)

	else:

		print "generating positive and negative samples"
		walks = load_walks(G, walk_file, args)
		positive_samples, ground_truth_negative_samples =\
		determine_positive_and_groud_truth_negative_samples(G, walks, args.context_size)

		print "saving positive and negative samples to file"
		with open(positive_samples_filename, "wb") as f:
			pkl.dump(positive_samples, f)
		with open(negative_samples_filename, "wb") as f:
			pkl.dump(ground_truth_negative_samples, f)


	return positive_samples, ground_truth_negative_samples

def determine_positive_and_groud_truth_negative_samples(G, walks, context_size):

	print "determining positive and negative samples"
	
	N = len(G)
	nodes = set(G.nodes())
	
	all_positive_samples = {n: set() for n in G.nodes()}
	positive_samples = []
	for num_walk, walk in enumerate(walks):
		for i in range(len(walk)):
			for j in range(i+1, min(len(walk), i+1+context_size)):
				u = walk[i]
				v = walk[j]

				positive_samples.append((u, v))
				positive_samples.append((v, u))
				
				all_positive_samples[u].add(v)
				all_positive_samples[v].add(u)
 
		if num_walk % 1000 == 0:  
			print "processed walk {}/{}".format(num_walk, len(walks))
			
	ground_truth_negative_samples = {n: sorted(list(nodes.difference(all_positive_samples[n]))) for n in G.nodes()}
	
	return positive_samples, ground_truth_negative_samples

def load_walks(G, walk_file, args):

	if not os.path.exists(walk_file):
		node2vec_graph = Graph(nx_G=G, is_directed=False, p=args.p, q=args.q)
		node2vec_graph.preprocess_transition_probs()
		walks = node2vec_graph.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length)
		with open(walk_file, "wb") as f:
			pkl.dump(walks, f)
		print "saved walks to {}".format(walk_file)
	else:
		print "loading walks from {}".format(walk_file)
		with open(walk_file, "rb") as f:
			walks = pkl.load(f)
	return walks


def create_neighbourhood_sample_list(nodes, neighbourhood_sample_sizes, neighbours):

	neighbourhood_sample_list = [nodes]

	for neighbourhood_sample_size in neighbourhood_sample_sizes[::-1]:

		neighbourhood_sample_list.append(np.array([np.concatenate([np.append(n, 
			np.random.choice(np.append(n, neighbours[n]), 
			replace=True, size=neighbourhood_sample_size)) for n in batch]) for batch in neighbourhood_sample_list[-1]]))

	# flip neighbour list
	neighbourhood_sample_list = neighbourhood_sample_list[::-1]


	return neighbourhood_sample_list





