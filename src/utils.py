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

	def save_positive_samples(positive_samples, positive_samples_filename):

		with open(positive_samples_filename, "w") as f:
			for positive_sample in positive_samples:
				f.write("{} {} ".format(positive_sample[0], positive_sample[1]))

	def load_positive_samples(positive_samples_filename):

		with open(positive_samples_filename, "r") as f:
			l = f.readline().rstrip()
			split = [int(n) for n in l.split(" ")]
			positive_samples = zip(split[::2], split[1::2]) 
		return positive_samples

	def save_negative_samples(negative_samples, negative_samples_filename):

		with open(negative_samples_filename, "w") as f:
			for k in negative_samples:
				f.write("{} ".format(k) + " ".join(str(v) for v in negative_samples[k]) + "\n")

	def load_negative_samples(negative_samples_filename):

		negative_samples = {}
		with open(negative_samples_filename, "r") as f:
			for l in f.readlines():
				split = l.split(" ")
				negative_samples.update({int(split[0]) : [int(n) for n in split[1:]]})
		return negative_samples

	
	if os.path.exists(positive_samples_filename):

		print "loading positive and negative samples from file"

		# with open(positive_samples_filename, "rb") as f:
		# 	positive_samples = pkl.load(f)
		# with open(negative_samples_filename, "rb") as f:
		# 	ground_truth_negative_samples = pkl.load(f)
		positive_samples = load_positive_samples(positive_samples_filename)
		ground_truth_negative_samples = load_negative_samples(negative_samples_filename) 

	else:

		print "generating positive and negative samples"
		walks = load_walks(G, walk_file, args)
		positive_samples, ground_truth_negative_samples =\
		determine_positive_and_groud_truth_negative_samples(G, walks, args.context_size)

		print "saving positive and negative samples to file"
		# with open(positive_samples_filename, "wb") as f:
		# 	pkl.dump(positive_samples, f)
		# with open(negative_samples_filename, "wb") as f:
		# 	pkl.dump(ground_truth_negative_samples, f)
		save_positive_samples(positive_samples, positive_samples_filename)
		save_negative_samples(ground_truth_negative_samples, negative_samples_filename)


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

	def save_walks_to_file(walks, walk_file):
		with open(walk_file, "w") as f:
			for walk in walks:
				for n in walk:
					f.write("{} ".format(n))

	def load_walks_from_file(walk_file, walk_length):

		with open(walk_file, "r") as f:
			l = f.readline().rstrip()
			l = [int(n) for n in l.split(" ")]
			walks = [l[i:i+walk_length] for i in range(0, len(l), walk_length)]
		return walks


	if not os.path.exists(walk_file):
		node2vec_graph = Graph(nx_G=G, is_directed=False, p=args.p, q=args.q)
		node2vec_graph.preprocess_transition_probs()
		walks = node2vec_graph.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length)
		# with open(walk_file, "wb") as f:
		# 	pkl.dump(walks, f)
		save_walks_to_file(walks, walk_file)
		print "saved walks to {}".format(walk_file)
	else:
		print "loading walks from {}".format(walk_file)
		# with open(walk_file, "rb") as f:
		# 	walks = pkl.load(f)
		walks = load_walks_from_file(walk_file, args.walk_length)
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





