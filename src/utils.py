import os
# import gzip
import random
import numpy as np
# import networkx as nx
from node2vec_sampling import Graph 

def load_positive_samples_and_ground_truth_negative_samples(G, args, walk_file,):# positive_samples_filename, negative_samples_filename):

	print ("generating positive and negative samples")
	walks = load_walks(G, walk_file, args)
	check_walks(G, walks)
	positive_samples, ground_truth_negative_samples =\
	determine_positive_and_groud_truth_negative_samples(G, walks, args.context_size)

	return positive_samples, ground_truth_negative_samples

def check_walks(G, walks):
	neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
	for walk in walks:
		for u, v in zip(walk, walk[1:]):
			assert v in neighbours[u], "{} is not the neighbour of {}".format(u, v)

	print ("CHECKED WALKS")
	# raise SystemExit


def determine_positive_and_groud_truth_negative_samples(G, walks, context_size):

	print ("determining positive and negative samples")
	
	# N = len(G)
	nodes = set(G.nodes())
	# neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
	
	all_positive_samples = {n: set() for n in G.nodes()}
	positive_samples = []
	for num_walk, walk in enumerate(walks):
		for i in range(len(walk)):
			for j in range(i+1, min(len(walk), i+1+context_size)):
				u = walk[i]
				v = walk[j]
				if u == v:
					continue
				# print u, v
				# print v, u

				positive_samples.append((u, v))
				positive_samples.append((v, u))
				# assert u in neighbours[v], "{} is not a neighbour of {}".format(u, v)
				
				all_positive_samples[u].add(v)
				all_positive_samples[v].add(u)
 
		if num_walk % 1000 == 0:  
			print ("processed walk {}/{}".format(num_walk, len(walks)))
			
	ground_truth_negative_samples = {n: sorted(list(nodes.difference(all_positive_samples[n]))) for n in G.nodes()}
	# print positive_samples
	# print 
	# print ground_truth_negative_samples 
	for u in ground_truth_negative_samples:
		# print u, len(ground_truth_negative_samples[u])
		assert len(ground_truth_negative_samples[u]) > 0, "node {} does not have any negative samples".format(u)
	# raise SystemExit
	print ("DETERMINED POSITIVE AND NEGATIVE SAMPLES")
	print ("found {} positive sample pairs".format(len(positive_samples)))
	# raise SystemExit
	
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
		print ("saved walks to {}".format(walk_file))


		walks2 = load_walks_from_file(walk_file, args.walk_length)
		for walk1, walk2 in zip(walks, walks2):
			for u, v in zip(walk1, walk2):
				assert u==v, "v does not equal v"
		# raise SystemExit
	else:
		print ("loading walks from {}".format(walk_file))
		# with open(walk_file, "rb") as f:
		# 	walks = pkl.load(f)
		walks = load_walks_from_file(walk_file, args.walk_length)
	return walks
