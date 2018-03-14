import time

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

import networkx as nx
# from networkx.readwrite import json_graph

import pickle as pkl

import os
import json

def load_reddit_data():
    
    G = nx.read_edgelist("../data/reddit/reddit_G.edg", delimiter=" ", data=True)
    print "loaded G"
    nx.set_edge_attributes(G=G, name="weight", values=1)
    print "set weights"

    feats = np.load("../data/reddit/reddit-feats.npy")
    id_map = json.load(open("../data/reddit/reddit-id_map.json"))
    conversion = lambda n : n
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    
    G = nx.relabel_nodes(G, id_map, )
    print "relabeled nodes"

    with open("../data/reddit/train_nodes", "rb") as f:
        train_nodes = pkl.load(f)
    with open("../data/reddit/val_nodes", "rb") as f:
        val_nodes = pkl.load(f)
    with open("../data/reddit/test_nodes", "rb") as f:
        test_nodes = pkl.load(f)
        
    train_idx = [id_map[n] for n in train_nodes]
    val_idx = [id_map[n] for n in val_nodes]
    test_idx = [id_map[n] for n in test_nodes]
    
    # normalize by training data
    train_feats = feats[train_idx]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    
    X = feats
    print "scaled features"
    
    class_map = json.load(open("../data/reddit/reddit-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)
    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}
    
    num_classes = max(class_map.values())
    class_map = {id_map[k]: v for k, v in class_map.items()}
    
    Y = np.zeros((len(class_map), num_classes))
    for n, c in class_map.items():
        Y[n, c] = 1

    print "build Y"
        
    train_G = G.subgraph(train_idx)
    val_G = G.subgraph(train_idx + val_idx)
    test_G = G
    print "build train/val/test networks "

    return train_G, val_G, test_G, X, Y, train_idx, val_idx, test_idx 


def main():

	train_G, val_G, test_G, X, Y, train_idx, val_idx, test_idx  = load_reddit_data()


if __name__ == "__main__":
	main()