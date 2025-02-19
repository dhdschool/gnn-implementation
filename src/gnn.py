import torch
from karateclub import GraphReader, LabelPropagation, Diff2Vec
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from time import time
import random

def gpu_check():
    print(f"GPU is avaliable: {torch.cuda.is_available()}")
    print(f"GPU device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"GPU device name: {torch.cuda.get_device_name(current_device)}")
        torch.set_default_device('cuda')
        print("Device set to GPU")
    else:
        print("Device default to CPU") 

def render_graph(graph, target):
    color_dct = {
        0:'red',
        1:'yellow',
        2:'blue',
        3:'green'
    }
    
    nodes = graph.nodes.keys()
    edges = graph.edges()
    
    nodes = [int(i) for i in nodes]
    colors = [color_dct[target[i]] for i in nodes]
    labels = [str(i) for i in nodes]
    
    net = Network(notebook=False,
                  bgcolor='#222222',
                  font_color = "white",
                  height='1000px',
                  width='1000px')
    
    net.add_nodes(nodes, color=colors, label=labels)
    net.add_edges(edges)
    
    net.force_atlas_2based()
    net.show('karateclub.html', notebook=False, local=False)
    
def render_graph_sample(graph, target, k=2):        
    sampler = KSampler(graph, k=k)
    sample = sampler.sample_as_graph()
    
    target = [target[i] for i in sample.nodes]
    render_graph(sample, target)
    
    
    
def format_kc(graph: nx.Graph):
    map = {}
    
    nodes = [i for i in range(len(graph.nodes))]
    for index, node in enumerate(graph.nodes.keys()):
        map[node] = index
    
    edges = []
    for edge in graph.edges:
        mapped_edge = (map[edge[0]], map[edge[1]])
        edges.append(mapped_edge)
    
    mapped_graph = nx.Graph()
    mapped_graph.add_nodes_from(nodes)
    mapped_graph.add_edges_from(edges)
    return mapped_graph, map

   
    
    

class KSampler:
    def __init__(self, graph: nx.Graph, target, k=2):
        self.graph = graph
        self.target = target
        self.k = k
        
    def sample(self, start_node, k=None):
        if k is None:
            k = self.k
        
        nodes = {start_node}
        depth = 0
        
        while depth < k:
            nodes_current = nodes.copy()
            for node in nodes_current:
                nodes = nodes.union(self.graph.neighbors(node))
                                
            depth += 1
        
        nodes = list(nodes)
        edges = self.generate_edges(nodes)
        return nodes, edges

    def generate_edges(self, nodes):
        edges = []
        n = len(nodes)
        if n <= 1:
            return
        
        l = 0
        r = 1
        while l < n - 1:
            edge_data = self.graph.get_edge_data(nodes[l], nodes[r])
            if edge_data is not None:
                edges.append((nodes[l], nodes[r]))
            
            r += 1
            if r >= n:
                l += 1
                r = l + 1
        
        return edges
    
    def sample_as_graph(self, k=None, kc=False):
        if k is None:
            k = self.k
            
        start_node = random.choice(list(self.graph.nodes))
        nodes, edges = self.sample(start_node, k)
        sample_graph = nx.Graph()
        
        sample_graph.add_nodes_from(nodes)
        sample_graph.add_edges_from(edges)
        sample_target = [self.target[i] for i in sample_graph.nodes.keys()]
        
        if kc:
            sample_graph, map = format_kc(sample_graph)
            return sample_graph, sample_target, map
            
        return sample_graph, sample_target

if __name__ == '__main__':
    gpu_check()
    
    reader = GraphReader('facebook')
    graph = reader.get_graph()
    target = reader.get_target()
    
    sampler = KSampler(graph, target, k=2)
    sample_graph, sample_target, sample_mapper = sampler.sample_as_graph(kc=True)
    
    test_nodes = set(sample_mapper.keys())
    train_nodes = set(graph.nodes()).difference(test_nodes)
    
    print()
    print(sample_graph)
    
    model = Diff2Vec(diffusion_number=2, diffusion_cover=20, dimensions=16)
    model.fit(graph)
    
    X = np.array(model.get_embedding())
    y = np.array(target)
    
    X_train, X_test = X[train_nodes], X[test_nodes]
    y_train, y_test = y[train_nodes], y[test_nodes]

    
    #render_graph(sample_graph, sample_target)