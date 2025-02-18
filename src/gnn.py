import torch
from karateclub import GraphReader, Diff2Vec
import networkx as nx
import numpy as np
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
    
    
    
    
   
    
   
    
    

class KSampler:
    def __init__(self, graph: nx.graph.Graph, k=2):
        self.graph = graph
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
    
    def sample_as_graph(self, k=None):
        if k is None:
            k = self.k
            
        start_node = random.choice(list(self.graph.nodes))
        nodes, edges = self.sample(start_node, k)
        sample_graph = nx.Graph()
        
        sample_graph.add_nodes_from(nodes)
        sample_graph.add_edges_from(edges)
        return sample_graph

if __name__ == '__main__':
    gpu_check()
    
    reader = GraphReader('facebook')
    graph = reader.get_graph()
    target = reader.get_target()
    
    sampler = KSampler(graph, k=3)
    sample = sampler.sample_as_graph()
    
    print()
    print(sample)
    #render_graph_sample(graph, target)
    
    model = Diff2Vec()
    model.fit(graph)
    
    membership = model.get_memberships()
    print(membership)
