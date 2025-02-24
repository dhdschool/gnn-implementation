import torch
from karateclub import GraphReader, LabelPropagation, Diff2Vec
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from time import time
import random
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

def render_graph(graph, target, name='karateclub'):
    color_dct = {
        0:'red',
        1:'green',
        2:'blue',
        3:'yellow'
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
    net.show(f'{name}.html', notebook=False, local=False)
    
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

def graph_split(graph : nx.Graph, target, mapper: dict, feature_size: int=16):
    test_nodes = np.array(list(mapper.keys())).astype(np.int32)
    train_nodes = np.array(list(set(graph.nodes()).difference(set(mapper.keys())))).astype(np.int32)
    
    label_count = len(pd.unique(target))
    y = np.zeros(shape=(target.shape[0], label_count))
    for index, label in enumerate(target):
        y[index, label] = 1
    
    embedder = Diff2Vec(diffusion_number=4, diffusion_cover=30, dimensions=feature_size)
    embedder.fit(graph)
    
    X = np.array(embedder.get_embedding())
    
    X_train, X_test = X[train_nodes], X[test_nodes]
    y_train, y_test = y[train_nodes, :], y[test_nodes, :]
    
    return X_train, y_train, X_test, y_test    

def train(model: nn.Module, dataloader: DataLoader):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    while True:
        for X, y in dataloader:
            optimizer.zero_grad()
            
            pred = model(X)
            loss = loss_fn(pred, y)
            
            loss.backward()
            optimizer.step()
        yield loss
          
class EmbeddedDataset(Dataset):
    def __init__(self, X, y):        
        self.features = X
        self.labels = y
        
        self.features = torch.Tensor(self.features).float().cuda()
        self.labels = torch.Tensor(self.labels).float().cuda()
        
    def __getitem__(self, key):
        return self.features[key], self.labels[key]
        
    def __len__(self):
        return len(self.features)



if __name__ == '__main__':
    gpu_check()
    
    embedding_features = 64
    epochs = 1
    
    reader = GraphReader('facebook')
    graph = reader.get_graph()
    target = reader.get_target()
    
    sampler = KSampler(graph, target, k=2)
    sample_graph, sample_target, sample_mapper = sampler.sample_as_graph(kc=True)
    
    print()
    print(f"Sample graph is a {sample_graph}")
    print()
    
    print("Embedding graph and preparing dataset, hold on this can take a while...")
    X_train, y_train, X_test, y_test = graph_split(graph, target, sample_mapper, feature_size=embedding_features)
    
    train_dataset = EmbeddedDataset(X_train, y_train)
    test_dataset = EmbeddedDataset(X_test, y_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=torch.Generator(device='cuda'))

    print("Dataset done loading!\n")

    model = nn.Sequential(
        nn.Linear(embedding_features, 32),
        nn.Softmax(dim=1),
        nn.Linear(32, 4),
        nn.Softmax(dim=1)
    ) 
    
    print("Training model...")
    trainer = train(model, train_dataloader)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        last_loss = next(trainer)
        print(f"Loss at epoch {epoch+1}: {last_loss}")
    print()
    
    X_test = test_dataset.features
    
    y_pred = np.argmax(model(X_test).cpu().detach(), axis=1)
    y_test = np.argmax(test_dataset.labels.cpu().detach(), axis=1)
    
    y_diff = np.array(y_pred == y_test).astype(np.int32)
    
    print("Evaluating model accuracy on sample...")
    f1 = f1_score(y_test, y_pred, average='micro')
    print(f"Model has an f1 score of: {f1}")
    print(f"Model has a test accuracy of {sum(y_diff) / len(y_diff)}\n")
    
    print("Rendering sample by label...")
    render_graph(sample_graph, sample_target, name="by_label")
    
    print()
    print("Rendering sample by prediction...")
    render_graph(sample_graph, y_diff, name="by_pred")