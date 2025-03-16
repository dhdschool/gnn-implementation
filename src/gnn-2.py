# Dataset: 
# Problem
# Embedding
# Predicting


import torch
from karateclub import GraphReader, MUSAE
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from time import time
import random
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def render_graph(graph, 
                 target, 
                 name='karateclub', 
                 color_dct={0:'red',1:'green'}):
    
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
    print()

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


class GraphDataset(Dataset):
    def __init__(self, graph: nx.Graph, target, features, mapper):
        self.graph = graph # G(V, E)
        self.target = target # t
        self.features = features # S
        self.features_sparse = coo_matrix(features) # S
        self.mapper = mapper
        
        self.adjacency = torch.tensor(nx.adjacency_matrix(self.graph).toarray()).float().cuda()
        self.target = torch.tensor(self.target).long().cuda()
        
        embedder = MUSAE()
        embedder.fit(self.graph, self.features_sparse)
        
        self.embedding = embedder.get_embedding()
        self.embedding = torch.tensor(self.embedding).float().cuda()
        
        
    def __len__(self):
        return len(self.embedding)
    
    def __getitem__(self, key):
        return self.embedding[key], self.adjacency[key], self.target[key]
        

class SBS:
    def __init__(self, reader: GraphReader):
        self.graph = reader.get_graph()
        self.target = reader.get_target()
        self.features = reader.get_features().toarray()
        self.V = set(list(self.graph.nodes()))
        
    def sample(self, t=4, k=4, k0=2):
        V0 = set(random.sample(list(self.V), k0))
        sampled_vertices = set()
        sampled_vertices.update(V0)
        for stage in range(t):
            V_i = set()
            for v in V0:
                neighbors = set(self.graph.neighbors(v))
                sampled = 0
                while sampled < k and len(neighbors) > 0:
                    V_i.add(neighbors.pop())
                    sampled += 1
            
            V0 = V_i
            sampled_vertices.update(V_i)
            
        self.V = self.V.difference(sampled_vertices)
        return sampled_vertices

    def generate_edges(self, V):
        vertices = list(V)
        edges = []
        n = len(vertices)
        if n <= 1:
            return
        
        l = 0
        r = 1
        while l < n - 1:
            edge_data = self.graph.get_edge_data(vertices[l], vertices[r])
            if edge_data is not None:
                edges.append((vertices[l], vertices[r]))
            
            r += 1
            if r >= n:
                l += 1
                r = l + 1
        
        return edges
    
    def sample_as_graph(self, t=4, k=4, k0=2, format_karateclub=False):
        vertices = self.sample(t, k, k0)
        edges = self.generate_edges(vertices)
        v_arr = np.array(list(vertices)).astype(np.int32)
        
        features = self.features[v_arr, :]
        target = np.array([self.target[i] for i in vertices])
        
        graph = nx.Graph()
        
        graph.add_nodes_from(vertices)
        graph.add_edges_from(edges)
        
        if format_karateclub:
            graph, map = format_kc(graph)
        else:
            map = {v:v for v in vertices}        

        return graph, target, features, map
    
    def sample_as_dataset(self, **kwargs):
        return GraphDataset(*self.sample_as_graph(**kwargs))

class EGCL(nn.Module):
    def __init__(self, embedding_dims,
                 in_channels, out_channels,
                 stride=1, kernel_size=1, padding=0):
        super(EGCL, self).__init__()
        self.weight = nn.Parameter(torch.empty(embedding_dims, embedding_dims))
        nn.init.xavier_uniform_(self.weight)
        
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              stride=stride,
                              kernel_size=kernel_size,
                              padding=padding)
        
    def forward(self, embeddings, normalized_adj):
        pooled = torch.mm(normalized_adj, embeddings)
        embedding_transform = nn.functional.relu(torch.mm(pooled, self.weight))
        
        
        return embedding_transform

class EGCN(nn.Module):
    def __init__(self, embedding_dims, gcl_count=2):
        super(EGCN, self).__init__()
        self.gcls = nn.ModuleList([EGCL(embedding_dims) for _ in range(gcl_count)])
        
    def forward(self, embeddings, adjacency):
        neighbor_count = torch.sum(adjacency, dim=1)
        neightbor_mat = torch.diag(neighbor_count)
        mean_denom = torch.inverse(neightbor_mat)
        
        normalized_adj = torch.mm(mean_denom, adjacency)
        for egcl in self.gcls:
            embeddings = egcl(embeddings, normalized_adj)
        return embeddings

class EGCN_Linear(nn.Module):
    def __init__(self, in_dimension, out_dimension, embedding_dimensions, gcl_count=2):
        self.egcns = nn.ModuleList([EGCN(embedding_dims=embedding_dimensions, gcl_count=gcl_count) for i in range(in_dimension)])
        self.weight = nn.Parameter(torch.empty(in_dimension, out_dimension))
        nn.init.normal_(self.weight)
    
    def forward(self, embeddings, adjacency):
        for index, mat in enumerate(embeddings):
            self.egcns[index](mat)
            
        
                
gpu_check()

reader = GraphReader('wikipedia')
sampler = SBS(reader) 
print("Sampling and embedding train data...")
train_data = sampler.sample_as_dataset(t=4, k=4, k0=4, format_karateclub=True)
print("Train data done!\n")
print("Sampling and embedding test data...")
test_data = sampler.sample_as_dataset(t=2, k=4, k0=2, format_karateclub=True)
print("Test data done!\n")

def train(dataset: GraphDataset, model: nn.Module):    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    while True:
        X, adj = dataset.embedding, dataset.adjacency
        y = dataset.target
        optimizer.zero_grad()
        
        pred = model(X, adj)        
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        yield loss

 
print(f"Train data is {train_data.graph}")
print(f"Test data is {test_data.graph}\n")

embedding_size = train_data.embedding.size(1)

epochs = 50

model = EGCN(embedding_dims=embedding_size, gcl_count=3)
trainer = train(dataset=train_data, model=model)  


print("Training model...")
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    last_loss = next(trainer)
    print(f"Loss at epoch {epoch+1}: {last_loss}")
print()


X_test_e = test_data.embedding
X_test_adj = test_data.adjacency
y_test = test_data.target.cpu().detach().numpy()

y_pred = np.argmax(model(X_test_e, X_test_adj).cpu().detach(), axis=1)
y_diff = np.array(y_pred == y_test).astype(np.int32)

print(f"\nModel has a test accuracy of {sum(y_diff) / len(y_diff)}\n")


render_graph(test_data.graph, y_test, name="by_label", color_dct={
    0:'blue',
    1:'yellow'
})

render_graph(test_data.graph, y_diff, name="by_pred")

