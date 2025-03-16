from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset, DataLoader
import random
from model import BinaryEncoder, GCNClassifer
from Render import render_graph
import os

EDGE_PATH = Path("data/edges.csv")
FEATURES_PATH = Path("data/features.csv")
TARGET_PATH = Path("data/target.csv")



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

class GraphReader:
    def __init__(self, *args, **kwargs):
        self.edge_df = pd.read_csv(EDGE_PATH)
        self.target_df = pd.read_csv(TARGET_PATH)
        self.features_df = pd.read_csv(FEATURES_PATH)
        
        self.graph = nx.graph.Graph()
        self.graph.add_nodes_from(self.target_df.index)
        self.graph.add_edges_from(zip(self.edge_df.T.loc['id_1'],
                                      self.edge_df.T.loc['id_2']))   

        self.target = self.target_df['target'].to_numpy(dtype=np.uint8)
        
        row = self.features_df["node_id"]
        col = self.features_df["feature_id"]
        values = self.features_df["value"]
        
        self.features = coo_matrix((values, 
                                    (row, col)), 
                            shape=(row.max() + 1, col.max() + 1)).toarray()
        
    def get_graph(self):
        return self.graph
    
    def get_target(self):
        return self.target
    
    def get_features(self):
        return self.features

class GraphDataset(Dataset):
    def __init__(self, graph: nx.Graph, target, features, mapper):
        self.graph = graph # G(V, E)
        self.target = target # t
        self.features = features # S
        self.mapper = mapper
        
        self.features = torch.tensor(self.features).float().cuda()
        self.adjacency = torch.tensor(nx.adjacency_matrix(self.graph).toarray()).float().to_sparse().cuda()
        self.adjacency = self.adjacency + torch.eye(self.adjacency.size(0)).to_sparse().cuda()
        
        self.target = torch.tensor(self.target).float().cuda()
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, key):
        return self.features[key], self.target[key]
        

class SBS:
    def __init__(self, reader: GraphReader):
        self.graph = reader.get_graph()
        self.target = reader.get_target()
        self.features = reader.get_features()
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
    
    
    def sample_compliment(self, format_karateclub = False):
        vertices = self.V
        self.V = set()
        
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

        return GraphDataset(graph, target, features, map)

def train_autoencoder(feature_size:int, embedding_dims=128, epochs=100):
    reader = GraphReader()
        
    dataset = GraphDataset(reader.get_graph(),
                           reader.get_target(),
                           reader.get_features(),
                           {})
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator('cuda')
    )
    
    feature_embedder = BinaryEncoder(feature_size, embedding_dims)
    feature_optim = torch.optim.Adam(feature_embedder.parameters(), lr=1e-3)
    feature_loss = torch.nn.BCELoss()
    feature_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=feature_optim, mode='min')

    feature_embedder.train()
    
    print(f"Training autoencoder to embed {feature_size} binary variables into a {embedding_dims} dimensional vector...")
    for epoch in range(epochs):
        epoch_loss=0
        
        for x, _, _ in dataloader:
            feature_optim.zero_grad()
            
            pred = feature_embedder(x)
            loss = feature_loss(pred, x)
            loss.backward()
            
            feature_optim.step()
            epoch_loss += loss.item()
            
        feature_scheduler.step(epoch_loss / len(dataloader))
        print(f"Embedding loss at step {epoch + 1}: {epoch_loss / len(dataloader)}")
    
    torch.save(feature_embedder.state_dict(), 'model/feature_embedder.pth')
    print("Training complete! Model saved to feature_embedder.pth.")
    
    feature_embedder.eval()
    return feature_embedder

def train_gcn(x,
              adj,
              y,
              embed_dim,
              epochs=150):
    
    feature_size = x.size(1)
    
    if os.path.exists('model/feature_embedder.pth'):
        encoder = BinaryEncoder(feature_size, embed_dim)
        encoder.load()
    else:
        encoder = train_autoencoder(feature_size, embed_dim)
        
    model = GCNClassifer(encoder,
                         embedding_dimension=embed_dim,
                         gcn_hidden_features=256,
                         gcn_out_features=128,
                         out_features=1)
        
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min')
    
    for epoch in range(epochs):
        optim.zero_grad()
            
        pred = model(x, adj)
        loss = loss_fn(pred, y)
        loss.backward()
            
        optim.step()
        epoch_loss = loss.item()
        
        scheduler.step(epoch_loss)
        print(f"Classification loss at step {epoch + 1}: {epoch_loss}")
    
    torch.save(model.state_dict(), 'model/gcn.pth')
    return model
    
if __name__ == '__main__':
    random.seed(0)
    torch.set_default_device('cuda')
    reader = GraphReader()
    
    sampler = SBS(reader)
    testdataset = sampler.sample_as_dataset(t=4, k=8, k0=1)
    traindataset = sampler.sample_compliment()

    print(f"Training graph is {traindataset.graph}")
    print(f"Testing graph is {testdataset.graph}")
    
    embed_dim = 128
    
    x = traindataset.features
    adj = traindataset.adjacency
    y = traindataset.target.unsqueeze(1)
    
    model = train_gcn(x, adj, y, embed_dim, epochs=1)
    model.eval()
    
    test_predict = model(testdataset.features, testdataset.adjacency)
    y_hat = torch.round(test_predict).float()
    
    y_diff = (y_hat == testdataset.target.unsqueeze(1)).squeeze().float().detach().tolist()
    print(y_diff)
    
    print(f'Test Accuracy: {sum(y_diff) / len(y_diff)}')
    render_graph(graph=testdataset.graph, target=y_diff)

    
    
    
