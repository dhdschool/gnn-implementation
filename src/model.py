import torch
import torch.nn as nn

class BinaryEncoder(nn.Module):
    def __init__(self, features, embedding_dim):
        super(BinaryEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(512, features),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        compressed_x = self.encode(x)
        recon_x = self.decode(compressed_x)
        return recon_x
    
    def load(self):
        self.load_state_dict(torch.load('model/feature_embedder.pth', weights_only=True))
        self.eval()
        
        
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, adj):
        degree = adj.sum(dim=1).to_dense()
        diag_inv_sqrt = torch.pow(degree, -.5)
        diag_inv_sqrt = diag_inv_sqrt.unsqueeze(1)
        diag_inv_sqrt = torch.diag(diag_inv_sqrt.view(-1))
        
        weighted_adj = diag_inv_sqrt @ adj @ diag_inv_sqrt
        
        weighted_x = x @ self.weight
        conv = weighted_adj @ weighted_x
        
        out = conv + self.bias
        return out
        
class GCNClassifer(nn.Module):
    def __init__(self,
                 encoder: BinaryEncoder,
                 embedding_dimension,
                 gcn_hidden_features,
                 gcn_out_features, 
                 out_features):
        super(GCNClassifer, self).__init__()
        self.feature_embedder = encoder        
        self.gcn1 = GCNLayer(embedding_dimension, gcn_hidden_features)
        self.gcn2 = GCNLayer(gcn_hidden_features, gcn_out_features)
        self.classifier = nn.Linear(gcn_out_features, out_features)
        
    def forward(self, feature, adj):
        embed = self.feature_embedder.encode(feature) 
        embed = nn.functional.relu(self.gcn1(embed, adj))
        embed = nn.functional.relu(self.gcn2(embed, adj))
        out = torch.sigmoid(self.classifier(embed)).clamp(1e-10, 1 - 1e-10)
        return out
    
    def load(self):
        self.load_state_dict(torch.load('model/gcn.pth', weights_only=True))
        self.eval()