"""
PyTorch Reproduction of GAT (Graph Attention Network)
Paper: Graph Attention Networks
Authors: Petar Veličković, Guillem Cucurull, Arantxa Casanova, et al.

Key idea: Each node attends over its neighbors, learning different importance
weights for different neighbors via a self-attention mechanism.
"""

import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import networkx as nx
import pickle as pkl
import sys


# ===================== Data Loading (same as GCN) =====================

def parse_index_file(filename: str):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_cora_data(path: str = "gcn/gcn/data"):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(f"{path}/ind.cora.{names[i]}", 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"{path}/ind.cora.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_features(features: sp.spmatrix) -> np.ndarray:
    """Row-normalize feature matrix."""
    rowsum = np.array(features.sum(1)).flatten()
    r_inv = np.power(rowsum, -1)
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv @ features
    return features.toarray()


# ========================= GAT Model Definition =========================

class GraphAttentionLayer(nn.Module):
    """
    Single GAT attention head.
    h_i' = σ(∑_{j ∈ N_i} α_ij W h_j)
    where α_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.6,
                 alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha  # negative slope for LeakyReLU
        self.concat = concat  # whether to apply ELU

        # Learnable parameters
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, h: torch.Tensor, adj: torch.sparse.FloatTensor):
        # h: [N, in_features]
        N = h.size(0)

        # Linear transformation: Wh
        Wh = h @ self.W  # [N, out_features]

        # Compute attention coefficients
        # e_ij = a^T [Wh_i || Wh_j] for all i,j
        Wh_i = Wh.unsqueeze(1).repeat(1, N, 1)   # [N, N, out_features]
        Wh_j = Wh.unsqueeze(0).repeat(N, 1, 1)   # [N, N, out_features]
        e = torch.cat([Wh_i, Wh_j], dim=-1)       # [N, N, 2*out_features]
        e = e @ self.a                             # [N, N, 1]
        e = e.squeeze(-1)                          # [N, N]

        # LeakyReLU
        e = F.leaky_relu(e, self.alpha)

        # Mask: only attend over neighbors (including self)
        # adj is sparse, convert to dense mask for element-wise operation
        adj_dense = adj.to_dense()  # [N, N], 1 where edge exists
        zero_vec = -9e15 * torch.ones_like(e)
        e = torch.where(adj_dense > 0, e, zero_vec)

        # Softmax over neighbors
        attention = F.softmax(e, dim=1)  # [N, N]

        # Dropout on attention coefficients
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Weighted sum of neighbor features
        h_prime = attention @ Wh  # [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (f"{self.__class__.__name__} ({self.in_features} -> "
                f"{self.out_features}, heads=1)")


class GAT(nn.Module):
    """
    2-layer GAT with multi-head attention.
    Architecture:
        Layer 1: K heads, each producing hid_dim features → concatenated
        Layer 2: 1 head, producing n_classes features → averaged
    """

    def __init__(self, n_features: int, n_classes: int,
                 hidden_dim: int = 8, n_heads: int = 8,
                 dropout: float = 0.6, alpha: float = 0.2):
        super().__init__()

        # Layer 1: n_heads independent attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(n_features, hidden_dim, dropout, alpha, concat=True)
            for _ in range(n_heads)
        ])

        # Layer 2: single attention head (output layer)
        self.out_attn = GraphAttentionLayer(
            hidden_dim * n_heads, n_classes,
            dropout=dropout, alpha=alpha, concat=False
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor):
        # Layer 1: concatenate all heads
        x = F.dropout(x, self.dropout.p, training=self.training)
        x = torch.cat([attn(x, adj) for attn in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout.p, training=self.training)

        # Layer 2: single head
        x = self.out_attn(x, adj)

        return F.log_softmax(x, dim=1)


# ========================== Training ==========================

def train():
    parser = argparse.ArgumentParser(description='PyTorch GAT on Cora')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--hidden', type=int, default=8, help='Hidden units per head')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='LeakyReLU negative slope')
    parser.add_argument('--epochs', type=int, default=500, help='Max epochs')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()

    # Device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print(f"Using device: {device}")
    print("Loading Cora dataset...")

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_cora_data()

    # Preprocess
    features_dense = normalize_features(features)
    n_nodes = adj.shape[0]
    n_features = features_dense.shape[1]
    n_classes = labels.shape[1]

    # Adjacency matrix with self-loops
    adj_selfloop = adj + sp.eye(n_nodes)

    # Convert to PyTorch sparse tensor (GAT needs dense mask anyway)
    # We'll work with dense adjacency for simplicity (Cora is only 2708x2708)
    adj_dense = torch.FloatTensor(adj_selfloop.toarray()).to(device)

    features_tensor = torch.FloatTensor(features_dense).to(device)
    labels_tensor = torch.LongTensor(labels.argmax(axis=1)).to(device)

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    # Model & optimizer
    model = GAT(
        n_features, n_classes,
        hidden_dim=args.hidden,
        n_heads=args.n_heads,
        dropout=args.dropout,
        alpha=args.alpha
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print(f"Model: GAT({n_features} -> {args.hidden}x{args.n_heads} heads -> {n_classes})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nTraining on {len(idx_train)} nodes, val on {len(idx_val)}, test on {len(idx_test)}")
    print(f"{'='*60}")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    t_total = time.time()

    for epoch in range(1, args.epochs + 1):
        t = time.time()

        model.train()
        output = model(features_tensor, adj_dense)
        loss_train = F.nll_loss(output[idx_train], labels_tensor[idx_train])
        acc_train = (output[idx_train].argmax(dim=1) == labels_tensor[idx_train]).float().mean()

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            output = model(features_tensor, adj_dense)
            loss_val = F.nll_loss(output[idx_val], labels_tensor[idx_val])
            acc_val = (output[idx_val].argmax(dim=1) == labels_tensor[idx_val]).float().mean()

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:04d} | "
                  f"train_loss={loss_train.item():.5f} | train_acc={acc_train.item():.4f} | "
                  f"val_loss={loss_val.item():.5f} | val_acc={acc_val.item():.4f} | "
                  f"time={time.time()-t:.4f}s")

        # Early stopping
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"{'='*60}")
    print(f"Total training time: {time.time()-t_total:.2f}s")

    # Test
    model.eval()
    with torch.no_grad():
        output = model(features_tensor, adj_dense)
        loss_test = F.nll_loss(output[idx_test], labels_tensor[idx_test])
        acc_test = (output[idx_test].argmax(dim=1) == labels_tensor[idx_test]).float().mean()

    print(f"\nTest results:")
    print(f"  Test loss: {loss_test.item():.5f}")
    print(f"  Test accuracy: {acc_test.item():.4f} ({acc_test.item()*100:.2f}%)")
    print(f"\nExpected accuracy on Cora: ~83.0-84.0% (GAT paper reports ~83.5%)")


if __name__ == '__main__':
    train()
