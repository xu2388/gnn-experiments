"""
PyTorch Reproduction of GCN (Graph Convolutional Network)
Paper: Semi-Supervised Classification with Graph Convolutional Networks
Author: Thomas N. Kipf & Max Welling

This is a clean PyTorch re-implementation using the Cora dataset.
"""

import numpy as np
import pickle as pkl
import sys
import time
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
import networkx as nx


# ===================== Data Loading (from original GCN code) =====================

def parse_index_file(filename: str):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_cora_data(path: str = "gcn/gcn/data") -> Tuple:
    """
    Load Cora dataset from pickle files.
    Returns: adj, features, labels, idx_train, idx_val, idx_test
    """
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


def normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
    """Symmetrically normalize adjacency matrix: D^{-1/2} A D^{-1/2}"""
    adj = adj + sp.eye(adj.shape[0])  # add self-loops
    d = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def normalize_features(features: sp.spmatrix) -> np.ndarray:
    """Row-normalize feature matrix."""
    rowsum = np.array(features.sum(1)).flatten()
    r_inv = np.power(rowsum, -1)
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv @ features
    return features.toarray()


def sparse_matrix_to_torch_sparse(sparse_matrix: sp.spmatrix) -> torch.sparse.FloatTensor:
    """Convert a scipy sparse matrix to a PyTorch sparse tensor."""
    sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_matrix.row, sparse_matrix.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_matrix.data)
    shape = torch.Size(sparse_matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ============================= GCN Model Definition =============================

class GraphConvolution(nn.Module):
    """Basic Graph Convolution layer: H' = ReLU(Ã @ H @ W)"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor):
        # x: [N, in_features], adj: [N, N] sparse
        support = x @ self.weight                     # [N, out_features]
        output = torch.sparse.mm(adj, support)        # Ã @ (X @ W)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCN(nn.Module):
    """
    2-layer GCN as described in the paper.
    Architecture: GCN(input) -> ReLU -> Dropout -> GCN(hidden) -> LogSoftmax
    """

    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 16, dropout: float = 0.5):
        super().__init__()
        self.gc1 = GraphConvolution(n_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.sparse.FloatTensor):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


# ================================== Training ====================================

def train():
    parser = argparse.ArgumentParser(description='PyTorch GCN on Cora')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss)')
    parser.add_argument('--hidden', type=int, default=16, help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--early_stopping', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()

    # Device setup
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
    adj_norm = normalize_adj(adj)
    features_dense = normalize_features(features)
    n_nodes = adj.shape[0]
    n_features = features_dense.shape[1]
    n_classes = labels.shape[1]

    # Convert to PyTorch tensors
    adj_tensor = sparse_matrix_to_torch_sparse(adj_norm).coalesce().to(device)
    features_tensor = torch.FloatTensor(features_dense).to(device)
    labels_tensor = torch.LongTensor(labels.argmax(axis=1)).to(device)

    # Masks
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    # Model, Optimizer
    model = GCN(n_features, n_classes, args.hidden, args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nTraining on {len(idx_train)} nodes, validating on {len(idx_val)}, testing on {len(idx_test)}")
    print(f"{'='*60}")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    t_total = time.time()

    for epoch in range(1, args.epochs + 1):
        t = time.time()

        # Forward
        model.train()
        output = model(features_tensor, adj_tensor)
        loss_train = F.nll_loss(output[idx_train], labels_tensor[idx_train])

        # Accuracy
        acc_train = (output[idx_train].argmax(dim=1) == labels_tensor[idx_train]).float().mean()

        # Backward
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            output = model(features_tensor, adj_tensor)
            loss_val = F.nll_loss(output[idx_val], labels_tensor[idx_val])
            acc_val = (output[idx_val].argmax(dim=1) == labels_tensor[idx_val]).float().mean()

        # Print
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
            if patience_counter >= args.early_stopping:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"{'='*60}")
    print(f"Total training time: {time.time()-t_total:.2f}s")

    # Test
    model.eval()
    with torch.no_grad():
        output = model(features_tensor, adj_tensor)
        loss_test = F.nll_loss(output[idx_test], labels_tensor[idx_test])
        acc_test = (output[idx_test].argmax(dim=1) == labels_tensor[idx_test]).float().mean()

    print(f"\nTest results:")
    print(f"  Test loss: {loss_test.item():.5f}")
    print(f"  Test accuracy: {acc_test.item():.4f} ({acc_test.item()*100:.2f}%)")
    print(f"\nExpected accuracy on Cora: ~81.5%")


if __name__ == '__main__':
    train()
