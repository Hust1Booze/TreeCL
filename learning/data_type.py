#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:08:53 2022

@author: aglabassi
"""

import torch
import torch_geometric

class BipartiteGraphPairData(torch_geometric.data.Data):
    """
    This class encode a pair of node bipartite graphs observation, s is graph0, t is graph1 
    """
    def __init__(self, constraint_features_s=None, edge_indices_s=None, edge_features_s=None, variable_features_s=None, bounds_s=None, depth_s=None, 
                 constraint_features_t=None, edge_indices_t=None, edge_features_t=None, variable_features_t=None,  bounds_t=None, depth_t=None,
                 y=None): 
        
        super().__init__()
        
        self.variable_features_s, self.constraint_features_s, self.edge_index_s, self.edge_attr_s, self.bounds_s, self.depth_s =  (
            variable_features_s, constraint_features_s, edge_indices_s, edge_features_s, bounds_s, depth_s)
        
        self.variable_features_t, self.constraint_features_t, self.edge_index_t, self.edge_attr_t, self.bounds_t, self.depth_t  = (
            variable_features_t, constraint_features_t, edge_indices_t, edge_features_t, bounds_t, depth_t)
        
        self.y = y
        

   
    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index_s':
            return torch.tensor([[self.variable_features_s.size(0)], [self.constraint_features_s.size(0)]])
        elif key == 'edge_index_t':
            return torch.tensor([[self.variable_features_t.size(0)], [self.constraint_features_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

class BipartiteGraphPairDataWithRoot(torch_geometric.data.Data):
    """
    This class encode a pair of node bipartite graphs observation, s is graph0, t is graph1 
    """
    def __init__(self, constraint_features_s=None, edge_indices_s=None, edge_features_s=None, variable_features_s=None, bounds_s=None, depth_s=None, 
                 constraint_features_t=None, edge_indices_t=None, edge_features_t=None, variable_features_t=None,  bounds_t=None, depth_t=None,
                 constraint_features_root=None, edge_indices_root=None, edge_features_root=None, variable_features_root=None,  bounds_root=None, depth_root=None,
                 y=None, optsol = None): 
        
        super().__init__()
        
        self.variable_features_s, self.constraint_features_s, self.edge_index_s, self.edge_attr_s, self.bounds_s, self.depth_s =  (
            variable_features_s, constraint_features_s, edge_indices_s, edge_features_s, bounds_s, depth_s)
        
        self.variable_features_t, self.constraint_features_t, self.edge_index_t, self.edge_attr_t, self.bounds_t, self.depth_t  = (
            variable_features_t, constraint_features_t, edge_indices_t, edge_features_t, bounds_t, depth_t)
        
        self.variable_features_root, self.constraint_features_root, self.edge_index_root, self.edge_attr_root, self.bounds_root, self.depth_root  = (
            variable_features_root, constraint_features_root, edge_indices_root, edge_features_root, bounds_root, depth_root)
        
        self.y = y
        self.optsol = optsol

   
    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index_s':
            return torch.tensor([[self.variable_features_s.size(0)], [self.constraint_features_s.size(0)]])
        elif key == 'edge_index_t':
            return torch.tensor([[self.variable_features_t.size(0)], [self.constraint_features_t.size(0)]])
        elif key == 'edge_index_root':
            return torch.tensor([[self.variable_features_root.size(0)], [self.constraint_features_root.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)
        


import torch
from torch_geometric.data import Data

class ContrastiveGraphData(torch_geometric.data.Data):
    """
    This class encodes a triplet (root, positive, negatives) for contrastive learning.
    Each instance includes:
      - root: the anchor graph
      - positive: the graph semantically close to the anchor
      - negatives: a list of graphs semantically unrelated to the anchor
    """
    def __init__(self, 
                 constraint_features_root=None, edge_indices_root=None, edge_features_root=None, variable_features_root=None, bounds_root=None, depth_root=None,
                 constraint_features_pos=None, edge_indices_pos=None, edge_features_pos=None, variable_features_pos=None, bounds_pos=None, depth_pos=None,
                 negatives=None,  # List of negative graph data (each negative is a dictionary with similar keys as root/positive)
                 y=None):  # Optional label, if any
        
        super().__init__()
        
        # Root graph features
        self.variable_features_root, self.constraint_features_root, self.edge_index_root, self.edge_attr_root, self.bounds_root, self.depth_root = (
            variable_features_root, constraint_features_root, edge_indices_root, edge_features_root, bounds_root, depth_root)
        
        # Positive graph features
        self.variable_features_pos, self.constraint_features_pos, self.edge_index_pos, self.edge_attr_pos, self.bounds_pos, self.depth_pos = (
            variable_features_pos, constraint_features_pos, edge_indices_pos, edge_features_pos, bounds_pos, depth_pos)
        
        # List of negative graph features
        self.negatives = negatives  # Each entry is a dictionary like: {'variable_features': ..., 'edge_index': ..., etc.}
        
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        """
        Overload the PyTorch Geometric method for incrementing indices when concatenating graphs in a batch.
        This applies to edge indices of root, positive, and negatives.
        """
        if key == 'edge_index_root':
            return torch.tensor([[self.variable_features_root.size(0)], [self.constraint_features_root.size(0)]])
        elif key == 'edge_index_pos':
            return torch.tensor([[self.variable_features_pos.size(0)], [self.constraint_features_pos.size(0)]])
        elif key.startswith('negatives'):
            neg_idx = int(key.split('_')[1])  # Get the index of the negative graph
            negative = self.negatives[neg_idx]
            return torch.tensor([[negative['variable_features'].size(0)], [negative['constraint_features'].size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, idx):
        data = torch.load(self.sample_files[idx])
        return data
