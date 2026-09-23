#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:38:45 2021

@author: abdel
"""


import os
import sys
import torch
import torch_geometric
from pathlib import Path
from model import GNNPolicy
from data_type import GraphDataset
from utils import process
import numpy as np

if __name__ == "__main__":
    problems = ["MIK"]
    problem_idx = 0
    problem = problems[problem_idx] 
    lr = 0.005
    n_epoch = 3
    n_sample = -1
    normalize = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_train = 64
    seed = 0
    lamda1 = 0.1
    lamda2 = 0

    data_type = 4

    loss_fn = torch.nn.BCELoss()
    optimizer_fn = torch.optim.Adam
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-lr':
            lr = float(sys.argv[i + 1])
        if sys.argv[i] == '-n_epoch':
            n_epoch = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_sample':
            n_sample = int(sys.argv[i + 1])
        if sys.argv[i] == '-normalize':
            normalize = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
        if sys.argv[i] == '-batch_train':
            batch_train = int(sys.argv[i + 1])
        if sys.argv[i] == '-seed':
            seed = int(sys.argv[i + 1])
        if sys.argv[i] == '-lamda1':
            lamda1 = float(sys.argv[i + 1])
        if sys.argv[i] == '-lamda2':
            lamda2 = float(sys.argv[i + 1])
        if sys.argv[i] == '-data_type':
            data_type = int(sys.argv[i + 1])
            
  
    # 设置随机种子
    torch.manual_seed(seed)                # 设置 PyTorch 的全局随机种子
    torch.cuda.manual_seed(seed)          # 设置 GPU 上的随机种子
    np.random.seed(seed)                  # 设置 NumPy 的随机种子
    #np.random.seed(999)   

    train_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data/{problem}/train")).glob("*.pt") ][:n_sample]
    
    train_aug_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data/{problem}/train_aug")).glob("*.pt") ][:n_sample]
    
    
    print(f'there are {len(train_files)} train samples and {len(train_aug_files)} train aug samples')


    train_samples = 0 
    aug_samples = 0
    if problem in ['GISP','WPMS']:
        train_samples = 41299
    elif problem in ['FCMCNF']:
        train_samples = 16285
    elif problem in ['COR_LAT','MIK']:
        train_samples = -1

    if data_type == 1:
        print("0.1 origin data")
        train_samples = 4130 #int(train_samples/10)
    elif data_type == 2:
        print("augment data only")
        aug_samples = train_samples
        train_samples = 0
    elif data_type == 3:
        print("0.1 origin data + 0.9 augment data")
        train_samples = int(train_samples/10)
        aug_samples = train_samples * 9
    elif data_type == 4:
        print("origin data only")
    elif data_type == 5:
        print("origin and augment data only")
        aug_samples = train_samples
    else:
        print("wrong data type")

    train_files = train_files[:train_samples] + train_aug_files[:aug_samples]
        

    train_data = GraphDataset(train_files)
    
    
# TO DO : learn something from the data
    train_loader = torch_geometric.loader.DataLoader(train_data, 
                                                     batch_size=batch_train, 
                                                     shuffle=True, 
                                                     follow_batch=['constraint_features_s', 
                                                                   'constraint_features_t',
                                                                   'constraint_features_root',
                                                                   'variable_features_s',
                                                                   'variable_features_t',
                                                                   'variable_features_root'])
    
    policy = GNNPolicy().to(device)
    optimizer = optimizer_fn(policy.parameters(), lr=lr) #ADAM is the best
    
    print("-------------------------")
    print(f"GNN for problem {problem}")
    print(f"Training on:          {len(train_data)} samples")
    print(f"Batch Size Train:     {batch_train}")
    print(f"Learning rate:        {lr} ")
    print(f"Number of epochs:     {n_epoch}")
    print(f"Normalize:            {normalize}")
    print(f"Device:               {device}")
    print(f"Loss fct:             {loss_fn}")
    print(f"Optimizer:            {optimizer_fn}")  
    print(f"Model's Size:         {sum(p.numel() for p in policy.parameters())} parameters ")
    print(f"seed:                 {seed}")
    print(f"lamda1:               {lamda1}")
    print(f"lamda2:               {lamda2}")
    print("-------------------------",flush=True) 
    
    for epoch in range(n_epoch):
        print(f"Epoch {epoch + 1}")
        
        train_loss, train_acc= process(policy, 
                                        train_loader, 
                                        loss_fn,
                                        device,
                                        optimizer=optimizer, 
                                        normalize=normalize,
                                        lamda1 = lamda1,
                                        lamda2 = lamda2)
        print(f"Train loss: {train_loss:0.3f}, accuracy1 {train_acc:0.3f}")

    torch.save(policy.state_dict(), f'policy_{problem}.pkl')
    print(f"Saved final-epoch model to policy_{problem}.pkl", flush=True)


