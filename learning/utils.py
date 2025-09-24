#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:04:12 2022

@author: aglabassi
"""

import torch
import torch_geometric
import torch.nn.functional as F
from tqdm import tqdm

def normalize_graph(constraint_features, 
                    edge_index,
                    edge_attr,
                    variable_features,
                    bounds,
                    depth,
                    bound_normalizor = 1000):
    
    
    #SMART
    obj_norm = torch.max(torch.abs(variable_features[:,2]), axis=0)[0].item()
    var_max_bounds = torch.max(torch.abs(variable_features[:,:2]), axis=1, keepdim=True)[0]  
    
    var_max_bounds.add_(var_max_bounds == 0)
    
    var_normalizor = var_max_bounds[edge_index[0]]
    cons_normalizor = constraint_features[edge_index[1], 0:1]
    normalizor = var_normalizor/(cons_normalizor + (cons_normalizor == 0))
    
    variable_features[:,2].div_(obj_norm)
    variable_features[:,:2].div_(var_max_bounds)
    constraint_features[:,0].div_(constraint_features[:,0] + (constraint_features[:,0] == 0) )
    edge_attr.mul_(normalizor)
    bounds.div_(bound_normalizor)
        
    
    
    #cheap 

    
    # #normalize objective
    # #obj_norm = torch.max(torch.abs(variable_features[:,2]), axis=0)[0].item()
    # #var_max_bounds = torch.max(torch.abs(variable_features[:,:2]), axis=1, keepdim=True)[0]  
    
    # #var_max_bounds.add_(var_max_bounds == 0)
    
    # #var_normalizor = var_max_bounds[edge_index[0]]
    # #cons_normalizor = constraint_features[edge_index[1]]
    # #normalizor = var_normalizor/(cons_normalizor)

    # variable_features[:,2].div_(100)
    # variable_features[:,:2].div_(300)
    # constraint_features.div_(300)
    # #edge_attr.mul_(normalizor)
    # bounds.div_(bound_normalizor)
    
    return (constraint_features, edge_index, edge_attr, variable_features, bounds, depth)



#function definition
# https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
def process(policy, data_loader, loss_fct, device, optimizer=None, normalize=True, lamda1= 1, lamda2 =1):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0
    n_samples_processed = 0

    with torch.set_grad_enabled(optimizer is not None):
        for idx,batch in enumerate(data_loader):
        #for idx, batch in enumerate(tqdm(data_loader, desc="Training:")):
        #for batch in tqdm(data_loader, desc="Training:"):  
            
            batch = batch.to(device)
            if normalize:
                #IN place operations
                (batch.constraint_features_s,
                 batch.edge_index_s, 
                 batch.edge_attr_s,
                 batch.variable_features_s,
                 batch.bounds_s,
                 batch.depth_s)  =  normalize_graph(batch.constraint_features_s,  batch.edge_index_s, batch.edge_attr_s,
                                                    batch.variable_features_s, batch.bounds_s,  batch.depth_s)
                
                (batch.constraint_features_t,
                 batch.edge_index_t, 
                 batch.edge_attr_t,
                 batch.variable_features_t,
                 batch.bounds_t,
                 batch.depth_t)  =  normalize_graph(batch.constraint_features_t,  batch.edge_index_t, batch.edge_attr_t,
                                                    batch.variable_features_t, batch.bounds_t,  batch.depth_t)
                
                (batch.constraint_features_root,
                 batch.edge_index_root, 
                 batch.edge_attr_root,
                 batch.variable_features_root,
                 batch.bounds_root,
                 batch.depth_root)  =  normalize_graph(batch.constraint_features_root,  batch.edge_index_root, batch.edge_attr_root,
                                                    batch.variable_features_root, batch.bounds_root,  batch.depth_root)
                                                    
        
            y_true = 0.5*batch.y + 0.5 #0,1 label from -1,1 label
            variable_avg1,variable_avg2,variable_avgroot = policy(batch)

            # infoNCE loss
            y_pred, loss, sim0, sim1 = infonce_loss(variable_avg1, variable_avg2, variable_avgroot, y_true, device)

            loss_value = loss.item() 
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            accuracy = (y_pred == y_true).float().mean().item()


            mean_loss += loss_value * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= (n_samples_processed + ( n_samples_processed == 0))
    mean_acc /= (n_samples_processed  + ( n_samples_processed == 0))

    return mean_loss, mean_acc

def select_opt_sample(y_true, predict_optsol1, predict_optsol2, index):
    predict_optsol = torch.zeros_like(predict_optsol1)
    for i in range(y_true.shape[0]):
        mask = (index == i)
        predict_optsol[mask] = torch.where(y_true[i]==0, predict_optsol1[mask],predict_optsol2[mask])

    return predict_optsol

def cal_variable_scores(y_true, variable_s, variable_t,variable_root,predict_optsol_root, index):
    score_s  = torch.zeros_like(index)
    score_t  = torch.zeros_like(index)

    score_0  = torch.zeros_like(y_true)
    score_1  = torch.zeros_like(y_true)
    for i in range(variable_root.shape[0]):
        #lb = 1
        if variable_s[0]!= variable_root[0]:
            score_s = predict_optsol_root[i]
        #up = 0 
        elif variable_s[1]!= variable_root[1]:
            score_s = 1 - predict_optsol_root[i]
        
        #lb = 1
        if variable_t[0]!= variable_root[0]:
            score_t = predict_optsol_root[i]
        #up = 0 
        elif variable_t[1]!= variable_root[1]:
            score_t = 1 - predict_optsol_root[i]


    for i in range(y_true.shape[0]):
        mask = (index == i)
        score_0[i] = torch.sum(score_s[mask])
        score_1[i] = torch.sum(score_t[mask])

    return score_0, score_1


def infonce_loss(embd0, embd1, embd_root, y_true, device):
    # 计算余弦相似度
    cos_sim_0 = F.cosine_similarity(embd_root, embd0)  # embd0 与 embd_true 的相似度
    cos_sim_1 = F.cosine_similarity(embd_root, embd1)  # embd1 与 embd_true 的相似度
    
    y_pred = torch.where(cos_sim_0 > cos_sim_1, torch.tensor(0).to(device), torch.tensor(1).to(device))

    # 根据 y_true 选择正样本的相似度
    sim_pos = torch.where(y_true == 0, cos_sim_0, cos_sim_1)  # 正样本相似度
    sim_neg = torch.where(y_true == 0, cos_sim_1, cos_sim_0)  # 负样本相似度

    # InfoNCE 损失计算
    tau = 0.07  # 温度系数
    loss = -torch.log(
        torch.exp(sim_pos / tau) / (torch.exp(sim_pos / tau) + torch.exp(sim_neg / tau))
    )

    loss = loss.mean() 

    return y_pred, loss , cos_sim_0, cos_sim_1

def process_ranknet(policy, X, y, loss_fct, device, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0
    n_samples_processed = 0
    X.to(device)

    with torch.set_grad_enabled(optimizer is not None):
        for idx,x in enumerate(X):
            yi = y[idx].to(device)
            y_true = 0.5*yi + 0.5 #0,1 label from -1,1 label
            y_proba = policy(x[:20].to(device), x[20:].to(device))
            y_pred = torch.round(y_proba)
            
            # Compute the usual cross-entropy classification loss
            #loss_fct.weight = torch.exp((1+torch.abs(batch.depth_s - batch.depth_t)) / 
                            #(torch.min(torch.vstack((batch.depth_s,  batch.depth_t)), axis=0)[0]))
            #print(y_proba)
            l = loss_fct(y_proba, y_true)
            #print(l)
            loss_value = l.item()
            if optimizer is not None:
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            
            accuracy = (y_pred == y_true).float().mean().item()

            mean_loss += loss_value
            mean_acc += accuracy 
            n_samples_processed += 1
            #print(y_proba.item(), y_true.item())

    mean_loss /= (n_samples_processed + ( n_samples_processed == 0))
    mean_acc /= (n_samples_processed  + ( n_samples_processed == 0))
    return mean_loss, mean_acc

