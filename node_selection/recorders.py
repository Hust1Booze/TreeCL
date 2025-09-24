#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:16:39 2021

@author: abdel

Contains utilities to save and load comparaison behavioural data


"""

import os
import imp
import torch
import numpy as np
import re
import time
import random
import copy

def load_src(name, fpath):
     return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("data_type", "../learning/data_type.py" )

from data_type import BipartiteGraphPairData,BipartiteGraphPairDataWithRoot


class CompFeaturizerSVM():
    def __init__(self, model,save_dir=None, instance_name=None):
        self.instance_name = instance_name
        self.save_dir = save_dir
        self.m = model
        
    def save_comp(self, model, node1, node2, comp_res, comp_id):
        
        f1,f2 = self.get_features(node1), self.get_features(node2)
        
        
        file_path = os.path.join(self.save_dir, f"{self.instance_name}_{comp_id}.csv")
        file = open(file_path, 'a')
        
        np.savetxt(file, f1, delimiter=',')
        np.savetxt(file, f2, delimiter=',')
        file.write(str(comp_res))
        file.close()
        
        return self
    
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        return self

    def get_features(self, node):
        
        model = self.m
        
        f = []
        feat = node.getHeHeaumeEisnerFeatures(model, model.getDepth()+1 )
        
        
        for k in ['vals', 'depth', 'maxdepth' ]:
            if k == 'vals':
                
                for i in range(1,19):
                    try:
                        f.append(feat[k][i])
                    except:
                        f.append(0)
                    
            else:
                f.append(feat[k])

        return f
    




class CompFeaturizer():
    
    def __init__(self, save_dir=None, instance_name=None):
        self.instance_name = instance_name
        self.save_dir = save_dir
        
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        return self

    
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.LP_feature_recorder = LP_feature_recorder
        return self
        
    
    def save_comp(self, model, node1, node2, comp_res, comp_id):
        
        BipartiteGraphs = self.get_torch_geometric_data(model, node1, node2, comp_res)
        # file_path = os.path.join(self.save_dir, f"{self.instance_name}_{comp_id}.pt")
        # torch.save(torch_geometric_data, file_path, _use_new_zipfile_serialization=False)

        i = 0 
        for BipartiteGraph in BipartiteGraphs:
            file_path = os.path.join(self.save_dir, f"{self.instance_name}_{comp_id}_{i}.pt")
            if i !=0:
                file_path = os.path.join(self.save_dir + '_aug', f"{self.instance_name}_{comp_id}_{i}.pt")
            torch.save(BipartiteGraph, file_path, _use_new_zipfile_serialization=False)
            i = i + 1
        
        return self
    
    def get_torch_geometric_data(self, model, node1, node2, comp_res=0):

        BipartiteGraphs = []
        
        triplet = self.get_triplet_tensors(model, node1, node2, comp_res)
        BipartiteGraphs.append(BipartiteGraphPairDataWithRoot(*triplet[0], *triplet[1], *triplet[2],triplet[3],triplet[4]))
        
        triplets = self.get_triplet_tensors_argument(model, node1, node2, comp_res)


        for triplet in triplets:
            BipartiteGraphs.append(BipartiteGraphPairDataWithRoot(*triplet[0], *triplet[1], *triplet[2],triplet[3],triplet[4]))
        #return BipartiteGraphPairDataWithRoot(*triplet[0], *triplet[1], *triplet[2],triplet[3],triplet[4])
        return BipartiteGraphs
    
    
    
    
    def get_graph_for_inf(self,model, node):
        
        gpu_gpu = time.time()
        
        self.LP_feature_recorder.record_sub_milp_graph(model, node)
        graphidx2graphdata = self.LP_feature_recorder.recorded_light
        all_conss_blocks = self.LP_feature_recorder.all_conss_blocks
        all_conss_blocks_features = self.LP_feature_recorder.all_conss_blocks_features
        
        g_idx = node.getNumber()
        
        var_attributes, cons_block_idxs = graphidx2graphdata[g_idx]
        
    
        g_data = self._get_graph_data(var_attributes, cons_block_idxs, all_conss_blocks, all_conss_blocks_features)
        
        variable_features = g_data[0]
        constraint_features = g_data[1]
        edge_indices = g_data[2]
        edge_features = g_data[3]
        
        lb, ub = node.getLowerbound(), node.getEstimate()
        depth = node.getDepth()
        
        if model.getObjectiveSense() == 'maximize':
            lb,ub = ub,lb
            
        g = (constraint_features,
              edge_indices, 
              edge_features, 
              variable_features, 
              torch.tensor([[lb, -1*ub]], device=self.LP_feature_recorder.device).float(),
              torch.tensor([depth], device=self.LP_feature_recorder.device).float()
              )
        
        gpu_gpu = (time.time() - gpu_gpu)
            
        return gpu_gpu, g
        
        
        
        
        
        
        
        
    
    
    
    def get_triplet_tensors(self, model, node1, node2, comp_res=0):
                
        self.LP_feature_recorder.record_sub_milp_graph(model, node1)
        self.LP_feature_recorder.record_sub_milp_graph(model, node2)

        graphidx2graphdata = self.LP_feature_recorder.recorded_light
        all_conss_blocks = self.LP_feature_recorder.all_conss_blocks
        all_conss_blocks_features = self.LP_feature_recorder.all_conss_blocks_features
        
        g0_idx, g1_idx, comp_res = node1.getNumber(), node2.getNumber(), comp_res

        #root_node = model.getRootNode()
        root_node =  self.LP_feature_recorder.find_deepest_ancestor(node1, node2)
        groot_idx = root_node.getNumber()
        
        # this line for debug info
        #self.LP_feature_recorder.record_compare(g0_idx, g1_idx, comp_res)

        var_attributes0, cons_block_idxs0 = graphidx2graphdata[g0_idx]
        var_attributes1, cons_block_idxs1 = graphidx2graphdata[g1_idx]
        var_attributes_root, cons_block_idxs_root = graphidx2graphdata[groot_idx]
        
        g_data = self._get_graph_pair_data(var_attributes0, 
                                           var_attributes1, 
                                           var_attributes_root,              
                                           cons_block_idxs0, 
                                           cons_block_idxs1, 
                                           cons_block_idxs_root,
                                           all_conss_blocks, 
                                           all_conss_blocks_features, 
                                           comp_res)
        
        bounds0 = [node1.getLowerbound(), node1.getEstimate()]
        bounds1 = [node2.getLowerbound(), node2.getEstimate()]
        bounds_root = [root_node.getLowerbound(), root_node.getEstimate()]
        if model.getObjectiveSense() == 'maximize':
            bounds0[1], bounds0[0] = bounds0
            bounds1[1], bounds1[0] = bounds1
            bounds_root[1], bounds_root[0] = bounds_root
        
        return self._to_triplet_tensors(g_data, node1.getDepth(), node2.getDepth(),root_node.getDepth(), bounds0, bounds1, bounds_root,self.LP_feature_recorder.device)
    

    
    
    def get_triplet_tensors_argument(self, model, node1, node2, comp_res=0):
        
        triplets = []
        graph0s, graph1s = self.LP_feature_recorder.generate_pos_neg_samples(model,node1,node2,comp_res)
        for graph0,graph1 in zip(graph0s, graph1s):

            graphidx2graphdata = self.LP_feature_recorder.recorded_light
            all_conss_blocks = self.LP_feature_recorder.all_conss_blocks
            all_conss_blocks_features = self.LP_feature_recorder.all_conss_blocks_features
            
            g0_idx, g1_idx, comp_res = node1.getNumber(), node2.getNumber(), comp_res

            #root_node = model.getRootNode()
            root_node =  self.LP_feature_recorder.find_deepest_ancestor(node1, node2)
            groot_idx = root_node.getNumber()
            
            # this line for debug info
            #self.LP_feature_recorder.record_compare(g0_idx, g1_idx, comp_res)

            var_attributes0, cons_block_idxs0 = graphidx2graphdata[g0_idx]
            var_attributes1, cons_block_idxs1 = graphidx2graphdata[g1_idx]
            var_attributes_root, cons_block_idxs_root = graphidx2graphdata[groot_idx]
            
            var_attributes0 = graph0.var_attributes
            var_attributes1 = graph1.var_attributes
            g_data = self._get_graph_pair_data(var_attributes0, 
                                            var_attributes1, 
                                            var_attributes_root,              
                                            cons_block_idxs0, 
                                            cons_block_idxs1, 
                                            cons_block_idxs_root,
                                            all_conss_blocks, 
                                            all_conss_blocks_features, 
                                            comp_res)
            
            bounds0 = [node1.getLowerbound(), node1.getEstimate()]
            bounds1 = [node2.getLowerbound(), node2.getEstimate()]
            bounds_root = [root_node.getLowerbound(), root_node.getEstimate()]
            if model.getObjectiveSense() == 'maximize':
                bounds0[1], bounds0[0] = bounds0
                bounds1[1], bounds1[0] = bounds1
                bounds_root[1], bounds_root[0] = bounds_root
        
            triplets.append(self._to_triplet_tensors(g_data, node1.getDepth(), node2.getDepth(),root_node.getDepth(), bounds0, bounds1, bounds_root,self.LP_feature_recorder.device))

        return triplets
  
       
    
    def _get_graph_pair_data(self, var_attributes0, var_attributes1, cons_block_idxs0, cons_block_idxs1, all_conss_blocks, all_conss_blocks_features, comp_res ):
        
        g1 = self._get_graph_data(var_attributes0, cons_block_idxs0, all_conss_blocks, all_conss_blocks_features)
        g2 = self._get_graph_data(var_attributes1, cons_block_idxs1, all_conss_blocks, all_conss_blocks_features)
     
        return list(zip(g1,g2)) + [comp_res]
    def _get_graph_pair_data(self, var_attributes0, var_attributes1, var_attributes_root,cons_block_idxs0, cons_block_idxs1, cons_block_idxs_root,all_conss_blocks, all_conss_blocks_features, comp_res ):
        
        g1 = self._get_graph_data(var_attributes0, cons_block_idxs0, all_conss_blocks, all_conss_blocks_features)
        g2 = self._get_graph_data(var_attributes1, cons_block_idxs1, all_conss_blocks, all_conss_blocks_features)
        g_root = self._get_graph_data(var_attributes_root, cons_block_idxs_root, all_conss_blocks, all_conss_blocks_features)
     
        return list(zip(g1,g2,g_root)) + [comp_res]

    def _get_graph_data(self, var_attributes, cons_block_idxs, all_conss_blocks, all_conss_blocks_features):
        
        
        adjacency_matrixes = map(all_conss_blocks.__getitem__, cons_block_idxs)
        
        cons_attributes_blocks = map(all_conss_blocks_features.__getitem__, cons_block_idxs)
        
        #TO DO ACCELERATE HSTACK VSTACK
        # adjacency_matrix = torch.hstack(tuple(adjacency_matrixes))
        # cons_attributes = torch.vstack(tuple(cons_attributes_blocks))
        adjacency_matrix = tuple(adjacency_matrixes)[0]
        cons_attributes = tuple(cons_attributes_blocks)[0]
        
        edge_idxs = adjacency_matrix._indices()
        edge_features =  adjacency_matrix._values().unsqueeze(1)
            
        
        return var_attributes, cons_attributes, edge_idxs, edge_features
        
    
    def _to_triplet_tensors(self, g_data, depth0, depth1,depth_root, bounds0, bounds1,bounds_root, device ):
        
        variable_features = g_data[0]
        constraint_features = g_data[1]
        edge_indices = g_data[2]
        edge_features = g_data[3]
        y = g_data[4]
        lb0, ub0 = bounds0
        lb1, ub1 = bounds1
        lb_root, ub_root = bounds_root
        
        g1 = (constraint_features[0],
              edge_indices[0], 
              edge_features[0], 
              variable_features[0], 
              torch.tensor([[lb0, -1*ub0]], device=device).float(),
              torch.tensor([depth0], device=device).float()
              )
        g2 = (constraint_features[1], 
              edge_indices[1], 
              edge_features[1], 
              variable_features[1], 
              torch.tensor([[lb1, -1*ub1]], device=device).float(),
              torch.tensor([depth1], device=device).float()
              )
        g_root = (constraint_features[2], 
              edge_indices[2], 
              edge_features[2], 
              variable_features[2], 
              torch.tensor([[lb_root, -1*ub_root]], device=device).float(),
              torch.tensor([depth_root], device=device).float()
              )
        
        
        
        return (g1,g2,g_root,y,self.LP_feature_recorder.var_optsol)
    
        
        
        

    


# params_to_set_false = ["constraints/linear/upgrade/indicator",
#                        "constraints/linear/upgrade/logicor",
#                        "constraints/linear/upgrade/knapsack",
#                        "constraints/linear/upgrade/setppc",
#                        "constraints/linear/upgrade/xor",
#                        "constraints/linear/upgrade/varbound"]
# model = scip.Model()


#Converts a branch and bound node, aka a sub-LP, to a bipartite var/constraint 
#graph representation
#1LP recorder per problem
class LPFeatureRecorder():
    
    def __init__(self, model, device, monitorBranchRule = None,optsol =None, augment_nums = 5):
        
        self.augment_nums = augment_nums
        self.optsol = optsol
        varrs = model.getVars()
        original_conss = model.getConss()
        self.monitorBranchRule = monitorBranchRule
        self.model = model
        
        self.n0 = len(varrs)
        
        self.varrs = varrs
        
        self.original_conss = original_conss
        
        self.recorded = dict()
        self.recorded_light = dict()
        self.all_conss_blocks = []
        self.all_conss_blocks_features = []
        self.obj_adjacency  = None
        
        self.device = device
        self.var_optsol = torch.zeros(self.n0,4, device=device).float()
        
        #INITIALISATION OF A,b,c into a graph
        self.init_time = time.time()
        self.var2idx = dict([ (str_var, idx) for idx, var in enumerate(self.varrs) for str_var in [str(var)]  ])
        self.idx2var = dict([ (idx,str_var) for idx, var in enumerate(self.varrs) for str_var in [str(var)]  ])
        root_graph = self.get_root_graph(model, device='cpu')
        self.init_time = (time.time() - self.init_time)
        
        
        self.init_cpu_gpu_time = time.time()
        root_graph.var_attributes = root_graph.var_attributes.to(device)
        for idx, _ in  enumerate(self.all_conss_blocks_features): #1 single loop
            self.all_conss_blocks[idx] = self.all_conss_blocks[idx].to(device)
            self.all_conss_blocks_features[idx] = self.all_conss_blocks_features[idx].to(device)
        
        self.init_cpu_gpu_time = (time.time() - self.init_cpu_gpu_time)
       
        self.recorded[1] = root_graph
        self.recorded_light[1] = (root_graph.var_attributes, root_graph.cons_block_idxs)

        self.node_parent = {}
        self.node_bound = {}
        
   
    def get_graph(self, model, sub_milp):
        
        sub_milp_number = sub_milp.getNumber()
        if sub_milp_number in self.recorded:
            return self.recorded[ sub_milp_number]
        else:
            self.record_sub_milp_graph(model, sub_milp)
            return self.recorded[ sub_milp_number ]
        
    def find_deepest_ancestor(self, node1, node2):
        depth1 = node1.getDepth()
        depth2 = node2.getDepth()
        while(depth1 >depth2):
            node1 = node1.getParent()
            depth1 -= 1
        while(depth1 <depth2):
            node2 = node2.getParent()
            depth2 -= 1
        
        while(node1.getParent().getNumber() != node2.getParent().getNumber()):
            node1 = node1.getParent()
            node2 = node2.getParent()
        
        idx = node1.getParent().getNumber()

        ancestro = node1.getParent()
        return ancestro
    def record_sub_milp_graph(self, model, sub_milp):
        
        node_lb, node_est = sub_milp.getLowerbound(), sub_milp.getEstimate()
        self.node_bound[sub_milp.getNumber()] = [node_lb,node_est]
        if sub_milp.getNumber() not in self.recorded:
            
            parent = sub_milp.getParent()
            if parent == None: #Root
                graph = self.get_root_graph(model)
                self.node_parent[sub_milp.getNumber()] = -1
                
            else:
                self.node_parent[sub_milp.getNumber()] = parent.getNumber()
                graph = self.get_graph(model, parent).copy()
                #self._add_conss_to_graph(graph, model, sub_milp.getAddedConss())
                self._change_branched_bounds(graph, sub_milp)
                
            #self._add_scip_obj_cons(model, sub_milp, graph)
            self.recorded[sub_milp.getNumber()] = graph
            self.recorded_light[sub_milp.getNumber()] = (graph.var_attributes, 
                                                         graph.cons_block_idxs)
            
    def generate_pos_neg_samples(self, model, node1, node2, res):

        depth1 = node1.getDepth()
        depth2 = node2.getDepth()
        
        number1 = node1.getNumber()
        number2 = node2.getNumber()
        node1_branch_decisions = {}
        node2_branch_decisions = {}
        node1_modified_branch_decisions = []
        node2_modified_branch_decisions = []

        while(depth1 >depth2):
            node1_branch_decisions[node1.getParent().getNumber()] = self._get_branched_info(node1)
            node1 = node1.getParent()
            depth1 -= 1
        while(depth1 <depth2):
            node2_branch_decisions[node2.getParent().getNumber()] = self._get_branched_info(node2)
            node2 = node2.getParent()
            depth2 -= 1
        
        while(node1.getNumber() != node2.getNumber()):
            node1_branch_decisions[node1.getParent().getNumber()] = self._get_branched_info(node1)
            node2_branch_decisions[node2.getParent().getNumber()] = self._get_branched_info(node2)
            node1 = node1.getParent()
            node2 = node2.getParent()

        assert node1.getNumber() == node2.getNumber(), "why not same ancestor"
        ancestor = node1
        assert res in [-1,1], "res not -1/1"
        node1_graphs = []
        node2_graphs = []

        decisions_num1 = len(node1_branch_decisions)
        decisions_num2 = len(node2_branch_decisions)
        for i in range(self.augment_nums):
            ancestor_graph =  self.recorded[ancestor.getNumber()].copy()

            if len(node1_branch_decisions) != 0:
                # modify branch decisions
                pos = res == -1
                node1_modified_branch_decisions = self.modify_branch(node1_branch_decisions, ancestor_graph, pos)

            if len(node2_branch_decisions) != 0:
                # modify branch decisions
                pos = res == 1
                node2_modified_branch_decisions = self.modify_branch(node2_branch_decisions, ancestor_graph, pos)

            node2_modified_branch_decisions[0][0] = node1_modified_branch_decisions[0][0]
            bvars, bbounds, btypes = node1_modified_branch_decisions[0] 
            _bvars, _bbounds, _btypes = node2_modified_branch_decisions[0] 
            _bbounds[0] = copy.deepcopy(bbounds[0])^1
            _btypes[0] = copy.deepcopy(btypes[0])^1

            origin_node1_branch_decisions = list(node1_branch_decisions.values())
            origin_node2_branch_decisions = list(node2_branch_decisions.values())
            origin_node1_branch_decisions[-1] = node1_modified_branch_decisions[0]
            origin_node2_branch_decisions[-1] = node2_modified_branch_decisions[0]

            _node1_graph = ancestor_graph.copy()
            self._change_branched_decision(_node1_graph, node1_modified_branch_decisions)
            _node2_graph = ancestor_graph.copy()
            self._change_branched_decision(_node2_graph, node2_modified_branch_decisions)
            node1_graphs.append(_node1_graph)
            node2_graphs.append(_node2_graph)
            #assert torch.equal(_node1_graph.var_attributes,self.recorded[number1].var_attributes), 'why not equal'
            #assert torch.equal(_node2_graph.var_attributes,self.recorded[number2].var_attributes), 'why not equal'
        return node1_graphs, node2_graphs
                
            
            

    
    
    def show_info(self):
        for node, parent in self.node_parent:
            for _node, _parent in self.node_parent:
                if node == _node:
                    continue
                if parent == _parent:
                    print(f'node:{node},{_node} have same parent {parent}')
                    self.record_compare(node, _node)


    def record_compare(self, node1_idx, node2_idx, comp_res):

        print(f"node{node1_idx} bound{self.node_bound[node1_idx]}, node{node2_idx} bound{self.node_bound[node2_idx]}, parent: node{self.node_parent[node1_idx]},  node{self.node_parent[node2_idx]}, comp_res{comp_res} ,")

        node1 = self.recorded[node1_idx]
        node2 = self.recorded[node2_idx]
        # this function use to show difference between two node record
        var_diff = node1.var_attributes != node2.var_attributes
        cons_diff = node1.cons_block_idxs != node2.cons_block_idxs
        num_var_diff = torch.sum(var_diff)
        #num_cons_diff = torch.sum(cons_diff)

        #if num_var_diff !=0 or num_cons_diff !=0:

        print(f"{num_var_diff} var diff, {cons_diff} cons diff")

        dim0,dim1 = node1.var_attributes.shape
        for row in range(dim0):
            if torch.sum(node1.var_attributes[row] != node2.var_attributes[row]) !=0:
                var_name = self.idx2var[row]
                print(f"{var_name} are diff,in node{node1_idx} {node1.var_attributes[row]}, node{node2_idx} :{node2.var_attributes[row]} ")
                if self.name2var(var_name) is not None:
                    print(f"optsol {var_name} is {self.optsol[self.name2var(var_name)]}, root is {self.recorded[1].var_attributes[row]}\n")

        print('-'*50 +'\n')

    def name2var(self,var_name):
        vars = self.model.getVars()
        for var in vars:
            if var.name == var_name:
                return var
        return None


    def get_root_graph(self, model, device=None):
        
        dev = device if device != None else self.device
        
        graph = BipartiteGraphStatic0(self.n0, dev)
        
        self._add_vars_to_graph(graph, model, dev)
        self._add_conss_to_graph(graph, model, self.original_conss, dev)
        if self.optsol is not None:
            self._vars_to_optsol()
        
        return graph
    
    
    # def _get_obj_adjacency(self, model):
    
    #    if self.obj_adjacency  == None:
    #        var_coeff = { self.var2idx[ str(t[0]) ]:c for (t,c) in model.getObjective().terms.items() if c != 0.0 }
    #        var_idxs = list(var_coeff.keys())
    #        weigths = list(var_coeff.values())
    #        cons_idxs = [0]*len(var_idxs)
           
    #        self.obj_adjacency =  torch.torch.sparse_coo_tensor([var_idxs, cons_idxs], weigths, (self.n0, 1), device=self.device)
    #        self.obj_adjacency = torch.hstack((-1*self.obj_adjacency, self.obj_adjacency))
           
    #    return self.obj_adjacency         
       
    
    # def _add_scip_obj_cons(self, model, sub_milp, graph):
    #     adjacency_matrix = self._get_obj_adjacency(model)
    #     cons_feature = torch.tensor([[ sub_milp.getEstimate() ], [ -sub_milp.getLowerbound() ]], device=self.device).float()
    #     graph.cons_block_idxs.append(len(self.all_conss_blocks_features))
    #     self.all_conss_blocks_features.append(cons_feature)
    #     self.all_conss_blocks.append(adjacency_matrix)
  
    
    def _vars_to_optsol(self):
        for idx, var in enumerate(self.varrs):
            binary, integer, continuous = self._one_hot_type(var)
            self.var_optsol[idx] = torch.tensor([self.optsol[var], binary, integer, continuous ], device=self.device).float()  
                
    def _add_vars_to_graph(self, graph, model, device=None):
        #add vars
        
        dev = device if device != None else self.device
        
        for idx, var in enumerate(self.varrs):
            graph.var_attributes[idx] = self._get_feature_var(model, var, dev)

    
    def _add_conss_to_graph(self, graph, model, conss, device=None):
        
        dev = device if device != None else self.device

        if len(conss) == 0:
            return

        cons_attributes = torch.zeros(len(conss), graph.d1, device=dev).float()
        var_idxs = []
        cons_idxs = []
        weigths = []
        for cons_idx, cons in enumerate(conss):

            cons_attributes[cons_idx] =  self._get_feature_cons(model, cons, dev)
          
            for var, coeff in model.getValsLinear(cons).items():

                if str(var) in self.var2idx:
                    var_idx = self.var2idx[str(var)]
                elif 't_'+str(var) in self.var2idx:
                    var_idx = self.var2idx['t_' + str(var)]
                else:
                    var_idx = self.var2idx[ '_'.join(str(var).split('_')[1:]) ] 
                    
                var_idxs.append(var_idx)
                cons_idxs.append(cons_idx)
                weigths.append(coeff)


        adjacency_matrix =  torch.sparse_coo_tensor([var_idxs, cons_idxs], weigths, (self.n0, len(conss)), device=dev) 
        
        #add idx to graph
        graph.cons_block_idxs.append(len(self.all_conss_blocks_features)) #carreful with parralelization
        #add appropriate structure to self
        self.all_conss_blocks_features.append(cons_attributes)
        self.all_conss_blocks.append(adjacency_matrix)
      

    def _change_branched_bounds(self, graph, sub_milp):
        
        bvars, bbounds, btypes = sub_milp.getParentBranchings()
        
        for bvar, bbound, btype in zip(bvars, bbounds, btypes): 
            
            if str(bvar) in self.var2idx:
                var_idx = self.var2idx[str(bvar)]
            elif 't_'+str(bvar) in self.var2idx:
                var_idx = self.var2idx['t_' + str(bvar)]
            elif '_'.join(str(bvar).split('_')[1:]) in self.var2idx:
                var_idx = self.var2idx[ '_'.join(str(bvar).split('_')[1:]) ] 
            else:
                var_idx = self.var2idx[ '_'.join(str(bvar).split('_')[1:2]) ]  
            
            graph.var_attributes[var_idx, int(btype) ] = bbound

    def _change_branched_decision(self, ancestor_graph, modified_branch_decisions):
        
        for decision in modified_branch_decisions:
            bvars, bbounds, btypes = decision
        
            for bvar, bbound, btype in zip(bvars, bbounds, btypes): 
                
                if str(bvar) in self.var2idx:
                    var_idx = self.var2idx[str(bvar)]
                elif 't_'+str(bvar) in self.var2idx:
                    var_idx = self.var2idx['t_' + str(bvar)]
                elif '_'.join(str(bvar).split('_')[1:]) in self.var2idx:
                    var_idx = self.var2idx[ '_'.join(str(bvar).split('_')[1:]) ] 
                else:
                    var_idx = self.var2idx[ '_'.join(str(bvar).split('_')[1:2]) ]  
                
                ancestor_graph.var_attributes[var_idx, int(btype) ] = bbound
    
    def _get_branched_info(self, node):
        bvars, bbounds, btypes = node.getParentBranchings()
        return bvars, bbounds, btypes
        
    
    def _get_feature_cons(self, model, cons, device=None):
        
        dev = device if device != None else self.device
        
        try:
            
            cons_n = str(cons)
            if re.match('flow', cons_n):
                
                rhs = model.getRhs(cons)
                leq = 0
                eq = 1
                geq = 0
            elif re.match('arc', cons_n):
                rhs = 0
                leq = eq =  1
                geq = 0
                
            else:
                rhs = model.getRhs(cons)
                leq = eq = 1
                geq = 0
        except:
            'logicor no repr'
            rhs = 0
            leq = eq = 1
            geq = 0
        
        
        return torch.tensor([ rhs, leq, eq, geq ], device=dev).float()

    def _get_feature_var(self, model, var, device=None):
        
        dev = device if device != None else self.device
        
        lb, ub = var.getLbOriginal(), var.getUbOriginal()
        
        if lb <= - 0.999e+20:
            lb = -300
        if ub >= 0.999e+20:
            ub = 300
            
        objective_coeff = model.getObjective()[var]
        
        binary, integer, continuous = self._one_hot_type(var)
    
        
        return torch.tensor([ lb, ub, objective_coeff, binary, integer, continuous ], device=dev).float()
    
    
    def _one_hot_type(self, var):
        vtype = var.vtype()
        binary, integer, continuous = 0,0,0
        
        if vtype == 'BINARY':
            binary = 1
        elif vtype == 'INTEGER':
            integer = 1
        elif vtype == 'CONTINUOUS':
            continuous = 1
            
        return binary, integer,  continuous
        

    def branch_once(self, graph, var):
        child1 = graph.copy()
        child2 = graph.copy()
        var_idx = self._get_var_idx(var)

        if (graph.var_attributes[var_idx, int(0) ] == graph.var_attributes[var_idx, int(1) ]) or graph.var_attributes[var_idx, int(0)] == 1 or \
        graph.var_attributes[var_idx, int(1)] == 0 :
            print('somthing not correct !')
            return None,None
        child1.var_attributes[var_idx, int(0) ] = 0
        child1.var_attributes[var_idx, int(1) ] = 0
        child2.var_attributes[var_idx, int(0) ] = 1
        child2.var_attributes[var_idx, int(1) ] = 1

        if self.optsol[var]>0.999:
            return child2, child1
        else:
            return child1, child2

    def sim_branch(self):
        root_graph = self.recorded[1]
        cur_node = root_graph.copy()

        idxs = list(range(0, len(self.model.getVars())))
        random.shuffle(idxs)

        for idx in idxs:
            if self.model.getVars()[idx].vtype() == 'BINARY':
                var = self.model.getVars()[idx]
                child1,child2 = self.branch_once(cur_node, var)
                if child1 != None:
                    self.positive_samples.append(child1)
                    self.negtive_samples.append(child2) 
                    self.parent_samples.append(cur_node.copy())   
                    self.node_idxs.append(idx)
                    cur_node = child1.copy()      

        self.root_graph = root_graph
        self.optsol_graph = cur_node

    def modify_branch(self, branch_decisions, co_parent_graph, pos_neg):

        modified_branch_decisions = []
        i = 0
        for parent_id, branch_decision in reversed(list(branch_decisions.items())):

            bvars, bbounds, btypes = branch_decision

            # jump first branch decision
            # if i ==0:
            #     i = i+1
            #     print(f"bvars: {bvars}, bbounds: {bbounds}, btypes:{btypes}")
            #     continue
            while(True):
                var = random.choice(self.monitorBranchRule.branch_cands[parent_id])

                var_idx = self._get_var_idx(var)
                
                # branched on this var before co_parent, jump
                if co_parent_graph.var_attributes[var_idx, int(0)] ==1 or co_parent_graph.var_attributes[var_idx, int(1)] ==0 :
                    continue
                else:
                    # make sure the modified branch decision make the node contain optsol
                    if pos_neg:
                        if self.optsol[var] >0.99:
                            modified_branch_decisions.append([[var], [0], [1]])
                        else:
                            modified_branch_decisions.append([[var], [1], [0]])
                    # random
                    else:
                        if np.random.rand() < 0.5:
                            modified_branch_decisions.append([[var], [0], [1]])
                        else:
                            modified_branch_decisions.append([[var], [1], [0]])
                    break
        

        return modified_branch_decisions
    

    def _get_var_idx(self,var):
        var_name = var.name
        if str(var_name) in self.var2idx:
            var_idx = self.var2idx[str(var_name)]
        elif 't_'+str(var_name) in self.var2idx:
            var_idx = self.var2idx['t_' + str(var_name)]
        elif '_'.join(str(var_name).split('_')[1:]) in self.var2idx:
            var_idx = self.var2idx[ '_'.join(str(var_name).split('_')[1:]) ] 
        else:
            var_idx = self.var2idx[ '_'.join(str(var_name).split('_')[1:2]) ]  

        
        return var_idx
   



        
class BipartiteGraphStatic0():
    
    #Defines the structure of the problem solved. Invariant toward problems
    def __init__(self, n0, device, d0=6, d1=4, allocate=True):
        
        self.n0, self.d0, self.d1 = n0, d0, d1
        self.device = device
        
        if allocate:
            self.var_attributes = torch.zeros(n0,d0, device=self.device)
            self.cons_block_idxs = []
        else:
            self.var_attributes = None
            self.cons_block_idxs = None
    
    
    def copy(self):
        
        copy = BipartiteGraphStatic0(self.n0, self.device, allocate=False)
        
        copy.var_attributes = self.var_attributes.clone()
        copy.cons_block_idxs = self.cons_block_idxs #no scip bonds
        
        return copy
