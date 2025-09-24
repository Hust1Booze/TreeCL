#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:43:54 2021

@author: abdel
"""

def load_src(name, fpath):
     import os, imp
     return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("data_type", "../learning/data_type.py" )
load_src("model", "../learning/model.py" )

import torch
import time
import numpy as np
from pyscipopt import Nodesel
from data_type import BipartiteGraphPairData
from model import GNNPolicy, RankNet
from line_profiler import LineProfiler
from joblib import dump, load
import torch.nn.functional as F



class CustomNodeSelector(Nodesel):

    def __init__(self, sel_policy='', comp_policy=''):
        self.sel_policy = sel_policy
        self.comp_policy = comp_policy
        self.sel_counter = 0
        self.comp_counter = 0

        
    def nodeselect(self):
        
        self.sel_counter += 1
        policy = self.sel_policy
        
        if policy == 'estimate':
            res = self.estimate_nodeselect()
        elif policy == 'dfs':
            res = self.dfs_nodeselect()
        elif policy == 'breadthfirst':
            res = self.breadthfirst_nodeselect()
        elif policy == 'bfs':
            res = self.bfs_nodeselect()
        elif policy == 'random':
            res = self.random_nodeselect()
        else:
            res = {"selnode": self.model.getBestNode()}
            # if self.model.getBestNode() is not None:
            #     print(f'sel node number : {self.model.getBestNode().getNumber()}')
            #     if self.model.getBestNode().getParent() is not None:
            #         print(f'sel node parent is : {self.model.getBestNode().getParent().getNumber()}')
        return res
    
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        policy = self.comp_policy
        
        if policy == 'estimate':
            res = self.estimate_nodecomp(node1, node2)
        elif policy == 'dfs':
            res = self.dfs_nodecomp(node1, node2)
        elif policy == 'breadthfirst':
            res = self.breadthfirst_nodecomp(node1, node2)
        elif policy == 'bfs':
            res = self.bfs_nodecomp(node1, node2)
        elif policy == 'random':
            res = self.random_nodecomp(node1, node2)
        else:
            res = 0
            
        return res
    
    #BFS
    def bfs_nodeselect(self):
        return {'selnode':self.model.getBfsSelNode() }
        
        
        
    #Estimate 
    def estimate_nodeselect(self):
        return {'selnode':self.model.getEstimateSelNode() }
    
    def estimate_nodecomp(self, node1,node2):
        
        #estimate 
        estimate1 = node1.getEstimate()
        estimate2 = node2.getEstimate()
        if (self.model.isInfinity(estimate1) and self.model.isInfinity(estimate2)) or \
            (self.model.isInfinity(-estimate1) and self.model.isInfinity(-estimate2)) or \
            self.model.isEQ(estimate1, estimate2):
                lb1 = node1.getLowerbound()
                lb2 = node2.getLowerbound()
                
                if self.model.isLT(lb1, lb2):
                    return -1
                elif self.model.isGT(lb1, lb2):
                    return 1
                else:
                    ntype1 = node1.getType()
                    ntype2 = node2.getType()
                    CHILD, SIBLING = 3,2
                    
                    if (ntype1 == CHILD and ntype2 != CHILD) or (ntype1 == SIBLING and ntype2 != SIBLING):
                        return -1
                    elif (ntype1 != CHILD and ntype2 == CHILD) or (ntype1 != SIBLING and ntype2 == SIBLING):
                        return 1
                    else:
                        return -self.dfs_nodecomp(node1, node2)
     
        
        elif self.model.isLT(estimate1, estimate2):
            return -1
        else:
            return 1
        
        
        
    # Depth first search        
    def dfs_nodeselect(self):
        
        selnode = self.model.getPrioChild()  #aka best child of current node
        if selnode == None:
            
            selnode = self.model.getPrioSibling() #if current node is a leaf, get 
            # a sibling
            if selnode == None: #if no sibling, just get a leaf
                selnode = self.model.getBestLeaf()
                
        return {"selnode": selnode}
    
    def dfs_nodecomp(self, node1, node2):
        return -node1.getDepth() + node2.getDepth()
    
    
    
    # Breath first search
    def breadthfirst_nodeselect(self):
        
        selnode = self.model.getPrioSibling()
        if selnode == None: #no siblings to be visited (all have been LP-solved), since breath first, 
        #we take the heuristic of taking the best leaves among all leaves
            
            selnode = self.model.getBestLeaf() #DOESTN INCLUDE CURENT NODE CHILD !
            if selnode == None: 
                selnode = self.model.getPrioChild()
        
        return {"selnode": selnode}
    
    def breadthfirst_nodecomp(self, node1, node2): 
        
        d1, d2 = node1.getDepth(), node2.getDepth()
        
        if d1 == d2:
            #choose the first created node
            return node1.getNumber() - node2.getNumber()
        
        #less deep node => better
        return d1 - d2
        
     
     #random
    def random_nodeselect(self):
        return {"selnode": self.model.getBestNode()}
    def random_nodecomp(self, node1,node2):
        return -1 if np.random.rand() < 0.5 else 1

    



class OracleNodeSelectorAbdel(CustomNodeSelector):

    def __init__(self, oracle_type, optsol=0, prune_policy='estimate', inv_proba=0, sel_policy=''):

        super().__init__(sel_policy=sel_policy)
        self.oracle_type = oracle_type
        self.optsol = optsol
        self.prune_policy = prune_policy 
        self.inv_proba = inv_proba
        self.sel_policy = sel_policy
        self.inf_counter  = 0
        
    
    def nodecomp(self, node1, node2, return_type=False):
        
        self.comp_counter += 1
        
        if self.oracle_type == "optimal_plunger":            
        
            d1 = self.is_sol_in_domaine(self.optsol, node1)
            d2 = self.is_sol_in_domaine(self.optsol, node2)
            inv = np.random.rand() < self.inv_proba
            
            if d1 and d2:
                res, comp_type = self.dfs_nodecomp(node1, node2), 0
            elif d1:
                res = comp_type = -1
                self.inf_counter += 1
                
            
            elif d2:
                res = comp_type = 1
                self.inf_counter += 1
            
            else:
                res, comp_type = self.estimate_nodecomp(node1, node2), 10         
            
            inv_res = -1 if res == 1 else 1
            res = inv_res if inv else res
            
            return res if not return_type  else  (res, comp_type)
        else:
            raise NotImplementedError

    
    def is_sol_in_domaine(self, sol, node):
        #By partionionning, it is sufficient to only check what variable have
        #been branched and if sol is in [lb, up]_v for v a branched variable
        
        bvars, bbounds, btypes = node.getAncestorBranchings()
        
        for bvar, bbound, btype in zip(bvars, bbounds, btypes): 
            if btype == 0:#LOWER BOUND
                if sol[bvar] < bbound:
                    return False
            else: #btype==1:#UPPER BOUND
                if sol[bvar] > bbound:
                    return False
        
        return True
            
            
    def setOptsol(self, optsol):
        self.optsol = optsol
        

class OracleNodeSelectorEstimator_RankNet(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, device, sel_policy='', n_primal=2):
        super().__init__(sel_policy=sel_policy)
        
        
        policy = RankNet()

        policy.load_state_dict(torch.load(f"./learning/policy_{problem}_ranknet.pkl", map_location=device)) #run from main

        policy.to(device)
        
        self.policy = policy
        self.device = device
        

        self.comp_featurizer = comp_featurizer
        
        self.inf_counter = 0
        
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0
        
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
         
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        
        decision =  self.policy(torch.tensor(f1, dtype=torch.float, device=self.device), torch.tensor(f2, dtype=torch.float, device=self.device))
        
        
    
        self.inf_counter += 1
        
        return -1 if decision < 0.5 else 1


class OracleNodeSelectorEstimator_SVM(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, sel_policy='', n_primal=2):
        super().__init__(sel_policy=sel_policy)
        
        self.policy = load(f'./learning/policy_{problem}_svm.pkl')
        self.comp_featurizer = comp_featurizer
        
        self.inf_counter = 0
        
        self.n_primal = n_primal
        self.best_primal = np.inf
        self.primal_changes = 0
        
    def nodecomp(self, node1, node2):
        
        self.comp_counter += 1
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
         
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        
        X = np.hstack((f1,f2))
        X = X[np.newaxis, :]
        
        decision = self.policy.predict(X)[0]
        self.inf_counter += 1
        
        return -1 if decision < 0.5 else 1

    
        
    
    
class OracleNodeSelectorEstimator(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, device, feature_normalizor, n_primal=2, use_trained_gnn=True, sel_policy='',avoid_same_comp =False):
        super().__init__(sel_policy=sel_policy)
        
        
        
        policy = GNNPolicy()
        if use_trained_gnn: 
            #policy.load_state_dict(torch.load(f"./learning/policy_{problem}.pkl", map_location=device)) #run from main
            policy.load_state_dict(torch.load(f"./policy_{problem}.pkl", map_location=device)) #run from main
        else:
            print("Using randomly initialized gnn")
            
        policy.to(device)
        
        self.policy = policy
        self.comp_featurizer = comp_featurizer
        self.device = device
        self.feature_normalizor = feature_normalizor
        
        self.n_primal = n_primal
        self.best_primal = np.inf #minimization
        self.primal_changes = 0
        
        self.fe_time = 0
        self.fn_time = 0
        self.inference_time = 0
        self.inf_counter = 0
        
        self.scores = dict()

        self.embd = dict()

        self.similarity = dict()

        self.variable_features = dict()

        self.var2idx = None

        self.comparison_results = {}

        # this use to not make diff nodecomp res, for (node1, node2) 
        # there may many time same nodecomp and SCIP will not work 
        # so here modify comp res make SCIP work
        self.avoid_same_comp = avoid_same_comp
        
        
        
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_featurizer.set_LP_feature_recorder(LP_feature_recorder)
        
        self.init_solver_cpu = LP_feature_recorder.init_time
        self.init_cpu_gpu = LP_feature_recorder.init_cpu_gpu_time
        
        self.fe_time = 0
        self.fn_time = 0
        self.inference_time = 0
        self.inf_counter = 0

        
    
    def nodecomp_back(self, node1,node2):
        
        self.comp_counter += 1        
        
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
            
            
        #begin inference process
        comp_scores = [-1,-1]
        
        for comp_idx, node in enumerate([node1, node2]):
            n_idx = node.getNumber()
        
            if n_idx in self.scores:
                comp_scores[comp_idx] = self.scores[n_idx]
            else:

                _time, g =  self.comp_featurizer.get_graph_for_inf(self.model, node)
                
                self.fe_time += _time

                
                start = time.time()
                g = self.feature_normalizor(*g)[:-1]
                self.fn_time += (time.time() - start)
                
                start = time.time()
                score = self.policy.forward_graph(*g).item()
                self.scores[n_idx] = score 
                comp_scores[comp_idx] = score
                self.inference_time += (time.time() - start)
                
        self.inf_counter += 1
        
        return -1 if comp_scores[0] > comp_scores[1] else 1

    
    def nodecomp(self, node1, node2):

        self.comp_counter += 1        
        
        if len(self.embd) >= 500:
            return self.estimate_nodecomp(node1, node2)
            
        if self.primal_changes >= self.n_primal: #infer until obtained nth best primal solution
            
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
            
        if self.var2idx is None:
            varrs = self.model.getVars()
            self.varrs = varrs
            self.var2idx = dict([ (str_var, idx) for idx, var in enumerate(self.varrs) for str_var in [str(var)]  ])
        #begin inference process
        comp_scores = [-1,-1]
        #
        if 1 not in self.embd.keys():
            root_node = self.model.getRootNode()
            _time, g =  self.comp_featurizer.get_graph_for_inf(self.model, root_node)
            g = self.feature_normalizor(*g)[:-1]
            variable_avg  = self.policy.forward_graph(*g)
            self.embd[1] = variable_avg
            #self.embd[1] = variable_avg.to('cpu')
            #self.scores[1] = score.item() 
            #self.variable_features[1] = variable_featuress
        n_idx1,n_idx2 = node1.getNumber(), node2.getNumber()

        if (n_idx1, n_idx2) in self.comparison_results:
            if self.avoid_same_comp:
                #print(f'same node compare, return diff result')
                return 1 if self.comparison_results[(n_idx1, n_idx2)] == -1 else -1


        for comp_idx, node in enumerate([node1, node2]):
            n_idx = node.getNumber()
        
            if n_idx in self.scores:
                comp_scores[comp_idx] = self.scores[n_idx]
            else:

                _time, g =  self.comp_featurizer.get_graph_for_inf(self.model, node)
                
                self.fe_time += _time

                
                start = time.time()
                g = self.feature_normalizor(*g)[:-1]
                self.fn_time += (time.time() - start)
                
                start = time.time()
                variable_avg  = self.policy.forward_graph(*g)
                #print(time.time() - start)
                self.inference_time += (time.time() - start)
                #variable_avg = variable_avg.to('cpu')
                # score = torch.linalg.norm(variable_avg, dim=1) + torch.linalg.norm(constraint_avg, dim=1) + torch.linalg.norm(bbounds, dim=1)
                # score = score.item()
                similarity = F.cosine_similarity(self.embd[self._find_co_parent(node1,node2)], variable_avg).item()
                #similarity = F.cosine_similarity(self.embd[1], variable_avg).item()
                self.embd[n_idx] = variable_avg
                #self.similarity[n_idx] = similarity
                self.scores[n_idx] = similarity #score 
                #self.variable_features[n_idx] = variable_features
                comp_scores[comp_idx] =  similarity #score
                #self.inference_time += (time.time() - start)
                    
                self.inf_counter += 1
        # if (self.scores[n_idx1]>self.scores[n_idx2] and self.similarity[n_idx1]<self.similarity[n_idx2]) \
        #     or (self.scores[n_idx1]<self.scores[n_idx2] and self.similarity[n_idx1]>self.similarity[n_idx2]):
        #     pass
            #print("different decision")
            #predict = self.find_co_parent(node1, node2)
            #return predict


        self.comparison_results[(n_idx1, n_idx2)] = -1 if comp_scores[0] > comp_scores[1] else 1
        return -1 if comp_scores[0] > comp_scores[1] else 1
    
    def find_co_parent(self, node1, node2):
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
        variable_features = self.variable_features[idx]

        bvars1, bbounds1, btypes1 = node1.getParentBranchings()
        bvars2, bbounds2, btypes2 = node2.getParentBranchings()
        
        bvars = bvars1[0]
        if str(bvars) in self.var2idx:
            var_idx = self.var2idx[str(bvars)]
        elif 't_'+str(bvars) in self.var2idx:
            var_idx = self.var2idx['t_' + str(bvars)]
        else:
            var_idx = self.var2idx[ '_'.join(str(bvars).split('_')[1:]) ] 

        if abs(bbounds1[0] - variable_features[var_idx]) <0.5:
            return -1
        else:
            return 1


    def _find_co_parent(self, node1, node2):
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
        return idx

            

