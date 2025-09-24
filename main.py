#!/usr/bin/env python
# coding: utf-8

# In[90]:


import sys
import os
import re
import numpy as np
import torch
# 移除多进程相关导入，改为单线程执行
from utils import record_stats, display_stats
from pathlib import Path 
import shutil
import time
 

import threading
from functools import partial
import torch
from utils import get_record_file,calwins

# 假设其他相关代码已定义，包括 record_stats 和 distribute

def get_solved_instances(problem, nodesels):
    """获取已经求解的实例列表"""
    solved_instances = set()
    stats_dir = Path(f"stats/{problem}")
    
    if not stats_dir.exists():
        return solved_instances
        
    for nodesel in nodesels:
        nodesel_dir = stats_dir / nodesel
        if nodesel_dir.exists():
            # 获取该nodesel下所有的.csv文件
            csv_files = list(nodesel_dir.glob("*.csv"))
            for csv_file in csv_files:
                # 将.csv文件名转换为.lp文件名（去掉.csv后缀）
                instance_name = csv_file.stem + ".lp"
                solved_instances.add(instance_name)
            
            csv_files = list(nodesel_dir.glob("*.mps"))
            for csv_file in csv_files:
                # 将.csv文件名转换为.lp文件名（去掉.csv后缀）
                instance_name = csv_file.stem + ".mps"
                solved_instances.add(instance_name)    
    return solved_instances

# 每个线程处理一个 instance
def process_instance(instance, nodesels, problem, device, normalize, verbose, default, avoid_same_comp = False):
    record_stats(nodesels=nodesels,
                 instances=[instance],  # 只处理一个实例
                 problem=problem,
                 device=device,
                 normalize=normalize,
                 verbose=verbose,
                 default=default,
                 avoid_same_comp = avoid_same_comp)

def run_in_parallel(instances, nodesels, problem, device, normalize, verbose, default, n_cpu):
    # 控制最大并发进程数为 n_cpu
    processes = []

    for p, instance in enumerate(instances):
        # 为每个实例创建一个新的进程
        process = Process(target=process_instance,
                                         args=(instance, nodesels, problem, device,
                                               normalize, verbose, default))
        processes.append(process)
        
        # 启动新进程
        process.start()

        # 如果已经有 n_cpu 个进程在运行，则等待这些进程完成
        if len(processes) >= n_cpu:
            for p in processes:
                p.join()  # 等待进程完成
            processes = []  # 清空进程列表，准备下一个批次

    # 确保最后剩下的进程也能完成
    for p in processes:
        p.join()

if __name__ == "__main__":
    
    n_cpu = 1
    n_instance = -1
    nodesels = ['gnn_dummy_nprimal=2']#['expert_dummy'] #['gnn_dummy_nprimal=2'] #['ranknet_dummy_nprimal=2']
    
    problem = 'MIK'
    normalize = True
    
    data_partition = 'test'
    device = 'cpu'
    verbose = False
    on_log = False
    default = False
    delete = False
    branching = None
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-n_cpu':
            n_cpu = int(sys.argv[i + 1])
        if sys.argv[i] == '-nodesels':
            nodesels = str(sys.argv[i + 1]).split(',')
        if sys.argv[i] == '-normalize':
            normalize = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-n_instance':
            n_instance = int(sys.argv[i + 1])
        if sys.argv[i] == '-data_partition':
            data_partition = str(sys.argv[i + 1])
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
        if sys.argv[i] == '-verbose':
            verbose = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-on_log':
            on_log = bool(int(sys.argv[i + 1]))    
        if sys.argv[i] == '-default':
            default = bool(int(sys.argv[i + 1]))  
        if sys.argv[i] == '-delete':
            delete = bool(int(sys.argv[i + 1]))  
        if sys.argv[i] == '-branch':
            branching = str(sys.argv[i + 1])




    if delete:
        try:
            import shutil
            shutil.rmtree(os.path.join(os.path.abspath(''), 
                                       f'stats/{problem}'))
        except:
            ''
    



    if problem == 'miplib' or problem == 'anonymous' or problem == 'MIK':
        instances = list(Path(os.path.join(os.path.abspath(''), 
                                        f"./problem_generation/data/{problem}/{data_partition}")).glob("*.mps"))
    else:
        instances = list(Path(os.path.join(os.path.abspath(''), 
                                    f"./problem_generation/data/{problem}/{data_partition}")).glob("*.lp"))
    if n_instance == -1 :
        
        n_instance = len(instances)
    import random
    random.shuffle(instances)

    # 找到已求解的实例并从instances中剔除
    # 获取已求解的实例
    solved_instances = get_solved_instances(problem, nodesels)
    print(f"Found {len(solved_instances)} already solved instances")
    
    # 从instances中剔除已求解的实例
    original_count = len(instances)
    #instances = [instance for instance in instances if instance.name not in solved_instances]
    filtered_count = len(instances)
    print(f"Filtered instances: {original_count} -> {filtered_count} (removed {original_count - filtered_count})")
    instances = instances[:n_instance]
    print(f"solving for {instances}")
    #instances = ['problem_generation/data/MIK/test/mik.250-10-75.1.mps']
    print("Evaluation")
    print(f"  Problem:                    {problem}")
    print(f"  n_instance/problem:         {len(instances)}")
    print(f"  Nodeselectors evaluated:    {','.join( ['default' if default else '' ] + nodesels)}")
    print(f"  Device for GNN inference:   {device}")
    print(f"  Normalize features:         {normalize}")
    print(f"  Branching:                  {branching}")
    print("----------------")



    # In[92]:


    #Run benchmarks - 单线程执行，无需多进程设置
    start_time = time.time()

    # 假设 instances 是你需要处理的实例列表，nodesels, problem 等是其他参数
    # 你可以选择合适的参数
    # run_in_parallel(instances=instances, 
    #                 nodesels=nodesels, 
    #                 problem=problem, 
    #                 device=device,  # 这里假设使用 CPU
    #                 normalize=normalize, 
    #                 verbose=verbose, 
    #                 default=default, 
    #                 n_cpu=8)  # 设置最多同时运行的进程数为 4

    # 单线程执行：直接调用record_stats函数处理所有实例
    # record_stats(nodesels=nodesels,
    #              instances=instances,
    #              problem=problem,
    #              device=torch.device(device),
    #              normalize=normalize,
    #              verbose=verbose,
    #              default=default)

    # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间
    execution_time = end_time - start_time
    print(f"程序运行时间：{execution_time}秒")

    if problem in ["COR_LAT","miplib",'anonymous','MIK']:
        min_n = 0
        max_n = 0
    else:
        min_n = min([ int( str(instance).split('=')[1 if problem == "GISP" else 2 ].split('_')[0] )  for instance in instances ] )
        max_n = max([ int( str(instance).split('=')[1 if problem == "GISP" else 2].split('_')[0] )  for instance in instances ] )

    # for instance in instances:
    #     try:
    #         file = get_record_file(problem, nodesels[0], instance)
    #         if not os.path.isfile(file): #no need to resolve 
    #             print(f'lack of file {file}')
    #             record_stats(nodesels=nodesels,
    #              instances=[instance],  # 只处理一个实例
    #              problem=problem,
    #              device=device,
    #              normalize=normalize,
    #              verbose=verbose,
    #              default=default,
    #              avoid_same_comp = True)
    #     except:
    #         print(f'some thing wrong in fix lack')

    means_nodes = display_stats(problem, nodesels, instances, min_n, max_n, default=default)

    #calwins(means_nodes)
    #shutil.rmtree('./stats')

    
   
