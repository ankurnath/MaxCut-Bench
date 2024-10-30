from  numba import njit
import numpy as np
import os
import pandas as pd
import glob
import random
import pickle
from multiprocessing.pool import Pool
from scipy.sparse import load_npz
from collections import defaultdict
import time

class GraphDataset(object):

    def __init__(self,folder_path,ordered=False):
        super().__init__()

        self.file_paths=glob.glob(f'{folder_path}/*.npz')
        self.file_paths.sort()
        self.ordered=ordered

        if self.ordered:
            self.i = 0

    def __len__(self):
        return len(self.file_paths)
    
    def get(self):
        if self.ordered:
            file_path = self.file_paths[self.i]
            self.i = (self.i + 1)%len(self.file_paths)
        else:
            file_path = random.sample(self.file_paths, k=1)[0]
        return load_npz(file_path).toarray()


@njit
def flatten_graph(graph):
    """
    Flatten a graph into matrices for adjacency, weights, start indices, and end indices.

    Parameters:
    - graph (adjacency matrix): The input graph to be flattened.

    Returns:
    - numpy.ndarray: Flattened adjacency matrix.
    - numpy.ndarray: Flattened weight matrix.
    - numpy.ndarray: Start indices for nodes in the flattened matrices.
    - numpy.ndarray: End indices for nodes in the flattened matrices.
    """
    flattened_adjacency = []
    flattened_weights = []
    num_nodes = graph.shape[0]
    
    node_start_indices = np.zeros(num_nodes,dtype=np.int64)
    node_end_indices = np.zeros(num_nodes,dtype=np.int64)
    
    for i in range(num_nodes):
        node_start_indices[i] = len(flattened_adjacency)
        for j in range(num_nodes):
            if graph[i, j] != 0:
                flattened_adjacency.append(j)
                flattened_weights.append(graph[i, j])
                
        node_end_indices[i] = len(flattened_adjacency)

    return (
        np.array(flattened_adjacency),
        np.array(flattened_weights),
        node_start_indices,
        node_end_indices
    )



@njit
def forward_greedy(graph):
    adj_matrix, weight_matrix, start_list, end_list=graph
    
    n=len(start_list)
    delta_local_cuts=np.zeros(n)
    spins=np.ones(n)
    
    
    curr_score=0
    for i in range(n):
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                  weight_matrix[start_list[i]:end_list[i]]):
                
            delta_local_cuts[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)
            curr_score+=weight*(spins[i]+spins[j]-2*spins[i]*spins[j])

    curr_score/=2    
    
    flag=True
    
    while flag:
        arg_gain=np.argsort(-delta_local_cuts)
        flag=False
        for v in arg_gain:
            if spins[v]:
                if delta_local_cuts[v]<=0:
                    flag=False
                    break
                    
                curr_score+=delta_local_cuts[v]
                delta_local_cuts[v]=-delta_local_cuts[v]
                
                for u,weight in zip(adj_matrix[start_list[v]:end_list[v]],
                                     weight_matrix[start_list[v]:end_list[v]]):

                    delta_local_cuts[u]+=weight*(2*spins[u]-1)*(2-4*spins[v])

                spins[v] = 1-spins[v]
                flag=True
                break
                  
    
    return curr_score

@njit
def standard_greedy(graph):
    adj_matrix, weight_matrix, start_list, end_list=graph
    
    n=len(start_list)
    delta_local_cuts=np.zeros(n)
    spins=np.ones(n)
    
    
    curr_score=0
    for i in range(n):
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                  weight_matrix[start_list[i]:end_list[i]]):
                
            delta_local_cuts[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)
            curr_score+=weight*(spins[i]+spins[j]-2*spins[i]*spins[j])

    curr_score/=2    

    
    while True:
        v=np.argmax(delta_local_cuts)

        if delta_local_cuts[v]<=0:
            break
                    
        curr_score+=delta_local_cuts[v]
        delta_local_cuts[v]=-delta_local_cuts[v]
        
        for u,weight in zip(adj_matrix[start_list[v]:end_list[v]],
                                weight_matrix[start_list[v]:end_list[v]]):

            delta_local_cuts[u]+=weight*(2*spins[u]-1)*(2-4*spins[v])

        spins[v] = 1-spins[v]

    return curr_score


@njit
def mca(graph,spins):
    adj_matrix, weight_matrix, start_list, end_list=graph
    
    n=len(start_list)
    delta_local_cuts=np.zeros(n)
    
    
    
    curr_score=0
    for i in range(n):
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],weight_matrix[start_list[i]:end_list[i]]):
                
            delta_local_cuts[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)
            curr_score+=weight*(spins[i]+spins[j]-2*spins[i]*spins[j])

    curr_score/=2   
    
    
    
    while True:
        v=np.argmax(delta_local_cuts)
        
        if delta_local_cuts[v]<=0:
            break
                    
        curr_score+=delta_local_cuts[v]
        delta_local_cuts[v]=-delta_local_cuts[v]

        for u,weight in zip(adj_matrix[start_list[v]:end_list[v]],
                                weight_matrix[start_list[v]:end_list[v]]):

            delta_local_cuts[u]+=weight*(2*spins[u]-1)*(2-4*spins[v])

        spins[v] =1-spins[v]

    return curr_score





from argparse import ArgumentParser

if __name__ == '__main__':


    parser = ArgumentParser()

    parser.add_argument("--test_distribution", type=str, help="Distribution of dataset")
    parser.add_argument("--num_repeat", type=int,default=50, help="Distribution of dataset")



    args = parser.parse_args()

    test_distribution = args.test_distribution
    num_repeat = args.num_repeat

    print('Distribution:',test_distribution)
    print('Number of repeat for MCA:',num_repeat)



    print(os.getcwd())

    save_folder = f'results/{test_distribution}'

    os.makedirs(save_folder,exist_ok=True)


    ####################
    dataset_path=os.path.join(f'../data/testing/{test_distribution}')

   
    test_dataset=GraphDataset(dataset_path,ordered=True)
    

    fg_cuts = []
    mca_cuts = []
    sg_cuts = []
    fg_times = []
    mca_times = []
    sg_times = []

    for i in range(len(test_dataset)):
        graph = test_dataset.get()
        g = flatten_graph(graph)

        # MCA
        start = time.time()
        mca_arguments = []
        for _ in range(args.num_repeat):
            spins = np.random.randint(2, size=graph.shape[0])
            mca_arguments.append((g, spins))

        with Pool() as pool:
            best_mca_cut = np.max(pool.starmap(mca, mca_arguments))
        end = time.time()
        elapsed_time_mca = end - start
        mca_cuts.append(best_mca_cut)
        mca_times.append(elapsed_time_mca)

        # Standard Greedy
        start = time.time()
        sg_cut = standard_greedy(g)
        end = time.time()
        elapsed_time_sg = end - start
        sg_cuts.append(sg_cut)
        sg_times.append(elapsed_time_sg)

        # Forward Greedy
        start = time.time()
        fg_cut = forward_greedy(g)
        end = time.time()
        elapsed_time_fg = end - start
        fg_cuts.append(fg_cut)
        fg_times.append(elapsed_time_fg)

    instances = [os.path.basename(file) for file in test_dataset.file_paths]

    # Saving MCA results
    mca_df = {'instances': instances, 'cut': mca_cuts, 'time': mca_times}
    mca_df = pd.DataFrame(mca_df)
    mca_df.to_pickle(os.path.join(save_folder, 'RG'))

    # Saving Standard Greedy results
    sg_df = {'instances': instances, 'cut': sg_cuts, 'time': sg_times}
    sg_df = pd.DataFrame(sg_df)
    


    
    try:
        OPT = pd.read_pickle(f'../data/testing/{test_distribution}/optimal')
        sg_df['OPT'] = OPT['OPT'].tolist()

        print('Mean',(sg_df['cut']/sg_df['OPT']).mean())
    except:
        # raise ValueError('')
        pass
    sg_df.to_pickle(os.path.join(save_folder, 'Standard Greedy'))
    # Saving Forward Greedy results
    fg_df = {'instances': instances, 'cut': fg_cuts, 'time': fg_times}
    fg_df = pd.DataFrame(fg_df)
    fg_df.to_pickle(os.path.join(save_folder, 'Forward Greedy'))


    # print(mca_df)
    print(sg_df)
    # print(fg_df)

















    



