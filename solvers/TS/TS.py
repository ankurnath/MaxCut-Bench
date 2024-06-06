import numpy as np
import glob
import numpy as np
from scipy.sparse import load_npz
import random
from  numba import njit
import os
import pandas as pd
from multiprocessing.pool import Pool

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
def tabu(graph,spins,tabu_tenure,max_steps):
    adj_matrix, weight_matrix, start_list, end_list=graph
    
    n=len(start_list)
    delta_local_cuts=np.zeros(n)
    
    tabu_list=np.ones(n)*-10000
    curr_score=0
    for i in range(n):
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                  weight_matrix[start_list[i]:end_list[i]]):
                
            delta_local_cuts[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)
            curr_score+=weight*(spins[i]+spins[j]-2*spins[i]*spins[j])
            

    curr_score/=2    
    best_score=curr_score

    for t in range(max_steps):
        arg_gain=np.argsort(-delta_local_cuts)
        for v in arg_gain:
            if (t-tabu_list[v]> tabu_tenure) or (best_score < curr_score + delta_local_cuts[v]):

                tabu_list[v] = t

                curr_score+=delta_local_cuts[v]
                delta_local_cuts[v]=-delta_local_cuts[v]
                
                for u,weight in zip(adj_matrix[start_list[v]:end_list[v]],
                                     weight_matrix[start_list[v]:end_list[v]]):

                    delta_local_cuts[u]+=weight*(2*spins[u]-1)*(2-4*spins[v])

                spins[v] = 1-spins[v]

                break

                
        best_score=max(curr_score,best_score)

    return best_score


# from argparse import ArgumentParser

# if __name__ == '__main__':

#     parser = ArgumentParser()
#     parser.add_argument("--distribution", type=str,default='Physics', help="Distribution of dataset")
#     parser.add_argument("--num_repeat", type=int,default=50, help="num_repeat")
#     parser.add_argument("--step_factor", type=int, default=2, help="Step factor")
#     # parser.add_argument ()
#     parser.add_argument("--save_file_path", type=str, default='', help="save file path")

#     parser.add_argument("--gamma", type=int, default=15, help="Tabu Tenure")
    

#     args = parser.parse_args()

#     save_folder=f'pretrained agents/{args.distribution}_Tabu/data'
#     os.makedirs(save_folder,exist_ok=True)

#     test_dataset=GraphDataset(f'../data/testing/{args.distribution}',ordered=True)

#     print("Number of test graphs:",len(test_dataset))
#     best_cuts=[]
#     for i in range(len(test_dataset)):
#         graph=test_dataset.get()
#         g=flatten_graph(graph)

#         n=graph.shape[0]

#         arguments=[]

#         for i in range(args.num_repeat):
#             spins= np.random.randint(2, size=graph.shape[0])
#             arguments.append((g,spins,args.gamma,graph.shape[0]*args.step_factor))
            
#         with Pool(40) as pool:
#             best_cut=np.max(pool.starmap(tabu, arguments))

#         best_cuts.append(best_cut)

#     best_cuts=np.array(best_cuts)

#     df={'Tabu':best_cuts,'tenure':[args.gamma]*best_cuts.shape[0]}
#     df['Instance'] = [os.path.basename(file) for file in test_dataset.file_paths]
#     df=pd.DataFrame(df)

#     print(df)

#     if args.save_file_path=='':
#         df.to_pickle(os.path.join(save_folder,f'results_gamma{args.gamma}_step{args.step_factor}'))
#         # print(f'Results are saved to {os.path.join(save_folder,f'results_tau{args.tau}_step{args.step_factor}')}')

#     else:
#         df.to_pickle(args.save_file_path)
#         # print(f'Results are saved to {os.path.join(save_folder,f'results_{args.save_file_path}')}')
    
#     # df.to_pickle(os.path.join(save_folder,f'results_gamma{args.gamma}_step{args.step_factor}'))


            

        












