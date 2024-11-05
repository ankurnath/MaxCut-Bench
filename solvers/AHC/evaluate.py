


from cim_optimizer.solve_Ising import *
import pickle
import numpy as np
import pandas as pd
import torch
import glob
from scipy.sparse import load_npz
import random
import os



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




def compute_cut(matrix,spins):
  return (1/4) * np.sum( np.multiply(matrix, 1 - np.outer(spins,spins)))

def load_pickle(file_path):
  with open(file_path, 'rb') as f:
      data = pickle.load(f)
  return data

from cim_optimizer.optimal_params import maxcut_100_params
from cim_optimizer.optimal_params import maxcut_200_params
from cim_optimizer.optimal_params import maxcut_500_params


def solve(graph,hyperparameters):
    

    result=Ising(-graph).solve(hyperparameters_autotune=True,
                           hyperparameters_randomtune=False,
                           return_lowest_energies_found_spin_configuration=True,
                           return_lowest_energy_found_from_each_run=False,
                           return_spin_trajectories_all_runs=False,
                           return_number_of_solutions=1,
                           suppress_statements=True,
                           **hyperparameters)

    
    spins=result.result['lowest_energy_spin_config']
    cut= (1/4) * np.sum( np.multiply(graph, 1 - np.outer(spins, spins) ) )
    return cut,spins

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_distribution', type=str,default="WattsStrogatz_200vertices_weighted",  help='Distribution of dataset')
    parser.add_argument("--device", type=int,default=None, help="cuda device")
    parser.add_argument("--num_repeat",type=int,default=10,help="Number of runs per instance")
    parser.add_argument("--time_span",type=int,default=1200,help="Time span")
    parser.add_argument("--num_parallel_runs",type=int,default=5,help="Number of parallel runs")
    

    args = parser.parse_args()

    hyperparameters=maxcut_200_params()
    hyperparameters['num_runs']=args.num_repeat
    hyperparameters['num_timesteps_per_run']=  args.time_span
    hyperparameters['num_parallel_runs']=  args.num_parallel_runs


    if torch.cuda.is_available():
        hyperparameters['use_GPU']=True
        if args.device is None:
            device = 'cuda:0' 
        else:
            device=f'cuda:{args.device}'
    else:
        device='cpu'

    hyperparameters['use_CAC']=False

    hyperparameters['chosen_device']=device

    dataset_path=os.path.join(os.getcwd(),f'data/testing/{args.test_distribution}')

    dataset=GraphDataset(dataset_path,ordered=True)


    print ('Number of test graphs:',len(dataset))
       
    cuts=[]
    spins=[]


    for _ in range(len(dataset)):
        graph=dataset.get()
        cut,spin=solve(graph,hyperparameters)
        cuts.append(cut)
        spins.append(spin)


    df={'cut':cuts}
    df=pd.DataFrame(df)

    test_distribution = args.distribution
    save_folder = os.path.join('results',test_distribution)
    os.makedirs(save_folder,exist_ok=True)
    file_path = os.path.join(save_folder,'AHC')
    df.to_pickle(file_path)

    
