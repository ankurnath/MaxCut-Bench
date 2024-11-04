import numpy as np
import torch

from torch.utils.data import DataLoader

from model import RUNCSP
from csp_data import CSP_Data
import os 


from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob
from utils import GraphDataset,mk_dir
import networkx as nx

import torch
from csp_data import CSP_Data
from timeit import default_timer as timer
from collections import defaultdict
import pandas as pd
import time 


def evaluate(model, loader, device, args):

    assignments=[]
    opt_steps=[]
    times = []

    with torch.inference_mode():
        for data in tqdm(loader):
            start = time.time()
            # path = data.path
            
            data = CSP_Data.collate([data for _ in range(args.num_repeat)])
            data.to(device)
            assignment = model(data, args.num_steps)
            assignments.append(data.hard_assign(assignment.squeeze()).cpu().numpy())

            end = time.time()
            times.append(end-start)
    return assignments,np.array(times)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # parser.add_argument("--distribution", type=str, default='ER_20', help="Distribution")
    parser.add_argument("--train_distribution",default=None,help='Train distribution (if train and test are not the same)')
    parser.add_argument("--test_distribution", type=str, help="Distribution of dataset")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of loader workers")

    parser.add_argument("--num_repeat", type=int, default=50, help="Number of parallel runs")
    parser.add_argument("--num_steps", type=int,required=True, default=250, help="Number of network steps")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dict_args = vars(args)

    train_distribution = args.train_distribution
    test_distribution = args.test_distribution

    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    model=RUNCSP.load(os.path.join(os.getcwd(),f'solvers/RUN-CSP/pretrained agents/{train_distribution}'))


    model.to(device)
    model.eval()

    # test_graph_gen=GraphDataset(folder_path=os.path.join(os.getcwd(),f'data/testing/{test_distribution}'),ordered=True)
    test_graph_gen=GraphDataset(folder_path=os.path.join(f'../data/testing/{test_distribution}'),ordered=True)
    print(f'Number of graphs:{len(test_graph_gen)}')
    graphs = [nx.from_numpy_array(test_graph_gen.get()) for _ in range(len(test_graph_gen))]
    n_tests = len(graphs)
    data = [CSP_Data.load_graph_weighted_maxcut(nx_graph)for nx_graph in graphs]
    const_lang = data[0].const_lang

    loader = DataLoader(
        data,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=CSP_Data.collate
    )

    assignments,times=evaluate(model, loader, device, args)
   

    df= defaultdict(list)
    for assignment,graph in zip(assignments,graphs):
        assignment=assignment.reshape(args.num_repeat,-1)
        numpy_graph=nx.to_numpy_array(graph)
        best_cut=0
        for i in range(args.num_repeat):
            spins=2*assignment[i]-1
            cut= (1/4) * np.sum( np.multiply( numpy_graph, 1 - np.outer(spins, spins) ) )
            best_cut=max(best_cut,cut)
        df['cut'].append(best_cut)
        
    df['time'] = times
    df['Train Distribution'] = [train_distribution]* n_tests
    df['Test Distribution'] = [test_distribution] * n_tests
    
    df = pd.DataFrame(df)

    # save_folder = os.path.join('results',test_distribution)
    save_folder = os.path.join('results',train_distribution+' '+test_distribution)
    mk_dir(save_folder)
    file_path = os.path.join(save_folder,'RUN-CSP')
    df.to_pickle(file_path)
    
    print(f'Data has been saved to {file_path}')
    print(df)

  
    

