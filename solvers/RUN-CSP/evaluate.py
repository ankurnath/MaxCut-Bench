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


def evaluate(model, loader, device, args):

    assignments=[]
    opt_steps=[]

    with torch.inference_mode():
        for data in loader:
            start = timer()
            path = data.path
            # print(args.num_boost)
            data = CSP_Data.collate([data for _ in range(args.num_boost)])
            data.to(device)

            # all_assignments = model(data, args.network_steps,eval=False)
            assignment = model(data, args.num_steps)
            assignments.append(data.hard_assign(assignment.squeeze()).cpu().numpy())
            # print(len(all_assignments))
            # assert len(all_assignments)==args.network_steps

            # assignment= assignment.reshape(args.num_boost,-1,args.network_steps)
            # print(assignment.shape)

            # best_unsat=1000000
            # opt_step=-1

            # for i in range(args.network_steps):

            #     assignment=all_assignments[-1]
            #     assignment = torch.cat([1.0-assignment, assignment], dim=2)

            #     num_unsat = data.count_unsat(assignment)
            #     min_unsat = num_unsat.min().cpu().numpy()

            #     if min_unsat<best_unsat:
            #         best_unsat=min_unsat
            #         # min_unsat=best_unsat
            #         opt_step=i
            #         best_assignment=assignment
            #     break


            # assignments.append(data.hard_assign(best_assignment.squeeze()).cpu().numpy())
            #     # assignment=data.hard_assign(all_assignments[i].squeeze()).cpu().numpy()

            # # print(opt_step)
            # opt_steps.append(opt_step)

            # assignment=all_assignments[-1]
            # # print(assignment.shape)
            # # if assignment.out_dim==1:
            # assignment = torch.cat([1.0-assignment, assignment], dim=2)

            # num_unsat = data.count_unsat(assignment)
            # min_unsat = num_unsat.min().cpu().numpy()
            # solved = min_unsat == 0
            # assignments.append(data.hard_assign(assignment.squeeze()).cpu().numpy())
            # assignment.append(all_assignments)




            # end = timer()
            # time = end - start

            # print(f'{path} -- Num Unsat: {min_unsat}')
            # print(f'{"Solved" if solved else "Unsolved"} {time:.2f}s')

    # return all_assignments
    return assignments


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--distribution", type=str, default='ER_20', help="Distribution")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of loader workers")

    parser.add_argument("--num_reapats", type=int, default=50, help="Number of parallel runs")
    # parser.add_argument("--network_steps", type=int, default=10000000, help="Number of network steps")
    parser.add_argument("--num_steps", type=int,required=True, default=250, help="Number of network steps")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dict_args = vars(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    try:

        model = RUNCSP.load(f'RUNCSP/pretrained agents/{args.distribution}')
    except:
        model= RUNCSP.load(f'pretrained agents/{args.distribution}')
    model.to(device)
    model.eval()

    # print(f'Loading Graphs from {args.data_path}...')
    # data = [CSP_Data.load_graph_maxcol(p, model.const_lang.domain_size) for p in tqdm(glob(args.data_path))]
    train_graph_gen=GraphDataset(folder_path=f'../data/testing/{args.distribution}',ordered=True)
    print(f'Number of graphs:{len(train_graph_gen)}')
    graphs = [nx.from_numpy_array(train_graph_gen.get()) for _ in range(len(train_graph_gen))]
    # graphs = [nx.from_numpy_array(train_graph_gen.get()) for _ in range(1)]
    data = [CSP_Data.load_graph_weighted_maxcut(nx_graph)for nx_graph in graphs]
    const_lang = data[0].const_lang

    loader = DataLoader(
        data,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=CSP_Data.collate
    )

    assignments=evaluate(model, loader, device, args)
    # print(assignments[0])

    df= defaultdict(list)
    for assignment,graph in zip(assignments,graphs):
        assignment=assignment.reshape(args.num_boost,-1)
        numpy_graph=nx.to_numpy_array(graph)
        best_cut=0
        for i in range(args.num_boost):
            
            spins=2*assignment[i]-1
            cut= (1/4) * np.sum( np.multiply( numpy_graph, 1 - np.outer(spins, spins) ) )
            # print(cut)
            best_cut=max(best_cut,cut)
        df['cut'].append(best_cut)
        # print('Best cut',best_cut)

    save_folder=f'RUNCSP/pretrained agents/{args.distribution}/data'
    # df['Opt Step']= opt_steps
    # df['Steps'] = [args.network_steps]*len(opt_steps)
    # mk_dir(save_folder)
    os.makedirs(save_folder,exist_ok=True)
    df=pd.DataFrame(df)
    file_name=os.path.join(save_folder,'results')
    print(df)
    df.to_pickle(file_name)
    
    

