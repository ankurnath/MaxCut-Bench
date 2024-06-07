import numpy as np
import torch

from argparse import ArgumentParser

from src.csp.csp_data import CSP_Data
from src.model.model import ANYCSP
from src.data.dataset import File_Dataset
from collections import defaultdict
import pandas as pd
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("--model_dir", type=str, help="Model directory")
    parser.add_argument("--distribution", type=str,help="Distribution of graphs")
    # parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--checkpoint_name", type=str, default='best', help="Checkpoint to be used")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    # parser.add_argument("--network_steps", type=int, default=100000, help="Number of network steps during evaluation")
    # parser.add_argument("--network_steps", type=int,required=True, default=1000, help="Number of network steps during evaluation")
    # parser.add_argument("--num_boost", type=int, default=50, help="Number of parallel evaluate runs")

    parser.add_argument("--num_steps", type=int,required=True, default=1000, help="Number of network steps during evaluation")
    parser.add_argument("--num_repeat", type=int, default=50, help="Number of parallel evaluate runs")
    parser.add_argument("--verbose", action='store_true', default=False, help="Output intermediate optima")
    parser.add_argument("--timeout", type=int, default=1200, help="Timeout in seconds")
    args = parser.parse_args()
    dict_args = vars(args)



    model_dir= os.path.join(os.getcwd(),f'solvers/ANYCSP/pretrained agents/{args.distribution}/network') 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ANYCSP.load_model(model_dir, args.checkpoint_name)
    model.eval()
    model.to(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    datapath=os.path.join(os.getcwd(),f'data/testing/{args.distribution}')

    dataset = File_Dataset(datapath)
    # print(dataset.files)

    num_solved = 0
    total_time = 0.0
    num_total = len(dataset)
    # print(num_total)
    df=defaultdict(list)

    for data in dataset:
        file = data.path
        max_val = data.constraints['ext'].cst_neg_mask.int().sum().cpu().numpy()
        # print(file)

        if args.num_repeat > 1:
            data = CSP_Data.collate([data for _ in range(args.num_repeat)])

        data.to(device)

        if args.verbose:
            print(f'Solving {file}:')

        with torch.inference_mode():
            data = model(
                data,
                args.num_steps,
                return_all_assignments=False,
                return_log_probs=False,
                stop_early=True,
                return_all_unsat=False,
                verbose=args.verbose,
                keep_time=True,
                timeout=args.timeout
            )

        best_per_run = data.best_num_unsat
        mean_best = best_per_run.mean()
        best = best_per_run.min().cpu().numpy()
        solved = best == 0
        num_solved += int(solved)
        best_cut_val = max_val - best

        print(
            f'{file}: {"Solved" if solved else "Unsolved"}, '
            f'Num Unsat: {int(best)}, '
            f'Cut Value: {best_cut_val}, '
            f'Steps: {data.num_steps}, '
            f'Opt Time: {data.opt_time:.2f}s, '
            f'Opt Step: {data.opt_step}'
        )
        df['cut'].append(best_cut_val)
        df['Opt Step'].append(data.opt_step)
        df['Opt Time'].append(data.opt_time)

        # break
    df=pd.DataFrame(df)
    data_folder=os.path.join(os.getcwd(),f'solvers/ANYCSP/pretrained agents/{args.distribution}','data')
    os.makedirs(data_folder,exist_ok=True)
    df.to_pickle(os.path.join(data_folder,'results'))
    # print(f'Solved {100 * num_solved / num_total:.2f}%, Average Time: {total_time / num_total:.2f}s')
    print(df)
