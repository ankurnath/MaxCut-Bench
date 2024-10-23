import os
from experiments.utils import  mk_dir
from argparse import ArgumentParser
from src.envs.utils import (RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            DEFAULT_OBSERVABLES,
                            Observable)


import torch
from src.networks.models import MPNN,LinearRegression

import os
import torch
from experiments.utils import test_network
from src.envs.utils import (
                            GraphDataset,
                            RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            DEFAULT_OBSERVABLES)
from collections import defaultdict


def test_GNN(train_distribution,test_distribution,num_repeat,num_steps,step_factor):
    
    current_directory=os.getcwd()
    model_load_path=os.path.join(current_directory,"solvers/ECO-DQN/pretrained agents",f'{train_distribution}')
    
    

    observables = DEFAULT_OBSERVABLES

    reward_signal = RewardSignal.BLS
    # Define a dictionary of whether spins are reversible for different models
    reversible_spin = True
    extra_action = ExtraAction.NONE
   

    basin_reward= True

    # Define a dictionary of whether spins are reversible for different models
    reversible_spin=True

    env_args = {        'observables':observables, # Get observables based on the 'model'
                        'reward_signal':reward_signal,# Get the reward signal based on the 'model'
                        'extra_action':extra_action,# Set extra action to None
                        'optimisation_target':OptimisationTarget.CUT, # Set the optimization target to CUT
                        'spin_basis':SpinBasis.BINARY,  # Set the spin basis to BINARY
                        'norm_rewards':True, # Normalize rewards (set to True)   
                        'stag_punishment':None, # Set stag punishment to None
                        'basin_reward':basin_reward, # Assign the 'basin_reward' based on the previous condition
                        'reversible_spins':reversible_spin,
                         'step_fact':step_factor,
                         'num_steps':num_steps}



    batched=True
    max_batch_size=None
    # data_folder = os.path.join(model_load_path, 'data')
    # network_folder = os.path.join(model_load_path, 'network')
    # mk_dir(data_folder)
    # mk_dir(network_folder)

    # print("data folder:", data_folder)
    # print("network folder:", network_folder)

    network_fn = lambda: MPNN(dim_in=7,
                             dim_embedding=64,
                            num_layers=3)

    #### CHANGE
    # graphs_test = GraphDataset(os.path.join(os.getcwd(),f'data/testing/{test_distribution}'), ordered=True)

    graphs_test = GraphDataset(f'../data/testing/{test_distribution}', ordered=True)
    n_tests=len(graphs_test)
    print(f'The number of test graphs:{n_tests}')


    graphs_test = [graphs_test.get() for _ in range(n_tests)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)
    print("Set torch default device to {}.".format(device))

    network = network_fn().to(device)

    network_save_path = os.path.join(model_load_path, 'network_best.pth')
    network.load_state_dict(torch.load(network_save_path,map_location=device))

    for param in network.parameters():
        param.requires_grad = False
    network.eval()

    results, _, _ = test_network(network, env_args, graphs_test, device, step_factor,n_attempts=num_repeat,
                                                return_raw=True, return_history=True,
                                                batched=batched, max_batch_size=max_batch_size,
                                                )
    

    save_folder = os.path.join('results',test_distribution)
    mk_dir(save_folder)

    results['Train Distribution'] = [train_distribution]* n_tests
    results['Test Distribution'] = [test_distribution] * n_tests
    results.drop(columns=['sol'], inplace=True)
    results.to_pickle(os.path.join(save_folder,'ECO-DQN'))
    print(results)
    
    
    # for res, label in zip([results],
    #                                       ["results"]):
    #     save_path = os.path.join(data_folder, label)
    #     res.to_pickle(save_path)
    #     print("{} saved to {}".format(label, save_path))
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_distribution",default=None,help='Train distribution (if train and test are not the same)')
    parser.add_argument("--test_distribution", type=str, help="Distribution of dataset")
    parser.add_argument("--num_repeat", type=int,default=50, help="Number of attempts")
    parser.add_argument("--num_steps",type=int, default=None, help="Number of steps in an episode")
    parser.add_argument("--step_factor",type=int, default=2, help="Step Factor")

    args = parser.parse_args()

    # Accessing arguments using attribute notation, not dictionary notation
    test_GNN(test_distribution=args.test_distribution,
             train_distribution = args.train_distribution,
             num_repeat=args.num_repeat,
              num_steps=args.num_steps, 
              step_factor=args.step_factor)