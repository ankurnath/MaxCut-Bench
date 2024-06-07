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


def test_GNN(distribution,model,step_factor=None):
    
    current_directory=os.getcwd()
    model_load_path=os.path.join(current_directory,"LocalSearch/pretrained agents",f'{distribution}_{model}')
    
    
    # gammas = {'ECO_DQN':0.95,'LinearRegression':0.95,'S2V':1,'LSDQN':0.9}
    step_factors={'ECO_DQN':2,'LinearRegression':2,'S2V':1,'LSDQN':2}
    clip_Q_targets=defaultdict(lambda: False)
    clip_Q_targets['S2V']=True
    # Define a dictionary of observables for different models
    observables={'LinearRegression':[Observable.SPIN_STATE,
                                     Observable.IMMEDIATE_REWARD_AVAILABLE,
                                     Observable.TIME_SINCE_FLIP],
                 'ECO_DQN':DEFAULT_OBSERVABLES,
                 'S2V':[Observable.SPIN_STATE],
                 'LSDQN':[Observable.SPIN_STATE]}

    reward_signal={'ECO_DQN':RewardSignal.BLS,
               'LinearRegression':RewardSignal.BLS,
               "S2V":RewardSignal.DENSE,
               'LSDQN':RewardSignal.DENSE
    #                'LSDQN':RewardSignal.NEGATIVE_DENSE
              }
    # Define a dictionary of whether spins are reversible for different models
    reversible_spins={'LinearRegression':True,"ECO_DQN":True,"S2V":False,"LSDQN":True}
    extra_action=defaultdict(lambda: ExtraAction.NONE)
    extra_action['LSDQN']=ExtraAction.DONE

    basin_reward={'LinearRegression':True,'ECO_DQN':True,'S2V':False,'LSDQN':False}

    # Define a dictionary of whether spins are reversible for different models
    reversible_spins={'LinearRegression':True,"ECO_DQN":True,"S2V":False,"LSDQN":True}
    if model =='S2V':
        step_factor=1

    if step_factor is None:
        raise ValueError('Step factor is not provided.')
    env_args = {        'observables':observables[model], # Get observables based on the 'model'
                        'reward_signal':reward_signal[model],# Get the reward signal based on the 'model'
                        'extra_action':extra_action[model],# Set extra action to None
                        'optimisation_target':OptimisationTarget.CUT, # Set the optimization target to CUT
                        'spin_basis':SpinBasis.BINARY,  # Set the spin basis to BINARY
                        'norm_rewards':True, # Normalize rewards (set to True)   
                        'stag_punishment':None, # Set stag punishment to None
                        'basin_reward':basin_reward[model], # Assign the 'basin_reward' based on the previous condition
                        'reversible_spins':reversible_spins[model],
                         'step_fact':step_factor}



    batched=True
    max_batch_size=None
    data_folder = os.path.join(model_load_path, 'data')
    network_folder = os.path.join(model_load_path, 'network')
    mk_dir(data_folder)
    mk_dir(network_folder)

    print("data folder:", data_folder)
    print("network folder:", network_folder)


    if model=='LinearRegression' :

        network_fn = lambda: LinearRegression(input_dim=len(observables[model])-1)


    elif model=='ECO_DQN' :
        network_fn = lambda: MPNN(dim_in=7,
                                    dim_embedding=64,
                                    num_layers=3)

    elif model=='S2V' or model=='LSDQN':
        network_fn = lambda: MPNN(dim_in=1,
                                    dim_embedding=64,
                                    num_layers=3)

    else:

        raise NotImplementError("Unknown Model Type")
    
    


    graphs_test = GraphDataset(f'../data/testing/{distribution}', ordered=True)
    n_tests=len(graphs_test)
    print(f'The number of test graphs:{n_tests}')


    graphs_test = [graphs_test.get() for _ in range(n_tests)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)
    print("Set torch default device to {}.".format(device))

    network = network_fn().to(device)

    network_save_path = os.path.join(network_folder, 'network_best.pth')
    network.load_state_dict(torch.load(network_save_path,map_location=device))

    for param in network.parameters():
        param.requires_grad = False
    network.eval()

    results, results_raw, history = test_network(network, env_args, graphs_test, device, step_factor,
                                                return_raw=True, return_history=True,
                                                batched=batched, max_batch_size=max_batch_size,
                                                )
    
    # for res, label in zip([results],
    #                       ["results"]):
    #     save_path = os.path.join(data_folder, label)
    #     res.to_pickle(save_path)
    #     print("{} saved to {}".format(label, save_path))

    # print(results['cut'].tolist())
    
    
    for res, label in zip([results, results_raw, history],
                                          ["results", "results_raw", "history"]):
        save_path = os.path.join(data_folder, label)
        res.to_pickle(save_path)
        print("{} saved to {}".format(label, save_path))
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--distribution", type=str, help="Distribution of dataset")
    parser.add_argument("--model", type=str, default=None, help="Model")
    parser.add_argument("--step_factor",type=int,required=True, help="Step factor")

    args = parser.parse_args()

    # Accessing arguments using attribute notation, not dictionary notation
    test_GNN(distribution=args.distribution,model=args.model,step_factor=args.step_factor)