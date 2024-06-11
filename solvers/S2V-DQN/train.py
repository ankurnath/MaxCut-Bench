import os
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import src.envs.core as ising_env
from experiments.utils import  mk_dir
from src.agents.dqn.dqn import DQN
from src.agents.dqn.utils import TestMetric
import torch
from src.envs.utils import (GraphDataset,
                            RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            Observable)
from src.networks.models import MPNN




def train_GNN(distribution):
    # Get the current working directory and store it in the variable current_directory
    current_directory = os.getcwd()
    # Create a directory path by joining the current directory with 'pretrained agents' using os.path.join()
    model_save_directory=os.path.join(current_directory,'solvers/S2V-DQN/pretrained agents')


    config_path=os.path.join(current_directory,'solvers','S2V-DQN/config.pkl')

    with open(config_path, 'rb') as f:
        train_config = pickle.load(f)
    
    
    gamma=1
    step_factor=1
    clip_Q_target=True
    observables=[Observable.SPIN_STATE]
    
    reward_signal=RewardSignal.DENSE,
    # Define a dictionary of whether spins are reversible for different models
    reversible_spin=False
    
    # Create the directory specified by model_save_directory if it doesn't already exist
    mk_dir(model_save_directory)

    
    extra_action=ExtraAction.NONE
    
    basin_reward=False
    
    # Define a dictionary of whether spins are reversible for different models
    reversible_spin=False
    
    env_args =              {'observables':observables, # Get observables based on the 'model'
                            'reward_signal':reward_signal,# Get the reward signal based on the 'model'
                            'extra_action':extra_action,# Set extra action to None
                            'optimisation_target':OptimisationTarget.CUT, # Set the optimization target to CUT
                            'spin_basis':SpinBasis.BINARY,  # Set the spin basis to BINARY
                            'norm_rewards':True, # Normalize rewards (set to True)   
                            'stag_punishment':None, # Set stag punishment to None
                            'basin_reward':basin_reward, # Assign the 'basin_reward' based on the previous condition
                            'reversible_spins':reversible_spin}
    
    
    train_graph_generator=GraphDataset(os.path.join(os.getcwd(),f'data/training/{distribution}'), ordered=False)
    test_graph_generator=GraphDataset(os.path.join(os.getcwd(),f'data/validation/{distribution}'), ordered=True)
    n_tests = len(test_graph_generator)
    
    save_loc= os.path.join(model_save_directory,f'{distribution}')
        
        
    mk_dir(save_loc)
    data_folder = os.path.join(save_loc,'data')
    network_folder = os.path.join(save_loc, 'network')


    mk_dir(data_folder)
    mk_dir(network_folder)
    
    train_env = ising_env.make("SpinSystem",
                                train_graph_generator,
                                step_factor,
                                **env_args)
                             
                             
    test_env = ising_env.make("SpinSystem",
                                test_graph_generator,
                                step_factor,
                                **env_args)
    
    network_save_path = os.path.join(network_folder,'network.pth')
    test_save_path = os.path.join(network_folder,'test_scores.pkl')
     
    network_fn = lambda: MPNN(dim_in=1,
                              dim_embedding=64,
                            num_layers=3)
    
    config_load=200




        
    print('Configuration load:',config_load)
    nb_steps=int(train_config['nb_steps'][config_load])
    agent = DQN(train_env,

                network_fn,

                init_network_params=None,
                init_weight_std=0.01,

                double_dqn=True,
                clip_Q_targets=clip_Q_target,

                replay_start_size=train_config["replay_start_size"][config_load],
                replay_buffer_size=train_config["replay_buffer_size"][config_load],  # 20000
                gamma=gamma,  # 1
                update_target_frequency=train_config["update_target_frequency"][config_load],  # 500

                update_learning_rate=False,
                initial_learning_rate=1e-4,
                peak_learning_rate=1e-4,
                peak_learning_rate_step=20000,
                final_learning_rate=1e-4,
                final_learning_rate_step=200000,


                update_frequency=32,  # 1
                minibatch_size=16,  # 128
    #                             minibatch_size=8,  # 128
                max_grad_norm=None,
                weight_decay=0,

                update_exploration=True,
                initial_exploration_rate=1,
                final_exploration_rate=0.05,  # 0.05
                final_exploration_step=train_config['final_exploration_step'][config_load],  # 40000

                adam_epsilon=1e-8,
                logging=False,
                loss="mse",

                save_network_frequency=train_config['save_network_frequency'][config_load],
                network_save_path=network_save_path,

                evaluate=True,
                test_env=test_env,
                test_episodes=n_tests,
                test_frequency=train_config['test_frequency'][config_load],  # 10000
                test_save_path=test_save_path,
                test_metric=TestMetric.MAX_CUT,

                seed=None,
                device=None
                )



   
    import time
    start = time.time()
    agent.learn(timesteps=nb_steps, verbose=False)
    agent.save()
    print(time.time() - start)






    


    
    
    
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--distribution", type=str, help="Distribution of dataset")
    


    args = parser.parse_args()
    # num_devices = torch.cuda.device_count()
    # for i in range(num_devices):
    #     device_name = torch.cuda.get_device_name(i)
    #     # print("CUDA Device {}: {}".format(i, device_name))
    # if True:
    #     pass

    # if torch.cuda.is_available():
    #     if args.device is None:
    #         device = 'cuda:0' 
    #     else:
    #         # pass
    #         device=f'cuda:{args.device}'



    # Accessing arguments using attribute notation, not dictionary notation
    train_GNN(distribution=args.distribution, 
              )