import sys, os
import gzip, pickle
from time import time, sleep
from tqdm import tqdm
from omegaconf import DictConfig, open_dict, OmegaConf

import random
import numpy as np
import torch
import dgl
from einops import rearrange, reduce, repeat

from data import get_data_loaders,get_test_data_loader
from util import seed_torch, TransitionBuffer, get_mdp_class
from algorithm import DetailedBalanceTransitionBuffer
import pandas as pd
from collections import defaultdict
import argparse

torch.backends.cudnn.benchmark = True


def get_alg_buffer(cfg, device):
    assert cfg.alg in ["db", "fl"]
    buffer = TransitionBuffer(cfg.tranbuff_size, cfg)
    alg = DetailedBalanceTransitionBuffer(cfg, device)
    return alg, buffer

def get_logr_scaler(cfg, process_ratio=1., reward_exp=None):
    if reward_exp is None:
        reward_exp = float(cfg.reward_exp)

    if cfg.anneal == "linear":
        process_ratio = max(0., min(1., process_ratio)) # from 0 to 1
        reward_exp = reward_exp * process_ratio +\
                     float(cfg.reward_exp_init) * (1 - process_ratio)
    elif cfg.anneal == "none":
        pass
    else:
        raise NotImplementedError

    # (R/T)^beta -> (log R - log T) * beta
    def logr_scaler(sol_size, gbatch=None):
        logr = sol_size
        return logr * reward_exp
    return logr_scaler

def refine_cfg(cfg):
    with open_dict(cfg):
        cfg.device = cfg.d
        cfg.work_directory = os.getcwd()

        if cfg.task in ["mis", "maxindset", "maxindependentset",]:
            cfg.task = "MaxIndependentSet"
            cfg.wandb_project_name = "MIS"
        elif cfg.task in ["mds", "mindomset", "mindominateset",]:
            cfg.task = "MinDominateSet"
            cfg.wandb_project_name = "MDS"
        elif cfg.task in ["mc", "maxclique",]:
            cfg.task = "MaxClique"
            cfg.wandb_project_name = "MaxClique"
        elif cfg.task in ["mcut", "maxcut",]:
            cfg.task = "MaxCut"
            cfg.wandb_project_name = "MaxCut"
        else:
            raise NotImplementedError

        # architecture
        assert cfg.arch in ["gin"]

        # log reward shape
        cfg.reward_exp = cfg.rexp
        cfg.reward_exp_init = cfg.rexpit
        if cfg.anneal in ["lin"]:
            cfg.anneal = "linear"

        # training
        cfg.batch_size = cfg.bs
        cfg.batch_size_interact = cfg.bsit
        cfg.leaf_coef = cfg.lc
        cfg.same_graph_across_batch = cfg.sameg

        # data
        cfg.test_batch_size = cfg.tbs
        cfg.data_type = cfg.input.upper()
        # if "rb" in cfg.input:
        #     cfg.data_type = cfg.input.upper()
        # elif "ba" in cfg.input:
        #     cfg.data_type = cfg.input.upper()
        # else:
        #     raise NotImplementedError

    del cfg.d, cfg.rexp, cfg.rexpit, cfg.bs, cfg.bsit, cfg.lc, cfg.sameg, cfg.tbs
    return cfg

@torch.no_grad()
def rollout(gbatch, cfg, alg):
    env = get_mdp_class(cfg.task)(gbatch, cfg)
    state = env.state

    ##### sample traj
    reward_exp_eval = None
    traj_s, traj_r, traj_a, traj_d = [], [], [], []
    while not all(env.done):
        action = alg.sample(gbatch, state, env.done, rand_prob=cfg.randp, reward_exp=reward_exp_eval)

        traj_s.append(state)
        traj_r.append(env.get_log_reward())
        traj_a.append(action)
        traj_d.append(env.done)
        state = env.step(action)

    ##### save last state
    traj_s.append(state)
    traj_r.append(env.get_log_reward())
    traj_d.append(env.done)
    assert len(traj_s) == len(traj_a) + 1 == len(traj_r) == len(traj_d)

    traj_s = torch.stack(traj_s, dim=1) # (sum of #node per graph in batch, max_traj_len)
    traj_r = torch.stack(traj_r, dim=1) # (batch_size, max_traj_len)
    traj_a = torch.stack(traj_a, dim=1) # (batch_size, max_traj_len-1)
    """
    traj_a is tensor like 
    [ 4, 30, 86, 95, 96, 29, -1, -1],
    [47, 60, 41, 11, 55, 64, 80, -1],
    [26, 38, 13,  5,  9, -1, -1, -1]
    """
    traj_d = torch.stack(traj_d, dim=1) # (batch_size, max_traj_len)
    """
    traj_d is tensor like 
    [False, False, False, False, False, False,  True,  True,  True],
    [False, False, False, False, False, False, False,  True,  True],
    [False, False, False, False, False,  True,  True,  True,  True]
    """
    traj_len = 1 + torch.sum(~traj_d, dim=1) # (batch_size, )

    ##### graph, state, action, done, reward, trajectory length
    batch = gbatch.cpu(), traj_s.cpu(), traj_a.cpu(), traj_d.cpu(), traj_r.cpu(), traj_len.cpu()
    return batch, env.batch_metric(state)



def main(cfg: DictConfig):
    cfg = refine_cfg(cfg)
    device = torch.device(f"cuda:{cfg.device:d}" if torch.cuda.is_available() and cfg.device>=0 else "cpu")
    print(f"Device: {device}")
    alg, buffer = get_alg_buffer(cfg, device)
    seed_torch(cfg.seed)
    print(str(cfg))
    print(f"Work directory: {os.getcwd()}")

    

    test_loader = get_test_data_loader(cfg)
    testset_size = len(test_loader.dataset)
    print(f"Testset size: {testset_size}")
    # alg_save_path = os.path.abspath(f"{cfg.input}/alg.pt")
    # alg_save_path_best = os.path.abspath(f"{cfg.input}/alg_best.pt")
    load_path=os.path.join('gflow/pretrained_agents',cfg.input)
    alg_load_path_best=os.path.join(load_path,"alg_best.pt")
    alg.load(alg_load_path_best)
    # cfg.sameg=True
    # os.makedirs(save_path,exist_ok = True)
    # train_data_used = 0
    # train_step = 0
    # train_logr_scaled_ls = []
    # train_metric_ls = []
    # metric_best = 0.
    
    
    @torch.no_grad()
    def evaluate(num_repeat):
        torch.cuda.empty_cache()
        # num_repeat = 50
        mis_ls, mis_top50_ls = [], []
        # best_cut=[]
        # result = {}
        result = defaultdict (list)

        pbar = tqdm(enumerate(test_loader))
        
        for batch_idx, gbatch in pbar:
            print('Batch idx',batch_idx)
            gbatch = gbatch.to(device)
            gbatch_rep = dgl.batch([gbatch] * num_repeat)

            env = get_mdp_class(cfg.task)(gbatch_rep, cfg)
            state = env.state
            # step=0
            rewards=env.get_log_reward()
            while not all(env.done):
                action = alg.sample(gbatch_rep, state, env.done, rand_prob=0.)
                # print(type(action))
                # print(action.shape)
                # print(action)
                state = env.step(action)
                # action=action.reshape(-1,num_repeat)
                # print(action)
                # print(state)
                # print(env.batch_metric(state))
                if torch.all(rewards<=env.get_log_reward()):
                    rewards=env.get_log_reward()
                else:
                    raise ValueError ("No reward")
                # print(env.get_log_reward())
                # step+=1
            
            # print (torch.sum(state)) 
            # print('Step',step)

            # logr_rep = logr_scaler(env.get_log_reward())
            # logr_ls += logr_rep.tolist()
            # print(state.mean())
            
            curr_mis_rep = torch.tensor(env.batch_metric(state))
            # print(curr_mis_rep.shape)
            curr_mis_rep = rearrange(curr_mis_rep, "(rep b) -> b rep", rep=num_repeat).float()
            # print(curr_mis_rep.shape)
            mis_ls += curr_mis_rep.mean(dim=1).tolist()
            mis_top50_ls += curr_mis_rep.max(dim=1)[0].tolist()
            # best_runs= torch.argmax (curr_mis_rep,dim=1)
            # state=state.reshape(125,-1,10)
            # for i in range(10):
            #     each_graph_state=state[:,i]

            # print(state.shape)
            # best_run=torch.argmax(curr_mis_rep,dim=1)
            # print(best_run)

        #     pbar.set_postfix({"Metric": f"{np.mean(mis_ls):.2f}+-{np.std(mis_ls):.2f}"})

        print(
              f"Metric={np.mean(mis_ls):.2f}+-{np.std(mis_ls):.2f}, "
              f"top50={np.mean(mis_top50_ls):.2f}, "
              )
        # print(state.shape)
        # state=state.reshape(10,125).tolist()

        # state=state.tolist()
        # graph_no,num_repeat,num_spin
        # state=state.reshape(10,50,125)
        # for i in range()

        

        result["cut"] = mis_top50_ls
        # for i in range(10):
        #     result['state'].append(state[i])

        # for j,best_run in enumerate(best_runs):
        #     result['state'].append(state[j][best_run][:].tolist())
        # print(result)
        result=pd.DataFrame(result)
        # print(result)
        data_folder=f'gflow/pretrained_agents/{cfg.input}/data'
        os.makedirs(data_folder,exist_ok=True)
        result.to_pickle(os.path.join(data_folder,'results'))

        # pickle.dump(result, gzip.open("./result.json", 'wb'))

    
    evaluate(cfg.num_repeat)
    # alg.save(alg_save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Argument parser for mcut task')
    # Task
    parser.add_argument('--task', type=str, default='mcut', help='Task name')

    # Inputs
    parser.add_argument('--distribution', type=str,required=True, help='Input graph')

    # WandB settings
    parser.add_argument('--wandb', type=int, default=0, help='Use Weights & Biases')
    
    # Device settings
    parser.add_argument('--d', type=int, default=0, help='Device to use: -1 for CPU')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Printing and evaluation frequency
    parser.add_argument('--print_freq', type=int, default=3, help='Frequency of printing status')
    parser.add_argument('--wandb_freq', type=int, default=None, help='Frequency of logging to Weights & Biases')
    parser.add_argument('--eval', type=bool, default=False, help='Evaluation mode')
    parser.add_argument('--eval_freq', type=int, default=200, help='Frequency of evaluation')

    # GIN architecture
    parser.add_argument('--arch', type=str, default='gin', help='Architecture type')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of hidden layers')
    parser.add_argument('--hidden_layer', type=int, default=5, help='Number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate')
    parser.add_argument('--aggr', type=str, default='sum', help='Aggregation method')
    parser.add_argument('--learn_eps', type=bool, default=True, help='Learn epsilon parameter')

    # GFlowNet algorithm parameters
    parser.add_argument('--alg', type=str, default='fl', help='Algorithm type')
    parser.add_argument('--onpolicy', type=bool, default=True, help='Use on-policy training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--trainsize', type=int, default=4000, help='Training size')
    parser.add_argument('--testsize', type=int, default=500, help='Test size')
    parser.add_argument('--tstep', type=int, default=30, help='Number of time steps')
    parser.add_argument('--bsit', type=int, default=8, help='Batch size for iterations')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--tbs', type=int, default=30, help='Batch size for training')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle training data')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--sameg', type=bool, default=False, help='Use same graph across one batch')
    parser.add_argument('--tranbuff_size', type=int, default=1000000, help='Transition buffer size')

    # Learning rates
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--zlr', type=float, default=1e-3, help='Z-learning rate')

    # Random probability
    parser.add_argument('--randp', type=float, default=0., help='Random probability')
    
    # Leaf coefficient
    parser.add_argument('--lc', type=int, default=1, help='Leaf coefficient')

    # Reward shaping
    parser.add_argument('--anneal', type=str, default='linear', help='Annealing strategy')
    parser.add_argument('--annend', type=int, default=40000, help='End of annealing period')
    parser.add_argument('--rexp', type=float, default=5e2, help='Reward exponent')
    parser.add_argument('--rexpit', type=int, default=1, help='Reward exponent iteration')

    # Back trajectory
    parser.add_argument('--back_trajectory', type=bool, default=False, help='Use back trajectory')

    # Number of tracjectories
    parser.add_argument('--num_repeat', type=int, default=50, help='Number of tracjectories')

    cfg = parser.parse_args()
    
    cfg= vars(cfg)
    cfg['input']=cfg['distribution']
    cfg.pop('distribution')
    cfg=OmegaConf.create(cfg)



    num_nodes = 5
    src = torch.tensor([0, 1, 1, 2, 3])
    dst = torch.tensor([1, 2, 3, 4, 0])
    graph = dgl.graph((src, dst), num_nodes=num_nodes)

    # Send the graph to a device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    main(cfg)