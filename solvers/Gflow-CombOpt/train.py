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

from data import get_data_loaders
from util import seed_torch, TransitionBuffer, get_mdp_class
from algorithm import DetailedBalanceTransitionBuffer

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
    # with open_dict(cfg):
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
    # overwrite
    device = torch.device(f"cuda:{cfg.device:d}" if torch.cuda.is_available() and cfg.device>=0 else "cpu")
    print(f"Device: {device}")
    alg, buffer = get_alg_buffer(cfg, device)
    seed_torch(cfg.seed)
    print(str(cfg))
    print(f"Work directory: {os.getcwd()}")

    train_loader, test_loader = get_data_loaders(cfg)
    trainset_size = len(train_loader.dataset)
    print(f"Trainset size: {trainset_size}")
    # alg_save_path = os.path.abspath(f"{cfg.input}/alg.pt")
    # alg_save_path_best = os.path.abspath(f"{cfg.input}/alg_best.pt")
    # save_path=os.path.join('pretrained_agents',cfg.input)

    save_path=os.path.join(os.getcwd(),f'solvers/Gflow-CombOpt/pretrained agents/{cfg.input}')
    alg_save_path_best=os.path.join(save_path,"alg_best.pt")
    os.makedirs(save_path,exist_ok = True)
    train_data_used = 0
    train_step = 0
    train_logr_scaled_ls = []
    train_metric_ls = []
    metric_best = 0.
    result = {"set_size": {}, "logr_scaled": {}, "train_data_used": {}, "train_step": {}, }

    @torch.no_grad()
    def evaluate(ep, train_step, train_data_used, logr_scaler):
        torch.cuda.empty_cache()
        num_repeat = 20
        mis_ls, mis_top20_ls = [], []
        logr_ls = []
        pbar = tqdm(enumerate(test_loader))
        pbar.set_description(f"Test Epoch {ep:2d} Data used {train_data_used:5d}")
        for batch_idx, gbatch in pbar:
            gbatch = gbatch.to(device)
            gbatch_rep = dgl.batch([gbatch] * num_repeat)

            env = get_mdp_class(cfg.task)(gbatch_rep, cfg)
            state = env.state
            while not all(env.done):
                action = alg.sample(gbatch_rep, state, env.done, rand_prob=0.)
                state = env.step(action)

            logr_rep = logr_scaler(env.get_log_reward())
            logr_ls += logr_rep.tolist()
            curr_mis_rep = torch.tensor(env.batch_metric(state))
            curr_mis_rep = rearrange(curr_mis_rep, "(rep b) -> b rep", rep=num_repeat).float()
            mis_ls += curr_mis_rep.mean(dim=1).tolist()
            mis_top20_ls += curr_mis_rep.max(dim=1)[0].tolist()
            pbar.set_postfix({"Metric": f"{np.mean(mis_ls):.2f}+-{np.std(mis_ls):.2f}"})

        print(f"Test Epoch{ep:2d} Data used{train_data_used:5d}: "
              f"Metric={np.mean(mis_ls):.2f}+-{np.std(mis_ls):.2f}, "
              f"top20={np.mean(mis_top20_ls):.2f}, "
              f"LogR scaled={np.mean(logr_ls):.2e}+-{np.std(logr_ls):.2e}")

        result["set_size"][ep] = np.mean(mis_ls)
        result["logr_scaled"][ep] = np.mean(logr_ls)
        result["train_step"][ep] = train_step
        result["train_data_used"][ep] = train_data_used
        print(result)
        pickle.dump(result, gzip.open("./result.json", 'wb'))

    for ep in range(cfg.epochs):
        for batch_idx, gbatch in enumerate(train_loader):
            reward_exp = None
            process_ratio = max(0., min(1., train_data_used / cfg.annend))
            logr_scaler = get_logr_scaler(cfg, process_ratio=process_ratio, reward_exp=reward_exp)

            train_logr_scaled_ls = train_logr_scaled_ls[-5000:]
            train_metric_ls = train_metric_ls[-5000:]
            gbatch = gbatch.to(device)
            if cfg.same_graph_across_batch:
                gbatch = dgl.batch([gbatch] * cfg.batch_size_interact)
            train_data_used += gbatch.batch_size

            ###### rollout
            batch, metric_ls = rollout(gbatch, cfg, alg)
            buffer.add_batch(batch)

            logr = logr_scaler(batch[-2][:, -1])
            train_logr_scaled_ls += logr.tolist()
            train_logr_scaled = logr.mean().item()
            train_metric_ls += metric_ls
            train_traj_len = batch[-1].float().mean().item()

            ##### train
            batch_size = min(len(buffer), cfg.batch_size)
            indices = list(range(len(buffer)))
            for _ in range(cfg.tstep):
                if len(indices) == 0:
                    break
                curr_indices = random.sample(indices, min(len(indices), batch_size))
                batch = buffer.sample_from_indices(curr_indices)
                train_info = alg.train_step(*batch, reward_exp=reward_exp, logr_scaler=logr_scaler)
                indices = [i for i in indices if i not in curr_indices]

            if cfg.onpolicy:
                buffer.reset()

            if train_step % cfg.print_freq == 0:
                print(f"Epoch {ep:2d} Data used {train_data_used:.3e}: loss={train_info['train/loss']:.2e}, "
                      + (f"LogZ={train_info['train/logZ']:.2e}, " if cfg.alg in ["tb", "tbbw"] else "")
                      + f"metric size={np.mean(train_metric_ls):.2f}+-{np.std(train_metric_ls):.2f}, "
                      + f"LogR scaled={train_logr_scaled:.2e} traj_len={train_traj_len:.2f}")

            train_step += 1

            ##### eval
            if batch_idx == 0 or train_step % cfg.eval_freq == 0:
                # alg.save(alg_save_path)
                metric_curr = np.mean(train_metric_ls[-1000:])
                if metric_curr > metric_best:
                    metric_best = metric_curr
                    print(f"best metric: {metric_best:.2f} at step {train_data_used:.3e}")
                    alg.save(alg_save_path_best)
                if cfg.eval:
                    evaluate(ep, train_step, train_data_used, logr_scaler)

    evaluate(cfg.epochs, train_step, train_data_used, logr_scaler)
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

    cfg = parser.parse_args()
    cfg= vars(cfg)
    cfg['input']=cfg['distribution']
    cfg.pop('distribution')
    cfg=OmegaConf.create(cfg)
    

    # Create a simple graph
    num_nodes = 5
    src = torch.tensor([0, 1, 1, 2, 3])
    dst = torch.tensor([1, 2, 3, 4, 0])
    graph = dgl.graph((src, dst), num_nodes=num_nodes)

    # Send the graph to a device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    
    main(cfg)