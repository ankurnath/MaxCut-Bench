import os
import pickle
import networkx as nx
import time
import numpy as np
import scipy as sp
import pandas as pd
import torch

import torch.nn.functional as F
from torch.distributions import Multinomial

from collections import namedtuple
from copy import deepcopy

import src.envs.core as ising_env
from src.envs.utils import (SingleGraphGenerator, SpinBasis)
from src.agents.solver import Network, Greedy
from torch_geometric.data import Batch

####################################################
# TESTING ON GRAPHS
####################################################

def test_network(network, env_args, graphs_test,device=None, step_factor=1, batched=True,
                 n_attempts=50, return_raw=False, return_history=False, max_batch_size=None):
    if batched:
        return __test_network_batched(network, env_args, graphs_test, device, step_factor,
                                      n_attempts, return_raw, return_history, max_batch_size)
    else:
        if max_batch_size is not None:
            print("Warning: max_batch_size argument will be ignored for when batched=False.")
        return __test_network_sequential(network, env_args, graphs_test, step_factor,
                                         n_attempts, return_raw, return_history)

def __test_network_batched(network, env_args, graphs_test, device=None, step_factor=1,
                           n_attempts=50, return_raw=False, return_history=False, max_batch_size=None
                           ):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)


    

    # HELPER FUNCTION FOR NETWORK TESTING

    acting_in_reversible_spin_env = env_args['reversible_spins']

    if env_args['reversible_spins']:
        # If MDP is reversible, both actions are allowed.
        if env_args['spin_basis'] == SpinBasis.BINARY:
            allowed_action_state = (0, 1)
    else:
        # If MDP is irreversible, only return the state of spins that haven't been flipped.
        if env_args['spin_basis'] == SpinBasis.BINARY:
            allowed_action_state = 1
        

    
    def get_greedy_actions(pred,batch):
            
        num_graphs = batch.max().item() + 1
        graph_sortidx=torch.argsort(batch).type(torch.int64)
        graph_ids, graph_counts = torch.unique_consecutive(batch[graph_sortidx],
                                                        return_counts=True)
        
        end_indices = torch.cumsum(graph_counts, dim=0).cpu().tolist()
        start_indices = [0] + end_indices[:-1]
        greedy_actions=torch.zeros((num_graphs,), dtype=torch.int).type(torch.int64)
        
        for graph_id, a, b in zip(graph_ids, start_indices, end_indices):
            indices = graph_sortidx[a:b]
            greedy_actions[graph_id] = indices[torch.argmax(pred[indices])]

        # assert torch.equal(global_max_pool(pred,batch),pred[greedy_actions])
            
        return greedy_actions

    def predict(states):

        qs = network(states)

        if acting_in_reversible_spin_env:
            actions=get_greedy_actions(qs,states.batch)
            
        else:
            if qs.dim() == 1:
                x = (states.squeeze()[:,0] == allowed_action_state).nonzero()
                actions = [x[qs[x].argmax().item()].item()]
            else:
                disallowed_actions_mask = (states.x[:, 0] != allowed_action_state).unsqueeze(-1)
                qs_allowed = qs.masked_fill(disallowed_actions_mask, -10000)
                actions=get_greedy_actions(qs_allowed,states.batch)
            if torch.is_tensor(actions):
                actions=actions.cpu()
        return actions

    # NETWORK TESTING

    results = []
    results_raw = []
    if return_history:
        history = []

    n_attempts = n_attempts if env_args["reversible_spins"] else 1

    for j, test_graph in enumerate(graphs_test):

        i_comp = 0
        i_batch = 0
        t_total = 0

        test_env = ising_env.make("SpinSystem",
                                  SingleGraphGenerator(test_graph),
                                  **env_args)

        if return_history:
            actions_history = []
            rewards_history = []
            scores_history = []

        best_cuts = []
        init_spins = []
        best_spins = []


        # print(f'Number of attempts:{n_attempts}')
        while i_comp < n_attempts:

            if max_batch_size is None:
                batch_size = n_attempts
            else:
                batch_size = min(n_attempts - i_comp, max_batch_size)

            i_comp_batch = 0

            if return_history:
                actions_history_batch = [[None]*batch_size]
                rewards_history_batch = [[None] * batch_size]
                scores_history_batch = []

            test_envs = [None] * batch_size
            best_cuts_batch = [-1e3] * batch_size
            init_spins_batch = [[] for _ in range(batch_size)]
            best_spins_batch = [[] for _ in range(batch_size)]

            

            obs_batch = [None] * batch_size

            print("Preparing batch of {} environments for graph {}.".format(batch_size,j), end="...")

            
            for i in range(batch_size):
                env = deepcopy(test_env)
                obs_batch[i] = env.reset(test=True)
                test_envs[i] = env
                init_spins_batch[i] = env.best_spins
            if return_history:
                # scores_history_batch.append([env.calculate_score() for env in test_envs])
                scores_history_batch.append([env.score for env in test_envs])

            print("done.")

            # Calculate the max cut acting w.r.t. the network
            t_start = time.time()

            # pool = mp.Pool(processes=16)
            # print(obs_batch[0].x)

            if not isinstance(obs_batch,list):
                raise ValueError('Not a list')
            

            k = 0
            while i_comp_batch < batch_size:
                t1 = time.time()
                # Note: Do not convert list of np.arrays to FloatTensor, it is very slow!
                # see: https://github.com/pytorch/pytorch/issues/13918
                # Hence, here we convert a list of np arrays to a np array.
                # obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
                # print(i_comp_batch)
                # print(obs_batch)
                # print(len(obs_batch))
                assert len(obs_batch)+i_comp_batch==batch_size
                obs_batch =Batch.from_data_list(data_list=obs_batch, follow_batch=None, exclude_keys=None).to(device)
                actions = predict(obs_batch)
                # print(len(actions))
                if torch.is_tensor(actions):
                    actions=actions.tolist()
                _, graph_counts = torch.unique_consecutive(obs_batch.batch,
                                                            return_counts=True)
                end_indices = torch.cumsum(graph_counts, dim=0).cpu().tolist()
                start_indices = [0] + end_indices[:-1]

                obs_batch = []

                if return_history:
                    scores = []
                    rewards = []

                for i,offset in enumerate(start_indices):
                    actions[i]-=offset

                actions_iter=iter(actions)
                i = 0
            
                # for env, action in zip(test_envs,actions):
                for env in test_envs:
                    # action=action-start_indices[i]
                    if env is not None:
                        action=next(actions_iter)

                        obs, rew, done, info = env.step(action)

                        if return_history:
                            scores.append(env.score)
                            rewards.append(rew)

                        if not done:
                            obs_batch.append(obs)
                        else:
                            best_cuts_batch[i] = env.get_best_cut()
                            best_spins_batch[i] = env.best_spins
                            i_comp_batch += 1
                            i_comp += 1
                            test_envs[i] = None
                    i+=1
                    k+=1

                if return_history:
                    actions_history_batch.append(actions)
                    scores_history_batch.append(scores)
                    rewards_history_batch.append(rewards)

                # print("\t",
                #       "Par. steps :", k,
                #       "Env steps : {}/{}".format(k/batch_size,n_steps),
                #       'Time: {0:.3g}s'.format(time.time()-t1))

            t_total += (time.time() - t_start)
            i_batch+=1
            print("Finished agent testing batch {}.".format(i_batch))



            if return_history:
                actions_history += actions_history_batch
                rewards_history += rewards_history_batch
                scores_history += scores_history_batch

            best_cuts += best_cuts_batch
            init_spins += init_spins_batch
            best_spins += best_spins_batch




            # print("\tGraph {}, par. steps: {}, comp: {}/{}".format(j, k, i_comp, batch_size),
            #       end="\r" if n_spins<100 else "")

        i_best = np.argmax(best_cuts)
        best_cut = best_cuts[i_best]
        sol = best_spins[i_best]

        mean_cut = np.mean(best_cuts)



        print('Graph {}, best(mean) cut: {}({}).  ({} attempts in {}s)\t\t\t'.format(
                j, best_cut, mean_cut,n_attempts, np.round(t_total,2)))

       
        
        results.append([best_cut, sol,
                        mean_cut,
                        t_total/(n_attempts)])

        
        results_raw.append([init_spins,
                            best_cuts, best_spins
                            ])

        # if return_history:
        #     history.append([np.array(actions_history).T.tolist(),
        #                     np.array(scores_history).T.tolist(),
        #                     np.array(rewards_history).T.tolist()])
        if return_history:
            history.append([actions_history,
                            scores_history,
                            rewards_history])


    
    results = pd.DataFrame(data=results, columns=["cut", "sol",
                                                "mean cut",
                                                "time"])

    results_raw = pd.DataFrame(data=results_raw, columns=["init spins",
                                                        "cuts", "sols"])


    if return_history:
        history = pd.DataFrame(data=history, columns=["actions", "scores", "rewards"])

    if return_raw==False and return_history==False:
        return results
    else:
        ret = [results]
        if return_raw:
            ret.append(results_raw)
        if return_history:
            ret.append(history)
        return ret


####################################################
# LOADING GRAPHS
####################################################

Graph = namedtuple('Graph', 'name n_vertices n_edges matrix bk_val bk_sol')



def load_graph_set(graph_save_loc):
    graphs_test = pickle.load(open(graph_save_loc,'rb'))

    def graph_to_array(g):
        if type(g) == nx.Graph:
            g = nx.to_numpy_array(g)
        elif type(g) == sp.sparse.csr_matrix:
            g = g.toarray()
        return g

    graphs_test = [graph_to_array(g) for g in graphs_test]
    print('{} target graphs loaded from {}'.format(len(graphs_test), graph_save_loc))
    return graphs_test







    




####################################################
# FILE UTILS
####################################################

def mk_dir(export_dir, quite=False):
    if not os.path.exists(export_dir):
            try:
                os.makedirs(export_dir)
                print('created dir: ', export_dir)
            except OSError as exc: # Guard against race condition
                 if exc.errno != exc.errno.EEXIST:
                    raise
            except Exception:
                pass
    else:
        print('dir already exists: ', export_dir)
