"""
Implements a DQN learning agent.
"""

import os
import pickle
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agents.dqn.utils import ReplayBuffer, Logger, TestMetric, set_global_seed
from src.envs.utils import ExtraAction

from torch_geometric.data import Batch
import time
# from torch_geometric.nn.pool import global_max_pool


class DQN:
    """
    # Required parameters.
    
    env : environment to use.
    network : Choice of neural network.

    # Initial network parameters.
    init_network_params : Pre-trained network to load upon initialisation.
    init_weight_std : Standard deviation of initial network weights.

    # DQN parameters
    double_dqn : Whether to use double DQN (DDQN).
    update_target_frequency : How often to update the DDQN target network.
    gamma : Discount factor.
    clip_Q_targets : Whether negative Q targets are clipped (generally True/False for irreversible/reversible agents).

    # Replay buffer.
    replay_start_size : The capacity of the replay buffer at which training can begin.
    replay_buffer_size : Maximum buffer capacity.
    minibatch_size : Minibatch size.
    update_frequency : Number of environment steps taken between parameter update steps.

    # Learning rate
    update_learning_rate : Whether to dynamically update the learning rate (if False, initial_learning_rate is always used).
    initial_learning_rate : Initial learning rate.
    peak_learning_rate : The maximum learning rate.
    peak_learning_rate_step : The timestep (from the start, not from when training starts) at which the peak_learning_rate is found.
    final_learning_rate : The final learning rate.
    final_learning_rate_step : The timestep of the final learning rate.

    # Optional regularization.
    max_grad_norm : The norm grad to clip gradients to (None means no clipping).
    weight_decay : The weight decay term for regularisation.

    # Exploration
    update_exploration : Whether to update the exploration rate (False would tend to be used with NoisyNet layers).
    initial_exploration_rate : Inital exploration rate.
    final_exploration_rate : Final exploration rate.
    final_exploration_step : Timestep at which the final exploration rate is reached.

    # Loss function
    adam_epsilon : epsilon for ADAM optimisation.
    loss="mse" : Loss function to use.

    # Saving the agent
    save_network_frequency : Frequency with which the network parameters are saved.
    network_save_path : Folder into which the network parameters are saved.

    # Testing the agent
    evaluate : Whether to test the agent during training.
    test_envs : List of test environments.  None means the training environments (envs) are used.
    test_episodes : Number of episodes at each test point.
    test_frequency : Frequency of tests.
    test_save_path : Folder into which the test scores are saved.
    test_metric : The metric used to quantify performance.

    # Other
    logging : Whether to log.
    seed : The global seed to set.  None means randomly selected.
    """
    def __init__(
        self,
        env,
        network,

        # Initial network parameters.
        init_network_params = None,
        init_weight_std = None,

        # DQN parameters
        double_dqn = True,
        update_target_frequency=10000,
        gamma=0.99,
        clip_Q_targets=False,

        # Replay buffer.
        replay_start_size=50000,
        replay_buffer_size=1000000,
        minibatch_size=32,
        update_frequency=1,

        # Learning rate
        update_learning_rate = True,
        initial_learning_rate = 0,
        peak_learning_rate = 1e-3,
        peak_learning_rate_step = 10000,
        final_learning_rate = 5e-5,
        final_learning_rate_step = 200000,

        # Optional regularization.
        max_grad_norm=None,
        weight_decay=0,

        # Exploration
        update_exploration=True,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        final_exploration_step=1000000,

        # Loss function
        adam_epsilon=1e-8,
        loss="mse",

        # Saving the agent
        save_network_frequency=10000,
        network_save_path='network',

        # Testing the agent
        evaluate=True,
        test_env=None,
        test_episodes=20,
        test_frequency=10000,
        test_save_path='test_scores',
        test_metric=TestMetric.ENERGY_ERROR,

        # Other
        logging=True,
        seed=None,
        device=None
    ):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device
        

        self.double_dqn = double_dqn

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.clip_Q_targets = clip_Q_targets
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size

        self.update_learning_rate = update_learning_rate
        self.initial_learning_rate = initial_learning_rate
        self.peak_learning_rate = peak_learning_rate
        self.peak_learning_rate_step = peak_learning_rate_step
        self.final_learning_rate = final_learning_rate
        self.final_learning_rate_step = final_learning_rate_step

        self.max_grad_norm = max_grad_norm
        self.weight_decay=weight_decay
        self.update_frequency = update_frequency
        self.update_exploration = update_exploration,
        self.initial_exploration_rate = initial_exploration_rate
        self.epsilon = self.initial_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.final_exploration_step = final_exploration_step
        self.adam_epsilon = adam_epsilon
        self.logging = logging
        if callable(loss):
            self.loss = loss
        else:
            try:
                self.loss = {'huber': F.smooth_l1_loss, 'mse': F.mse_loss}[loss]
            except KeyError:
                raise ValueError("loss must be 'huber', 'mse' or a callable")


        self.env, self.acting_in_reversible_spin_env  = env,env.reversible_spins

        self.replay_buffer=ReplayBuffer(self.replay_buffer_size)

        self.seed = random.randint(0, 1e6) if seed is None else seed

        set_global_seed(self.seed, self.env)

        self.network = network().to(self.device)
        self.init_network_params = init_network_params
        self.init_weight_std = init_weight_std
        if self.init_network_params != None:
            print("Pre-loading network parameters from {}.\n".format(init_network_params))
            self.load(init_network_params)
        else:
            if self.init_weight_std != None:
                def init_weights(m):
                    if type(m) == torch.nn.Linear:
                        print("Setting weights for", m)
                        m.weight.normal_(0, init_weight_std)
                with torch.no_grad():
                    self.network.apply(init_weights)

        self.target_network = network().to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.network.parameters(), 
                                    lr=self.initial_learning_rate, 
                                    eps=self.adam_epsilon,
                                    weight_decay=self.weight_decay)

        self.evaluate = evaluate
        self.test_env=test_env
        self.test_episodes = int(test_episodes)
        self.test_frequency = test_frequency
        self.test_save_path = test_save_path
        self.test_metric = test_metric

        self.losses_save_path = os.path.join(os.path.split(self.test_save_path)[0], "losses.pkl")

        if not self.acting_in_reversible_spin_env:
            assert self.env.extra_action==ExtraAction.NONE, "For deterministic MDP, no extra action is allowed."
            assert self.test_env.extra_action==ExtraAction.NONE, "For deterministic MDP, no extra action is allowed."


        self.allowed_action_state = self.env.get_allowed_action_states()

        self.save_network_frequency = save_network_frequency
        self.network_save_path = network_save_path



    def learn(self, timesteps, verbose=False):

        if self.logging:
            logger = Logger()


        self.env.train()

        state=self.env.reset() # data object
        score = 0
        losses_eps = []
        t1 = time.time()

        test_scores = []
        losses = []

        is_training_ready = False

        for timestep in range(timesteps):

            if not is_training_ready:

                if len(self.replay_buffer)>=self.replay_start_size :
                    print('\n The buffer has {} transitions stored - training is starting!\n'.format(
                        self.replay_start_size))
                    is_training_ready=True

                
            _state=Batch.from_data_list([state], follow_batch=None, exclude_keys=None)
            action = self.act(_state, is_training_ready=is_training_ready)

            if torch.is_tensor(action):
                action=action.item()

            # Update epsilon
            if self.update_exploration:
                self.update_epsilon(timestep)

            # Update learning rate
            if self.update_learning_rate:
                self.update_lr(timestep)

            # Perform action in environment
            # print(action)
            state_next, reward, done, _ = self.env.step(action)

            score += reward

            

            # Store transition in replay buffer
            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)


            done = torch.as_tensor([done], dtype=torch.float)


            self.replay_buffer.add(state.cpu(), action, reward, state_next, done)

            if done:
                # Reinitialise the state
                if verbose:
                    loss_str = "{:.2e}".format(np.mean(losses_eps)) if is_training_ready else "N/A"
                    print("timestep : {}, episode time: {}, score : {}, mean loss: {}, time : {} s".format(
                        (timestep+1),
                         self.env.current_step,
                         np.round(score,3),
                         loss_str,
                         round(time.time() - t1, 3)))

                if self.logging:
                    logger.add_scalar('Episode_score', score, timestep)

                state=self.env.reset()
                score = 0
                losses_eps = []
                t1 = time.time()

            else:
                state = state_next

            if is_training_ready:
                
                
                # Update the main network
                if timestep % self.update_frequency == 0:
                    start_time = time.perf_counter()
                    # Sample a batch of transitions
                    
                    transitions = self.replay_buffer.sample(self.minibatch_size, self.device)

                    # Train on selected batch
                    loss = self.train_step(transitions)
                    losses.append([timestep,loss])
                    losses_eps.append(loss)

                    if self.logging:
                        logger.add_scalar('Loss', loss, timestep)
                    end_time = time.perf_counter()

                    elapsed_time = end_time - start_time

                    # print(f"Elapsed Time to update the main network: {elapsed_time} seconds")


                # Periodically update target network
                if timestep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            if (timestep+1) % self.test_frequency == 0 and self.evaluate and is_training_ready:
                start_time = time.perf_counter()
                test_score = self.evaluate_agent()
                end_time = time.perf_counter()

                elapsed_time = end_time - start_time
                print(f"Elapsed Time to evalute network: {elapsed_time} seconds")
                print('\nTest score: {}\n'.format(np.round(test_score,3)))

                if self.test_metric in [TestMetric.FINAL_CUT,TestMetric.MAX_CUT,TestMetric.CUMULATIVE_REWARD,TestMetric.KNAPSACK,TestMetric.MAXSAT,TestMetric.MAXCOVER]:
                    best_network = all([test_score > score for t,score in test_scores])
                elif self.test_metric in [TestMetric.ENERGY_ERROR, TestMetric.BEST_ENERGY]:
                    best_network = all([test_score < score for t, score in test_scores])
                else:
                    raise NotImplementedError("{} is not a recognised TestMetric".format(self.test_metric))

                if best_network:
                    path = self.network_save_path
                    path_main, path_ext = os.path.splitext(path)
                    path_main += "_best"
                    if path_ext == '':
                        path_ext += '.pth'
                    self.save(path_main + path_ext)

                test_scores.append([timestep+1,test_score])

            # if (timestep + 1) % self.save_network_frequency == 0 and is_training_ready:
            #     path = self.network_save_path
            #     path_main, path_ext = os.path.splitext(path)
            #     path_main += str(timestep+1)
            #     if path_ext == '':
            #         path_ext += '.pth'
            #     self.save(path_main+path_ext)

        if self.logging:
            logger.save()

        path = self.test_save_path
        if os.path.splitext(path)[-1] == '':
            path += '.pkl'

        with open(path, 'wb+') as output:
            pickle.dump(np.array(test_scores), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('test_scores saved to {}'.format(path))

        with open(self.losses_save_path, 'wb+') as output:
            pickle.dump(np.array(losses), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('losses saved to {}'.format(self.losses_save_path))


    @torch.no_grad()
    def __only_bad_actions_allowed(self, state, network):
        x = (state[0, :] == self.allowed_action_state).nonzero()
        q_next = network(state.to(self.device).float())[x].max()
        return True if q_next < 0 else False
    
    @staticmethod
    def get_greedy_actions(pred=None,batch=None,offset=True):
        # print(batch)
            
        num_graphs = batch[-1].item() + 1
        # graph_sortidx=torch.argsort(batch).type(torch.int64)
        graph_ids, graph_counts = torch.unique_consecutive(batch,
                                                           return_counts=True)
        
        # if not offset:
        
        #     print(graph_counts)
        
        end_indices = torch.cumsum(graph_counts, dim=0).cpu().tolist()
        start_indices = [0] + end_indices[:-1]
        # if not offset:
        #     print(start_indices)
        greedy_actions = torch.zeros((num_graphs,), dtype=torch.int64)
        
        for graph_id, a, b in zip(graph_ids, start_indices, end_indices):
            # indices = graph_sortidx[a:b]
            # print(indices)
            # print(x)
            # greedy_actions[graph_id] = indices[torch.argmax(pred[indices])]
            if offset:
                greedy_actions[graph_id] = torch.argmax(pred[a:b])+a
            else:
                greedy_actions[graph_id] = torch.argmax(pred[a:b])
        
        # if not offset:
        #     print('Size of graphs',graph_counts)
        #     print('Greedy actions',greedy_actions)


        # assert torch.equal(global_max_pool(pred,batch),pred[greedy_actions])
        # if not offset:
        
        #     print(greedy_actions)
        
            
        return greedy_actions

    def train_step(self, transitions):

        states, actions, rewards, states_next, dones = transitions
        
        states=Batch.from_data_list(states, follow_batch=None, exclude_keys=None)

        states_next=Batch.from_data_list(states_next, follow_batch=None, exclude_keys=None)

        
        
        

        if self.acting_in_reversible_spin_env:
            # Calculate target Q
            with torch.no_grad():
                if self.double_dqn:
                    
                    network_preds=self.network(states_next.to(self.device))
                    greedy_actions = DQN.get_greedy_actions(network_preds,states_next.batch).to(self.device)
                    q_value_target = self.target_network(states_next.to(self.device))[greedy_actions]
                    
                    

                else:
#                   
                    q_value_target = self.target_network(states_next.to(self.device)).max(1, True)[0]

        else:

            target_preds = self.target_network(states_next.to(self.device))
            disallowed_actions_mask = (states_next.x[:,0] != self.allowed_action_state).unsqueeze(-1)
            
            # Calculate target Q, selecting ONLY ALLOWED ACTIONS greedily.
            with torch.no_grad():
                if self.double_dqn:
#                     network_preds = self.network(states_next.float())
                    network_preds = self.network(states_next.to(self.device))
                    # print(network_preds.shape)
                    # Set the Q-value of disallowed actions to a large negative number (-10000) so they are not selected.
                    network_preds_allowed = network_preds.masked_fill(disallowed_actions_mask,-10000)
                    # print(network_preds_allowed.shape)
                    greedy_actions = DQN.get_greedy_actions(network_preds_allowed,states_next.batch)
#                     greedy_actions = network_preds_allowed.argmax(1, True)
                    # q_value_target = target_preds.gather(1, greedy_actions)
                    q_value_target = target_preds[greedy_actions]
                else:
                    q_value_target = target_preds.masked_fill(disallowed_actions_mask,-10000).max(1, True)[0]

        if self.clip_Q_targets:
            q_value_target[q_value_target < 0] = 0

        # Calculate TD target
            
        rewards=rewards.to(self.device)
        dones=dones.to(self.device)
        # print(rewards.device)
        # print(dones.device)
        # print(q_value_target.device)
        # print(self.gamma.device)

        
        td_target = rewards + (1 - dones) * self.gamma * q_value_target

        # Calculate Q value

        

        
        # graph_sortidx=torch.argsort(states.batch)
        _, graph_counts = torch.unique_consecutive(states.batch,
                                                    return_counts=True)
        
        end_indices = torch.cumsum(graph_counts, dim=0).cpu().tolist()
        start_indices = torch.tensor([0] + end_indices[:-1]).type(torch.int)
        q_value = self.network(states.to(self.device))[actions.squeeze(-1)+start_indices]
        
        

        # Calculate loss
        loss = self.loss(q_value, td_target, reduction='mean')

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm is not None: #Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def act(self, state, is_training_ready=True):
        if is_training_ready and random.uniform(0, 1) >= self.epsilon:
            # Action that maximises Q function
            action = self.predict(state)
        else:
            if self.acting_in_reversible_spin_env:
                # Random random spin.
                action = np.random.randint(0, self.env.action_space.n)
            else:
                # Flip random spin from that hasn't yet been flipped.
                # print(self.allowed_action_state)
                x = (state.x[:,0] == self.allowed_action_state).nonzero()
                # print(state.x[:,0])
                # print(x)
                action = x[np.random.randint(0, len(x))].item()
        return action

    def update_epsilon(self, timestep):
        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (
            timestep / self.final_exploration_step
        )
        self.epsilon = max(eps, self.final_exploration_rate)

    def update_lr(self, timestep):
        if timestep <= self.peak_learning_rate_step:
            lr = self.initial_learning_rate - (self.initial_learning_rate - self.peak_learning_rate) * (
                    timestep / self.peak_learning_rate_step
                )
        elif timestep <= self.final_learning_rate_step:
            lr = self.peak_learning_rate - (self.peak_learning_rate - self.final_learning_rate) * (
                    (timestep - self.peak_learning_rate_step) / (self.final_learning_rate_step - self.peak_learning_rate_step)
                )
        else:
            lr = None

        if lr is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = lr


    @torch.no_grad()
    def predict(self, states, acting_in_reversible_spin_env=None):

        if acting_in_reversible_spin_env is None:
            acting_in_reversible_spin_env = self.acting_in_reversible_spin_env

        qs = self.network(states.to(self.device))

        if acting_in_reversible_spin_env:
            actions=DQN.get_greedy_actions(qs,states.batch,offset=False)
            return actions
        else:
            disallowed_actions_mask = (states.x[:, 0] != self.allowed_action_state).unsqueeze(-1)
            qs_allowed = qs.masked_fill(disallowed_actions_mask, -10000)
            actions=DQN.get_greedy_actions(qs_allowed,states.batch,offset=False)
            if torch.is_tensor(actions):
                actions=actions.cpu()
            return actions

    @torch.no_grad()
    def evaluate_agent(self, batch_size=None):
        """
        Evaluates agent's current performance.  Run multiple evaluations at once
        so the network predictions can be done in batches.
        """

        print('Started evalution')
        # print('Test episodes',self.test_episodes)
        if batch_size is None:
            batch_size = self.minibatch_size

        i_test = 0
        i_comp = 0
        test_scores = []
        batch_scores = [0]*batch_size

        test_envs = np.array([None]*batch_size)
        
        # actions_envs=np.array([None]*batch_size)
        obs_batch = []
        mapping={}
        # flag=True
        

        while i_comp < self.test_episodes:

            for i, env in enumerate(test_envs):
                if env is None and i_test < self.test_episodes:
#                     test_env, testing_in_reversible_spin_env = self.get_random_env(self.test_envs)
                    # test_env, testing_in_reversible_spin_env = self.test_env,self.test_env.reversible_spins
                    # this gets a new graph
                    # print('get new observation')
                    new_obs = self.test_env.reset(test=True)
                    # print(new_obs)
                    test_env = deepcopy(self.test_env)
                    # test_env.test()

                    test_envs[i] = test_env
                    
                    obs_batch.append(new_obs)
                    mapping[i]=len(obs_batch)-1

                    i_test += 1


            graph_batch=Batch.from_data_list(obs_batch, follow_batch=None, exclude_keys=None).to(self.device)
            # print(graph_batch.batch)
            # print('*'*20)
            # for temp in test_envs:
            #     try:
            #         print(temp.n_spins,end=',')
            #     except:
            #         print('None',end=',')
            # print()
            
            actions = self.predict(graph_batch,self.test_env.reversible_spins)

            if torch.is_tensor(actions):
                actions=actions.tolist()

            # try:
            #     action_iter = iter(actions)
            #     # start_indices_iter = iter(start_indices) 
            # except:
            #     action_iter = iter([actions])
            #     # start_indices_iter = iter(start_indices)
                

            # for i, test_env in enumerate(test_envs):
            #     actions_envs[i] = next(action_iter) if test_env is not None else None



            obs_batch = []

            i = 0

            # for env, action in zip(test_envs, actions_envs):
            for idx,env in enumerate(test_envs):
                if env is not None:

                    # if env.n_spins<action+1:

                    #     for temp in test_envs:
                    #         try:
                    #             print(temp.n_spins,end=',')
                    #         except:
                    #             print('None',end=',')
                    #     print('')
                    #     print('Number of vertices',env.n_spins)
                    #     print('action',action)
                    #     raise ValueError('Wrong Environment')
                    action=actions[mapping[idx]]
                    obs, rew, done, info = env.step(action)

                    if self.test_metric == TestMetric.CUMULATIVE_REWARD:
                        batch_scores[i] += rew

                    if done:
                        # print('DONE')
                        if self.test_metric == TestMetric.BEST_ENERGY:
                            batch_scores[i] = env.best_energy
                        elif self.test_metric == TestMetric.ENERGY_ERROR:
                            batch_scores[i] = abs(env.best_energy - env.calculate_best()[0])
                        elif self.test_metric == TestMetric.MAX_CUT:
                            batch_scores[i] = env.get_best_cut()

                        elif self.test_metric==TestMetric.KNAPSACK:
                            batch_scores[i] = env.get_best_cut()
                            
                 
                        elif self.test_metric == TestMetric.FINAL_CUT:
                            batch_scores[i] = env.calculate_cut()
                            
                        elif self.test_metric == TestMetric.MAXSAT:
                            batch_scores[i] = env.get_best_cut()
                            
                        elif self.test_metric==TestMetric.MAXCOVER:
                            batch_scores[i] = env.get_best_cut()
                            
                            
                        else:
                            
                            raise NotImplementedError()
                        

                        test_scores.append(batch_scores[i])

                        if self.test_metric == TestMetric.CUMULATIVE_REWARD:
                            batch_scores[i] = 0

                        i_comp += 1
                        test_envs[i] = None
                    else:

                        obs_batch.append(obs)
                        mapping[idx]=len(obs_batch)-1

                i += 1

            # if flag:
            #     end_time=time.perf_counter()
            #     print('Elapsed time:',end_time-start_time)
            #     flag=False

        print(test_scores)
        return np.mean(test_scores)

    def save(self, path='network.pth'):
        if os.path.splitext(path)[-1]=='':
            path + '.pth'
        torch.save(self.network.state_dict(), path)

    def load(self,path):
        self.network.load_state_dict(torch.load(path,map_location=self.device))
