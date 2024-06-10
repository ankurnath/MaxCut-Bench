from abc import ABC, abstractmethod
from collections import namedtuple
from operator import matmul

import numpy as np
# import torch.multiprocessing as mp
# from numba import jit, float64, int64


import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from src.envs.utils import (
                            RewardSignal,
                            ExtraAction,
                            OptimisationTarget,
                            Observable,
                            SpinBasis,
                            DEFAULT_OBSERVABLES,
                            HistoryBuffer)

# A container for get_result function below. Works just like tuple, but prettier.
ActionResult = namedtuple("action_result", ("snapshot","observation","reward","is_done","info"))

class SpinSystemFactory(object):
    '''
    Factory class for returning new SpinSystem.
    '''

    @staticmethod
    def get(graph_generator=None,
            step_fact=2,
            observables = DEFAULT_OBSERVABLES,
            reward_signal = RewardSignal.DENSE,
            extra_action = ExtraAction.PASS,
            optimisation_target = OptimisationTarget.ENERGY,
            spin_basis = SpinBasis.SIGNED,
            norm_rewards=False,
            # memory_length=None,  # None means an infinite memory.            
            stag_punishment=None, # None means no punishment for re-visiting states.
            basin_reward=None, # None means no reward for reaching a local minima.
            reversible_spins=True, # Whether the spins can be flipped more than once (i.e. True-->Georgian MDP).
            seed=None,
            num_steps=None):

        
        return SpinSystem(          graph_generator,step_fact,
                                    observables,reward_signal,
                                    extra_action,optimisation_target,
                                    spin_basis,norm_rewards,
                                    # memory_length,
                                    stag_punishment,
                                    basin_reward,reversible_spins,
                                    seed,
                                    num_steps)

class SpinSystem(object):
    '''
    SpinSystemBase implements the functionality of a SpinSystem that is common to both
    biased and unbiased systems.  Methods that require significant enough changes between
    these two case to not readily be served by an 'if' statement are left abstract, to be
    implemented by a specialised subclass.
    '''

    # Note these are defined at the class level of SpinSystem to ensure that SpinSystem
    # can be pickled.
    class get_action_space():
        def __init__(self, n_actions):
            self.n = n_actions
            self.actions = np.arange(self.n)

        def sample(self, n=1):
            return np.random.choice(self.actions, n)

    class get_observation_space():
        def __init__(self, n_spins, n_observables):
            self.shape = [n_spins, n_observables]

    def __init__(self,
                 graph_generator=None,
                 step_fact=2,
                 observables=DEFAULT_OBSERVABLES,
                 reward_signal = RewardSignal.DENSE,
                 extra_action = ExtraAction.PASS,
                 optimisation_target=OptimisationTarget.CUT,
                 spin_basis=SpinBasis.SIGNED,
                 norm_rewards=False,
                 stag_punishment=None,
                 basin_reward=False,
                 reversible_spins=False,
                 seed=None,
                 num_steps=None):
        '''
        Init method.

        Args:
            graph_generator: A GraphGenerator (or subclass thereof) object.
            max_steps: Maximum number of steps before termination.
            reward_signal: RewardSignal enum determining how and when rewards are returned.
            extra_action: ExtraAction enum determining if and what additional action is allowed,
                          beyond simply flipping spins.
            seed: Optional random seed.
        '''

        if seed != None:
            np.random.seed(seed)

        # Ensure first observable is the spin state.
        # This allows us to access the spins as self.state[0,:self.n_spins.]
        assert observables[0] == Observable.SPIN_STATE, "First observable must be Observation.SPIN_STATE."
        

        self.observables = list(enumerate(observables))

        self.extra_action = extra_action

        self.gg = graph_generator
        self.step_fact=step_fact
        

        self.reward_signal = reward_signal
        self.norm_rewards = norm_rewards
        self.extra_action=extra_action
        

        self.optimisation_target = optimisation_target
        self.spin_basis = spin_basis

        self.stag_punishment = stag_punishment
        self.basin_reward_flag = basin_reward
        self.reversible_spins = reversible_spins
        self.num_steps=num_steps
        

    
    def adjacency_list_from_numpy_array(self):
        num_nodes = self.matrix.shape[0]
        adjacency_list = {}

        for i in range(num_nodes):
            neighbors = np.where(self.matrix[i] != 0)[0]
            adjacency_list[i] = list(neighbors)
        return adjacency_list



    def reset(self, spins=None,test=False):
        """
        Explanation here
        """

        if test:
            self.test()
        self.current_step = 0
        self.matrix = self.gg.get()
        self.max_local_reward_available=np.max(np.sum(self.matrix,axis=1))


        if self.max_local_reward_available==0:
            self.reset(spins=spins,test=test)

        # self._reset_graph_observables()
        self.adj_list=self.adjacency_list_from_numpy_array()
        

        ### Modification
        self.n_spins =self.matrix.shape[0]

        if self.basin_reward_flag:
            self.basin_reward=1/self.n_spins

        if self.num_steps:
            self.max_steps=self.num_steps
            
        else:
            self.max_steps=self.n_spins*self.step_fact
        # print(self.max_steps)
        self.n_actions = self.n_spins
        if self.extra_action != ExtraAction.NONE:
            self.n_actions+=1

        edge_indices, edge_attr=dense_to_sparse(torch.from_numpy(self.matrix))
        edge_attr=edge_attr.unsqueeze(-1).type(torch.float)


        self.action_space = self.get_action_space(self.n_actions)
        self.observation_space = self.get_observation_space(self.n_spins, len(self.observables))
        self.horizon_length=self.max_steps


        self.max_local_reward_available=np.max(np.sum(self.matrix,axis=1))

       

        state=torch.zeros((self.n_actions,self.observation_space.shape[1]))

        
        

        if self.reversible_spins:
            self.spins=np.random.randint(2,size=(self.n_spins,))
        else:
            self.spins=np.ones((self.n_spins,))

        self.data=Data(x=state,edge_index=edge_indices,edge_attr=edge_attr)

        self.merginal_gain=np.zeros((self.n_spins,))
        self.score=0
        for u in range(self.n_spins):
            for v in self.adj_list[u]:
                self.merginal_gain[u]+=self.matrix[u,v]*(2*self.spins[u]-1)*(2*self.spins[v]-1)
                self.score+=self.matrix[u,v]*(self.spins[u]+self.spins[v]-2*self.spins[u]*self.spins[v])
        self.score/=2

        for idx, obs in self.observables:
            if obs==Observable.SPIN_STATE:
                # The returned tensor and ndarray share the same memory. Modifications to the tensor will be reflected in the ndarray and vice versa. 
                self.data.x[:self.n_spins,idx]=torch.from_numpy(self.spins)

            if obs==Observable.IMMEDIATE_REWARD_AVAILABLE:
                self.data.x[:self.n_spins,idx]=torch.from_numpy(self.merginal_gain/self.max_local_reward_available)
                

            elif obs==Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE:

                self.data.x[:self.n_spins,idx] = 1 - np.sum(self.merginal_gain <= 0) / self.n_spins


        

        self.best_score = self.score
        # self.best_obs_score = self.score
        self.best_spins = self.spins.copy()
        # self.best_obs_spins = state[:self.n_spins,0].clone()

        # spins=2*self.spins-1
        # assert self.score== (1/4) * np.sum( np.multiply( self.matrix, 1 - np.outer(spins, spins) ) )

        if (self.stag_punishment is not None or self.basin_reward_flag) :

            #  We need the buffer only when are training
            self.history_buffer = HistoryBuffer()

        return self.get_observation()



    def seed(self, seed):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)


    def train(self):
        self.testing=False
        self.training=True

    def test(self):
        self.training=False
        self.testing=True

    def step(self, action):
        done = False

        rew = 0 # Default reward to zero.
        # randomised_spins = False
        self.current_step += 1

        if self.current_step > self.max_steps:
            print("The environment has already returned done. Stop it!")
            raise NotImplementedError
        

        # Try to avoid copy

        # new_state = np.copy(self.state)

        ############################################################
        # 1. Performs the action and calculates the score change. #
        ############################################################

        if action==self.n_spins:
            if self.extra_action==ExtraAction.DONE:
                rew=0
                done=True
                return (self.get_observation(), rew, done, None)
 
        else:
            # Perform the action and calculate the score change.

            # if not self.reversible_spins:
            #     assert self.spins[action]==self.get_allowed_action_states()

            delta_score=self.merginal_gain[action]
            # self.score+=delta_score
            self.score+=delta_score
            self.merginal_gain[action]=-self.merginal_gain[action]
            # self.data.x[action,1]=-self.data.x[action,1]
            
            for v in self.adj_list[action]:
                self.merginal_gain[v]+=self.matrix[action,v]*(2*self.spins[v]-1)*(2-4*self.spins[action])

            self.spins[action]=1-self.spins[action]


            # self.data.x[0,action] = 1-self.state[0,action]
            # delta_score = self._calculate_score_change(self.state[0,action], self.matrix, action)
            # self.score += delta_score

        #############################################################################################
        # 2. Calculate reward for action and update anymemory buffers.                              #
        #   a) Calculate reward (always w.r.t best observable score).                              #
        #   b) If new global best has been found: update best ever score and spin parameters.      #
        #   c) If the memory buffer is finite (i.e. self.memory_length is not None):                #
        #          - Add score/spins to their respective buffers.                                  #
        #          - Update best observable score and spins w.r.t. the new buffers.                #
        #      else (if the memory is infinite):                                                    #
        #          - If new best has been found: update best observable score and spin parameters. #                                                                        #
        #############################################################################################

        # self.state = new_state
        # immeditate_rewards_avaialable = self.get_immeditate_rewards_avaialable()

        if self.score > self.best_score:
            if self.reward_signal == RewardSignal.BLS:
                rew = self.score - self.best_score
            elif self.reward_signal == RewardSignal.CUSTOM_BLS:
                rew = self.score - self.best_score
                rew = rew / (rew + 0.1)

        if self.reward_signal == RewardSignal.DENSE:
            rew = delta_score
        elif self.reward_signal == RewardSignal.NEGATIVE_DENSE:
            rew = -delta_score
        


        if self.norm_rewards:
            rew /= self.n_spins

        if (self.stag_punishment is not None or self.basin_reward_flag) and self.training :
            visiting_new_state = self.history_buffer.update(action)

        if self.stag_punishment is not None and self.training:
            if not visiting_new_state:
                rew -= self.stag_punishment

        if self.basin_reward_flag and self.training:
            if np.all(self.merginal_gain <= 0):
                # All immediate score changes are +ive <--> we are in a local minima.
                if visiting_new_state:
                    # #####TEMP####
                    # if self.reward_signal != RewardSignal.BLS or (self.score > self.best_obs_score):
                    # ####TEMP####
                    rew += self.basin_reward

        
        # check again
        if self.score > self.best_score:
            self.best_score = self.score
            # self.best_spins = self.data.x[:self.n_spins,0].clone()
            self.best_spins = self.spins.copy()

        #############################################################################################
        # 3. Updates the state of the system (except self.state[0,:] as this is always the spin     #
        #    configuration and has already been done.                                               #
        #   a) Update self.state local features to reflect the chosen action.                       #                                                                  #
        #   b) Update global features in self.state (always w.r.t. best observable score/spins)     #
        #############################################################################################

        for idx, observable in self.observables:

            ### Local observables ###
            if observable==Observable.SPIN_STATE:
                self.data.x[:self.n_spins,idx] = torch.from_numpy(self.spins)


            if observable==Observable.IMMEDIATE_REWARD_AVAILABLE:
                self.data.x[:self.n_spins,idx] = torch.from_numpy(self.merginal_gain / self.max_local_reward_available)

            if observable==Observable.TIME_SINCE_FLIP:
                self.data.x[:,idx] += (1. / self.max_steps)
                self.data.x[action,idx] = 0

            ### Global observables ###
            elif observable==Observable.EPISODE_TIME:
                self.data.x[:,idx]  += (1. / self.max_steps)

            elif observable==Observable.TERMINATION_IMMANENCY:
                # Update 'Immanency of episode termination'
                self.data.x[:,idx]  = max(0, ((self.current_step - self.max_steps) / self.horizon_length) + 1)

            elif observable==Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE:
                self.data.x[:,idx] = 1 - np.sum(self.merginal_gain <= 0) / self.n_spins

            elif observable==Observable.DISTANCE_FROM_BEST_SCORE:
                self.data.x[:,idx] = np.abs(self.score - self.best_score) / self.max_local_reward_available

            elif observable==Observable.DISTANCE_FROM_BEST_STATE:
                self.data.x[:self.n_spins,idx] = np.count_nonzero(self.best_spins- self.spins)

        # spins=2*self.spins-1
        # assert self.score== (1/4) * np.sum( np.multiply( self.matrix, 1 - np.outer(spins, spins) ) )

        #############################################################################################
        # 4. Check termination criteria.                                                            #
        #############################################################################################
        if self.current_step == self.max_steps:
            # Maximum number of steps taken --> done.
            # print("Done : maximum number of steps taken")
            done = True

        if not self.reversible_spins:
            # if len((self.data.x[:,0] > 0).nonzero()) == 0 or np.all(self.merginal_gain<=0):
            if len((self.data.x[:,0] > 0).nonzero()) == 0 :
                # If no more spins to flip --> done.
                # print("Done : no more spins to flip")
                done = True

        return (self.get_observation(), rew, done, None)

    def get_observation(self):
        if self.training:
            return self.data.clone()
        else:
            return self.data


    def get_allowed_action_states(self):
        if self.reversible_spins:
            # If MDP is reversible, both actions are allowed.
            if self.spin_basis == SpinBasis.BINARY:
                return (0,1)
            else:
                raise NotImplementedError()
        else:
            # If MDP is irreversible, only return the state of spins that haven't been flipped.
            if self.spin_basis==SpinBasis.BINARY:
                return 1
            else:
                raise NotImplementedError()
            
    def get_best_cut(self):
        if self.optimisation_target==OptimisationTarget.CUT:
            return self.best_score
        else:
            raise NotImplementedError("Can't return best cut when optimisation target is set to energy.")

    

