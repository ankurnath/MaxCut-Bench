import math
import pickle
import random
import threading
from collections import namedtuple
from enum import Enum

import numpy as np
import torch





Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'state_next', 'done')
)

class TestMetric(Enum):

    CUMULATIVE_REWARD = 1
    BEST_ENERGY = 2
    ENERGY_ERROR = 3
    MAX_CUT = 4
    FINAL_CUT = 5
    KNAPSACK=6
    MAXSAT=7
    MAXCOVER=8

def set_global_seed(seed, env):
    torch.manual_seed(seed)
    env.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    


class ReplayBuffer:
    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = {}
        self._position = 0

        self.next_batch_process=None
        self.next_batch_size=None
        self.next_batch_device=None
        self.next_batch = None

    def add(self, *args):
        """
        Saves a transition.
        """
        if self.next_batch_process is not None:
            # Don't add to the buffer when sampling from it.
            self.next_batch_process.join()
        self._memory[self._position] = Transition(*args)
        self._position = (self._position + 1) % self._capacity
        


    def _prepare_sample(self, batch_size, device=None):
        
        self.next_batch_size = batch_size
        self.next_batch_device = device

        batch = random.sample(list(self._memory.values()), batch_size)

        self.next_batch = [torch.stack(tensors) if torch.is_tensor(tensors[0]) else 
                           tensors for tensors in zip(*batch)]
        self.next_batch_ready = True

    def launch_sample(self, *args):
        self.next_batch_process = threading.Thread(target=self._prepare_sample, args=args)
        self.next_batch_process.start()

    def sample(self, batch_size, device=None):
        """
        Samples a batch of Transitions, with the tensors already stacked
        and transfered to the specified device.
        Return a list of tensors in the order specified in Transition.
        """
        if self.next_batch_process is not None:
            self.next_batch_process.join()
        else:
            self.launch_sample(batch_size, device)
            self.sample(batch_size, device)

        if self.next_batch_size==batch_size and self.next_batch_device==device:
            next_batch = self.next_batch
            self.launch_sample(batch_size, device)
            return next_batch
        else:
            self.launch_sample(batch_size, device)
            self.sample(batch_size, device)

    def __len__(self):
        return len(self._memory)




class Logger:
    def __init__(self):
        self._memory = {}
        self._saves = 0
        self._maxsize = 1000000
        self._dumps = 0

    def add_scalar(self, name, data, timestep):
        """
        Saves a scalar
        """
        if isinstance(data, torch.Tensor):
            data = data.item()

        self._memory.setdefault(name, []).append([data, timestep])

        self._saves += 1
        if self._saves == self._maxsize - 1:
            with open('log_data_' + str((self._dumps + 1) * self._maxsize) + '.pkl', 'wb') as output:
                pickle.dump(self._memory, output, pickle.HIGHEST_PROTOCOL)
            self._dumps += 1
            self._saves = 0
            self._memory = {}

    def save(self):
        with open('log_data.pkl', 'wb') as output:
            pickle.dump(self._memory, output, pickle.HIGHEST_PROTOCOL)
