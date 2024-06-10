import random
from abc import ABC, abstractmethod
from enum import Enum

import networkx as nx
import numpy as np

from scipy.spatial.distance import pdist, squareform

class EdgeType(Enum):

    UNIFORM = 1
    DISCRETE = 2
    RANDOM = 3

class RewardSignal(Enum):

    DENSE = 1
    BLS = 2
    SINGLE = 3
    CUSTOM_BLS = 4
    NEGATIVE_DENSE=5

class ExtraAction(Enum):

    PASS = 1
    RANDOMISE = 2
    NONE = 3
    DONE = 4

class OptimisationTarget(Enum):

    CUT = 1
    ENERGY = 2
    COLOR = 3

class SpinBasis(Enum):

    SIGNED = 1
    BINARY = 2

class Observable(Enum):
    # Local observations that differ between nodes.
    SPIN_STATE = 1
    IMMEDIATE_REWARD_AVAILABLE = 2
    TIME_SINCE_FLIP = 3

    # Global observations that are the same for all nodes.
    EPISODE_TIME = 4
    TERMINATION_IMMANENCY = 5
    NUMBER_OF_GREEDY_ACTIONS_AVAILABLE = 6
    DISTANCE_FROM_BEST_SCORE = 7
    DISTANCE_FROM_BEST_STATE = 8

DEFAULT_OBSERVABLES = [Observable.SPIN_STATE,
                       Observable.IMMEDIATE_REWARD_AVAILABLE,
                       Observable.TIME_SINCE_FLIP,
                       Observable.DISTANCE_FROM_BEST_SCORE,
                       Observable.DISTANCE_FROM_BEST_STATE,
                       Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE,
                       Observable.TERMINATION_IMMANENCY]

class GraphGenerator(ABC):

    def __init__(self, n_spins, edge_type):

        self.n_spins = n_spins
        self.edge_type = edge_type

        if type(n_spins) in [list,tuple]:
            assert len(n_spins)==2 and n_spins[1]>n_spins[0],"Invalid range of number of nodes."
            self.get_spin=lambda : np.random.randint(n_spins[0],n_spins[1]+1)
        else:
           self.get_spin=lambda: self.n_spins

        

        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda n: np.ones((n,n))
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask(n):
                mask = 2. * np.random.randint(2, size=(n,n)) - 1.
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask
            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask(n):
                # mask = 2.*np.random.rand(n,n)-1
                mask = np.random.rand(n,n)
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask
            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()
        
    @abstractmethod
    def get(self):
        raise NotImplementedError

###################
# Unbiased graphs #
###################
    
class RandomCompleteGraphGenerator(object):

    def __init__(self,n_spins,dimension) -> None:

        self.n_spins=n_spins
        self.dimension=dimension


    def get(self):

        # Generate num_nodes uniformly sampled nodes in [0, 1] in num_dimensions
        nodes = np.random.rand(self.n_spins, self.dimension)

        # Calculate pairwise Euclidean distances
        adjacency_matrix = squareform(pdist(nodes))

        return adjacency_matrix
        

class RandomErdosRenyiGraphGenerator(GraphGenerator):

    def __init__(self, n_spins,p_connection, edge_type):

        super().__init__(n_spins, edge_type)
        self.p_connection=p_connection


    def get(self):
        n=self.get_spin()
        g = nx.erdos_renyi_graph(n, self.p_connection)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask(n))
        # No self-connections (this modifies adj in-place).
        np.fill_diagonal(adj, 0)
        return adj

class RandomBarabasiAlbertGraphGenerator(GraphGenerator):

    def __init__(self, n_spins, m_insertion_edges,edge_type):
        super().__init__(n_spins, edge_type)

        self.m_insertion_edges = m_insertion_edges

    def get(self):

        n=self.get_spin()
        g = nx.barabasi_albert_graph(n, self.m_insertion_edges)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask(n))
        # No self-connections (this modifies adj in-place).
        np.fill_diagonal(adj, 0)
        return adj
    
class RandomRegularGraphGenerator(GraphGenerator):

    def __init__(self, n_spins, d,edge_type):
        super().__init__(n_spins, edge_type)

        # self.m_insertion_edges = m_insertion_edges
        self.d=d

    def get(self):

        n=self.get_spin()
        g=nx.random_regular_graph(d=self.d,n=self.n_spins)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask(n))
        np.fill_diagonal(adj, 0)
        return adj


class RandomWattsStrogatzGraphGenerator(GraphGenerator):

    def __init__(self,n_spins,k,p, edge_type):
        super().__init__(n_spins, edge_type)
        self.k  = k
        self.p  = p
    def get(self):
        
        n=self.get_spin()
        
        g= nx.watts_strogatz_graph(n=n, k=self.k, p=self.p)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask(n))
        np.fill_diagonal(adj, 0)

        return adj
    
class RandomHomleKimGraphGenerator(GraphGenerator):

    def __init__(self,n_spins,m,p, edge_type):
        super().__init__(n_spins, edge_type)
        self.m  = m
        self.p  = p
    def get(self):
        
        n=self.get_spin()
        
        # g= nx.watts_strogatz_graph(n=n, k=self.k, p=self.p)
        g=nx.powerlaw_cluster_graph(n=n, m=self.m, p=self.p)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask(n))
        np.fill_diagonal(adj, 0)

        return adj





        
################
# Known graphs #
################
class SingleGraphGenerator(object):

    def __init__(self, matrix):
        self.matrix = matrix
        

    def get(self):

        return self.matrix
    
import glob
import numpy as np
from scipy.sparse import load_npz
class GraphDataset(object):

    def __init__(self,folder_path,ordered=False):
        super().__init__()

        self.file_paths=glob.glob(f'{folder_path}/*.npz')
        self.file_paths.sort()
        self.ordered=ordered

        if self.ordered:
            self.i = 0

    def __len__(self):
        return len(self.file_paths)
    
    def get(self):
        if self.ordered:
            file_path = self.file_paths[self.i]
            self.i = (self.i + 1)%len(self.file_paths)
        else:
            file_path = random.sample(self.file_paths, k=1)[0]
        return load_npz(file_path).toarray()





class SetGraphGenerator(object):

    def __init__(self, matrices, ordered=False):
        super().__init__()
        self.graphs = matrices
    
        self.ordered = ordered
        if self.ordered:
            self.i = 0

    def get(self):
        if self.ordered:
            m = self.graphs[self.i]
            self.i = (self.i + 1)%len(self.graphs)
        else:
            m = random.sample(self.graphs, k=1)[0]
        return m


class HistoryBuffer():
    def __init__(self):
        self.buffer = {}
        self.current_action_hist = set([])
        self.current_action_hist_len = 0

    def update(self, action):
        new_action_hist = self.current_action_hist.copy()
        if action in self.current_action_hist:
            new_action_hist.remove(action)
            self.current_action_hist_len -= 1
        else:
            new_action_hist.add(action)
            self.current_action_hist_len += 1

        try:
            list_of_states = self.buffer[self.current_action_hist_len]
            if new_action_hist in list_of_states:
                self.current_action_hist = new_action_hist
                return False
        except KeyError:
            list_of_states = []

        list_of_states.append(new_action_hist)
        self.current_action_hist = new_action_hist
        self.buffer[self.current_action_hist_len] = list_of_states
        return True