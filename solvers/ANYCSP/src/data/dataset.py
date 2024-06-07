from src.csp.csp_data import CSP_Data
from src.utils.data_utils import load_dimacs_cnf, load_mtx, load_mc
from src.data.xparser import XParser
from scipy.sparse import load_npz
import networkx as nx

# from glob import glob
import glob
import numpy as np
from tqdm import tqdm
import os

import torch
from torch.utils.data import Dataset

import glob
import random
import numpy as np
from scipy.sparse import load_npz
import pandas as pd
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
        return nx.from_numpy_array(load_npz(file_path).toarray())


def load_cnf_file(path):
    cnf = load_dimacs_cnf(path)
    cnf = [np.int64(c) for c in cnf]

    num_var = np.max([np.abs(c).max() for c in cnf])
    num_const = len(cnf)

    arity = np.int64([c.size for c in cnf])
    const_idx = np.arange(0, num_const, dtype=np.int64)
    tuple_idx = np.repeat(const_idx, arity)

    cat = np.concatenate(cnf, axis=0)
    var_idx = np.abs(cat) - 1
    val_idx = np.int64(cat > 0).reshape(-1)

    data = CSP_Data(num_var=num_var, domain_size=2, path=path)
    data.add_constraint_data(
        True,
        torch.tensor(const_idx),
        torch.tensor(tuple_idx),
        torch.tensor(var_idx),
        torch.tensor(val_idx)
    )
    return data


def nx_to_col(nx_graph, num_colors):
    num_vert = nx_graph.order()
    num_edges = nx_graph.number_of_edges()

    idx_map = {v: i for i, v in enumerate(nx_graph.nodes())}

    const_idx = np.repeat(np.arange(0, num_edges, dtype=np.int64), num_colors)
    tuple_idx = np.repeat(np.arange(0, num_edges * num_colors, dtype=np.int64), 2)

    vertex_idx = np.int64([[idx_map[u], idx_map[v]] for u, v in nx_graph.edges()])
    vertex_idx = np.tile(vertex_idx, (1, num_colors))
    vertex_idx = vertex_idx.reshape(-1)

    val_idx = np.repeat(np.arange(0, num_colors, dtype=np.int64), 2)
    val_idx = np.tile(val_idx, (num_edges,))

    data = CSP_Data(num_var=num_vert, domain_size=num_colors)
    data.add_constraint_data(
        True,
        torch.tensor(const_idx),
        torch.tensor(tuple_idx),
        torch.tensor(vertex_idx),
        torch.tensor(val_idx)
    )
    return data


def nx_to_maxcut(nx_graph, edge_weights=None, path=None):
    num_vert = nx_graph.order()
    num_edges = nx_graph.number_of_edges()

    idx_map = {v: i for i, v in enumerate(nx_graph.nodes())}

    const_idx = np.repeat(np.arange(0, num_edges, dtype=np.int64), 2)
    tuple_idx = np.repeat(np.arange(0, num_edges * 2, dtype=np.int64), 2)

    vertex_idx = np.int64([[idx_map[u], idx_map[v]] for u, v in nx_graph.edges()])
    vertex_idx = np.tile(vertex_idx, (1, 2))
    vertex_idx = vertex_idx.reshape(-1)

    val_idx = np.repeat(np.arange(0, 2, dtype=np.int64), 2)
    val_idx = np.tile(val_idx, (num_edges,))

    if edge_weights is None:
        edge_weights = np.ones((num_edges,), dtype=np.int64)
    cst_type = (edge_weights + 1) // 2

    data = CSP_Data(num_var=num_vert, domain_size=2, path=path)
    data.add_constraint_data(
        True,
        torch.tensor(const_idx),
        torch.tensor(tuple_idx),
        torch.tensor(vertex_idx),
        torch.tensor(val_idx),
        cst_type=torch.tensor(cst_type)
    )
    return data


def load_xcsp3(path):
    parser = XParser(path)
    data = parser.to_CSP_data()
    return data


def load_mc_file(path):
    g, edge_weights = load_mc(path)
    data = nx_to_maxcut(g, edge_weights, path=path)
    return data


def load_mtx_file(path):
    g, edge_weights = load_mtx(path)
    data = nx_to_maxcut(g, edge_weights, path=path)
    return data


def load_npz_file(path,task,num_colors=3):
    matrix=load_npz(path).toarray()
    g = nx.from_numpy_array(matrix)
    edge_weights=[]
    for edge in g.edges():
        edge_weights.append(g.edges[edge]['weight'])
    edge_weights=np.array(edge_weights)
    if task=="maxcut":
    
        data = nx_to_maxcut(g, edge_weights, path=path)
    elif task=="coloring":
        data = nx_to_col(g,num_colors=num_colors)
    else:
        raise ValueError("Unknown objective function")
    return data





class File_Dataset(Dataset):

    def __init__(self, path, preload=False,task='maxcut'):
        super(File_Dataset, self).__init__()
        self.path = path
        self.files=glob.glob(path + '/*')
        self.task=task
        

        remove_files=[]
        for file_path in self.files:
            postfix = file_path.split('.')[-1]
            if postfix not in ['xml','cnf','mc','mtx','npz']:
                remove_files.append(file_path)
        
        for remove_file in remove_files:
            self.files.remove(remove_file)


        self.files.sort()

        self.preload = preload
        if self.preload:
            print('Loading Data:')
            self.data = [self.load_file(file_path) for file_path in tqdm(self.files)]

    def load_file(self, file_path):
        postfix = file_path.split('.')[-1]
        if postfix == 'xml':
            data = load_xcsp3(file_path)
        elif postfix == 'cnf':
            data = load_cnf_file(file_path)
        elif postfix == 'mc':
            data = load_mc_file(file_path)
        elif postfix == 'mtx':
            data = load_mtx_file(file_path)

        elif postfix=='npz':
            data=load_npz_file(file_path,self.task)
        else:
            raise ValueError(f'File type {postfix} is not supported.')
        return data

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if self.preload:
            data = self.data[item]
        else:
            file_path = self.files[item]
            data = self.load_file(file_path)
        return data


class Generator_Dataset(Dataset):

    def __init__(self, generators, epoch_samples=1000):
        super(Generator_Dataset, self).__init__()
        self.generators = generators
        self.num_gen = len(self.generators)
        self.epoch_samples = epoch_samples

    def __len__(self):
        return self.epoch_samples

    def __getitem__(self, item):
        i = np.random.randint(self.num_gen)
        return self.generators[i].create_random_instance()
