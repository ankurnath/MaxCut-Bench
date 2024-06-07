import sys, os
import pathlib
from pathlib import Path
import functools
import gzip, pickle

import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
from scipy.sparse import load_npz


def read_dgl_from_graph(graph_path):
    _g=load_npz(graph_path).toarray()
    _g=nx.from_numpy_array(_g)
    _g=_g.to_directed()
    # _g = nx.read_gpickle(graph_path)
    labelled = "optimal" in graph_path.name or "non-optimal" in graph_path.name
    # if labelled:
    #     g = dgl.from_networkx(_g, node_attrs=['label'])
    # else:
    g = dgl.from_networkx(_g, edge_attrs=['weight'])
    return g

class GraphDataset(Dataset):
    def __init__(self, data_dir=None, size=None):
        assert data_dir is not None
        self.data_dir = data_dir
        # print(data_dir)
        self.graph_paths = sorted(list(self.data_dir.rglob("*.npz")))
        if size is not None:
            assert size > 0
            self.graph_paths = self.graph_paths[:size]
        self.num_graphs = len(self.graph_paths)
        # print(self.num_graphs)

    def __getitem__(self, idx):
        return read_dgl_from_graph(self.graph_paths[idx])

    def __len__(self):
        return self.num_graphs

# def _prepare_instances(instance_directory: pathlib.Path, cache_directory: pathlib.Path, **kwargs):
#     cache_directory.mkdir(parents=True, exist_ok=True)
#     # # glob only searches the first level
#     # # for graph_path in instance_directory.rglob("*.gpickle"):
#     # for graph_path in tqdm(instance_directory.glob("*.gpickle")):
#     #     # pathlib resolve: absolute path
#     #     _prepare_instance(graph_path.resolve(), cache_directory, **kwargs)

#     resolved_graph_paths = [graph_path.resolve() for graph_path in instance_directory.glob("*.npz")]
#     prepare_instance = functools.partial(
#         _prepare_instance,
#         cache_directory=cache_directory,
#         **kwargs,
#     )
#     imap_unordered_bar(prepare_instance, resolved_graph_paths, n_processes=None)

from multiprocessing import Pool
from tqdm import tqdm
# def imap_unordered_bar(func, args, n_processes=2):
#     p = Pool(n_processes)
#     args = list(args)
#     res_list = []
#     with tqdm(total=len(args)) as pbar:
#         for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
#             pbar.update()
#             res_list.append(res)
#     pbar.close()
#     p.close()
#     p.join()
#     return res_list

# def _prepare_instance(source_instance_file: pathlib.Path, cache_directory: pathlib.Path):
#     cache_directory.mkdir(parents=True, exist_ok=True)
#     dest_path = cache_directory / (source_instance_file.stem + ".graph")
#     if os.path.exists(dest_path):
#         source_mtime = os.path.getmtime(source_instance_file)
#         last_updated = os.path.getmtime(dest_path)
#         if source_mtime <= last_updated:
#             return  # we already have an up2date version of that file as matrix

#     try:
#         g=load_npz(source_instance_file).toarray()
#         g=nx.from_numpy_array(g)
#         # g = nx.read_gpickle(source_instance_file)
#     except:
#         print(f"Failed to read {source_instance_file}.")
#         return
#     # g.remove_edges_from(nx.selfloop_edges(g)) # remove self loops
#     # nx.write_gpickle(g, dest_path)
#     # print(f"Updated graph file: {source_instance_file}.")

def collate_fn(graphs):
    return dgl.batch(graphs)


def get_test_data_loader(cfg):
    data_path = Path(__file__).parent.parent.parent / "data"
    test_data_path = data_path / "testing" / (cfg.input)
    print(f"Loading data from {test_data_path}.")
    testset = GraphDataset(test_data_path)
    test_loader = DataLoader(testset, batch_size=cfg.test_batch_size,
             shuffle=False, collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=True)
    return test_loader



def get_data_loaders(cfg,add_task=False):
    if add_task:
        folder=f'data_{cfg.task}'
    else:
        folder='data'
    # data_path = Path(__file__).parent.parent.parent / "data"
    data_path = Path(__file__).parent.parent.parent / folder
    # data_path = data_path / pathlib.Path(cfg.input)  # string to pathlib.Path
    print(f"Loading data from {data_path}.")

    # preprocessed_name = "gfn"
    train_data_path = data_path / "training" / (cfg.input)
    # train_cache_directory = train_data_path / "preprocessed" / preprocessed_name
    # _prepare_instances(train_data_path, train_cache_directory)

    test_data_path = data_path / "validation" / (cfg.input)
    # test_cache_directory = test_data_path / "preprocessed" / preprocessed_name
    # _prepare_instances(test_data_path, test_cache_directory)

    trainset = GraphDataset(train_data_path)
    testset = GraphDataset(test_data_path)
    print(len(testset))
    

    # collate_fn = lambda graphs: dgl.batch(graphs)
    train_batch_size = 1 if cfg.same_graph_across_batch else cfg.batch_size_interact
    train_loader = DataLoader(trainset, batch_size=train_batch_size,
            shuffle=cfg.shuffle, collate_fn=collate_fn, drop_last=False,
            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=cfg.test_batch_size,
             shuffle=False, collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader