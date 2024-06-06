import numpy as np
import torch
from torch_geometric.utils import degree
from torch_scatter import scatter_sum

from data_utils import load_dimacs_cnf, load_dimacs_graph
from const_language import Constraint_Language


class CSP_Data:
    """ Class to represent a binary CSP instance """

    def __init__(self, num_vars, const_lang, edges, batch=None, path=None):
        """
        :param num_vars: Size of the underlying domain
        :param const_lang: A Constraint_Language object that specifies the language of the instance
        :param edges: A dict of edge tensors. edges[rel] is a torch long tensor of shape 2 x m_{rel} where edges[rel]_i is the i-th edge of relation rel.
        :param batch: optional long tensor that indicates the instance in the batch which each variable belongs to.
        :path path: Optional string that holds the original path of an instance loaded from disc.
        """
        self.num_vars = num_vars
        self.const_lang = const_lang
        self.edges = edges
        self.path = path

        self.batch = torch.zeros((num_vars,), dtype=torch.int64) if batch is None else batch
        self.batch_size = self.batch.max() + 1

        self.device = 'cpu'

        # degree and inverse degree needed for mean pooling
        self.var_deg = degree(torch.cat([e.reshape(-1) for e in edges.values()]), dtype=torch.float32, num_nodes=self.num_vars)
        self.var_reg = 1.0 / (self.var_deg + 1.0e-6).view(-1, 1)

    def to(self, device):
        # move data to given device
        self.device = device
        self.var_deg = self.var_deg.to(device)
        self.var_reg = self.var_reg.to(device)
        self.batch = self.batch.to(device)

        self.const_lang.to(device)

        for k in self.edges.keys():
            self.edges[k] = self.edges[k].to(device)

    @staticmethod
    def collate(data_list):
        # merge instances into one batch

        num_vars = sum([d.num_vars for d in data_list])
        const_lang = data_list[0].const_lang
        path = data_list[0].path
        batch = torch.cat([d.batch + i for i, d in enumerate(data_list)])

        # combine edges and shift variables to batch offset
        var_offset = 0
        edges = {rel: [] for rel in const_lang.relations.keys()}
        for data in data_list:
            for rel, edge_idx in data.edges.items():
                edges[rel].append(edge_idx + var_offset)
            var_offset += data.num_vars

        edges = {rel: torch.cat(edge_idx, dim=1) for rel, edge_idx in edges.items() if len(edge_idx) > 0}

        # create merged instance
        data = CSP_Data(num_vars, const_lang, edges, batch, path)
        return data

    def hard_assign(self, soft_assignment):
        # assign value with larges prob to each variable
        return torch.argmax(soft_assignment, dim=-1)

    def constraint_sat_prob(self, soft_assignment):
        """
        :param soft_assignment: a soft variable assignment
        :return sat_prob: dictionary where sat_prob[rel] is a torch float tensor such that sat_prob[rel]_{i,t}. is the prob of edge i being satisfied in time step t.
        """

        soft_assignment = soft_assignment.view(self.num_vars, -1, self.const_lang.domain_size)
        sat_prob = {}
        for rel, edge_idx in self.edges.items():
            # characteristic matrix
            R = self.const_lang.char_matrices[rel]

            # get soft assignments at each edge
            p1 = soft_assignment[edge_idx[0]]
            p2 = soft_assignment[edge_idx[1]]

            # compute probability
            sat_prob[rel] = (torch.matmul(p1, R) * p2).sum(dim=2)

        return sat_prob

    def count_unsat(self, soft_assignment):
        """
        :param soft_assignment: a soft variable assignment
        :return num_unsat: tensor such that num_unsat_{i,t} is the number of unsatisfied constraints on instance i in time step t.
        """
        hard_assignment = self.hard_assign(soft_assignment)
        num_unsat = torch.zeros((self.batch_size, hard_assignment.shape[1]), dtype=torch.int64, device=self.device)
        for rel, edge_idx in self.edges.items():
            R = self.const_lang.char_matrices[rel]
            v1 = hard_assignment[edge_idx[0]]
            v2 = hard_assignment[edge_idx[1]]
            edge_unsat = (1.0 - R[v1, v2]).long()
            num_unsat += scatter_sum(edge_unsat, self.batch[edge_idx[0]], dim=0)
        return num_unsat

    @staticmethod
    def load_2cnf(path):
        # load 2sat formula from disc

        const_lang = Constraint_Language.get_2sat_language()
        cnf = load_dimacs_cnf(path)
        cnf = [np.int64(c) for c in cnf]
        num_var = np.max([np.abs(c).max() for c in cnf])

        def clause_type(clause):
            # returns the relation type for a given clause
            if clause[0] * clause[1] < 0:
                return 'IMPL'
            elif clause[0] > 0:
                return 'OR'
            else:
                return 'NAND'

        # fill unit clauses
        cnf = [[c[0], c[0]] if len(c) == 1 else c for c in cnf]

        # normalize implication clauses
        cnf = [[c[1], c[0]] if clause_type(c) == 'IMPL' and c[0] > 0 else c if len(c) == 1 else c for c in cnf]

        edges = {rel: [] for rel in {'OR', 'IMPL', 'NAND'}}
        for i, c in enumerate(cnf):
            u = abs(c[0]) - 1
            v = abs(c[1]) - 1
            rel = clause_type(c)
            edges[rel].append([u, v])

        edges = {rel: torch.tensor(e).transpose(0, 1) for rel, e in edges.items() if len(e) > 0}
        data = CSP_Data(num_vars=num_var, const_lang=const_lang, edges=edges, path=path)
        return data

    @staticmethod
    def load_graph_maxcol(path, num_colors):
        # load graph from disc and create coloring instance

        nx_graph = load_dimacs_graph(path)
        const_lang = Constraint_Language.get_coloring_language(num_colors)

        num_vert = nx_graph.order()
        idx_map = {v: i for i, v in enumerate(nx_graph.nodes())}

        edge_idx = torch.tensor([[idx_map[u], idx_map[v]] for u, v in nx_graph.edges()])
        edge_idx = edge_idx.transpose(0, 1)
        edges = {'NEQ': edge_idx}

        data = CSP_Data(num_vars=num_vert, const_lang=const_lang, edges=edges, path=path)
        return data

    @staticmethod
    def load_graph_weighted_maxcut(nx_graph):
        # load graph from disc and create weighted maxcut instance
        # nx_graph = load_dimacs_graph(path)
        const_lang = Constraint_Language.get_maxcut_language()

        num_vert = nx_graph.order()
        idx_map = {v: i for i, v in enumerate(nx_graph.nodes())}

        edges = {'EQ': [], 'NEQ': []}
        for u, v, w in nx_graph.edges(data='weight'):
            rel = 'NEQ' if w > 0 else 'EQ'
            edges[rel].append([idx_map[u], idx_map[v]])

        edges = {rel: torch.tensor(e).transpose(0, 1) for rel, e in edges.items() if len(e) > 0}

        data = CSP_Data(num_vars=num_vert, const_lang=const_lang, edges=edges, path=None)
        return data
    
    @staticmethod
    def load_graph_unweigted_maxcut(nx_graph):
        # load graph from disc and create weighted maxcut instance
        # nx_graph = load_dimacs_graph(path)
        const_lang = Constraint_Language.get_coloring_language(2)

        num_vert = nx_graph.order()
        idx_map = {v: i for i, v in enumerate(nx_graph.nodes())}

        edges = {'EQ': [], 'NEQ': []}
        for u, v, w in nx_graph.edges(data='weight'):
            rel = 'NEQ' if w > 0 else 'EQ'
            edges[rel].append([idx_map[u], idx_map[v]])

        edges = {rel: torch.tensor(e).transpose(0, 1) for rel, e in edges.items() if len(e) > 0}

        data = CSP_Data(num_vars=num_vert, const_lang=const_lang, edges=edges, path=None)
        return data
