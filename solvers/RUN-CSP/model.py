import torch
from torch.nn import Module, ModuleDict, LSTMCell, Linear, Softmax, Sigmoid, BatchNorm1d, Sequential
from torch_scatter import scatter_sum, scatter_mean

import os


class Message_Network(Module):

    def __init__(self, rel_name, hidden_dim):

        super(Message_Network, self).__init__()
        self.rel_name = rel_name
        self.hidden_dim = hidden_dim
        self.linear = Linear(2 * hidden_dim, 2 * hidden_dim, bias=False)

    def forward(self, csp_data, x):
        # get
        edge_idx = csp_data.edges[self.rel_name]

        x = torch.cat([x[edge_idx[0]], x[edge_idx[1]]], dim=1)
        x = self.linear(x)

        r = scatter_sum(x[:, :self.hidden_dim], edge_idx[0], dim=0, dim_size=csp_data.num_vars) + \
            scatter_sum(x[:, self.hidden_dim:], edge_idx[1], dim=0, dim_size=csp_data.num_vars)

        return r


class Symmetric_Message_Network(Module):
    def __init__(self, rel_name, hidden_dim):
        super(Symmetric_Message_Network, self).__init__()
        self.rel_name = rel_name
        self.hidden_dim = hidden_dim
        self.linear = Linear(2 * hidden_dim, hidden_dim, bias=False)

    def forward(self, csp_data, x):
        edge_idx = csp_data.edges[self.rel_name]
        edge_idx = torch.cat([edge_idx, torch.stack([edge_idx[1], edge_idx[0]], dim=0)], dim=1)

        x = torch.cat([x[edge_idx[0]], x[edge_idx[1]]], dim=1)
        x = self.linear(x)

        r = scatter_sum(x, edge_idx[1], dim=0, dim_size=csp_data.num_vars)
        return r


class RUNCSP(Module):

    def __init__(self, model_dir, hidden_dim, const_lang):
        super(RUNCSP, self).__init__()
        self.model_dir = model_dir
        self.hidden_dim = hidden_dim
        self.const_lang = const_lang
        self.out_dim = const_lang.domain_size if const_lang.domain_size > 2 else 1

        # init message passing modules (one per allowed relation)
        msg = {}
        for rel in const_lang.relations.keys():
            if const_lang.is_symmetric[rel]:
                msg[rel] = Symmetric_Message_Network(rel, hidden_dim)
            else:
                msg[rel] = Message_Network(rel, hidden_dim)
        self.msg = ModuleDict(msg)

        self.norm = BatchNorm1d(hidden_dim)
        self.cell = LSTMCell(hidden_dim, hidden_dim)

        # use sigmoid for 2-d domains and softmax otherwise
        if self.out_dim == 1:
            self.soft_assign = Sequential(Linear(hidden_dim, 1, bias=False), Sigmoid())
        else:
            self.soft_assign = Sequential(Linear(hidden_dim, self.out_dim, bias=False), Softmax(dim=1))

    def save(self):
        # save_folder=os.path.join(self.model_dir,'network')

        # os.makedirs(save_folder, exist_ok=True)
        os.makedirs(self.model_dir,exist_ok=True)

        torch.save(self, os.path.join(self.model_dir, 'best.pkl'))

    @staticmethod
    def load(model_dir):
        return torch.load(os.path.join(model_dir, 'best.pkl'))

    def forward(self, csp_data, steps):
        # init recurrent states
        h = torch.normal(0.0, 1.0, (csp_data.num_vars, self.hidden_dim), device=csp_data.device)
        c = torch.zeros((csp_data.num_vars, self.hidden_dim), dtype=torch.float32, device=csp_data.device)

        assignments = []
        for _ in range(steps):

            # aggregate msg passed for each relation
            rec = torch.zeros((csp_data.num_vars, self.hidden_dim), dtype=torch.float32, device=csp_data.device)
            for rel in csp_data.edges.keys():
                rec = rec + self.msg[rel](csp_data, h)

            # divide by degree (mean pooling) and normalize
            rec = rec * csp_data.var_reg
            rec = self.norm(rec)

            # update recurrent states
            h, c = self.cell(rec, (h, c))

            # predict soft assignment
            y = self.soft_assign(h)
            assignments = y.unsqueeze(1)
            #assignments.append(y)

            num_unsat = csp_data.count_unsat(assignments)
            min_unsat = num_unsat.cpu().numpy().min()
            if min_unsat == 0:
                break

        # combine all assignments
        #assignments = torch.stack(assignments, dim=1)

        # turn 1-d output into 2-d (needed for loss and evaluation)
        if self.out_dim == 1:
            assignments = torch.cat([1.0-assignments, assignments], dim=2)

        return assignments