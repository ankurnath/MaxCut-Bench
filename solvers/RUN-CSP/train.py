import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import RUNCSP
from loss import csp_loss
from csp_data import CSP_Data

from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob

# from train import train

from utils import GraphDataset,mk_dir
import networkx as nx

def train(model, opt, loader, device, args):
    # writer = SummaryWriter(args.model_dir)
    step = 0
    best_conflict_ratio = float('inf')
    for e in range(args.epochs):
        num_unsat_list = []
        # solved_list = []
        for data in tqdm(loader):
            opt.zero_grad()
            data.to(device)

            assignment = model(data, args.network_steps)
            loss = csp_loss(data, assignment, discount=args.discount)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            num_unsat = data.count_unsat(assignment)
            num_unsat = num_unsat.min(dim=1)[0]
            # solved = num_unsat == 0
            num_unsat_list.append(num_unsat)
            # solved_list.append(solved)

            

            # if (step + 1) % args.logging_steps == 0:
                
            #     writer.add_scalar('Train/Loss', loss.mean(), step)
            #     writer.add_scalar('Train/Solved_Ratio', solved.float().mean(), step)
            #     writer.add_scalar('Train/Unsat_Count', num_unsat.float().mean(), step)
            #     num_unsat_list = []
            #     solved_list = []

            step += 1

        num_unsat = torch.cat(num_unsat_list, dim=0)
        # solved = torch.cat(solved_list, dim=0)
        conflict_ratio=num_unsat.float().mean()
        print('Conflict_ratio:',conflict_ratio)
        if conflict_ratio < best_conflict_ratio:
            model.save()
            best_conflict_ratio = conflict_ratio
        # model.save()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--distribution", type=str, default='ER_20', help="Path to the training data")
    parser.add_argument("--seed", type=int, default=0, help="the random seed for torch and numpy")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of loader workers")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")

    parser.add_argument("--batch_size", type=int, default=10, help="The batch size used for training")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--logging_steps", type=int, default=10, help="Training steps between logging")
    parser.add_argument("--discount", type=float, default=0.9, help="Discount factor")    
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden Dimension of the network")
    parser.add_argument("--network_steps", type=int, default=30, help="Number of network steps during training")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dict_args = vars(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # print(f'Loading Graphs from {args.data_path}...')

    train_graph_gen=GraphDataset(folder_path=f'../data/training/{args.distribution}')
    print(f'Number of graphs:{len(train_graph_gen)}')
    graphs = [nx.from_numpy_array(train_graph_gen.get()) for _ in range(len(train_graph_gen))]
    # graphs = [nx.from_numpy_array(train_graph_gen.get()) for _ in range(20)]


    # data = [CSP_Data.load_graph_maxcol(p, args.num_col) for p in tqdm(glob(args.data_path))]
    # if args.weighted:
    #     data = [CSP_Data.load_graph_weighted_maxcut(nx_graph)for nx_graph in graphs]
    # else:
    #     data = [CSP_Data.load_graph_unweighted_maxcut(nx_graph)for nx_graph in graphs]
    data = [CSP_Data.load_graph_weighted_maxcut(nx_graph)for nx_graph in graphs]
    const_lang = data[0].const_lang

    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=CSP_Data.collate
    )

    model = RUNCSP(f'pretrained agents/{args.distribution}', args.hidden_dim, const_lang)
    model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train(model, opt, loader, device, args)
