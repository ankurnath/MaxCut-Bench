from argparse import ArgumentParser
import os
import numpy as np

from EO import *
import pickle

from smartprint import smartprint as sprint



if __name__ == '__main__':


    parser = ArgumentParser()

    parser.add_argument("--train_distribution", type=str,default='BA_20',required=True, help="Distribution of training dataset")

    parser.add_argument("--num_repeat", type=int,default=50, help="Number of attempts")

    parser.add_argument("--num_steps", type=int, default=2, help="Number of steps")

    parser.add_argument("--low", type=float, default=1.2, help="lower bound of tau")

    parser.add_argument("--high", type=float, default=1.4, help="higher bound of tau")

    parser.add_argument('--step',type=float, default=0.1, help="Step size of tau")


    args = parser.parse_args()

    distribution =args.train_distribution
    num_repeat = args.num_repeat
    num_steps = args.num_steps
    low = args.low
    high = args.high
    step = args.step

    sprint(distribution)
    sprint(num_repeat)
    sprint(num_steps)
    sprint(low)
    sprint(high)
    sprint(step)





    save_folder=f'pretrained agents/{args.train_distribution}'
    save_folder=os.path.join(os.getcwd(),'solvers/EO',save_folder)

    


    os.makedirs(save_folder,exist_ok=True)
    


    

    dataset_path=os.path.join(os.getcwd(),f'data/validation/{args.train_distribution}')

    dataset=GraphDataset(dataset_path,ordered=True)

    print("Number of graphs:",len(dataset))
    for tau in np.arange(args.low,args.high,args.step):

        best_mean=0
        best_tau=None

        best_cuts=[]
        for i in range(len(dataset)):
            graph=dataset.get()
            g=flatten_graph(graph)

            n=graph.shape[0]

            indices = np.arange(1, n + 1, dtype='float')
            pmf = 1 / (indices **tau)
            pmf /= pmf.sum()
            num_samples = args.num_steps*args.num_repeat
            actions = np.random.choice(indices-1, size=num_samples, p=pmf)
            actions=actions.reshape(args.num_repeat,args.num_steps).astype(int)

            arguments=[]

            for i in range(args.num_repeat):
                spins= np.random.randint(2, size=graph.shape[0])
                arguments.append((g,spins,actions[i]))
            
            with Pool() as pool:
                best_cut=np.max(pool.starmap(EO, arguments))

            best_cuts.append(best_cut)

        best_cuts=np.array(best_cuts)

        if best_mean<best_cuts.mean():
            best_mean=best_cuts.mean()
            best_tau=tau
            tau_save_path=os.path.join(save_folder,'best_tau')

            with open(tau_save_path, "wb") as file:
                pickle.dump(best_tau, file)

    
    print('Best Tau:',best_tau)

    



    











