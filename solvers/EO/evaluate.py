from argparse import ArgumentParser
import os
import numpy as np

from EO import *
import pickle
import time


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--train_distribution",type=str,default=None,help='Train distribution (if train and test are not the same)')

    parser.add_argument("--distribution", type=str,default='Physics',required=True, help="Distribution of test dataset")

    parser.add_argument("--num_repeat", type=int,default=50,required=True, help="Number of attempts")

    parser.add_argument("--num_steps", type=int, default=2,required=True, help="Number of steps")

    parser.add_argument("--tau", type=float, default=1.4, help="Default tau")

    parser.add_argument("--num_threads",type=int,default=20,help="Number of threads")

    args = parser.parse_args()

    print(args.train_distribution)

    #### loading the data
    if args.train_distribution:

        save_folder = f'pretrained agents/{args.train_distribution}'
        print('Generelization Experiment')
        train_distribution = args.train_distribution
    else:
        save_folder = f'pretrained agents/{args.distribution}'
        train_distribution = args.distribution


    save_folder = os.path.join(os.getcwd(),'solvers/EO',save_folder)
    os.makedirs(save_folder,exist_ok=True)

    print('Distribution:',args.distribution)


    dataset_path=os.path.join(os.getcwd(),f'data/testing/{args.distribution}')

    dataset=GraphDataset(dataset_path,ordered=True)

    print("Number of graphs:",len(dataset))

    
    best_tau_path=os.path.join(save_folder,'best_tau')

    # Load the pickle file
    try:
        with open(best_tau_path, 'rb') as file:
            best_tau = pickle.load(file)
        print('Loaded best value of tau from training')
    except:
        best_tau = args.tau
        train_distribution = 'Default'
        print('Loaded default tau:',best_tau)


    best_cuts=[]
    elapsed_times =[]
    for i in range(len(dataset)):
        graph=dataset.get()
        start = time.time()
        g=flatten_graph(graph)

        n=graph.shape[0]

        indices = np.arange(1, n + 1, dtype='float')
        pmf = 1 / (indices **best_tau)
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
        end = time.time()
        elapsed_time = end-start
        elapsed_times.append(elapsed_time)

    best_cuts=np.array(best_cuts)
    elapsed_times = np.array(elapsed_times)

    df={'cut':best_cuts,'tau':[best_tau]*best_cuts.shape[0]}
    df['Instance'] = [os.path.basename(file) for file in dataset.file_paths]
    df['Train Distribution'] = [train_distribution]*best_cuts.shape[0]
    df['Test Distribution'] = [args.distribution]*best_cuts.shape[0]
    df['Time'] = elapsed_times
    df['Threads'] = [args.num_threads] * best_cuts.shape[0]
    df=pd.DataFrame(df)

    print(df)

    # df.to_pickle(os.path.join(data_folder,'results'))
    results_save_folder = os.path.join('results',args.distribution)
    os.makedirs(results_save_folder,exist_ok=True)
    df.to_pickle(os.path.join(results_save_folder,'EO'))



