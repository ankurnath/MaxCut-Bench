from argparse import ArgumentParser
import os
import numpy as np

from TS import *
import pickle
import time

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--train_distribution",type=str,help='Train distribution (if train and test are not the same)')

    parser.add_argument("--test_distribution", type=str, help="Distribution of test dataset")

    parser.add_argument("--num_repeat", type=int,default=50,required=True, help="Number of attempts")

    parser.add_argument("--num_steps", type=int, default=2,required=True, help="Number of steps")

    parser.add_argument("--tabu_tenure", type=int, default=100, help="Tabu Tenure")

    parser.add_argument("--num_threads",type=int,default=20,help="Number of threads")

    args = parser.parse_args()




    save_folder = f'pretrained agents/{args.train_distribution}'
    save_folder=os.path.join(os.getcwd(),'solvers/TS',save_folder)

    os.makedirs(save_folder,exist_ok=True)

    dataset_path=os.path.join(os.getcwd(),f'data/testing/{args.test_distribution}')

    dataset=GraphDataset(dataset_path,ordered=True)

    print("Number of graphs:",len(dataset))


    best_tabu_tenure_path=os.path.join(save_folder,'best_tabu_tenure')

    # Load the pickle file
    try:
        with open(best_tabu_tenure_path, 'rb') as file:
            best_tabu_tenure = pickle.load(file)
        print('Loaded pretrained value:',best_tabu_tenure)
    except:
        best_tabu_tenure = args.tabu_tenure
        print('Loaded default value:',best_tabu_tenure)



    best_cuts=[]
    elapsed_times =[]
    for i in range(len(dataset)):
        graph=dataset.get()
        start = time.time()

        g=flatten_graph(graph)

        n=graph.shape[0]



        arguments=[]

        for i in range(args.num_repeat):
            spins= np.random.randint(2, size=graph.shape[0])
            arguments.append((g,spins,best_tabu_tenure,args.num_steps))
        
        with Pool(args.num_threads) as pool:
            best_cut=np.max(pool.starmap(tabu, arguments))

        best_cuts.append(best_cut)
        end = time.time()
        elapsed_time = end-start
        elapsed_times.append(elapsed_time)


    best_cuts=np.array(best_cuts)
    elapsed_times = np.array(elapsed_times)

    df={'cut':best_cuts,'tabu_tenure':[best_tabu_tenure]*best_cuts.shape[0]}
    df['Instance'] = [os.path.basename(file) for file in dataset.file_paths]
    df['Train Distribution'] = [args.train_distribution]*best_cuts.shape[0]
    df['Test Distribution'] = [args.test_distribution]*best_cuts.shape[0]
    df['Time'] = elapsed_times
    df['Threads'] = [args.num_threads] * best_cuts.shape[0]
    df=pd.DataFrame(df)

    print(df)

    df.to_pickle(os.path.join('results',args.test_distribution,'TS'))



