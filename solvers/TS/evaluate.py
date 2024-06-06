from argparse import ArgumentParser
import os
import numpy as np

from TS import *
import pickle


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--distribution", type=str,default='Physics',required=True, help="Distribution of test dataset")

    parser.add_argument("--num_repeat", type=int,default=50,required=True, help="Number of attempts")

    parser.add_argument("--num_steps", type=int, default=2,required=True, help="Number of steps")



    args = parser.parse_args()

    save_folder=f'pretrained agents/{args.distribution}_TS'
    save_folder=os.path.join(os.getcwd(),'solvers/TS',save_folder)

    network_folder=os.path.join(save_folder,'network')
    data_folder=os.path.join(save_folder,'data')


    dataset_path=os.path.join(os.getcwd(),f'data/testing/{args.distribution}')

    dataset=GraphDataset(dataset_path,ordered=True)

    print("Number of graphs:",len(dataset))


    best_tabu_tenure_path=os.path.join(network_folder,'best_tabu_tenure')

    # Load the pickle file
    with open(best_tabu_tenure_path, 'rb') as file:
        best_tabu_tenure = pickle.load(file)

    best_cuts=[]
    for i in range(len(dataset)):
        graph=dataset.get()
        g=flatten_graph(graph)

        n=graph.shape[0]



        arguments=[]

        for i in range(args.num_repeat):
            spins= np.random.randint(2, size=graph.shape[0])
            arguments.append((g,spins,best_tabu_tenure,args.num_steps))
        
        with Pool() as pool:
            best_cut=np.max(pool.starmap(tabu, arguments))

        best_cuts.append(best_cut)

    best_cuts=np.array(best_cuts)

    df={'cut':best_cuts,'tabu_tenure':[best_tabu_tenure]*best_cuts.shape[0]}
    df['Instance'] = [os.path.basename(file) for file in dataset.file_paths]
    df=pd.DataFrame(df)

    print(df)

    df.to_pickle(os.path.join(data_folder,'results'))



