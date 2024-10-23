from argparse import ArgumentParser
import os
import numpy as np

from TS import *
import pickle



if __name__ == '__main__':


    parser = ArgumentParser()

    parser.add_argument("--distribution", type=str,default='Physics',required=True, help="Distribution of training dataset")

    parser.add_argument("--num_repeat", type=int,default=50,required=True, help="Number of attempts")

    parser.add_argument("--num_steps", type=int, default=2,required=True, help="Number of steps")

    parser.add_argument("--low", type=float, default=20,required=True, help="lower bound of tabu tenure")

    parser.add_argument("--high", type=float, default=150,required=True, help="higher bound of tabu tenure")

    parser.add_argument('--step',type=float, default=10,required=True, help="Step size of tabu tenure")


    args = parser.parse_args()


    save_folder=f'pretrained agents/{args.distribution}'
    save_folder=os.path.join(os.getcwd(),'solvers/TS',save_folder)
    os.makedirs(save_folder,exist_ok=True)
    


    

    dataset_path=os.path.join(os.getcwd(),f'data/validation/{args.distribution}')

    dataset=GraphDataset(dataset_path,ordered=True)

    print("Number of graphs:",len(dataset))
    for tabu_tenure in np.arange(args.low,args.high,args.step):

        best_mean=0
        best_tabu_tenure=None

        best_cuts=[]
        for i in range(len(dataset)):
            graph=dataset.get()
            g=flatten_graph(graph)

            

            arguments=[]

            for i in range(args.num_repeat):
                spins= np.random.randint(2, size=graph.shape[0])
                arguments.append((g,spins,tabu_tenure,args.num_steps))
            
            with Pool() as pool:
                best_cut=np.max(pool.starmap(tabu, arguments))

            best_cuts.append(best_cut)

        best_cuts=np.array(best_cuts)
        # print(best_cuts.mean())

        # print(best_mean<best_cuts.mean())
        if best_mean<best_cuts.mean():
            # print(best_cuts.mean())

            best_mean=best_cuts.mean()
            best_tabu_tenure=tabu_tenure
            # print(best_mean)

    # print(best_cuts)
    print('Best Tabu Tenure:',best_tabu_tenure)

    tabu_tenure_save_path=os.path.join(save_folder,'best_tabu_tenure')

    with open(tabu_tenure_save_path, "wb") as file:
        # Use pickle.dump() to save the integer to the file
        pickle.dump(tabu_tenure, file)



    











