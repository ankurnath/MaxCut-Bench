import gurobipy as gp
from gurobipy import GRB

from utils import *
import os
import sys



def gurobi_solver(graph,max_time = None,max_threads = None):
    

    model = gp.Model()
    model.setParam("OutputFlag", 0)

    if max_time:
        model.setParam('TimeLimit', max_time)

    if max_threads:
        model.setParam('Threads', max_threads)

    vdict= model.addVars(graph.number_of_nodes(), vtype=GRB.BINARY, name="Build")

    cut = [data['weight']*(vdict[i] + vdict[j] - 2*vdict[i]*vdict[j]) for i,j,data in graph.edges(data=True)]

    model.setObjective(sum(cut), gp.GRB.MAXIMIZE)
    

    model.optimize()
    
    
    
    return model.ObjVal, [key for key in vdict.keys() if abs(vdict[key].x) > 1e-6]



    
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument( "--test_distribution", type=str, default='BA_200vertices_weighted', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--time_limit", type=float, default= 10, help="Maximum Time Limit" )
    parser.add_argument( "--threads", type=int, default= 10, help="Maximum number of threads" )
  
    args = parser.parse_args()

    distribution = args.test_distribution
    time_limit = args.time_limit
    threads = args.threads

    sprint(distribution)
    sprint(time_limit)
    sprint(threads)

    # sys.stdout = open('out.dat', 'w')
    test_dataset = GraphDataset(f'data/testing/{distribution}',ordered=True)


    
    df = defaultdict(list)

    for _ in tqdm(range(len(test_dataset))):
    # for _ in tqdm(range(1)):

        graph = test_dataset.get()
        graph = nx.from_numpy_array(graph)
        objVal, solution = gurobi_solver(graph=graph,max_time=time_limit,max_threads=threads)
        df['cut'].append(objVal)
        df['time'].append(time_limit)
        df['threads'].append(threads)
        

    folder_name = f'data/results/{distribution}'

    os.makedirs(folder_name,exist_ok=True)

    file_path = os.path.join(folder_name,'Gurobi') 

    df = pd.DataFrame(df)
    try:
        OPT = load_from_pickle(f'../data/testing/{distribution}/optimal')
        df['OPT'] = OPT['OPT']
        df['Ratio'] = df['cut']/df['OPT']
        sprint(df['Ratio'].mean())
    except:
        pass
    # OPT = load_from_pickle(f'../data/testing/{distribution}/optimal')
    # df['Approx. ratio'] = df['cut']/OPT['OPT'].values
    # print(df)
    print(df)

    df.to_pickle(file_path)


        






    # train(dataset=args.dataset,budget=args.budget)