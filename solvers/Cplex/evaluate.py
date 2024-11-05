from docplex.mp.model import Model
from utils import *
import os


def cplex_solver(graph,max_time,max_threads):
    model = Model(name='Maximum Cut')
    model.parameters.timelimit.set(max_time)
    model.parameters.threads(max_threads)

    x = {node: model.binary_var(name=f"x_{node}") for node in graph.nodes()}

    model.maximize(model.sum(data['weight']*x[u]+data['weight']*x[v]-2*data['weight']*x[u]*x[v] for u,v, data in graph.edges(data=True)))

    model.print_information()

    model.solve()

    
    return model._objective_value()



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument( "--test_distribution", type=str, default='ER_200vertices_weighted', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--time_limit", type=float, default= 600, help="Maximum Time Limit" )
    parser.add_argument( "--threads", type=int, default= 1, help="Maximum number of threads" )
  
    args = parser.parse_args()

    distribution = args.test_distribution
    time_limit = args.time_limit
    threads = args.threads

    sprint(distribution)
    sprint(time_limit)
    sprint(threads)

    

    

    test_dataset = GraphDataset(f'data/testing/{distribution}',ordered=True)

    df = defaultdict(list)

    for _ in range(len(test_dataset)):
   

        graph = test_dataset.get()
        graph = nx.from_numpy_array(graph)
        objVal= cplex_solver(graph=graph,max_time=time_limit,max_threads=threads)
        df['cut'].append(objVal)
        df['time'].append(time_limit)
        df['threads'].append(threads)
        
        

    

    folder_name = f'results/{distribution}'

    os.makedirs(folder_name,exist_ok=True)

    file_path = os.path.join(folder_name,'Cplex') 

    df = pd.DataFrame(df)
    try:
        OPT = load_from_pickle(f'data/testing/{distribution}/optimal')
        df['OPT'] = OPT['OPT']
        df['Ratio'] = df['cut']/df['OPT']
        print(df['Ratio'].mean())
    except:
        pass
    


    
    print(df)

    df.to_pickle(file_path)
    
    print(f'Data has been saved to {file_path}')









# tms = model.solve()
# assert tms
# tms.display()
# # Add binary variables for each node in the graph
# # Create a dictionary to map the node index to CPLEX variable names
# vdict = {i: f"Build_{i}" for i in graph.nodes()}
# for var_name in vdict.values():
#     model.variables.add(names=[var_name], types=[model.variables.type.binary])

# # Setting the objective function
# objective_terms = []
# for i, j, data in graph.edges(data=True):
#     weight = data.get('weight', 1)  # Default to 1 if 'weight' is not in data
    
#     # Add objective components: weight * (v_i + v_j - 2 * v_i * v_j)
#     # objective_terms.append((vdict[i]+vdict[j] - 2* vdict[i]* vdict[j], weight))
#     objective_terms.append((vdict[j]+'*'+vdict[j], weight))
#     break
#     # objective_terms.append((vdict[i], -2 * weight))
#     # objective_terms.append((vdict[j], -2 * weight))

# # Setting the objective
# # model.objective.set_linear(objective_terms)
# model.objective.set_quadratic(objective_terms)
# # # Set the objective in CPLEX model
# # model.objective.set_linear([(term, 1) for term in objective_terms])

# # objective = model.sum(vdict[i]*data['weight']+vdict[j]*data['weight']-2*data['weight']*vdict[i]*vdict[j] for i, j ,data in graph.edges(data=True))
# # print(model.objective.set)