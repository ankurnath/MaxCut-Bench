from utils import *





if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument( "--distribution", type=str, default='BA_200vertices_weighted', help="Name of the dataset to be used (default: 'Facebook')" )
    # parser.add_argument( "--time_limit", type=float, default=600, help="Maximum Time Limit" )
    parser.add_argument( "--threads", type=int, default= 10, help="Maximum number of threads" )

    args = parser.parse_args()

    distribution = args.distribution
    
    threads = args.threads


    import os
    # This is to limit the number of threads
    os.environ["OMP_NUM_THREADS"] = str(threads)



    import cvxpy as cp
    from scipy.linalg import sqrtm
    import time


    test_dataset = GraphDataset(f'../data/testing/{distribution}',ordered=True)


    df = defaultdict(list)
    for i in tqdm(range(len(test_dataset))):
    # for i in range(10):

        
        start = time.time()

        graph = test_dataset.get()

        n = len(graph)

        matrix = cp.Variable((n , n ), PSD=True)

        cut = .25 * cp.sum(cp.multiply(graph, 1 - matrix))

        problem = cp.Problem(cp.Maximize(cut), [cp.diag(matrix) == 1])

        problem.solve(verbose=False)
        # print(problem.solver_stats)

        # sprint(problem.solver_stats.solve_time)

        if problem.solver_stats.extra_stats['info']['status'] =='solved':

            vectors = matrix.value
            random = np.random.normal(size=vectors.shape[1])
            random /= np.linalg.norm(random, 2)

            spins = np.sign(np.dot(vectors, random))
            cut = (1/4) * np.sum( np.multiply( graph, 1 - np.outer(spins, spins) ) )

            end = time.time()

            elapesed_time = end -start

            df['cut'].append(cut)
            df['Time'].append(elapesed_time)

        else:
            df['cut'].append(0)
            df['cut'].append(elapesed_time)

        # break

        
    folder_name = f'data/SDP/{distribution}'

    os.makedirs(folder_name,exist_ok=True)

    file_path = os.path.join(folder_name,'results') 

    df = pd.DataFrame(df)
    # # OPT = load_from_pickle(f'../data/testing/{distribution}/optimal')
    # # df['Approx. ratio'] = df['cut']/OPT['OPT'].values
    print(df)

    df.to_pickle(file_path)