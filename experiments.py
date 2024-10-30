import subprocess
import os



for algorithm in  [
                    'S2V-DQN',
                    # 'SoftTabu',
                    # 'ECO-DQN',
                    # 'LS-DQN',
                    
                    'Greedy',
                    # 'TS',
                    # 'EO',
                    # 'ANYCSP',
                    # 'RUN-CSP',
                    # 'Gflow-CombOpt'
                  ]:
    
    for dist in [
                # 'ER',
                # 'planar',
                'torodial'
                ]:

        for suffix in [
            'weighted',
            # 'unweighted'
            ]:


            train_model_path = f'{dist}_800vertices_{suffix}'
            
            for n in [
                      800,1000,2000,
                      3000,
                      5000,7000,8000,9000,10000
                      ]:
                test_dist_path = f'{dist}_{n}vertices_{suffix}'
                if not os.path.exists(f'../data/testing/{test_dist_path}'):
                    continue
                # test_dist_path = f'torodial_3000vertices_unweighted'
                if algorithm == 'Greedy':
                    command = f'python evaluation.py --algorithm {algorithm}  --test_distribution {test_dist_path}'
                    subprocess.run(command,shell=True)

                elif algorithm == 'TS' or algorithm == 'EO':
                    command = f'python evaluation.py --algorithm {algorithm} --train_distribution {train_model_path} --test_distribution {test_dist_path}'
                    subprocess.run(command,shell=True)

                elif os.path.exists(os.path.join('solvers',algorithm,'pretrained agents',train_model_path)):
                    
                    if os.path.exists(f'../data/testing/{test_dist_path}'):
                        
                        if algorithm == 'RUN-CSP':
                            command = f'python evaluation.py --algorithm {algorithm} --train_distribution {train_model_path} --test_distribution {test_dist_path} --num_steps {1000}'
                            subprocess.run(command,shell=True)
                        else:
                            command = f'python evaluation.py --algorithm {algorithm} --train_distribution {train_model_path} --test_distribution {test_dist_path} --num_steps {2*n}'
                            subprocess.run(command,shell=True)

                    else:
                        print(f'Data not found for testing for {test_dist_path}')
                else:
                    print(f'No pretrained Network found for dist {train_model_path} for {algorithm}')