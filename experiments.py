import subprocess
import os



for algorithm in  [
                    # 'S2V-DQN',
                    # 'SoftTabu',
                    # 'ECO-DQN',
                    'LS-DQN',
                    
                    # 'Greedy',
                    # 'TS',
                    # 'EO',
                    # 'ANYCSP',
                    # 'RUN-CSP',
                    # 'Gflow-CombOpt'
                  ]:
    
    for train_dist in [
                'ER',
                'planar',
                'torodial'
                ]:
            train_suffix = 'weighted'
        


            train_model_path = f'{train_dist}_800vertices_{train_suffix}'
            
            # for n in [
            #           2000
            #           ]:
            n = 2000
            for test_dist in ['ER','planar','torodial']:
                for test_suffix in [
                            'weighted',
                            'unweighted'
                            ]:
                    test_dist_path = f'{test_dist}_{n}vertices_{test_suffix}'
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