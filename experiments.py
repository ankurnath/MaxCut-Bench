import subprocess
import os



for algorithm in  [
                    'S2V-DQN',
                    'ECO-DQN',
                    'LS-DQN',
                    'SoftTabu',
                    'Greedy'
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
                    

                elif os.path.exists(os.path.join('solvers',algorithm,'pretrained agents',train_model_path)):
                    
                    if os.path.exists(f'../data/testing/{test_dist_path}'):
                        
                        
                        command = f'python evaluation.py --algorithm {algorithm} --train_distribution {train_model_path} --test_distribution {test_dist_path}'
                        subprocess.run(command,shell=True)

                    else:
                        print(f'Data not found for testing for {test_dist_path}')
                else:
                    print(f'No pretrained Network found for dist {train_model_path} for {algorithm}')