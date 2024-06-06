import subprocess
import numpy as np

for distribution in [ 
                    #  'Physics',
                    #  'rnd_graph_800vertices_unweighted',
                    #  'rnd_graph_800vertices_weighted',
                    #  'toroidal_grid_2D_800vertices_weighted',
                    #  'planar_800vertices_unweighted',
                    #  'planar_800vertices_weighted',
                    #  'ER_200',
                    #  'BA_200',
                     'HomleKim_800vertices_unweighted',
                     'HomleKim_800vertices_weighted',
                     'BA_800vertices_unweighted',
                     'BA_800vertices_weighted',
                     'WattsStrogatz_800vertices_unweighted',
                     'WattsStrogatz_800vertices_weighted',
                    #  'SK_spin_70_100vertices_weighted',
                    #  'dense_MC_100_200vertices_unweighted',
                     ]:
    for gamma in np.arange(120,150,10):
        print(f'Distribution:{distribution} Tau:{gamma}')
        command= f'python Tabu.py --distribution {distribution} --gamma {gamma}'

        subprocess.run(command,shell=True,check=True)
    # break
