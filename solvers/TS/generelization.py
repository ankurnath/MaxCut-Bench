import os
import pickle
import subprocess

from argparse import ArgumentParser

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':



    for train_distribution in [ 'rnd_graph_800vertices_weighted',
                               'planar_800vertices_weighted',
                               'toroidal_grid_2D_800vertices_weighted'

                            ]:
        for test_distribution in [
                              'rnd_graph_2000vertices_unweighted',
                              'rnd_graph_2000vertices_weighted',
                              'planar_2000vertices_unweighted',
                              'planar_2000vertices_weighted',
                              'toroidal_grid_2D_2000vertices_weighted'

                            ]:

            df=load_pickle(f'pretrained agents/{train_distribution}_Tabu/data/results')

            gamma=df['tenure'].iloc[0]

            save_folder=f'generalization/{train_distribution}_Tabu/'

            os.makedirs(save_folder,exist_ok=True)
            save_file_path=os.path.join(save_folder,f'results_{test_distribution}')

            command=f'python Tabu.py --distribution {test_distribution} --gamma {gamma} --save_file_path {save_file_path}'

            subprocess.run(command,shell=True,check=True)