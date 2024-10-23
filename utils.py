import numpy as np
import glob
from scipy.sparse import load_npz
import random
from argparse import ArgumentParser
import networkx as nx
from collections import defaultdict
from smartprint import smartprint as sprint
import pandas as pd
import time
import os
import pickle
from multiprocessing.pool import Pool
import re
from tqdm import tqdm
import seaborn as sns

def save_to_pickle(data, file_path):
    """
    Save data to a pickle file.

    Parameters:
    - data: The data to be saved.
    - file_path: The path to the pickle file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f'Data has been saved to {file_path}')


def load_from_pickle(file_path,quiet = True):
    """
    Load data from a pickle file.

    Parameters:
    - file_path: The path to the pickle file.

    Returns:
    - loaded_data: The loaded data.
    """
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    if not quiet:
        print(f'Data has been loaded from {file_path}')
    return loaded_data



