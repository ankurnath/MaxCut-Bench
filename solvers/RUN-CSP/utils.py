
import os
import glob
from scipy.sparse import load_npz
import random


class GraphDataset(object):

    def __init__(self,folder_path,ordered=False):
        super().__init__()

        self.file_paths=glob.glob(f'{folder_path}/*.npz')
        self.file_paths.sort()
        self.ordered=ordered

        if self.ordered:
            self.i = 0

    def __len__(self):
        return len(self.file_paths)
    
    def get(self):
        if self.ordered:
            file_path = self.file_paths[self.i]
            self.i = (self.i + 1)%len(self.file_paths)
        else:
            file_path = random.sample(self.file_paths, k=1)[0]
        return load_npz(file_path).toarray()

def mk_dir(path):
  try:
    os.mkdir(path)
  except:
    print("Folder already created")