import os
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
root_folder = 'results'

all_folders = os.listdir(root_folder)

gaint_df = defaultdict(list)

for folder in all_folders:

    train_dist,test_dist = folder.split(' ')
    # print(train_dist)
    # print(test_dist)

    folder_path = os.path.join(root_folder,folder)
    OPT = load_from_pickle(f'../data/testing/{test_dist}/optimal')

    for algorithm in os.listdir(folder_path):

        file = load_from_pickle(os.path.join(folder_path,algorithm))

        size = len(file)
        

        gaint_df['algorithm'] += [algorithm]*size
        gaint_df['Ratio'] +=  list(file['cut'].values/ OPT['OPT'].values ) 
        gaint_df['train_Dist'] += [train_dist]*size
        gaint_df['test_Dist'] += [test_dist]*size





df= pd.DataFrame(gaint_df)
print(gaint_df)

# Get unique train distributions
train_dists = df['train_Dist'].unique()

fontsize = 20
# Loop through each train distribution
for train_dist in train_dists:
    # Filter the data for the current train distribution
    train_dist_df = df[df['train_Dist'] == train_dist]

    train_dist,_,suffix = train_dist.split('_')

    if not train_dist.isupper():
        train_dist = train_dist.capitalize()

    if train_dist =='Torodial':
        train_dist = 'Toroidal'
    
    # Get unique test distributions for the current train distribution
    test_dists = train_dist_df['test_Dist'].unique()
    
    # Create subplots
    n_tests = len(test_dists)
    fig, axes = plt.subplots(nrows=1, ncols=n_tests, figsize=(5 * n_tests, 6), sharey=False)
    
    # Loop through each test distribution and plot
    for i,(ax, test_dist) in enumerate(zip(axes, test_dists)):
        # Filter the data for the current test distribution
        test_dist_df = train_dist_df[train_dist_df['test_Dist'] == test_dist]
        
        # Create the bar plot in the corresponding subplot
        sns.barplot(data=test_dist_df, x='algorithm', y='Ratio', ax=ax)
        
        # Add titles and labels
        test_dist,_,suffix = test_dist.split('_')

        if not test_dist.isupper():
            test_dist = test_dist.capitalize()

        if test_dist =='Torodial':
            test_dist = 'Toroidal'


        ax.set_title(f'{test_dist+" ("+suffix+")"}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel(f'Mean Approx. Ratio',fontsize=fontsize)
        ax.set_xlabel(f'')
        sns.despine()
    
    # Set a common y-label
    fig.suptitle(f'Train Dist: {train_dist}', fontsize=16)
    # plt.ylabel('Ratio')
    # plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title

    # Show the plots
    plt.show()



