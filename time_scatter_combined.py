import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
import re
root_folder = 'results'
save_folder = 'plots_scatter_png'
os.makedirs(save_folder, exist_ok=True)

dataset_titles = {  
    'ER_800vertices_unweighted': 'ER800 (Unweighted)',
    'planar_800vertices_unweighted': 'Planar800 (Unweighted)',
    'BA_800vertices_unweighted': 'BA800 (Unweighted)',
    'WattsStrogatz_800vertices_unweighted': 'WS800 (Unweighted)',
    'HomleKim_800vertices_unweighted': 'HK800 (Unweighted)',
    'dense_MC_100_200vertices_unweighted':'Phase Transition (Unweighted)',
    'ER_800vertices_weighted': 'ER800 (Weighted)',
    'planar_800vertices_weighted': 'Planar800 (Weighted)',
    'torodial_800vertices_weighted': 'Torodial800 (Weighted)',
    'BA_800vertices_weighted': 'BA800 (Weighted)',
    'WattsStrogatz_800vertices_weighted': 'WS800 (Weighted)',
    'HomleKim_800vertices_weighted': 'HK800 (Weighted)',
    'ER_200vertices_weighted': 'ER200 (Weighted)',
    'BA_200vertices_weighted': 'BA200 (Weighted)',
    'SK_spin_70_100vertices_weighted': 'SK70-100 (Weighted)',
    'Physics': 'Physics (Weighted)',


}
print(len(dataset_titles))


algorithm_names = {
    
    'Gurobi': 'Gurobi',
    'Cplex': 'CPLEX',
    'LS_Simplified': 'Greedy',
    'TS': 'TS',
    'EO': 'EO',
    'S2V-DQN': 'S2V-DQN',
    'ECO-DQN': 'ECO-DQN',
    'SoftTabu': 'ECO+LR',
    'LS-DQN': 'LS-DQN',
    'Gflow': 'Gflow-CombOpt',
    'RUN-CSP': 'RUN-CSP',
    'ANYCSP': 'ANYCSP',
    'SDP':'SDP'
    
    
}


print(len(os.listdir(root_folder)))
results = []
for dataset in os.listdir(root_folder):

    dataset_folder = os.path.join(save_folder, dataset)

    # Load optimal values
    OPT = pd.read_pickle(os.path.join('data/testing', dataset, 'optimal'))
    OPT = OPT['OPT'].values
    

    # Compute optimal values
    for algorithm in os.listdir(os.path.join(root_folder, dataset)):
        if dataset.endswith('weighted') and algorithm == 'Gflow':
            continue
        data = pd.read_pickle(os.path.join(root_folder, dataset, algorithm))
        if algorithm not in ['Gurobi', 'Cplex','SDP']:
            OPT = np.maximum(OPT, data['cut'].values)

    # Collect results for scatter plot
    for algorithm in algorithm_names.keys():
        if dataset.endswith('unweighted'):
            pass

        elif (dataset.endswith('weighted') or dataset == 'Physics') and algorithm == 'Gflow':
            continue
        data = pd.read_pickle(os.path.join(root_folder, dataset, algorithm))
        mean_approx_ratio = (data['cut'].values / OPT[:len(data)]).mean()
        mean_time = data['time'].mean() if 'time' in data else data['Time'].mean()

        match = re.search(r"\((.*?)\)", dataset_titles[dataset])
        if match:
            extracted_text = match.group(1)

        if extracted_text == 'Weighted':
            bool_val = True
        else:
            bool_val = False
        results.append((algorithm_names[algorithm],dataset_titles[dataset],bool_val, mean_time, mean_approx_ratio))
    
df = pd.DataFrame(results, columns=['Algorithm','Dataset','Weighted','Time', 'Approximation Ratio'])


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Compute mean Approximation Ratio & Time for each Algorithm separately for weighted and unweighted datasets
df_grouped = df.groupby(['Algorithm', 'Weighted']).agg({'Time': 'mean', 'Approximation Ratio': 'mean'}).reset_index()

# Split into weighted and unweighted
df_weighted = df_grouped[df_grouped['Weighted'] == True]
df_unweighted = df_grouped[df_grouped['Weighted'] == False]

# Function to plot scatter plot
def plot_scatter(df, title, save_name):
    fig, ax = plt.subplots(figsize=(8, 5))

    algorithm_colors = {
    'Gurobi': '#1f77b4',  # Blue
    'CPLEX': '#ff7f0e',   # Orange
    'Greedy': '#2ca02c',  # Green
    'TS': '#d62728',      # Red
    'EO': '#9467bd',      # Purple
    'S2V-DQN': '#8c564b', # Brown
    'ECO-DQN': '#e377c2', # Pink
    'ECO+LR': '#7f7f7f',  # Gray
    'LS-DQN': '#bcbd22',  # Yellow-Green
    'Gflow-CombOpt': '#17becf', # Cyan
    'RUN-CSP': '#fec76f', # Dark Red
    'ANYCSP': '#ff9896',  # Light Red
    'SDP': '#670067'      # Light Blue
    }


    ax = sns.scatterplot(
        x='Time', 
        y='Approximation Ratio', 
        hue='Algorithm', 
        size='Approximation Ratio',  
        sizes=(50, 500),  
        data=df, 
        palette=algorithm_colors,  # Use custom colors
        edgecolor='black', 
        legend=False
    )

    # # Scatter plot
    # sns.scatterplot(
    #     x='Time', 
    #     y='Approximation Ratio', 
    #     hue='Algorithm', 
    #     size='Approximation Ratio',
    #     sizes=(50, 500),  
    #     data=df, 
    #     palette='tab10',  
    #     edgecolor='black', 
    #     legend=False
    # )

    

    # Log scale for better visualization
    if df['Time'].max() / df['Time'].min() > 100:
        ax.set_xscale('log')


    # for _, row in df.iterrows():
    #     ax.text(row['Time'],  # Shift right (increase x slightly)
    #             row['Approximation Ratio'],  # Shift up
    #             row['Algorithm'],
    #             fontsize=12, ha='left', va='bottom')

    # Labels and formatting
    fontsize = 16
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Mean Time (s)', fontsize=fontsize)
    plt.ylabel('Mean Approximation Ratio', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    sns.despine()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save plot
    plt.tight_layout()
    plt.savefig(save_name, format='png', bbox_inches='tight', dpi=300)
    plt.close()

# Plot and save figures
plot_scatter(df_weighted, "Weighted Datasets", "scatter_weighted.png")
plot_scatter(df_unweighted, "Unweighted Datasets", "scatter_unweighted.png")

