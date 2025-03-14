import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
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
    
    
}


print(len(os.listdir(root_folder)))
for dataset in os.listdir(root_folder):
    dataset_folder = os.path.join(save_folder, dataset)

    # Load optimal values
    OPT = pd.read_pickle(os.path.join('data/testing', dataset, 'optimal'))
    OPT = OPT['OPT'].values
    results = []

    # Compute optimal values
    for algorithm in os.listdir(os.path.join(root_folder, dataset)):
        if dataset.endswith('weighted') and algorithm == 'Gflow':
            continue
        data = pd.read_pickle(os.path.join(root_folder, dataset, algorithm))
        if algorithm not in ['Gurobi', 'Cplex']:
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
        results.append((algorithm_names[algorithm], mean_time, mean_approx_ratio))
    
    # Convert to DataFrame
    df = pd.DataFrame(results, columns=['Algorithm', 'Time', 'Approximation Ratio'])

    # Plot scatter plot
    fig, ax = plt.subplots(figsize=(8, 5))
    # ax = sns.scatterplot(x='Time', y='Approximation Ratio', hue='Algorithm', data=df, palette='tab10', s=200, edgecolor='black',legend=False)
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
    }

    # Scatter plot with custom colors
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

    # ax = sns.scatterplot(x='Time', 
    #                  y='Approximation Ratio', 
    #                  hue='Algorithm', 
    #                  size='Approximation Ratio',  # Scale by Approximation Ratio
    #                  sizes=(50, 500),  # Set min and max marker size
    #                  data=df, 
    #                  palette='tab10', 
    #                  edgecolor='black', 
    #                  legend=False
    #                  )

    # Annotate each point
    # Annotate each point with a slight shift
    # for _, row in df.iterrows():
    #     ax.text(row['Time'],  # Shift right (increase x slightly)
    #             row['Approximation Ratio'],  # Shift up
    #             row['Algorithm'],
    #             fontsize=12, ha='left', va='bottom')
        
    # texts = []
    # for _, row in df.iterrows():
    #     texts.append(ax.text(
    #         row['Time'], row['Approximation Ratio'], row['Algorithm'],
    #         fontsize=12, ha='left', va='bottom'
    #     ))

    # adjust_text(texts, ax=ax, expand=(1.2, 1.5))

    # Log-scale for better visualization if needed
    if df['Time'].max() / df['Time'].min() > 100:
        ax.set_xscale('log')

    fontsize = 14
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Time', fontsize=fontsize)
    plt.ylabel('Approximation Ratio', fontsize=fontsize)
    plt.title(dataset_titles[dataset], fontsize=fontsize)
    sns.despine()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save figure
    save_path = os.path.join(save_folder, f'scatter_{dataset}.png')
    plt.tight_layout()
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()
