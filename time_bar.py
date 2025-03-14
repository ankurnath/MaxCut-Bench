import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

root_folder = 'results'
save_folder = 'plots'
os.makedirs(save_folder, exist_ok=True)

dataset_titles = {  
    'ER_800vertices_unweighted': 'ER800 (Unweighted)',
    'planar_800vertices_unweighted': 'Planar800 (Unweighted)',
    'BA_800vertices_unweighted': 'BA800 (Unweighted)',
    'WattsStrogatz_800vertices_unweighted': 'WS800 (Unweighted)',
    'HomleKim_800vertices_unweighted': 'HK800 (Unweighted)',
    'dense_MC_100_200vertices_unweighted':'Phase Transition (Unweighted)',
    'torodial_800vertices_weighted': 'Torodial800 (Weighted)',
    'ER_800vertices_weighted': 'ER800 (Weighted)',
    'planar_800vertices_weighted': 'Planar800 (Weighted)',
    'BA_800vertices_weighted': 'BA800 (Weighted)',
    'WattsStrogatz_800vertices_weighted': 'WS800 (Weighted)',
    'HomleKim_800vertices_weighted': 'HK800 (Weighted)',
    'ER_200vertices_weighted': 'ER200 (Weighted)',
    'BA_200vertices_weighted': 'BA200 (Weighted)',
    'SK_spin_70_100vertices_weighted': 'SK70-100 (Weighted)',
    'Physics': 'Physics (Weighted)',


}


algorithm_names = {
    
    'Gurobi': 'Gurobi',
    'Cplex': 'CPLEX',
    'LS_Simplified': 'Greedy',
    'SDP':'SDP',
    'TS': 'TS',
    'EO': 'EO',
    'S2V-DQN': 'S2V-DQN',
    'ECO-DQN': 'ECO-DQN',
    'SoftTabu': 'ECO+LR',
    'LS-DQN': 'LS-DQN',
    'Gflow': 'Gflow',
    'RUN-CSP': 'RUN-CSP',
    'ANYCSP': 'ANYCSP',
    
    
    
}

for dataset in os.listdir(root_folder):
# for dataset in ['ER_800vertices_unweighted']:
    dataset_folder = os.path.join(save_folder, dataset)
    # os.makedirs(dataset_folder, exist_ok=True)
    
    OPT = pd.read_pickle(os.path.join('data/testing', dataset, 'optimal'))
    OPT = OPT['OPT'].values
    results = []

    # Compute optimal values
    for algorithm in os.listdir(os.path.join(root_folder, dataset)):

        
    # for algorithm in algorithm_names.keys():
        data = pd.read_pickle(os.path.join(root_folder, dataset, algorithm))

        if algorithm == 'Gurobi' or algorithm == 'Cplex' or algorithm == 'SDP':
            pass
        else:
            OPT = np.maximum(OPT, data['cut'].values)
    
    # Collect results for plotting
    # for algorithm in os.listdir(os.path.join(root_folder, dataset)):
    for algorithm in algorithm_names.keys():
        if dataset.endswith('unweighted'):
            pass

        elif (dataset.endswith('weighted') or dataset == 'Physics') and algorithm == 'Gflow':
            continue
        data = pd.read_pickle(os.path.join(root_folder, dataset, algorithm))
        mean_approximation_ratio = (data['cut'].values / OPT[:len(data)]).mean()
        # mean_approximation_ratio = (data['cut'].values / OPT).mean()
        try:
            mean_time = data['time'].mean()
        except:
            mean_time = data['Time'].mean()
            # print(data.columns)
            # print(f'{algorithm} {dataset}')
            # raise ValueError
        # mean_time = data['time'].mean()
        results.append((algorithm_names[algorithm], mean_time, mean_approximation_ratio))
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame(results, columns=['Algorithm', 'Time', 'Approximation Ratio'])

    
    
    # Plot
    # plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax = sns.barplot(x='Algorithm', y='Time', data=df, palette='rocket',hue='Algorithm',legend=False)

    if df['Time'].max() / df['Time'].min() > 100:
        ax.set_yscale('log')
    
    # Annotate bars with Approximation Ratio
    for i, (algorithm, time, ratio) in enumerate(results):
        ax.text(i, time, f'{ratio:.3f}', ha='center', va='bottom', fontsize=11, color='black')
    
    fontsize = 16
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{dataset_titles[dataset]}', fontsize=fontsize)
    # plt.xlabel('Algorithm',fontsize=fontsize)
    plt.ylabel('Time (s)',fontsize=fontsize)
    plt.xlabel('',fontsize=fontsize)
    plt.xticks(rotation=45)
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    # save_path = os.path.join(dataset_folder, f'{dataset}.pdf')
    save_path = os.path.join(save_folder, f'{dataset}.pdf')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight',dpi=300)
    plt.close()
    # break
