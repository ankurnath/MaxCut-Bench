import matplotlib.pyplot as plt
from utils import *

distributions =[
                'ER_800vertices_unweighted',
                'planar_800vertices_unweighted',
                'BA_800vertices_unweighted',
                'WattsStrogatz_800vertices_unweighted',
                # 'HomleKim_800vertices_unweighted',
                
                
                ]

mapping={
    'WattsStrogatz_800vertices_unweighted':'WS800',
    'BA_800vertices_unweighted':'BA800',
    'HomleKim_800vertices_unweighted':'HK800',
    'ER_800vertices_unweighted':'Gset(ER800)',
    'planar_800vertices_unweighted':'Gset(Planar800)'

}


# Set font size parameters
font_size = 20
title_size = 24
tick_size = 16

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=len(distributions), figsize=(22, 6), dpi=200)
# fig, axes = plt.subplots(nrows=1, ncols=len(distributions),  dpi=300)

for i, (ax, distribution) in enumerate(zip(axes, distributions)):
    folder_path = os.path.join('results', distribution)
    files = os.listdir(folder_path)
    df = {}
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        data = load_from_pickle(file_path=file_path)
        df[file] = data['cut']
    
    df = pd.DataFrame(df)
    print(df)
    selected_algorithms = ['Standard Greedy','TS','EO','GFlow','ANYCSP']  # Corrected the typo
    df_clean = df[selected_algorithms]

    df_clean = df_clean.rename(columns={
            'GFlow': 'GFlow-CombOPT',
            'ANYCSP': 'ANYCSP',
            'Standard Greedy': 'Greedy',
            'TS': 'TS',
            'EO': 'EO'
        })
    
    # Melt the cleaned dataframe for plotting
    df_long = pd.melt(df_clean, var_name='Algorithms', value_name='Value')
    font_size = 20
    sns.violinplot(x='Algorithms', y='Value', data=df_long, ax=ax, 
                   inner='box', fill=False, palette='rocket',
                   linewidth=2)
    
    if i == 0:
        ax.set_ylabel('Objective value', fontsize=font_size)
    else:
        ax.set_ylabel('')  # Set an empty label if needed
    ax.set_xlabel('')
    # Set the tick font size for both axes
    # Increase tick font sizes
    ax.tick_params(axis='x', labelsize=tick_size,labelrotation=90)
    ax.tick_params(axis='y', labelsize=tick_size)
    
    # Set the title with increased font size
    ax.set_title(mapping.get(distribution, distribution), fontsize=title_size)
    sns.despine()
    # ax.set_title(mapping.get(distribution, distribution))

# plt.tight_layout()
# plt.show()
plt.savefig('violent_unweighted.pdf',dpi=300,bbox_inches='tight')