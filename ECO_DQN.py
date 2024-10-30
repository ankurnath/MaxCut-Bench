import matplotlib.pyplot as plt
import numpy as np
from utils import *
import matplotlib.ticker as ticker

for i,dist in enumerate([
                        'ER',
                        'planar',
                        'torodial'
                        ]):
    for suffix in [
                    'unweighted',
                    'weighted'
                    ]:
        distributions = []
        df = defaultdict(list)
        for n in [800,1000,2000,3000,5000,7000,10000]:
            distribution = f'{dist}_{n}vertices_{suffix}'

            try:
                OPT = load_from_pickle(f'../data/testing/{distribution}/optimal')
                print(OPT)
            except:
                # raise ValueError
                pass

            for algorithm in [
                        'ECO-DQN',
                        'SoftTabu',
                        
                        'LS-DQN',
                        # 'Forward Greedy',
                        
                        # 'RG',
                        # 'Forward Greedy',
                        # 'Standard Greedy',
                        'TS',
                        # 'S2V-DQN',
                        # 'EO',
                        # 'ANYCSP'
                        ]:
                
                try:
                    df_ = load_from_pickle(f'results/{distribution}/{algorithm}')

                    if df_.empty:
                        pass
                    else:
                        df ['Ratio'].append((df_['cut'].values/OPT['OPT'].values).mean())
                        df['N'].append(n)

                        if algorithm == 'SoftTabu':
                            df['algorithm'].append('ECO+LR')

                        elif algorithm == 'RG':
                            df['algorithm'].append('LS-Simplified')
                        else:
                            df['algorithm'].append(algorithm)
                except:
                    pass

        # print(df)
        df = pd.DataFrame(df)
        print(df)

        if df.empty:
            continue               
        fontsize = 30
        markersize = 45

        plt.figure(figsize=(8,6),dpi=200)
        
        markers = {'S2V-DQN': '*','Greedy':'<','ECO-DQN':'P',
                   'S2V-Simplified':'p','LS-DQN':'*','LS-Simplified':'>',
                   'ECO+LR':'<','TS':'.','EO':'H','ANYCSP':'.'}  # specify markers for each algorithm
        sns.lineplot(x='N', y='Ratio', hue='algorithm', style='algorithm', 
                     markers=markers, data=df, markersize=markersize,
                     legend=True)
        # sns.lineplot(x='N', y='Ratio', hue='algorithm', style='algorithm',data=df, markersize=20)
        
        plt.xlabel('Graph Size,|V|',fontsize=fontsize+10)
        plt.ylabel('Approx. Ratio',fontsize=fontsize+10)

        plt.xticks(fontsize=fontsize)
        # plt.xticks([20,40,60,100,200,500],fontsize=fontsize)
        plt.yticks(fontsize=fontsize+10)

        title = dist+ ' '+ suffix 
        # plt.title(dist+ ' '+ suffix)

        # plt.locator_params(nbins=6)
        # plt.xticks([20,40,60,100,200,500])
        # plt.xscale('log')
        # # Set custom xticks on a logarithmic scale using LogLocator
        # ax = plt.gca()  # Get current axis
        # ax.set_xticks([20, 40, 60, 100, 200, 500])  # Specify the exact tick locations
        # ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Ensure the tick labels are formatted properly
        # plt.xticks([20,40,60,100,200,500])

        # ax = plt.gca()  # Get the current axis
        # ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[0.0,1.0, 2.0, 5.0]))  # Major ticks at log scale

        
        plt.grid(True,linestyle='--', alpha=0.7)

        plt.locator_params(nbins=6)
        if dist == 'torodial':
            plt.legend(frameon=False,fontsize=fontsize+5,ncol=1,loc='best')
            # plt.legend(['Label 1', 'Label 2', 'Label 3'])
            # plt.legend(['S2V-DQN','S2V-Simplified','Greedy'],frameon=False,fontsize=fontsize,ncol=1,loc='best')
        else:
            # pass
            # # Add a legend
            legend = plt.legend()

            # # Remove the legend if needed
            legend.remove()  # This will remove the legend
        if dist == 'torodial':
            plt.title(f'Toroidal ({suffix.capitalize()})',fontsize=fontsize+10)

        elif dist == 'planar':
            plt.title(f'Planar ({suffix.capitalize()})',fontsize=fontsize+10)
        else:
            plt.title(f'{dist} ({suffix.capitalize()})',fontsize=fontsize+10)
        #     dist = 'Toroidal'
        # plt.title(f'{dist}({suffix})',fontsize=fontsize)
        sns.despine()
        save_folder = 'ECO_DQN'

        os.makedirs(save_folder,exist_ok=True)
        file_path = os.path.join(save_folder,f'{title}.pdf')
        # plt.savefig(f'{title}.png') 
        # plt.savefig(f'{title}.png', bbox_inches='tight')
        plt.savefig(file_path, bbox_inches='tight')
        # plt.show()