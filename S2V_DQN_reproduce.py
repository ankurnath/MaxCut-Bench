import matplotlib.pyplot as plt
import numpy as np
from utils import *
import matplotlib.ticker as ticker

for i,dist in enumerate([
                        'ER',
                        'BA',
                        # 'torodial'
                        ]):
    for suffix in [
                    # 'unweighted',
                    'weighted'
                    ]:
        distributions = []
        df = defaultdict(list)
        for n in [20,40,60,100,200]:
            distribution = f'{dist}_{n}vertices_{suffix}'

            try:
                OPT = load_from_pickle(f'../data/testing/{distribution}/optimal')
                print(OPT)
            except:
                # raise ValueError
                pass

            for algorithm in [
                        # 'ECO-DQN',
                        # 'SoftTabu',
                        
                        # 'LS-DQN',
                        # 'Forward Greedy',
                        
                        # 'RG',
                        'S2V-DQN',
                        'Forward Greedy',
                        'Standard Greedy',
                        # 'TS',
                        
                        # 'EO',
                        # 'ANYCSP'
                        ]:
                
                try:
                    df_ = load_from_pickle(f'results/{distribution}/{algorithm}')

                    if df_.empty:
                        pass
                    else:
                        # print(df_['cut'].values/OPT['OPT'].values)
                        df ['Ratio'].append((df_['cut'].values/OPT['OPT'].values).mean())
                        df ['Ratio (STD)'].append((df_['cut'].values/OPT['OPT'].values).std())
                        df['N'].append(n)

                        if algorithm == 'Forward Greedy':
                            df['algorithm'].append('S2V-Simplified')

                        elif algorithm == 'Standard Greedy':
                            df['algorithm'].append('Greedy')

                        elif algorithm == 'S2V-DQN':
                            df['algorithm'].append('S2V-DQN (OURS)')
                        else:
                            df['algorithm'].append(algorithm)
                except:
                    pass

        # print(df)
        if dist =='ER':
            df['algorithm']+= ['S2V-DQN (Pre-trained)']*5
            df['N'] += [20,40,60,100,200]
            df['Ratio']+= [0.97,0.98,0.97,0.92,0.95]
            df ['Ratio (STD)'] += [0.031,0.039,0.0412,0.03,0.036]
        elif dist =='BA':
            df['algorithm']+= ['S2V-DQN (Pre-trained)']*4
            df['N'] += [20,40,100,200]
            df['Ratio']+= [0.97,0.96,0.95,0.93]
            df ['Ratio (STD)'] +=[0.03,0.032,0.041,0.03]
        df = pd.DataFrame(df)
        print(df)

        if df.empty:
            continue               
        fontsize = 30
        markersize = 20

        plt.figure(dpi=300)
        
        markers = {'S2V-DQN (OURS)': '*','Greedy':'<',
                   'S2V-Simplified':'p','LS-DQN':'*','RG':'o',
                   'SoftTabu':'p','TS':'.','S2V-DQN (Pre-trained)':'.','ANYCSP':'.'}  # specify markers for each algorithm
        sns.lineplot(x='N', y='Ratio', hue='algorithm', style='algorithm', 
                     markers=markers, data=df, markersize=markersize,
                     legend=True)
        
        algorithms = df['algorithm'].unique()  # Get all the unique algorithms
        for algorithm in algorithms:
            subset = df[df['algorithm'] == algorithm]
            plt.fill_between(
                subset['N'], 
                subset['Ratio'] - subset['Ratio (STD)']/2, 
                subset['Ratio'] + subset['Ratio (STD)']/2, 
                alpha=0.1  # Adjust transparency
            )
        # sns.lineplot(x='N', y='Ratio', hue='algorithm', style='algorithm',data=df, markersize=20)
        
        plt.xlabel('Graph Size,|V|',fontsize=fontsize)
        plt.ylabel('Mean Approx. Ratio',fontsize=fontsize-2)

        plt.xticks(fontsize=fontsize)
        # plt.xticks([20,40,60,100,200,500],fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

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
        if dist == 'BA':
            plt.legend(frameon=False,fontsize=fontsize-15,ncol=1,loc='best')
            # plt.legend(['Label 1', 'Label 2', 'Label 3'])
            # plt.legend(['S2V-DQN','S2V-Simplified','Greedy'],frameon=False,fontsize=fontsize,ncol=1,loc='best')
        else:
            # pass
            # # Add a legend
            legend = plt.legend()

            # # Remove the legend if needed
            legend.remove()  # This will remove the legend
        # if dist == 'torodial':
        #     plt.title(f'Toroidal ({suffix.capitalize()})',fontsize=fontsize+10)

        # elif dist == 'planar':
        #     plt.title(f'Planar ({suffix.capitalize()})',fontsize=fontsize+10)
        # else:
        #     plt.title(f'{dist} ({suffix.capitalize()})',fontsize=fontsize+10)
        #     dist = 'Toroidal'
        # plt.title(f'{dist}({suffix})',fontsize=fontsize)
        sns.despine()
        save_folder = 'S2V_DQN'

        os.makedirs(save_folder,exist_ok=True)
        title = dist+'_reproduce'
        file_path = os.path.join(save_folder,f'{title}.pdf')
        # plt.savefig(f'{title}.png') 
        # plt.savefig(f'{title}.png', bbox_inches='tight')
        plt.savefig(file_path, bbox_inches='tight')
        # plt.show()