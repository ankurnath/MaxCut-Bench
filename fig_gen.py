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
                # OPT['OPT'] = OPT['OPT'].values
                # OPT = OPT['OPT'].tolist()
                # print('Loaded data')
            except:
                # raise ValueError
                pass

            for algorithm in [
                        'ECO-DQN',
                        'S2V-DQN',
                        # 'LS-DQN',
                        'Forward Greedy',
                        # 'TS',
                        # 'Standard Greedy',
                        # 'Forward Greedy'
                        ]:
                
                try:
                    df_ = load_from_pickle(f'results/{distribution}/{algorithm}')

                    if df_.empty:
                        pass
                    else:
                        df ['Ratio'].append((df_['cut'].values/OPT['OPT'].values).mean())
                        df['N'].append(n)
                        df['algorithm'].append(algorithm)
                    # print(df_['cut'].values)
                    # print(OPT['OPT'].values)      
                    
                except:
                    pass

        # print(df)
        df = pd.DataFrame(df)
        print(df)

        if df.empty:
            continue               
        fontsize = 20
        

        plt.figure(dpi=200)
        markers = {'S2V-DQN': 'p','Standard Greedy':'D','ECO-DQN':'P','Forward Greedy':'s'}  # specify markers for each algorithm
        sns.lineplot(x='N', y='Ratio', hue='algorithm', style='algorithm', markers=markers, data=df, markersize=10)
        # sns.lineplot(x='N', y='Ratio', hue='algorithm', style='algorithm',data=df, markersize=20)
        
        plt.xlabel('Graph Size,|V|',fontsize=fontsize)
        plt.ylabel('Approx. Ratio',fontsize=fontsize)

        plt.xticks(fontsize=fontsize)
        # plt.xticks([20,40,60,100,200,500],fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(distribution)

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


        sns.despine()
        plt.show()