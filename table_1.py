from utils import *


for dist in ['BigMac','BQP']:
# for dist in [
#     'BA_800vertices_unweighted',
#     'HomleKim_800vertices_unweighted',
#     'WattsStrogatz_800vertices_unweighted',
#     'BA_800vertices_weighted',
#     'HomleKim_800vertices_weighted',
#     'WattsStrogatz_800vertices_weighted',
#     ]:
    root_folder = f'results/{dist}'
    files = os.listdir(root_folder)
    # files.sort()
    
    

    # df = defaultdict(list)
    df ={}
    try:
        OPT = pd.read_pickle(f'../data/testing/{dist}/optimal')
        df['OPT'] = OPT['OPT']
    except:
        pass

    

    for file in files:
        file_path = os.path.join(root_folder,file)

        data = load_from_pickle(file_path=file_path)

        df[file] = data['cut']

        # data['ratio'] = data['cut']/OPT['OPT']
        # print(data)
        # break

    df = pd.DataFrame(df)

    fontsize = 20

    # Normalize each row by its maximum
    normalized_df = df.div(df.max(axis=1), axis=0)
    selected_columns = ['EO','TS','S2V-DQN','ECO-DQN','SoftTabu','ANYCSP','RUN-CSP']
    normalized_df = normalized_df[selected_columns]

    # Rename 'SoftTabu' to 'ECO+LR'
    normalized_df = normalized_df.rename(columns={'SoftTabu': 'ECO+LR'})
    print(normalized_df)
    # df = normalized_df
    plt.figure(dpi=200)


    # # Calculate mean and standard deviation
    mean_values = normalized_df.mean()
    std_values = normalized_df.std()

    print(mean_values)
    print(std_values)

    bar_plot = sns.barplot(x=mean_values.index, y=mean_values.values, palette='Spectral',linewidth=1.5)
    # Adding edge color to each bar
    for bar in bar_plot.patches:
        bar.set_edgecolor('black')
    # Add error bars manually using matplotlib's errorbar function
    plt.errorbar(x=range(len(mean_values)), y=mean_values.values, 
                yerr=std_values.values, fmt='none', c='black',elinewidth=2)
    # sns.barplot(x=mean_values.index, y=mean_values.values, palette="rocket")
    plt.xlabel('')
    plt.ylabel('Mean Approx. Ratio',fontsize=fontsize)
   
    sns.despine()

    # Rotate x-axis labels
    plt.xticks(rotation=90,fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Remove top and right spines
    sns.despine()

    # Save the figure as PDF
    plt.savefig(f'{dist}.pdf', format='pdf', bbox_inches='tight')

    # # Show the plot
    # plt.show()
    # plt.show()
    

    # # Prepare the DataFrame for plotting
    # plot_data = pd.DataFrame({
    #     'Algorithm': mean_values.index,
    #     'Mean': mean_values.values,
    #     'Std': std_values.values
    # })

    

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # # sns.barplot(x='Algorithm', y='Mean', data=plot_data, yerr=plot_data['Std'], capsize=0.1, palette='Blues')
    
    # bar_plot = sns.barplot(x='Algorithm', y='Mean', data=plot_data, palette='Blues')

  
        

    # sns.barplot(data=normalized_df, x='algorithm', y='Ratio', ax=ax,palette='rocket')

    # print(normalized_df)

    # # Calculate mean and std of each column
    # mean_values = normalized_df.mean()
    # std_values = normalized_df.std()

    # # Create a new DataFrame with mean and std as columns
    # # Create a new DataFrame with mean and std, rounded to 3 decimal places
    # summary_df = pd.DataFrame({
    #     'Mean': mean_values.round(3),
    #     'Std': std_values.round(3)
    # })

    # df = summary_df
   
    # df.rename(columns={'index': 'Algorithm'}, inplace=True)
    # sns.barplot(x='Algorithm', y='Mean', data=df, yerr=df['Std'], capsize=0.1, palette='Blues')

    # # Print the resulting DataFrame

    
    # print(df)

    

    