from utils import *


# for dist in ['BigMac','BQP']:
for dist in ['BA_800vertices_unweighted']:
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

    # Normalize each row by its maximum
    normalized_df = df.div(df.max(axis=1), axis=0)

    # Calculate mean and std of each column
    mean_values = normalized_df.mean()
    std_values = normalized_df.std()

    # Create a new DataFrame with mean and std as columns
    # Create a new DataFrame with mean and std, rounded to 3 decimal places
    summary_df = pd.DataFrame({
        'Mean': mean_values.round(3),
        'Std': std_values.round(3)
    })

    # Print the resulting DataFrame
    print(summary_df)

    # normalized_df = df.div(df.max(axis=1), axis=0)

    # # Calculate mean and std of each column
    # mean_values = normalized_df.mean()
    # std_values = normalized_df.std()

    # # Print the results
    # print("Mean of each column:\n", mean_values)
    # print("\nStandard deviation of each column:\n", std_values)

    # df['OPT_MAX']= df.max(axis=1)

    # print(df.mean(axis=0))

    # print(df)

    