import os
import pandas as pd
from tabulate import tabulate  # Install using: pip install tabulate
import numpy as np



root_folder = 'results'
dataset_results = {}

# Collecting data
for dataset in os.listdir(root_folder):
    OPT = pd.read_pickle(os.path.join('data/testing', dataset, 'optimal'))
    OPT = OPT['OPT'].values
    dataset_results[dataset] = {}

    for algorithm in os.listdir(os.path.join(root_folder, dataset)):
        data = pd.read_pickle(os.path.join(root_folder, dataset, algorithm))
        OPT = np.maximum(OPT,data['cut'].values)
        

    for algorithm in os.listdir(os.path.join(root_folder, dataset)):
        data = pd.read_pickle(os.path.join(root_folder, dataset, algorithm))
        mean_approximation_ratio = (data['cut'].values / OPT).mean()
        dataset_results[dataset][algorithm] = f"{mean_approximation_ratio:.4f}"

# Dynamically determine all algorithm names for consistent columns
all_algorithms = sorted({algo for results in dataset_results.values() for algo in results})

# Prepare table data
table_data = []
for dataset, results in dataset_results.items():
    row = [dataset] + [results.get(algo, "N/A") for algo in all_algorithms]
    table_data.append(row)

# Print as a formatted table
headers = ["Dataset"] + all_algorithms
print(tabulate(table_data, headers=headers, tablefmt="pretty"))



# import os
# import pandas as pd

# root_folder = 'results'

# for dataset in os.listdir(root_folder):
#     print("=" * 50)
#     print(f"📂 Dataset: {dataset}")
#     print("=" * 50)

#     OPT = pd.read_pickle(os.path.join('data/testing', dataset, 'optimal'))

#     for algorithm in os.listdir(os.path.join(root_folder, dataset)):
#         print(f"\n🔹 Algorithm: {algorithm}")
        
#         data = pd.read_pickle(os.path.join(root_folder, dataset, algorithm))
#         mean_approximation_ratio = (data['cut'] / OPT['OPT']).mean()

#         print(f"   ➤ Mean Approximation Ratio: {mean_approximation_ratio:.4f}")

# print("\n✅ Processing complete!")
