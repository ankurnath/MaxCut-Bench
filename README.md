# Maximum Cut Benchmarking Suite

This repository contains a benchmarking suite for maximum cut
solvers. The suite bundles several solvers and datasets with a unified
interface. Currently, we support the following solvers:

- S2V-DQN
- ECO-DQN
- LS-DQN
- SoftTabu (ECO+LR)
- RUN-CSP
- ANYCSP
- Gflow-CombOpt
- Amplitude Heterogeneity Correction (AHC)
- Chaotic Amplitude Control (CAC)
- Tabu Search
- Extremal Optimization
- Greedy
- Semidefinite Programming
- Gurobi
- Cplex



## Repository Contents

In `solvers`, you can find wrappers for the currently supported solvers. In `data`, there are three folders for training, testing, and validation. You need to save your graph in .npz format. For demonstration purposes, we provide an example of a small dataset with a BA distribution.

For using this suite, `conda` is required. You can download it from here : https://www.anaconda.com/download .

## Setup

The included [``environment.yml``](environment.yml) file will produce a working environment called ``benchenv``.

    >>> git clone https://github.com/ankurnath/MaxCut-Bench.git
    >>> cd MaxCut-Bench
    >>> conda env create -f environment.yml 
    >>> conda activate benchenv



## Details on how to use the Solvers

### Gurobi

To evaluate 

    >>> python solvers/Gurobi/evaluate.py  --test_distribution BA_20 --time_limit 10 --threads 10

### Cplex

To evaluate 

    >>> python solvers/Cplex/evaluate.py  --test_distribution BA_20 --time_limit 10 --threads 10

### SDP

To evaluate 

    >>> python solvers/SDP/evaluate.py  --test_distribution BA_20  --threads 10


### S2V-DQN

To train 

    >>> python solvers/S2V-DQN/train.py  --distribution BA_20

To evaluate 

    >>> python solvers/S2V-DQN/evaluate.py --train_distribution BA_20 --test_distribution BA_20


### ECO-DQN

To train 

    >>> python solvers/ECO-DQN/train.py --distribution BA_20

To evaluate 

    >>> python solvers/ECO-DQN/evaluate.py --train_distribution BA_20 --test_distribution BA_20


### LS-DQN

To train 

    >>> python solvers/LS-DQN/train.py --distribution BA_20

To evaluate 

    >>> python solvers/LS-DQN/evaluate.py --train_distribution BA_20 --test_distribution BA_20 

### SoftTabu

To train 

    >>> python solvers/SoftTabu/train.py --distribution BA_20

To evaluate 

    >>> python solvers/SoftTabu/evaluate.py --train_distribution BA_20 --test_distribution BA_20


### RUN-CSP

To train 

    >>> python solvers/RUN-CSP/train.py --distribution BA_20

To evaluate 

    >>> python solvers/RUN-CSP/evaluate.py --train_distribution BA_20 --test_distribution BA_20 --num_repeat 50 --num_steps 40


### ANYCSP

To train 

    >>> python solvers/ANYCSP/train.py --distribution BA_20 

To evaluate 

    >>> python solvers/ANYCSP/evaluate.py --train_distribution BA_20 --test_distribution BA_20 --num_repeat 50 --num_steps 40

### Gflow-CombOpt

To train 

    >>> python solvers/Gflow-CombOpt/train.py --distribution BA_20

To evaluate 

    >>> python solvers/Gflow-CombOpt/evaluate.py --train_distribution BA_20 --test_distribution BA_20 --num_repeat 50 


### Amplitude Heterogeneity Correction

To evaluate 

    >>> python solvers/AHC/evaluate.py  --test_distribution BA_20

Hyperparameter tuning for each instance is done on the fly.

### Chaotic Amplitude Control

To evaluate 

    >>> python solvers/CAC/evaluate.py  --test_distribution BA_20

Hyperparameter tuning for each instance is done on the fly.

### Tabu Search

To train

    >>> python solvers/TS/train.py  --train_distribution BA_20

To test

    >>> python solvers/TS/evaluate.py --train_distribution BA_20 --test_distribution BA_20


### Extremal Optimization

To train

    >>> python solvers/EO/train.py  --train_distribution BA_20

To test

    >>> python solvers/EO/evaluate.py --train_distribution BA_20 --test_distribution BA_20

### Greedy

To test

    >>> python solvers/Greedy/evaluate.py --test_distribution BA_20  





