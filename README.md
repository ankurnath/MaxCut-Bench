# Maximum Cut Benchmarking Suite

This repository contains a benchmarking suite for maximum cut
solvers. The suite bundles several solvers and datasets with a unified
interface. Currently, we support the following solvers:

- S2V-DQN
- ECO-DQN
- LS-DQN
- SoftTabu
- RUN-CSP
- ANYCSP
- Gflow-CombOpt
- Amplitude Heterogeneity Correction (AHC)
- Chaotic Amplitude Control (CAC)
- Tabu Search
- Extremal Optimization
- Standard Greedy 
- Stochastic Greedy
- Forward Greedy


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

    >>> python solvers/Gurobi/evaluate.py  --test_distribution BA_20


### S2V-DQN

To train 

    >>> python train.py --algorithm S2V-DQN --distribution BA_20

To evaluate 

    >>> python evaluation.py --algorithm S2V-DQN --train_distribution BA_20 --test_distribution BA_20


### ECO-DQN

To train 

    >>> python train.py --algorithm ECO-DQN --distribution BA_20 --num_steps 40

To evaluate 

    >>> python evaluation.py --algorithm ECO-DQN --train_distribution BA_20 --test_distribution BA_20 --num_repeat 50 --num_steps 40


### LS-DQN

To train 

    >>> python train.py --algorithm LS-DQN --distribution BA_20 --num_steps 40

To evaluate 

    >>> python evaluation.py --algorithm LS-DQN --train_distribution BA_20 --test_distribution BA_20 --num_repeat 50 --num_steps 40

### SoftTabu

To train 

    >>> python train.py --algorithm SoftTabu --distribution BA_20 --num_steps 40

To evaluate 

    >>> python evaluation.py --algorithm SoftTabu --train_distribution BA_20 --test_distribution BA_20 --num_repeat 50 --num_steps 40



### RUN-CSP

To train 

    >>> python train.py --algorithm RUN-CSP --distribution BA_20 

To evaluate 

    >>> python evaluation.py --algorithm RUN-CSP --train_distribution BA_20 --test_distribution BA_20 --num_repeat 50 --num_steps 40


### ANYCSP

To train 

    >>> python train.py --algorithm ANYCSP --distribution BA_20 

To evaluate 

    >>> python evaluation.py --algorithm ANYCSP --train_distribution BA_20 --test_distribution BA_20 --num_repeat 50 --num_steps 40

### Gflow-CombOpt

To train 

    >>> python train.py --algorithm Gflow-CombOpt --distribution BA_20 

To evaluate 

    >>> python evaluation.py --algorithm Gflow-CombOpt --train_distribution BA_20 --test_distribution BA_20 --num_repeat 50 


### Amplitude Heterogeneity Correction

To evaluate 

    >>> python evaluation.py --algorithm AHC --test_distribution BA_20 

Hyperparameter tuning for each instance is done on the fly.

### Chaotic Amplitude Control

To evaluate 

    >>> python evaluation.py --algorithm CAC --test_distribution BA_20 

Hyperparameter tuning for each instance is done on the fly.

### Tabu Search

To train

    >>> python train.py --algorithm TS --distribution BA_20 --num_repeat 50 --num_steps 40 --low 20 --high 150 --step 10

To test

    >>> python evaluation.py --algorithm TS --train_distribution BA_20 --test_distribution BA_20 --num_repeat 50  --num_steps 40


### Extremal Optimization

To train

    >>> python train.py --algorithm EO --distribution BA_20 --num_repeat 50 --num_steps 40 --low 1.1 --high 2 --step 0.1

To test

    >>> python evaluation.py --algorithm EO --train_distribution BA_20 --test_distribution BA_20 --num_repeat 50  --num_steps 40

### Greedy

To test

    >>> python evaluation.py --algorithm Greedy --test_distribution BA_20 --num_repeat 50  


### Data and Models

We cannot reupload publicly available datasets, as we do not have permission to redistribute them. Therefore, we only provide synthetic datasets generated with the best-known solutions, along with all the [pretrained models](https://drive.google.com/file/d/1gGoIQ1LhzomLS0hhzpAjnslVet4faZIo/view?usp=sharing). Additionally, we offer all the [data](https://drive.google.com/file/d/1LJ0kjavA9wIjnIpmkI38-wlf2C77p3dy/view?usp=sharing) needed for training and validation across all distributions.


