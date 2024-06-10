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

In `solvers`, you can find the wrappers for the currently supported solvers. In `data`, you have to three folders for training, testing and validation. You have to save graph your graph in .npz format.

For using this suite, `conda` is required. You can download it from here : https://www.anaconda.com/download

## Setup

The included [``environment.yml``](environment.yml) file will produce a working environment called ``benchenv``.

    >>> git clone https://github.com/ankurnath/MaxCut-Bench.git
    >>> cd MaxCut-Bench
    >>> conda env create -f environment.yml 
    >>> conda active benchenv



## Details how to use the Solvers

### S2V-DQN

To train 

    >>> python train.py --algorithm S2V-DQN --distribution BA_20

To evaluate 

    >>> python evaluation.py --algorithm S2V-DQN --distribution BA_20


### ECO-DQN

To train 

    >>> python train.py --algorithm ECO-DQN --distribution BA_20 --num_steps 40

To evaluate 

    >>> python evaluation.py --algorithm ECO-DQN --distribution BA_20 --num_repeat 50 --num_steps 40


### LS-DQN

To train 

    >>> python train.py --algorithm LS-DQN --distribution BA_20 --num_steps 40

To evaluate 

    >>> python evaluation.py --algorithm LS-DQN --distribution BA_20 --num_repeat 50 --num_steps 40

### SoftTabu

To train 

    >>> python train.py --algorithm SoftTabu --distribution BA_20 --num_steps 40

To evaluate 

    >>> python evaluation.py --algorithm SoftTabu --distribution BA_20 --num_repeat 50 --num_steps 40



### RUN-CSP

To train 

    >>> python train.py --algorithm RUN-CSP --distribution BA_20 

To evaluate 

    >>> python evaluation.py --algorithm RUN-CSP --distribution BA_20 --num_repeat 50 --num_steps 40


### ANYCSP

To train 

    >>> python train.py --algorithm ANYCSP --distribution BA_20 

To evaluate 

    >>> python evaluation.py --algorithm ANYCSP --distribution BA_20 --num_repeat 50 --num_steps 40

### Gflow-CombOpt

To train 

    >>> python train.py --algorithm Gflow-CombOpt --distribution BA_20 

To evaluate 

    >>> python evaluation.py --algorithm Gflow-CombOpt --distribution BA_20 --num_repeat 50 


### Amplitude Heterogeneity Correction

To evaluate 

    >>> python evaluation.py --algorithm AHC--distribution BA_20 

Hyperparameter tuning for each instance is done on the fly.

### Chaotic Amplitude Control

To evaluate 

    >>> python evaluation.py --algorithm CAC --distribution BA_20 

Hyperparameter tuning for each instance is done on the fly.

### Tabu Search

To train

    >>> python train.py --algorithm TS --distribution BA_20 --num_repeat 50 --num_steps 40 --low 20 --high 150 --step 10

To test

    >>> python evaluation.py --algorithm TS --distribution BA_20 --num_repeat 50  --num_steps 40


### Extremal Optimization

To train

    >>> python train.py --algorithm EO --distribution BA_20 --num_repeat 50 --num_steps 40 --low 1.1 --high 2 --step 0.1

To test

    >>> python evaluation.py --algorithm EO --distribution BA_20 --num_repeat 50  --num_steps 40

### Greedy

To test

    >>> python evaluation.py --algorithm Greedy --distribution BA_20 --num_repeat 50  


### Data and Models

We cannot reupload publicly available datasets, as we do not have any permission to re-distribute them. Hence we only provide the synthetic datasets we generated with with best known solutions along with all pretrained models.
