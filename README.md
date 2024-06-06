# Maximum Cut Benchmarking Suite

This repository contains a benchmarking suite for maximum cut
solvers. The suite bundles several solvers and datasets with a unified
interface. Currently, we support the following solvers:

- S2V-DQN
- ECO-DQN
- LS-DQN
- RUN-CSP
- ANYCSP
- Gflownet-CombOpt
- ECORD
- Amplitude Heterogeneity Correction (AHC)
- Chaotic Amplitude Control (CAC)
- Tabu Search
- Extremal Optimization
- Standard Greedy 
- Stochastic Greedy
- Forward Greedy


## Details on the Solvers

### S2V-DQN


### ECO-DQN


### LS-DQN

### RUN-CSP


### ANYCSP

### Gflownet-CombOpt

### ECORD

### Amplitude Heterogeneity Correction (AHC)

### Chaotic Amplitude Control

### Tabu Search

### Extremal Optimization

### Greedy


## Repository Contents

In `solvers`, you can find the wrappers for the currently supported solvers. In `data_generation`, you find the code required for generating random and real-world graphs.

For using this suite, `conda` is required. You can the `setup_bm_env.sh` script which will setup the conda environment with all required dependencies. You can find out more about the usage using `python main.py -h`. The `main.py` file is the main interface you will call for data generation, solving, and training.

In the `helper_scripts` folder, you find some scripts that could be helpful when doing analyses with this suite.

## Publication

You can find our ICLR 2022 conference paper [here](https://openreview.net/forum?id=mk0HzdqY7i1).

If you use this in your work, please cite us and the papers of the solvers that you use.

```bibtex
@inproceedings{boether_dltreesearch_2022,
  author = {Böther, Maximilian and Kißig, Otto and Taraz, Martin and Cohen, Sarel and Seidel, Karen and Friedrich, Tobias},
  title = {What{\textquoteright}s Wrong with Deep Learning in Tree Search for Combinatorial Optimization},
  booktitle = {Proceedings of the International Conference on Learning Representations ({ICLR})},
  year = {2022}
}
```

If you have questions you are welcome to reach out to [@MaxiBoether](https://github.com/MaxiBoether) and [@EightSQ](https://github.com/EightSQ).

### Data and Models

On popular request, we provide the (small) random graphs with labels and the models we trained [here](https://owncloud.hpi.de/s/cv6szEJtSs8UGju) ([backup location](https://mboether.com/paper-models-randomgraphs.zip)).
The Intel tree search model that was trained by Li et al. can be downloaded from [the original repository](https://github.com/isl-org/NPHard/tree/master/model).
Note that we cannot reupload the labeled real world graphs, as we do not have any permission to re-distribute them.
However, the benchmarking suite supports the automatic download and labeling of _all_ random and real world graphs used in the paper.
Please do not only rely on the data we provide and instead use this suite to generate graphs and train models on your own, as there is no guarantee that our evaluation is fully correct.

## Contributions

There are (of course) some improvements that can be made. For example, the argument parsing requires a major refactoring, and the output formats are currently not fully harmonized. We are open for pull requests, if you want to contribute. Thank you very much!