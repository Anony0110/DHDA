
# Dually Hierarchical Drift Adaptation for Online Configuration Performance Learning

This repository contains the data and code for the following paper: 
> Dually Hierarchical Drift Adaptation for Online Configuration Performance Learning

## Introduction
> Modern configurable software systems need to learn models that correlate configuration and performance. However, when the system operates in dynamic environments, the workload variations, hardware changes, and system updates will inevitably introduce concept drifts at different levels. As such, existing offline and transfer learning approaches can struggle to adapt to these implicit and unpredictable changes in real-time, rendering configuration performance learning challenging. 
>
> To address this, we propose DHDA, an online configuration performance learning framework designed to capture and adapt to these drifts at different levels. The key idea is that DHDA adapts to both the local and global drifts using dually hierarchical adaptation: at the upper level, we redivide the data into different divisions, within each of which the local model is retrained, to handle global drifts only when necessary. At the lower level, the local models of the divisions can detect local drifts and adapt themselves asynchronously. To balance responsiveness and efficiency, DHDA combines incremental updates with periodic full retraining to minimize redundant computation when no drifts are detected.
>
> Through evaluating eight software systems and against state-of-the-art approaches, we show that DHDA achieves considerably better accuracy and can effectively adapt to drifts with up to 2.91 times improvements, while incurring reasonable overhead and is able to improve different local models in handling concept drift. 

This repository contains the **key codes**, **full data used**, and the **raw experiment results** for the paper.

# Documents
- **base_model**:
local model used in DHDA

- **drift_data**:
configuration datasets of eight subject systems as specified in the paper.

- **results**:
contains the raw experiment results for all the research questions.

- **utils**:
contains utility functions to build DHDA.

- **main.py**: 
the *main program* for using DHDA, which automatically reads data from csv files, trains and evaluates the model, and saves the results.

- **requirements.txt**:
the required packages for running main.py.

# Prerequisites and Installation
1. Download all the files into the same folder/clone the repository.

2. Install the specified version of Python:
the codes have been tested with **Python 3.9**, **tensorflow 2.12 - 2.16**, and **keras < 3.0**, other versions might cause errors.

3. Using the command line: cd to the folder with the codes, and install all the required packages by running:

        pip install -r requirements.txt

# Run *DHDA*

- **Command line**: cd to the folder with the codes, input the command below, and the rest of the processes will be fully automated.

        python main.py
        
- **Python IDE (e.g. Pycharm)**: Open the *main.py* file on the IDE, and simply click 'Run'.


# Change Experiment Settings
To run more complicated experiments, alter the codes following the instructions below and comments in *main.py*.
#### To switch between subject systems
    Change the line 110 in main.py

    E.g., to run SeMPL with DeepArch, simply write 'env_list = env_list8'.

#### To change the number of experiments for specified sample size(s)
    Change 'run_time' at line 118.

#### To change the dataset sampling ratio for each experiment:
    Change the 'max_sampling_ratio' and 'min_sampling_ratio' on lines 116 and 117.

#### To change the number of test environments per experiment
    Change n_environment in line 111

# State-of-the-art Performance Prediction Models
Below are the repositories of the SOTA performance prediction models, which are evaluated and compared with *DHDA* in the paper.

- [BEETLE](https://github.com/ai-se/BEETLE)

   A model that selects the bellwether environment for transfer learning.

- [SeMPL](https://github.com/ideas-labo/SeMPL)

  A meta-learning approach for configuration performance learning across multiple environments
- [ARF](https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.meta.AdaptiveRandomForestRegressor.html)

  A general adaptive learning method designed for online regression tasks using an ensemble of randomized trees with adaptive mechanisms to handle concept drift
- [SRP](https://riverml.xyz/0.22.0/api/ensemble/SRPRegressor/)

  A general online learning approach that dynamically creates and updates ensemble models using subsets of features and data patches
