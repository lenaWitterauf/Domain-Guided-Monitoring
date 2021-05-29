# Domain-Guided-Monitoring
Can we use domain knowledge to better monitor complex systems?

## Repository Structure
This repo is structured as follows:
- `notebooks/` contains some jupyter notebooks used for simple exploration and experimentation.
- `data/` should be used for the data used for and created during training and preprocessing steps. 
- `artifacts/` is used for storing the outputs of experiment runs
- `src/` contains all the code used for our experiments:
  - `src/features` includes code handling the features of our experiments. It is separated into
    - `src/features/preprocessing`: any preprocessing steps necessary to use the raw data. will produce intermediate results so that we don't have to run this step every time,
    - `src/features/knowledge`: code for handling the **Expert Knowledge** part of the training,
    - `src/features/sequences`: code handling the sequences (eg big data part) used during training.
  - `src/training` contains code used for the actual training part
    - `src/training/models` defines the model structures used for training.
    - `src/training/analysis` defines training analysis such as printing learned embeddings.
- `environment.yml` defines the anaconda environment and dependencies used for running the experimentation code.

### Supported Datasets
For now, this repository supports two types of datasets:
- [MIMIC dataset](https://mimic.physionet.org/about/mimic/) and
- Huawei log dataset.
If you want to add a new type of dataset, look at the preprocessors implemented in `src/features/preprocessing/` and put your own implementation there.

### Supported Expert Knowledge
With this repository, the following types of expert knowledge are supported:
- **Hierarchical Expert Knowledge** (original idea see `CHOI, Edward, et al. GRAM: graph-based attention model for healthcare representation learning. In: Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining. 2017. S. 787-795.`)
- **Textual Expert Knowledge** (original idea see `MA, Fenglong, et al. Incorporating medical code descriptions for diagnosis prediction in healthcare. BMC medical informatics and decision making, 2019, 19. Jg., Nr. 6, S. 1-13.`)
- **Causal Expert Knowledge** (original idea see `YIN, Changchang, et al. Domain Knowledge guided deep learning with electronic health records. In: 2019 IEEE International Conference on Data Mining (ICDM). IEEE, 2019. S. 738-747.`)

## How to Run this Code
In order to run this code, you need Anaconda + Python >= 3.8. This repository contains a makefile that simplifies executing the code on a remote server using miniconda. Below are step-by-step descriptions of what to do in order to run the code from either the Makefile or manual setup.

### Run via Makefile
- **Create and activate conda environment**: run `make install` 
- **Get the data**: We don't include training data into this repo, so you have to download it yourself and move it to the `data/` folder. For now, we support training on the MIMIC dataset and on Huawei log data. 
  - **Get MIMIC**: In order to access MIMIC dataset, you need a credentialed account on physionet. You can request access [here](https://mimic.physionet.org/gettingstarted/access/). Run `make install_mimic PHYSIONET_USER="<yourphysionetusername>"` to download the required physionet files from their website. 
  - **Get Huawei Log Data**: Download the Huawei `concurrent data` dataset and move the file `concurrent data/logs/logs_aggregated_concurrent.csv` to the `data/` directory
- **Run the code**: To run the experiment, execute `make run ARGS="<yourargshere>"`. There are a bunch of commandline options, the most important ones are:
  -  `--experimentconfig_sequence_type`: dataset to use, for now valid values here are `mimic` and `huawei_logs`
  -  `--experimentconfig_model_type`: use this to choose the knowledge model you want to run; valid values are `simple`, `gram`, `text` and `causal`
  -  to see the full list of options run `python main.py -h`

### Run manual SetUp
- **Create and activate conda environment**: run `conda env update -f environment.yml` to create (or update) an anaconda environment from the given `environment.yml` file. Activate the environment by running `conda activate healthcare-aiops`
- **Get the data**: We don't include training data into this repo, so you have to download it yourself and move it to the `data/` folder. For now, we support training on the MIMIC dataset and on Huawei log data. 
  - **Get MIMIC**: In order to access MIMIC dataset, you need a credentialed account on physionet. You can request access [here](https://mimic.physionet.org/gettingstarted/access/) or use the example MIMIC dataset available [here](https://mimic.physionet.org/gettingstarted/demo/). Once you have access, move `ADMISSIONS.csv` and `DIAGNOSES_ICD.csv` into the `data/` directory
  - **Get Huawei Log Data**: Download the Huawei `concurrent data` dataset and move the file `concurrent data/logs/logs_aggregated_concurrent.csv` to the `data/` directory
- **Run the code**: To run the experiment from the command line, execute `python main.py`. There are (amongst others) the following commandline options:
  -  `--experimentconfig_sequence_type`: dataset to use, for now valid values here are `mimic` and `huawei_logs`
  -  `--experimentconfig_model_type`: use this to choose the knowledge model you want to run; valid values are `simple`, `gram`, `text` and `causal`
  -  to see the full list of options run `python main.py -h`
