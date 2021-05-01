# Domain-Guided-Monitoring
Can we use domain knowledge to better monitor complex systems?

## Repository Structure
This repo is structured as follows:
- `notebooks/` contains some jupyter notebooks used for simple exploration and experimentation.
- `data/` should be used for the data used for and created during training and preprocessing steps. 
- `src/` contains all the code used for our experiments:
  - `src/features` includes code handling the features of our experiments. It is separated into
    - `src/features/preprocessing`: any preprocessing steps necessary to use the raw data. will produce intermediate results so that we don't have to run this step every time,
    - `src/features/knowledge`: code for handling the **Expert Knowledge** part of the training,
    - `src/features/sequences`: code handling the sequences (eg big data part) used during training.
  - `src/models` defines the model structures used for training.
- `environment.yml` defines the anaconda environment and dependencies used for running the experimentation code.

### Supported Data
For now, this repository only works on the [MIMIC dataset](https://mimic.physionet.org/about/mimic/). Further dataset support for monitoring data as well as a description on how to add custom datasets tbd soon.

### Supported Models
This repo implements
- a very simple LSTM model (add argument `--experimentrunner_model_type simple` when executing code)
- an implementation using hierarchical embeddings based on GRAM (add argument `--experimentrunner_model_type gram` when executing code)
  - in order to run this model, you'll need a file containing expert knowledge. For the MIMIC dataset, hierarchical expert knowledge can be gathered from the ICD9 structure. To use the ICD9 hierarchy, you'll need to download [this archive](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip) and place the extracted files in your `data/` directory.

### Supported Expert Knowledge
We will add the following types of expert knowledge:
- [x] **Hierarchical Expert Knowledge** (see `CHOI, Edward, et al. GRAM: graph-based attention model for healthcare representation learning. In: Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining. 2017. S. 787-795.`)
- [ ] **Textual Expert Knowledge** (see `MA, Fenglong, et al. Incorporating medical code descriptions for diagnosis prediction in healthcare. BMC medical informatics and decision making, 2019, 19. Jg., Nr. 6, S. 1-13.`)
- [ ] **Causal Expert Knowledge** (see `YIN, Changchang, et al. Domain Knowledge guided deep learning with electronic health records. In: 2019 IEEE International Conference on Data Mining (ICDM). IEEE, 2019. S. 738-747.`)

## How to Run this Code
In order to run this code, you need Anaconda + Python >= 3.8.
- **Create and activate conda environment**: run `conda env update -f environment.yml` to create (or update) an anaconda environment from the given `environment.yml` file. Activate the environment by running `conda activate healthcare-aiops`
- **Get the data**: We don't include training data into this repo, so you have to download it yourself and move it to the `data/` folder. For now, we only support training on the MIMIC dataset. You can request access [here](https://mimic.physionet.org/gettingstarted/access/) or download the example MIMIC dataset [here](https://mimic.physionet.org/gettingstarted/demo/)
  - optional: **Get Expert Knowledge**: see previous section on how to add the expert knowledge required for the respective model types!
- **Run the code**: To run the experiment from the command line, execute `python main.py`. There are (amongst others) the following commandline options:
  -  `experimentrunner_need_sequence_preprocessing`: enable if you need to preprocess the raw MIMIC dataset first (will be the case if you haven't run this code before)
  -  `experimentrunner_sequence_type`: type of sequences to predict. for now, only MIMIC is allowed here.
  - `--experimentrunner_model_type`: use this to choose the model you want to run
  -  to see the full list of options run `python main.py -h`
