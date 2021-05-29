# Usage:
# make install			# downloads miniconda and initializes conda environment
# make install_mimic	# downloads required mimic files from physionet (physionet credentialed account required)
# make ui	  			# starts mlflow ui at port 5000
# make run  			# executes main.py within the conda environment \
				  			example: make run ARGS="--experimentconfig_sequence_type huawei_logs"
# make run_all 			# executes main.py within the conda environment for all knowledge - datatype combinations
# make run_mimic 		# executes main.py within the conda environment for all knowledge types on mimic dataset
# make run_huawei		# executes main.py within the conda environment for all knowledge types on huawei dataset

CONDA_ENV_NAME = healthcare-aiops
CONDA_URL = https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
CONDA_SH = Miniconda3-latest-Linux-x86_64.sh
CONDA_DIR = .tmp

DATA_DIR = data
DATA_TYPES = huawei_logs mimic
KNOWLEDGE_TYPES = simple gram text causal

install:
ifneq (,$(wildcard ${CONDA_DIR}))
	@echo "Remove old install files"
	@rm -Rf ${CONDA_DIR}
endif
	@echo "Downloading miniconda..."
	@mkdir ${CONDA_DIR}
	@cd .tmp && wget -nc ${CONDA_URL} > /dev/null
	@chmod +x ./${CONDA_DIR}/${CONDA_SH}
	@./${CONDA_DIR}/${CONDA_SH} -b -u -p ./${CONDA_DIR}/miniconda3/ > /dev/null
	@echo "Initializing conda environment..."
	@./${CONDA_DIR}/miniconda3/bin/conda env create -q --force -f environment.yml > /dev/null
	@echo "Finished!"

install_mimic:
	@cd ${DATA_DIR}
	@wget -N -c --user ${PHYSIONET_USER} --ask-password https://physionet.org/files/mimiciii/1.4/ADMISSIONS.csv.gz
	@wget -N -c --user ${PHYSIONET_USER} --ask-password https://physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv.gz
	@gzip -d ADMISSIONS.csv.gz
	@gzip -d DIAGNOSES_ICD.csv.gz

ui:
	@echo "Starting MLFlow UI at port 5000"
	PATH="${PATH}:$(shell pwd)/${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin" ; \
	./${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin/mlflow ui

run: 
	./${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python main.py ${ARGS}

run_all: 
	for data_type in ${DATA_TYPES} ; do \
		for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
			echo "Starting experiment for " $$data_type " with knowledge type " $$knowledge_type "....." ; \
			./${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python main.py \
				--experimentconfig_sequence_type $$data_type \
				--experimentconfig_model_type $$knowledge_type \
				${ARGS} ; \
		done ; \
	done

run_mimic: 
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
		echo "Starting experiment for mimic with knowledge type " $$knowledge_type "....." ; \
		./${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python main.py \
			--experimentconfig_sequence_type mimic \
			--experimentconfig_model_type $$knowledge_type \
			${ARGS} ; \
	done ; \

run_huawei:
	for knowledge_type in ${KNOWLEDGE_TYPES} ; do \
		echo "Starting experiment for huawei_logs with knowledge type " $$knowledge_type "....." ; \
		./${CONDA_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python main.py \
			--experimentconfig_sequence_type huawei_logs \
			--experimentconfig_model_type $$knowledge_type \
			${ARGS} ; \
	done ; \
