# SynthEHRella: Synthetic Electronic Health Record Benchmarking

## Package Overview

```
SynthEHRella/
│
├── SynthEHRella/
│   ├── __init__.py              
│   ├── data/
|   |   ├── methods/
│   |   |   ├── cor-gan
│   |   |   ├── plasmode
│   |   |   ├── synthea
│   |   |   └── ...
│   |   ├── scripts/
│   |   |   ├── gen_corgan.sh
│   |   |   ├── gen_plasmode.sh
│   |   |   ├── gen_synthea.sh
│   |   |   └── ...
│   │   ├── __init__.py
│   │   ├── data_generator.py
│   │   ├── utils.py    
│   ├── evaluation/
│   │   ├── __init__.py          
│   │   ├── fidelity.py
│   │   ├── utility.py
│   │   ├── privacy.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config_loader.py
│   ├── run_generation.py           
│   ├── run_evaluation.py
│   └── real_data_preprocessing.py
├── tests/
├── setup.py
├── README.md
└── requirements.txt
```


## Installation

Before installing the package, make sure you have the following dependencies installed:

```bash
conda env create -f environment.yaml
```

To install the package, run the following command:

```bash
pip install .
```


## Usage

### Obtain MIMIC-III/IV Data

MIMIC-III and MIMIC-IV data are open-source and can be obtained by requesting via \url{https://physionet.org/content/mimiciii/1.4/} and \url{https://physionet.org/content/mimiciv/3.1/}. After obtaining the data, you can preprocess the data using the following steps.

### Preprocessing Real EHR Data

To preprocess real EHR data, run the following command:

```bash
python -m synthEHRella.run_preprocessing <admissionFile> <diagnosisFile> <patientsFile> <outPath> binary <dataset>
```

Here, `<admissionFile>`, `<diagnosisFile>`, and `<patientsFile>` are the paths to the admission, diagnosis, and patients files in MIMIC-III/IV, respectively. `<outPath>` is the path to save the preprocessed data. `<dataset>` is the name of the dataset (e.g., `mimic3`, `mimic4`). The preprocessing will output multiple files: 

For `mimic3`:
- `processed_mimic3.matrix`: the processed EHR data in a matrix format coded by ICD-9
- `processed_mimic3.pid`: the patient IDs (most of the time we don't need this)
- `processed_mimic3.types`: the disease type of each dimension in the matrix
- `mimic3-real-phecodex.npy`: the processed EHR data with PhecodeX coding system (for visualization in the paper)
- `mimic3-real-phecodexm.npy`: the processed EHR data PhecodeX coding system with higher hierachy (for quantitative evaluation)

For `mimic4`:
- `mimic4-real-phecodex.npy`: the processed EHR data with PhecodeX coding system (for visualization in the paper)
- `mimic4-real-phecodexm.npy`: the processed EHR data PhecodeX coding system with higher hierachy (for quantitative evaluation)

The processed MIMIC-III dataset will be used as the training data for all the methods except `ehrdiff`. For `ehrdiff`, the training data provided by their authors are used to train the model to guarantee only minor change on their codebase. 


### Generating Synthetic EHR Data

To generate synthetic EHR data using a specific method, run the following command:

```bash
python -m synthEHRella.run_generation <method> --params data.real_training_data_path=<path_to_real_data> generation.ckpt_dir=<path_to_save_ckpt> generation.params.num_gen_samples=<num_gen_samples>
```

Here, `<method>` is the name of the method to generate synthetic EHR data; `<path_to_real_data>` is the path to the preprocessed real EHR data; `<path_to_save_ckpt>` is the path to save the checkpoint, synthetic data, and any other necessary files generated during the model training process; `<num_gen_samples>` is the number of synthetic samples to generate.


Currently, the following methods are supported:

- `corgan`: CorGAN
- `plasmode`: Plasmode
- `synthea`: Synthea
- `ehrdiff`: EHRDiff
- `medgan`: MedGAN
- `vae`: VAE
- `promptehr`: PromptEHR
- `resample`: Resample (baseline methods)
- `prevalence-based-random`: Prevalence-based Random (baseline methods)


### Post-Processing

Postprocessing pipeline is required for all the pipelines to transform from ICD-9 / SNOMED-CT to PhecodeX.

```bash
python -m synthEHRella.run_postprocessing <method> --synthetic_data_path <path_to_synthetic_data> --output_path <path_to_save_results>
```

Here, `<path_to_synthetic_data>` is the path to the synthetic EHR data file; `<method>` is the name of the method used to generate the synthetic EHR data; `<path_to_save_results>` is the path to save the post-processed synthetic EHR data.

Instruction on finding synthetic data path for each method:

`corgan`, `medgan`, and `vae`: the synthetic data are contained in a `.npy` file under the `{ckpt_dir}` set in generation called `synthetic-{gen_sample_size}.npy`. `gen_sample_size` is the number of samples generated.

`ehrdiff`: the synthetic data are contained in a file called 'all_x.npy' under the '{ckpt_dir}/samples' folder.

`plasmode`: the synthetic data are all the files under the `{ckpt_dir}` folder.

`synthea`: the synthetic data are contained in a file called `conditions.csv` under the `{ckpt_dir}/csv` folder.

`promptehr`: the synthetic data are contained in a file called `promptehr-synthetic.npy` under the `{ckpt_dir}` folder.

`resample` and `prevalence-based-random`: the synthetic data are contained in a file called `{resample/pbr}-synthetic.npy` under the `{ckpt_dir}` folder. 


### Evaluation

To evaluate the fidelity, utility, and privacy of the synthetic EHR data, run the following command:

```bash
python -m SynthEHRella.run_evaluation <method> --synthetic_data_path <path_to_synthetic_data> --real_eval_data_path <path_to_real_eval_data> --output_dir <path_to_save_results>
```

Here, `<method>` is the name of the method to evaluate; `<path_to_synthetic_data>` is the path to the synthetic EHR data; `<path_to_real_eval_data>` is the path to the real evaluation data; `<path_to_save_results>` is the path to save the evaluation results. Note that the real_eval_data_path should be the path to the `.npy` file containing real data coded by PhecodeX (i.e., `mimic3-real-phecodexm.npy` or `mimic4-real-phecodexm.npy`).

