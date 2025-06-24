# SynthEHRella

A Python package for synthetic Electronic Health Records (EHR) data generation benchmarking.

## Cite

|       | Citation     |
| -------------  | -------------  | 
| paper | Chen X, Wu Z*, Shi X, Cho H, Mukherjee B $^\dagger$ (2025). [Generating synthetic electronic health record data: a methodological scoping review with benchmarking on phenotype data and open-source software](https://doi.org/10.1093/jamia/ocaf082). *Journal of the American Medical Informatics Association*, Volume 32, Issue 7, July 2025, Pages 1227–1240. [*: correspdonding author (zhenkewu@umich.edu); $^\dagger$: senior author (bhramar.mukherjee@yale.edu)].      |

## Package Overview

```
synthEHRella/
│
├── synthEHRella/
│   ├── __init__.py              
│   ├── data/
|   |   ├── methods/
│   |   |   ├── cor-gan
│   |   |   ├── plasmode
│   |   |   ├── synthea
│   |   |   └── ...
│   │   ├── __init__.py
│   │   ├── data_generator.py
│   │   ├── utils.py    
│   ├── evaluation/
│   │   ├── __init__.py          
│   │   ├── fidelity.py
│   │   ├── utility.py
│   │   └── privacy.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_transform.py
│   │   └── ... (mapping files)
│   ├── run_generation.py           
│   ├── run_evaluation.py
│   ├── run_postprocessing.py
│   └── run_preprocessing.py
├── setup.py
├── LICENSE
├── README.md
└── environmental.yaml
```



## Preliminary

You should have the following files before installing the package. 

### Obtain MIMIC-III/IV Data

MIMIC-III and MIMIC-IV data are open source and available through PhysioNet. Request access here: 

[MIMIC-III v1.4](https://physionet.org/content/mimiciii/1.4/) 

[MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) 

After obtaining the data, you can preprocess it for use with SynthEHRella. Note that in the paper, we used MIMIC-III v1.4 and MIMIC-IV v2.2 for the evaluation.

### (Optional) Obtain Synthea Software

NOTE: Synthea software is optional and only required if you want to generate synthetic data using Synthea. 

Synthea is an open-source synthetic EHR data generation method that generates synthetic data by simulating the history of synthetic patients. You can download the software from their [Github Repository](https://github.com/synthetichealth/synthea/releases). The file you need to download is called `synthea-with-dependencies.jar`. After obtaining the software, you could place it in the `synthEHRella/synthEHRella/data/methods/synthea` folder.

## Installation

Before installing the package, make sure you have the following dependencies installed:

```bash
conda env create -f environment.yaml
```

To install the `synthEHRella` package, run the following command:

```bash
pip install .
```

## Usage

### STEP 1: Preprocessing Real EHR Data

To preprocess real EHR data, run:

```bash
python -m synthEHRella.run_preprocessing <admissionFile> <diagnosisFile> <patientsFile> <outPath> binary <dataset>
```

Arguments:

- `<admissionFile>`, `<diagnosisFile>`, `<patientsFile>`: Paths to the admission, diagnosis, and patients files in MIMIC-III/IV.

- `<outPath>`: Path to save the preprocessed data.

- `<dataset>`: Name of the dataset (i.e., `mimic3`, `mimic4`).

The preprocessing will output multiple files: 

For `mimic3`:
- `processed_mimic3.matrix`: Matrix-format EHR data coded by ICD-9.
- `processed_mimic3.pid`: Patient IDs (most of the time we don't need this).
- `processed_mimic3.types`: Disease types for each dimension.
- `mimic3-real-phecodex.npy`: EHR data in PhecodeX coding (used for visualization in the paper).
- `mimic3-real-phecodexm.npy`: EHR data in PhecodeX coding with higher hierachy (for quantitative evaluation in the paper).

For `mimic4`:
- `mimic4-real-phecodex.npy`: EHR data in PhecodeX coding (used for visualization in the paper).
- `mimic4-real-phecodexm.npy`: EHR data in PhecodeX coding with higher hierachy (for quantitative evaluation in the paper).

The processed MIMIC-III dataset will be used as the training data for all the methods except `ehrdiff`. For `ehrdiff`, the training data provided by their authors are used to train the model to guarantee only minor change on their codebase. The data can be downloaded from their [Github Repository](https://github.com/sczzz3/EHRDiff).


### STEP 2: Generating Synthetic EHR Data

To generate synthetic EHR data, run:

```bash
python -m synthEHRella.run_generation <method> --params data.real_training_data_path=<path_to_real_data> generation.ckpt_dir=<path_to_save_ckpt> generation.params.num_gen_samples=<num_gen_samples>
```

Arguments:

- `<method>`: The method for generating synthetic data.

- `<path_to_real_data>`: Path to preprocessed real EHR data.

- `<path_to_save_ckpt>`: Directory to save the generated files (e.g., model checkpoints, synthetic data).

- `<num_gen_samples>`: Number of synthetic samples to generate.

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


### STEP 3: Post-Processing

To transform synthetic data from ICD-9 / SNOMED-CT to PhecodeX, run:

```bash
python -m synthEHRella.run_postprocessing <method> --synthetic_data_path <path_to_synthetic_data> --output_path <path_to_save_results>
```

Arguments:

- `<method>`: Method used to generate synthetic data.

- `<path_to_synthetic_data>`: Path to the synthetic EHR data.

- `<path_to_save_results>`: Path to save the post-processed results.


Instruction on finding synthetic data path for each method:

- `corgan`, `medgan`, and `vae`: the synthetic data are contained in a `.npy` file under the `{ckpt_dir}` set in generation called `synthetic-{gen_sample_size}.npy`. `gen_sample_size` is the number of samples generated.

- `ehrdiff`: the synthetic data are contained in a file called 'all_x.npy' under the '{ckpt_dir}/samples' folder.

- `plasmode`: the synthetic data are all the files under the `{ckpt_dir}` folder.

- `synthea`: the synthetic data are contained in a file called `conditions.csv` under the `{ckpt_dir}/csv` folder.

- `promptehr`: the synthetic data are contained in a file called `promptehr-synthetic.npy` under the `{ckpt_dir}` folder.

- `resample` and `prevalence-based-random`: the synthetic data are contained in a file called `{resample/pbr}-synthetic.npy` under the `{ckpt_dir}` folder. 


### STEP 4: Evaluation

To evaluate the fidelity, utility, and privacy of synthetic EHR data, run:

```bash
python -m SynthEHRella.run_evaluation <method> --synthetic_data_path <path_to_synthetic_data> --real_eval_data_path <path_to_real_eval_data> --output_dir <path_to_save_results>
```

Arguments:

- `<method>`: Method used to generate synthetic data.

- `<path_to_synthetic_data>`: Path to synthetic EHR data.

- `<path_to_real_eval_data>`: Path to real evaluation data (should be .npy with PhecodeX coding, such as mimic3-real-phecodexm.npy or mimic4-real-phecodexm.npy).

- `<path_to_save_results>`: Path to save evaluation results.

## Contributing

SynthEHRella is designed to be an **open-source** benchmark for synthetic EHR generation methods. We welcome all kinds of suggestions, and contributions! Please feel free to open an issue or submit a pull request.

## License

SynthEHRella is licensed under the MIT License. See LICENSE for more information.

## Acknowledgments

This work was supported through grant DMS1712933 from the National Science Foundation and MI-CARES grant 1UG3CA267907 from the National Cancer Institute. The funders had no role in the design of the study; collection, analysis, or interpretation of the data; writing of the report; or the decision to submit the manuscript for publication.

