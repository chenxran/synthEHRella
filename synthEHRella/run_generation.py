import sys
import os
import argparse
import subprocess
from omegaconf import OmegaConf
import pickle
import numpy as np


def random_sample_from_data(real_data, num_samples):
    """
    Generate synthetic data by randomly sampling from the real data.

    Parameters:
        real_data (numpy.ndarray): The real data matrix (n x p).
        num_samples (int): Number of samples to generate.

    Returns:
        synthetic_data (numpy.ndarray): The synthetic data matrix (num_samples x p).
    """
    n, p = real_data.shape
    indices = np.random.choice(n, num_samples, replace=True)
    synthetic_data = real_data[indices]
    return synthetic_data


def randomly_generate_data(real_data, num_samples):
    """
    Generate synthetic data by randomly generating binary data with the same dimensionality as the real data.

    Parameters:
        real_data (numpy.ndarray): The real data matrix (n x p).
        num_samples (int): Number of samples to generate.

    Returns:
        synthetic_data (numpy.ndarray): The synthetic data matrix (num_samples x p).
    """
    _, d = real_data.shape
    # use real_data to calculate the probability of 1 in each dimension
    p_ones = np.mean(real_data, axis=0)
    synthetic_data = np.zeros((num_samples, d), dtype=int)

    for i in range(d):
        synthetic_data[:, i] = np.random.binomial(1, p_ones[i], size=num_samples)

    return synthetic_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', help='The generation method (e.g., corgan, plasmode, synthea)')
    parser.add_argument('--real_training_data_path', default=None, help='Path to the real training data')
    parser.add_argument('--ckpt_dir', default=None, help='Path to the directory containing the model checkpoints')
    parser.add_argument('--num_gen_samples', type=int, default=50000, help='Number of samples to generate')
    
    # define an argument to receive arbitrary string as additional arguments
    parser.add_argument('--params', nargs='*', help='Additional parameters to override from the default config')
    
    # Parse arguments
    args = parser.parse_args()
    if args.params:
        args.params = ' '.join(args.params)

    # Load the default config from the specified method YAML file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # config = OmegaConf.load(f"{base_dir}/config/{args.method}.yaml")

    if args.method not in ['resample', 'prevalence-based-random']:
        command_list = []
        if args.method == 'corgan':
            command = f"python {base_dir}/data/methods/cor-gan/Generative/corGAN/pytorch/CNN/MIMIC/wgancnnmimic.py"
            command += f" --DATASETPATH {args.real_training_data_path}"
            command += f" --expPATH {args.ckpt_dir}"
            command += f" --gen_sample_size {args.num_gen_samples}"
            if args.params is not None:
                command += f" {args.params}"
            command += " --training True --generate True"
            
            command_list.append(command.split())
        
        elif args.method == 'medgan':
            command = f"python {base_dir}/data/methods/cor-gan/Generative/medGAN/MIMIC/pytorch/MLP/medGAN.py"
            command += f" --DATASETPATH {args.real_training_data_path}"
            command += f" --PATH {args.ckpt_dir}"
            command += f" --num_fake_samples {args.num_gen_samples}"
            if args.params is not None:
                command += f" {args.params}"
            command += " --training True"
            
            command_list.append(command.split())
        
        elif args.method == 'plasmode':
            command = f"Rscript {base_dir}/data/methods/plasmode/plasmode.R"
            command += f" {args.real_training_data_path} {args.num_gen_samples} {args.ckpt_dir}"
            if args.params is not None:
                command += f" {args.params}"
            command_list.append(command.split())
        
        elif args.method == 'ehrdiff':
            command = f"python {base_dir}/data/methods/EHRDiff/main.py"
            command += f" --mode train --config {base_dir}/data/methods/EHRDiff/configs/mimic/train_edm.yaml"
            command += f" --workdir {args.ckpt_dir}"
            if args.params is not None:
                command += f" {args.params}"
            command += f" data.path={args.real_training_data_path}"
            
            command_list.append(command.split())
            command = f"python {base_dir}/data/methods/EHRDiff/main.py"
            command += f" --mode eval --config {base_dir}/data/methods/EHRDiff/configs/mimic/sample_edm.yaml"
            command += f" --workdir {args.ckpt_dir}"
            command += f" test.n_samples={args.num_gen_samples}"
            
            command_list.append(command.split())
            
        elif args.method == 'synthea':
            command = f"java -jar {base_dir}/data/methods/synthea/synthea-with-dependencies.jar"
            command += f" -p {args.num_gen_samples}"
            command += f" --exporter.csv.export=true"
            command += f" --exporter.baseDirectory={args.ckpt_dir}"
            if args.params is not None:
                command += f" {args.params}"

            command_list.append(command.split())
            
        elif args.method == 'promptehr':
            command = f"python {base_dir}/data/methods/PromptEHR/gen_promptEHR.py"
            command += f" --num_gen_samples {args.num_gen_samples}"
            command += f" --save_path {args.ckpt_dir}"
            if args.params is not None:
                command += f" {args.params}"

            command_list.append(command.split())
        
        elif args.method == 'vae':
            command = f"python {base_dir}/data/methods/cor-gan/Generative/VAE/MIMIC/vaeConvolutional.py"
            command += f" --DATASETPATH {args.real_training_data_path}"
            command += f" --MODELPATH {args.ckpt_dir}"
            command += f" --num_fake_samples {args.num_gen_samples}"
            if args.params is not None:
                command += f" {args.params}"
            command += " --train True"
            
            command_list.append(command.split())
        
        try:
            for command in command_list:
                print(f"Executing command: {command}")
                result = subprocess.run(command, check=True, stdout=sys.stdout, stderr=sys.stderr)
            print(f"Synthetic data generated successfully.")
                
        except subprocess.CalledProcessError as e:
            print(f"Error in generating synthetic data: {e.stderr}")
            raise
    else:
        with open(args.real_training_data_path, 'rb') as f:
            real_data = pickle.load(f)
        if args.method == 'resample':
            matrix = random_sample_from_data(real_data, args.num_gen_samples)
        elif args.method == 'pbr':
            matrix = randomly_generate_data(real_data, args.num_gen_samples)

        # avoid prevalence = 0
        for i in range(matrix.shape[1]):
            if np.all(matrix[:, i] == 0) or np.all(matrix[:, i] == 1):
                idx = np.random.randint(matrix.shape[0])
                matrix[idx, i] = 1 - matrix[idx, i]
                
        np.save(os.path.join(args.ckpt_dir, f"{args.method}-synthetic.npy"), matrix)

                
if __name__ == "__main__":
    main()
