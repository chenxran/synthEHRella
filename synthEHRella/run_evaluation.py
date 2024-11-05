import sys
import json
import argparse
import numpy as np
from omegaconf import OmegaConf
from synthEHRella.evaluation.fidelity import discriminative_score, compute_correlation, compute_prevalence, coappearance_matrix
from synthEHRella.evaluation.privacy import membership_inference_attack, attribute_inference_attack
from synthEHRella.evaluation.utility import trtr, tsrtr, tstr
from synthEHRella.evaluation.utils import mean_ignore_nan_inf

# fidelity evaluation
def fidelity_evaluation(real_data, synthetic_data):
    real_prevalence = compute_prevalence(real_data)
    synthetic_prevalence = compute_prevalence(synthetic_data)

    results = {}
    results["mmd"] = np.abs(real_prevalence - synthetic_prevalence).max()
    results["rmspe"] = np.sqrt(mean_ignore_nan_inf(np.square((real_prevalence - synthetic_prevalence) / real_prevalence)))
    results["mape"] = mean_ignore_nan_inf(np.abs((real_prevalence - synthetic_prevalence) / real_prevalence)) * 100

    real_corr = compute_correlation(real_data.astype(float))
    synthetic_corr = compute_correlation(synthetic_data.astype(float))
    results["corr_fro_dist"] = np.linalg.norm(real_corr - synthetic_corr, 'fro')

    real_coappear = coappearance_matrix(real_data.astype(float))
    synthetic_coappear = coappearance_matrix(synthetic_data.astype(float))
    results["coappear_fro_dist"] = np.linalg.norm(real_coappear - synthetic_coappear, 'fro')

    score = discriminative_score(real_data, synthetic_data)
    results["discriminative_auc"] = score["auc"]
    results["discriminative_accuracy"] = score["accuracy"]
    
    return results


def utility_evaluation(real_data, synthetic_data):
    # real_prevalence = compute_prevalence(real_data)
    index = 0 # np.argsort(np.abs(real_prevalence - 0.5))[0]
    # print(index)
    trtr_results = trtr(real_data, index=index)
    tsrtr_results = tsrtr(real_data, synthetic_data, index=index)
    tstr_results = tstr(real_data, synthetic_data, index=index)

    # calculate the difference of accuracy and auc between trtr-tstr and trtr-tsrtr.
    results = {}
    results["trtr_accuracy"] = trtr_results["accuracy"]
    results["trtr_auc"] = trtr_results["auc"]
    results["tsrtr_accuracy"] = tsrtr_results["accuracy"]
    results["tsrtr_auc"] = tsrtr_results["auc"]
    results["tstr_accuracy"] = tstr_results["accuracy"]
    results["tstr_auc"] = tstr_results["auc"]
    results["tstr-trtr_accuracy"] = tstr_results["accuracy"] - trtr_results["accuracy"]
    results["tstr-trtr_auc"] = tstr_results["auc"] - trtr_results["auc"]
    results["tsrtr-trtr_accuracy"] = tsrtr_results["accuracy"] - trtr_results["accuracy"]
    results["tsrtr-trtr_auc"] = tsrtr_results["auc"] - trtr_results["auc"]

    return results


def privacy_evaluation(real_data, synthetic_data):
    # select 10 most balanced features + 10 most imbalanced features as unknown features (in total 20 unknown features)
    real_prevalence = compute_prevalence(real_data)
    sorted_indices = np.argsort(np.abs(real_prevalence - 0.5))
    unknown_features_idx = sorted_indices[:10].tolist() + sorted_indices[-10:].tolist()
    known_features_idx = list(set(range(real_data.shape[1])) - set(unknown_features_idx))

    attribute_inference_results = attribute_inference_attack(real_data, synthetic_data, known_features_idx, unknown_features_idx)
    membership_inference_results = membership_inference_attack(real_data, synthetic_data)
    
    results = {}
    results["attribute_inference_attack"] = attribute_inference_results
    results["membership_inference_attack"] = membership_inference_results
    
    return results


def evaluation(args):
    real_data = np.load(args.real_eval_data_path)
    synthetic_data = np.load(args.synthetic_data_path)

    results = {}
    if args.fidelity:
        results = {"fidelity": fidelity_evaluation(real_data, synthetic_data)}
    if args.utility:
        results = {**results, "utility": utility_evaluation(real_data, synthetic_data)}
    if args.privacy:
        results = {**results, "privacy": privacy_evaluation(real_data, synthetic_data)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', help='The generation method (e.g., cor-gan, plasmode, synthea)')
    parser.add_argument('--output_dir', default=None, help='Path to the output directory')
    parser.add_argument('--real_eval_data_path', default=None, help='Path to the real evaluation data')
    parser.add_argument('--synthetic_data_path', default=None, help='Path to the synthetic data')
    parser.add_argument('--fidelity', default=True, help='Whether to evaluate fidelity')
    parser.add_argument('--utility', default=True, help='Whether to evaluate utility')
    parser.add_argument('--privacy', default=True, help='Whether to evaluate privacy')
    
    # Parse arguments
    args = parser.parse_args()

    results = evaluation(args)
    # save the results as a json file in config.evaluation.output_dir
    with open(args.output_dir, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()