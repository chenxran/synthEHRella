import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, pairwise_distances
from sklearn.model_selection import train_test_split


def attribute_inference_attack(real_data, synthetic_data, known_features_idx, unknown_features_idx, k=1):
    """
    Perform an attribute inference attack.

    Parameters:
    real_data (numpy.ndarray): The real EHR data matrix.
    synthetic_data (numpy.ndarray): The synthetic EHR data matrix.
    known_features_idx (list): Indices of features known by the attackers.
    unknown_features_idx (list): Indices of features unknown by the attackers.
    k (int): Number of nearest neighbors to use.

    Returns:
    float: The F1-score of the prediction of the unknown code features.
    """
    known_real = real_data[:, known_features_idx]
    known_synthetic = synthetic_data[:, known_features_idx]
    unknown_real = real_data[:, unknown_features_idx]
    
    # Find k nearest neighbors in synthetic data for each real data point
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(known_synthetic)
    distances, indices = nbrs.kneighbors(known_real)
    
    # Recover unknown features by majority voting (k=1 simplifies this to direct assignment)
    recovered_features = synthetic_data[indices.flatten()][:, unknown_features_idx]

    # Evaluate the performance using F1-score
    f1 = f1_score(unknown_real.flatten(), recovered_features.flatten(), average='macro')
    
    return f1

def membership_inference_attack(real_data, synthetic_data):
    """
    Perform a membership inference attack.

    Parameters:
    real_data (numpy.ndarray): The real EHR data matrix.
    synthetic_data (numpy.ndarray): The synthetic EHR data matrix.

    Returns:
    float: The F1-score of the membership prediction.
    """
    # mixed_real_data = np.vstack((real_train_data, real_test_data))
    # labels = np.hstack((np.ones(real_train_data.shape[0]), np.zeros(real_test_data.shape[0])))
    
    # Compute minimum L2 distance to synthetic data
    indices = np.random.choice(real_data.shape[0], min(synthetic_data.shape[0], real_data.shape[0]), replace=False)
    real_data = real_data[indices]
    distances = pairwise_distances(real_data, synthetic_data, metric='euclidean')
    min_distances = np.min(distances, axis=1)
    
    results = {}
    results["min_min_distances"] = min_distances.min()
    results["max_min_distances"] = min_distances.max()
    results["mean_min_distances"] = min_distances.mean()
    results["median_min_distances"] = np.median(min_distances)
    # results["min_distances"] = min_distances

    return results