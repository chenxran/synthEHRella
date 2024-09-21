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


def compute_prevalence(data):
    """
    Compute the prevalence of each code (column-wise mean) in the dataset.
    """
    return data.mean(axis=0)

def compute_correlation(data):
    # input: n x p
    # output: p x p
    """
    Compute the correlation matrix of the dataset.
    """
    # for each column, if the values in this column are all the same, randomly add a small noise
    for i in range(data.shape[1]):
        if len(np.unique(data[:, i])) == 1:
            data[:, i] += np.random.normal(0, 1e-6, data.shape[0])

    return np.corrcoef(data, rowvar=False)

def discriminative_score(real_data, synthetic_data, model_type='logistic'):
    """
    Evaluate the ability of a linear model or random forest to discriminate between real and synthetic data.

    Parameters:
    real_data (numpy.ndarray): The real data matrix (n x p).
    synthetic_data (numpy.ndarray): The synthetic data matrix (m x p).
    model_type (str): The type of model to use ('logistic' for logistic regression, 'random_forest' for random forest).

    Returns:
    dict: A dictionary with cross-validation scores for accuracy and AUC.
    """
    # Combine real and synthetic data
    n, p = real_data.shape
    m, _ = synthetic_data.shape
    data = np.vstack((real_data, synthetic_data))
    labels = np.hstack((np.ones(n), np.zeros(m)))  # 1 for real data, 0 for synthetic data

    # Choose the model
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    else:
        raise ValueError("Invalid model_type. Choose 'logistic' or 'random_forest'.")

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    auc_scores = []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

        # Evaluate the model
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_prob))

    # Return the average scores
    results = {
        'accuracy': np.mean(accuracy_scores),
        'auc': np.mean(auc_scores)
    }
    
    return results