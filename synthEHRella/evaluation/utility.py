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


def evaluate_model(train_data, train_labels, test_data, test_labels, model_type='logistic'):
    """
    Train and evaluate the model.

    Parameters:
    train_data (numpy.ndarray): Training data.
    train_labels (numpy.ndarray): Labels for the training data.
    test_data (numpy.ndarray): Test data.
    test_labels (numpy.ndarray): Labels for the test data.
    model_type (str): The type of model to use ('logistic' or 'random_forest').

    Returns:
    dict: A dictionary with accuracy and AUC.
    """
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    else:
        raise ValueError("Invalid model_type. Choose 'logistic' or 'random_forest'.")
    # if train_label are all the same, then randomly flip one label
    if len(np.unique(train_labels)) == 1:
        indices = np.random.choice(train_labels.shape[0], 1, replace=False)
        train_labels[indices] = 1 - train_labels[indices]
    model.fit(train_data, train_labels)
    y_pred = model.predict(test_data)
    y_prob = model.predict_proba(test_data)[:, 1]  # Probability estimates for the positive class

    results = {
        'accuracy': accuracy_score(test_labels, y_pred),
        'auc': roc_auc_score(test_labels, y_prob)
    }
    return results


def tstr(real_data, synthetic_data, model_type='logistic', index=None):
    """
    Train a model on synthetic data and evaluate on real data.

    Parameters:
    real_data (numpy.ndarray): The real data matrix.
    synthetic_data (numpy.ndarray): The synthetic data matrix.
    model_type (str): The type of model to use ('logistic' or 'random_forest').

    Returns:
    dict: A dictionary with accuracy and AUC.
    """
    assert index is not None
    real_labels = real_data[:, index]
    real_data = np.delete(real_data, index, axis=1)
    synthetic_labels = synthetic_data[:, index]
    synthetic_data = np.delete(synthetic_data, index, axis=1)

    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(real_data, real_labels, test_size=0.2, random_state=42, shuffle=False)
    return evaluate_model(synthetic_data, synthetic_labels, X_test_real, y_test_real, model_type)


def tsrtr(real_data, synthetic_data, model_type='logistic', index=None):
    """
    Train a model on a mixture of synthetic and real data and evaluate on real data.

    Parameters:
    real_data (numpy.ndarray): The real data matrix.
    synthetic_data (numpy.ndarray): The synthetic data matrix.
    model_type (str): The type of model to use ('logistic' or 'random_forest').

    Returns:
    dict: A dictionary with accuracy and AUC.
    """
    assert index is not None
    real_labels = real_data[:, index]
    real_data = np.delete(real_data, index, axis=1)
    synthetic_labels = synthetic_data[:, index]
    synthetic_data = np.delete(synthetic_data, index, axis=1)

    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(real_data, real_labels, test_size=0.2, random_state=42, shuffle=False)

    X_train_mix = np.vstack((X_train_real, synthetic_data))
    y_train_mix = np.hstack((y_train_real, synthetic_labels))

    return evaluate_model(X_train_mix, y_train_mix, X_test_real, y_test_real, model_type)


def trtr(real_data, model_type='logistic', index=None):
    """
    Train a model on real data and evaluate on real data.

    Parameters:
    real_data (numpy.ndarray): The real data matrix.
    model_type (str): The type of model to use ('logistic' or 'random_forest').

    Returns:
    dict: A dictionary with accuracy and AUC.
    """
    # index-th dimension will be treated as label, the rest will be treated as features
    assert index is not None
    real_labels = real_data[:, index]
    real_data = np.delete(real_data, index, axis=1)

    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(real_data, real_labels, test_size=0.2, random_state=42, shuffle=False)

    return evaluate_model(X_train_real, y_train_real, X_test_real, y_test_real, model_type)


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