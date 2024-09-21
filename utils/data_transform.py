import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class DataTransform(object):
    def __init__(self):
        with open('ICD9toICD10Mapping.json', 'r') as f:
            self.icd92icd10 = json.load(f)
        
        with open('ICD10types.json', 'r') as f:
            self.icd10types = json.load(f)

        with open('icd10_to_phecodex_mapping.json') as f:
            self.icd10_to_phecodex = json.load(f)
        
        with open('phecodex_types.json', 'r') as f:
            self.phecodex_types = json.load(f)
        
        with open('phecodex_to_phecodexm_mapping.json', 'r') as f:
            self.phecodex_to_phecodexm = json.load(f)
        
        with open('phecodexm_types.json', 'r') as f:
            self.phecodexm_types = json.load(f)

        with open("snomed2icd10.json", "r") as f:
            self.snomed2icd10_dict = json.load(f)

    def snomedtoicd10(self, data_path='results/synthea-synthetic/csv/conditions.csv'):
        data = pd.read_csv(data_path)

        # group data by "PATIENT". For "CODE" column, Please keep only unique code
        df_filtered = data[['PATIENT', 'CODE']]
        patient_to_codes = df_filtered.groupby('PATIENT')['CODE'].apply(lambda codes: list(codes.unique())).to_dict()

        patient_to_icd10codes = {}
        for k, v in tqdm(patient_to_codes.items()):
            patient_to_icd10codes[k] = []
            for code in v:
                if code in self.snomed2icd10_dict:
                    patient_to_icd10codes[k].extend(self.snomed2icd10_dict[code])
            patient_to_icd10codes[k] = list(set(patient_to_icd10codes[k]))

        matrix = np.zeros((len(patient_to_icd10codes), len(self.icd10types)))
        for i, (k, v) in tqdm(enumerate(patient_to_icd10codes.items())):
            for code in v:
                code = str(code)
                code = code.replace(".", "")
                if code in self.icd10types:
                    matrix[i, self.icd10types[code]] = 1
        return matrix

    def snomedtophecode(self, data_path='results/synthea-synthetic/csv/conditions.csv'):
        icd10_data = self.snomedtoicd10(data_path)
        phecode_data = self.icd10tophecode(icd10_data)
        return phecode_data

    def icd9toicd10(self, data):
        """
        Transform the data matrix to a binary matrix with ICD-10 codes.
        """
        n, p = data.shape
        
        # Initialize the icd10_data matrix
        icd10_data = np.zeros((n, len(self.icd10types)), dtype=int)
        
        # Create a mapping from ICD-9 indices to ICD-10 codes
        icd9_to_icd10 = []
        for j in range(p):
            if str(j) in self.icd92icd10:
                for code in self.icd92icd10[str(j)]:
                    icd9_to_icd10.append((j, code))
        
        # Convert the list to a NumPy array for efficient processing
        icd9_to_icd10 = np.array(icd9_to_icd10)
        
        # Use advanced indexing to set the corresponding ICD-10 codes
        rows, cols = np.nonzero(data[:, icd9_to_icd10[:, 0]])
        icd10_data[rows, icd9_to_icd10[cols, 1]] = 1

        return icd10_data

    def icd10tophecode(self, data):
        """
        Transform the data matrix to a binary matrix with Phecode codes.
        """
        n, p = data.shape
        
        # Initialize the phecode_data matrix
        phecode_data = np.zeros((n, len(self.phecodex_types)), dtype=int)
        
        # Create a mapping from ICD-10 indices to Phecode codes
        icd10_to_phecode = []
        for j in range(p):
            if str(j) in self.icd10_to_phecodex:
                for code in self.icd10_to_phecodex[str(j)]:
                    icd10_to_phecode.append((j, code))
        
        # Convert the list to a NumPy array for efficient processing
        icd10_to_phecode = np.array(icd10_to_phecode)
        
        # Use advanced indexing to set the corresponding Phecode codes
        rows, cols = np.nonzero(data[:, icd10_to_phecode[:, 0]])
        phecode_data[rows, icd10_to_phecode[cols, 1]] = 1

        return phecode_data
    
    def icd9tophecode(self, data):
        icd10_data = self.icd9toicd10(data)
        phecode_data = self.icd10tophecode(icd10_data)
        return phecode_data

    def phecodextophecodexm(self, data):
        n, p = data.shape
        phecodexm = np.zeros((n, len(self.phecodexm_types)), dtype=int)

        phecodex_to_phecodexm = []
        for j in range(p):
            if str(j) in self.phecodex_to_phecodexm:
                phecodex_to_phecodexm.append((j, self.phecodex_to_phecodexm[str(j)]))
        
        phecodex_to_phecodexm = np.array(phecodex_to_phecodexm)
        # print(phecodex_to_phecodexm.shape)
        # print(data.shape)


        rows, cols = np.nonzero(data[:, phecodex_to_phecodexm[:, 0]])
        phecodexm[rows, phecodex_to_phecodexm[cols, 1]] = 1

        return phecodexm

def compute_prevalence(data):
    """
    Compute the prevalence of each code (column-wise mean) in the dataset.
    """
    return data.mean(axis=0)

def compute_num_of_codes_per_patient(data):
    """
    Compute the number of codes per patient (row-wise sum) in the dataset.
    """
    return data.sum(axis=1)


# depict box-plot for each row of data, if have k rows, then generate a plot with k boxes
def depict_box_plot(data):
    """
    Depict the box plot for each row of the data matrix.
    """
    # n, p = data.shape
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, vert=False)
    ax.set_yticklabels([])
    ax.set_xlabel('Number of Codes')
    plt.show()