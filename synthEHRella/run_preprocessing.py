import re
import os
import sys
import copy
import json
import _pickle as pickle
import numpy as np
from datetime import datetime
from synthEHRella.utils.data_transform import DataTransform
import pandas as pd
import numpy as np
from tqdm import tqdm


# This script processes MIMIC-III dataset and builds a binary matrix or a count matrix depending on your input.
# The output matrix is a Numpy matrix of type float32, and suitable for training medGAN.
# Written by Edward Choi (mp2893@gatech.edu)
# Usage: Put this script to the folder where MIMIC-III CSV files are located. Then execute the below command.
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv <output file> <"binary"|"count">
# Note that the last argument "binary/count" determines whether you want to create a binary matrix or a count matrix.

# Output files
# <output file>.pids: cPickled Python list of unique Patient IDs. Used for intermediate processing
# <output file>.matrix: Numpy float32 matrix. Each row corresponds to a patient. Each column corresponds to a ICD9 diagnosis code.
# <output file>.types: cPickled Python dictionary that maps string diagnosis codes to integer diagnosis codes.

def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr
    
def convert_to_3digit_icd9(dxStr):  # merge into broader category (because the last two digits generally have no use.)
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3]
        else: return dxStr


def preprocessing_mimic3(
    admissionFile,
    diagnosisFile,
    outPath,
    binary_count,
):
    if binary_count != 'binary' and binary_count != 'count':
        print('You must choose either binary or count.')
        sys.exit()

    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    infd = open(admissionFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap: pidAdmMap[pid].append(admId)
        else: pidAdmMap[pid] = [admId]
    infd.close()

    print('Building admission-dxList mapping')

    originalTo3digit = {}
    threeDigitToOriginal = {}

    admDxMap = {}
    infd = open(diagnosisFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        #dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
        dxStr = 'D_' + convert_to_3digit_icd9(tokens[4][1:-1])

        if convert_to_3digit_icd9(tokens[4][1:-1]) not in originalTo3digit:
            originalTo3digit[tokens[4][1:-1]] = convert_to_3digit_icd9(tokens[4][1:-1])
        
        if convert_to_3digit_icd9(tokens[4][1:-1]) not in threeDigitToOriginal:
            threeDigitToOriginal[convert_to_3digit_icd9(tokens[4][1:-1])] = []
            threeDigitToOriginal[convert_to_3digit_icd9(tokens[4][1:-1])].append(tokens[4][1:-1])
        else:
            threeDigitToOriginal[convert_to_3digit_icd9(tokens[4][1:-1])].append(tokens[4][1:-1])

        if admId in admDxMap: admDxMap[admId].append(dxStr)
        else: admDxMap[admId] = [dxStr]
    infd.close()

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    for pid, admIdList in pidAdmMap.items():
        #if len(admIdList) < 2: continue
        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList])
        pidSeqMap[pid] = sortedList
    
    print('Building pids, dates, strSeqs')
    pids = []
    dates = []
    seqs = []
    for pid, visits in pidSeqMap.items():
        pids.append(pid)
        seq = []
        date = []
        for visit in visits:
            date.append(visit[0])  # date (string)
            seq.append(visit[1])  # diagnosis (list)
        dates.append(date)
        seqs.append(seq)
    
    print('Converting strSeqs to intSeqs, and making types')
    types = {}
    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:  # visit is a list of diagnosis codes in this single visit
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    print('Constructing the matrix')
    numPatients = len(newSeqs)
    numCodes = len(types)
    matrix = np.zeros((numPatients, numCodes)).astype('float32')
    for i, patient in enumerate(newSeqs):
        for visit in patient:
            for code in visit:
                if binary_count == 'binary':
                    matrix[i][code] = 1.
                else:
                    matrix[i][code] += 1.

    pickle.dump(pids, open(outPath + '/processed_mimic3.pids', 'wb'), -1)
    pickle.dump(matrix, open(outPath +'/processed_mimic3.matrix', 'wb'), -1)
    pickle.dump(types, open(outPath +'/processed_mimic3.types', 'wb'), -1)

    transform = DataTransform()
    mimic3_phecodex = transform.icd9tophecode(matrix)
    mimic3_phecodexm = transform.phecodextophecodexm(mimic3_phecodex)

    np.save(f'{outPath}/mimic3-real-phecodex.npy', mimic3_phecodex)
    np.save(f'{outPath}/mimic3-real-phecodexm.npy', mimic3_phecodexm)


def preprocessing_mimic4(
    admissionFile,
    diagnosisFile,
    outPath,
    binary_count,
):
    if binary_count != 'binary' and binary_count != 'count':
        print('You must choose either binary or count.')
        sys.exit()

    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    infd = open(admissionFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[0])
        admId = int(tokens[1])
        admTime = datetime.strptime(tokens[2], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap: 
            pidAdmMap[pid].append(admId)
        else: 
            pidAdmMap[pid] = [admId]
    infd.close()

    print('Building admission-dxList mapping')

    # originalTo3digit = {}
    # threeDigitToOriginal = {}

    admDxMap = {}
    infd = open(diagnosisFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[1])
        #dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
        dxStr = 'D_' + convert_to_3digit_icd9(tokens[3]) + f'_icd_{tokens[-1]}'

        # if convert_to_3digit_icd9(tokens[3][1:-1]) not in originalTo3digit:
        #     originalTo3digit[tokens[3][1:-1]] = convert_to_3digit_icd9(tokens[3][1:-1])
        
        # if convert_to_3digit_icd9(tokens[4][1:-1]) not in threeDigitToOriginal:
        #     threeDigitToOriginal[convert_to_3digit_icd9(tokens[3][1:-1])] = []
        #     threeDigitToOriginal[convert_to_3digit_icd9(tokens[3][1:-1])].append(tokens[3][1:-1])
        # else:
        #     threeDigitToOriginal[convert_to_3digit_icd9(tokens[3][1:-1])].append(tokens[3][1:-1])

        if admId in admDxMap: 
            admDxMap[admId].append(dxStr)
        else: 
            admDxMap[admId] = [dxStr]
    infd.close()

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    for pid, admIdList in pidAdmMap.items():
        #if len(admIdList) < 2: continue
        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList if admId in admDxMap])
        pidSeqMap[pid] = sortedList
    
    print('Building pids, dates, strSeqs')
    pids_mimic4 = []
    dates = []
    seqs = []
    for pid, visits in pidSeqMap.items():
        pids_mimic4.append(pid)
        seq = []
        date = []
        for visit in visits:
            date.append(visit[0])  # date (string)
            seq.append(visit[1])  # diagnosis (list)
        dates.append(date)
        seqs.append(seq)
    
    print('Converting strSeqs to intSeqs, and making types')
    types_mimic4 = {}
    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:  # visit is a list of diagnosis codes in this single visit
            newVisit = []
            for code in visit:
                if code in types_mimic4:
                    newVisit.append(types_mimic4[code])
                else:
                    types_mimic4[code] = len(types_mimic4)
                    newVisit.append(types_mimic4[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    print('Constructing the matrix')
    numPatients = len(newSeqs)
    numCodes = len(types_mimic4)
    matrix_mimic4 = np.zeros((numPatients, numCodes)).astype('float32')
    for i, patient in enumerate(newSeqs):
        for visit in patient:
            for code in visit:
                if binary_count == 'binary':
                    matrix_mimic4[i][code] = 1.
                else:
                    matrix_mimic4[i][code] += 1.

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(f'{base_dir}/utils/ICD9toICD10Mapping.json', 'r') as f:
        icd92icd10 = json.load(f)

    with open(f'{base_dir}/utils/ICD10types.json', 'r') as f:
        icd10types = json.load(f)
    
    with open(f'{base_dir}/utils/processed_mimic3.types', 'rb') as f:  # TODO: add path
        types_mimic3 = pickle.load(f)

    # pickle.dump(pids_mimic4, open(outFile+'_include_icd10.pids', 'wb'), -1)
    # pickle.dump(matrix_mimic4, open(outFile+'_include_icd10.matrix', 'wb'), -1)
    # pickle.dump(types_mimic4, open(outFile+'_include_icd10.types', 'wb'), -1)

    def icd9toicd10(data):
        """
        Transform the data matrix to a binary matrix with ICD-10 codes.
        """
        n, p = data.shape
        
        # Initialize the icd10_data matrix
        icd10_data = np.zeros((n, len(icd10types)), dtype=int)
        
        # Create a mapping from ICD-9 indices to ICD-10 codes
        icd9_to_icd10 = []
        for j in range(p):
            if str(j) in icd92icd10:
                for code in icd92icd10[str(j)]:
                    icd9_to_icd10.append((j, code))
        
        # Convert the list to a NumPy array for efficient processing
        icd9_to_icd10 = np.array(icd9_to_icd10)
        
        # Use advanced indexing to set the corresponding ICD-10 codes
        rows, cols = np.nonzero(data[:, icd9_to_icd10[:, 0]])
        icd10_data[rows, icd9_to_icd10[cols, 1]] = 1

        return icd10_data

    matrix_mimic4_mimic4_icd10_columns = []
    matrix_mimic4_mimic3_icd9_only = np.zeros((matrix_mimic4.shape[0], len(types_mimic3)))
    for diag_type, index in types_mimic4.items():
        if diag_type.endswith("_icd_9"):
            diag_type = re.sub("_icd_9", "", diag_type)
            if diag_type in types_mimic3:
                mimic3_index = types_mimic3[diag_type]
                matrix_mimic4_mimic3_icd9_only[:, mimic3_index] = matrix_mimic4[:, index]
        elif diag_type.endswith("_icd_10"):
            diag_type = re.sub("_icd_10", "", diag_type)
            matrix_mimic4_mimic4_icd10_columns.append((diag_type, index))
    
    matrix_mimic4_icd10 = icd9toicd10(matrix_mimic4_mimic3_icd9_only)
    matrix_mimic4_icd10_update = copy.deepcopy(matrix_mimic4_icd10)

    for diag_type, index in matrix_mimic4_mimic4_icd10_columns:
        diag_type = re.sub("D_", "", diag_type)
        if diag_type in icd10types:
            matrix_mimic4_icd10_update[:, icd10types[diag_type]] = matrix_mimic4_icd10_update[:, icd10types[diag_type]] + matrix_mimic4[:, index]

    matrix_mimic4_icd10_update[matrix_mimic4_icd10_update != 0] = 1

    transform = DataTransform()
    new_matrix_mimic4_phecodex = transform.icd10tophecode(matrix_mimic4_icd10_update)
    new_matrix_mimic4_phecodexm = transform.phecodextophecodexm(new_matrix_mimic4_phecodex)

    np.save(f'{outPath}/mimic4-real-phecodex.npy', new_matrix_mimic4_phecodex)
    np.save(f'{outPath}/mimic4-real-phecodexm.npy', new_matrix_mimic4_phecodexm)


def preprocessing_mimic3_survival(
    admissionFile,
    diagnosisFile,
    patientsFile,
    outPath,
):
    admissions_df = pd.read_csv(admissionFile)
    diagnosis_df = pd.read_csv(diagnosisFile)
    patients_df = pd.read_csv(patientsFile)

    # merge admissions and diagnosis by SUBJECT_ID and HADM_ID
    merged_df = pd.merge(admissions_df, diagnosis_df, on=['SUBJECT_ID', 'HADM_ID'])
    # only keep SUBJECT_ID, HADM_ID, ICD9_CODE, ADMITTIME
    merged_df = merged_df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'ICD9_CODE']]

    def convert_to_3digit_icd9(dxStr):  # merge into broader category (because the last two digits generally have no use.)
        if dxStr.startswith('E'):
            if len(dxStr) > 4: return dxStr[:4]
            else: return dxStr
        else:
            if len(dxStr) > 3: return dxStr[:3]
            else: return dxStr

    # apply this function to the ICD9_CODE column (transform to str first)
    merged_df['ICD9_CODE'] = merged_df['ICD9_CODE'].astype(str)
    merged_df['ICD9_CODE'] = merged_df['ICD9_CODE'].apply(convert_to_3digit_icd9)

    # add "D_" to all ICD9_CODE.
    merged_df['ICD9_CODE'] = 'D_' + merged_df['ICD9_CODE']
    # replace 'D_nan' with 'D_'
    merged_df['ICD9_CODE'] = merged_df['ICD9_CODE'].replace('D_nan', 'D_')

    df_pivot = merged_df.pivot_table(index=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME'], columns='ICD9_CODE', aggfunc=lambda x: int(any(x)), fill_value=0)

    df_pivot.reset_index(inplace=True)
    df_pivot.columns.name = None  # Remove the categorization name to flatten the DataFrame structure for clarity

    # construct a new column called DAY. Group rows by SUBJECT_ID and DAY is the difference between ADMITTIME and the earliest ADMITTIME for each SUBJECT_ID
    df_pivot['ADMITTIME'] = pd.to_datetime(df_pivot['ADMITTIME'])
    df_pivot['DAY'] = df_pivot.groupby('SUBJECT_ID')['ADMITTIME'].transform(lambda x: (x - x.min()).dt.days)


    # prepare socio-demographic variables
    patients_df = patients_df.dropna(subset=['DOB'])
    admissions_df = admissions_df.dropna(subset=['ADMITTIME'])

    # admissions_df for each SUBJECT_ID, only keep the row with earliest ADMITTIME
    admissions_df = admissions_df.sort_values('ADMITTIME').groupby('SUBJECT_ID').head(1)

    # merge
    demo_df = pd.merge(patients_df, admissions_df, on='SUBJECT_ID')
    assert len(demo_df) == 46520 # the number of rows in real data

    # Convert to datetime, coercing errors
    demo_df['DOB'] = pd.to_datetime(demo_df['DOB'], errors='coerce')
    demo_df['ADMITTIME'] = pd.to_datetime(demo_df['ADMITTIME'], errors='coerce')

    # Define a function to calculate age safely
    def calculate_age(dob, admit_time):
        if pd.isnull(dob) or pd.isnull(admit_time):
            return np.nan  # Return NaN if either date is missing
        try:
            # Calculate timedelta and convert to years
            age_years = (admit_time - dob).days / 365.25
            if age_years < 0:
                return np.nan  # Return NaN if age is negative
            return age_years
        except OverflowError:
            return np.nan  # Return NaN if there is an overflow error

    # Apply the function to each row
    for i, row in tqdm(demo_df.iterrows()):
        try:
            demo_df.at[i, 'AGE_IN_YEARS'] = int(calculate_age(row['DOB'], row['ADMITTIME']))
        except: # add na
            demo_df.at[i, 'AGE_IN_YEARS'] = 86  # add 86 to the row with age >= 85

    # group age into 0-9, 10-19, ..., 70-79, 80+
    demo_df['AGE_GROUP'] = pd.cut(demo_df['AGE_IN_YEARS'], bins=[-1, 9, 19, 29, 39, 49, 59, 69, 79, 1000], labels=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'])

    # Define a function to map subcategories to the specified main categories
    def map_ethnicity(ethnicity):
        if 'ASIAN' in ethnicity:
            return 'ASIAN'
        elif 'BLACK' in ethnicity:
            return 'BLACK'
        elif 'HISPANIC' in ethnicity or 'LATINO' in ethnicity:
            return 'HISPANIC'
        elif 'WHITE' in ethnicity:
            return 'WHITE'
        else:
            return 'OTHER'

    # Apply the mapping function to the ETHNICITY column
    demo_df['MERGED_ETHNICITY'] = demo_df['ETHNICITY'].apply(map_ethnicity)

    # merge socio-demographic and the survival dataset
    df_pivot_2 = pd.merge(df_pivot, demo_df, on='SUBJECT_ID')

    # Filter the DataFrame
    df_pivot_3 = df_pivot_2.groupby('SUBJECT_ID').tail(1)
    df_pivot_3 = df_pivot_3[df_pivot_3.columns[~df_pivot_3.columns.str.startswith('D_')]]

    # Compute the maximum DAY for each SUBJECT_ID
    max_days = df_pivot_2.groupby('SUBJECT_ID')['DAY'].max()

    # Filter out D_ columns
    D_columns = df_pivot_2.columns[df_pivot_2.columns.str.startswith('D_')]

    # Initialize a dictionary to store event and TimeToEvent data
    events_data = {}

    for col in D_columns:
        events_data['event_' + col] = df_pivot_2.groupby('SUBJECT_ID')[col].max()
        events_data['TimeToEvent_' + col] = df_pivot_2[df_pivot_2[col] == 1].groupby('SUBJECT_ID')['DAY'].min()

    # Create a DataFrame from the dictionary
    events_df = pd.DataFrame(events_data)
    events_df = events_df.reset_index()

    # Fill missing TimeToEvent values with the maximum DAY for each SUBJECT_ID where the event did not occur
    for col in D_columns:
        mask = events_df['event_' + col] == 0
        events_df.loc[mask, 'TimeToEvent_' + col] = events_df.loc[mask, 'SUBJECT_ID'].map(max_days)

    # Merge the data
    df_pivot_3 = df_pivot_3.merge(events_df, on='SUBJECT_ID', how='left')
    df_pivot_3.to_csv(f"{outPath}/preprocessed_mimiciii_for_plasmode_with_demo_survival.csv")


if __name__ == '__main__':
    admissionFile = sys.argv[1]
    diagnosisFile = sys.argv[2]
    patientsFile = sys.argv[3]
    outPath = sys.argv[4]
    binary_count = sys.argv[5]
    dataset = sys.argv[6]
    
    if dataset == "mimic3":
        preprocessing_mimic3(admissionFile, diagnosisFile, outPath, binary_count)
        preprocessing_mimic3_survival(admissionFile, diagnosisFile, patientsFile, outPath)
    elif dataset == "mimic4":
        preprocessing_mimic4(admissionFile, diagnosisFile, outPath, binary_count)
    