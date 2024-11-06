MIMIC_III_ADMISSION_FILE="data/physionet.org/files/mimiciii/1.4/ADMISSIONS.csv"
MIMIC_III_DIAGNOSIS_FILE="data/physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv"
MIMIC_III_PATIENTS_FILE="data/physionet.org/files/mimiciii/1.4/PATIENTS.csv"
MIMIC_IV_ADMISSION_FILE="data/physionet.org/files/mimiciv/2.2/hosp/admissions.csv"
MIMIC_IV_DIAGNOSIS_FILE="data/physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv"
MIMIC_IV_PATIENTS_FILE="data/physionet.org/files/mimiciv/2.2/hosp/patients.csv"
OUTPUT="testing_output/"

python -m synthEHRella.run_preprocessing $MIMIC_III_ADMISSION_FILE $MIMIC_III_DIAGNOSIS_FILE $MIMIC_III_PATIENTS_FILE $OUTPUT binary mimic3
python -m synthEHRella.run_preprocessing $MIMIC_IV_ADMISSION_FILE $MIMIC_IV_DIAGNOSIS_FILE $MIMIC_IV_PATIENTS_FILE $OUTPUT binary mimic4