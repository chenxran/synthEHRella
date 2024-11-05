import os
import pickle
import numpy as np
import pandas as pd
from utils.data_transform import DataTransform
# implement
    # real_data = load_real_data(config)
    # synthetic_data = load_synthetic_data(config)

# def load_real_data(config):
#     if config.data.real_data_phecodexm_path is None or not os.path.exists(config.data.real_data_phecodexm_path):
#         if config.data.real_data_icd9_path is None or not os.path.exists(config.data.real_data_icd9_path):
#             raise ValueError("Real data path is not provided or does not exist.")
#         else:
#             with open(config.data.real_data_icd9_path, 'rb') as f:
#                 real_data_icd9 = pickle.load(f)
            
#         transform = DataTransform()
#         real_data_phecodexm = transform.icd9tophecodexm(real_data_icd9)
#         if config.data.real_data_phecodexm_path is not None:
#             np.save(config.data.real_data_phecodexm_path, real_data_phecodexm)        
#     else:
#         real_data_phecodexm = np.load(config.data.real_data_phecodexm_path)
#     return real_data_phecodexm


# def load_synthetic_data(config):
#     if config.data.synthetic_data_path is None or not os.path.exists(config.data.synthetic_data_path):
#         raise ValueError("Synthetic data path is not provided or does not exist.")
#     else:
#         if config.data.synthetic_data_type not in ["snomed", 'plasmode']:       
#             synthetic_data = np.load(config.data.synthetic_data_path)
            
#             transform = DataTransform()
#             if config.data.synthetic_data_type == 'icd9':
#                 synthetic_data = transform.icd9tophecodexm(synthetic_data)
#             elif config.data.synthetic_data_type == 'icd10':
#                 synthetic_data = transform.icd10tophecodexm(synthetic_data)
#             elif config.data.synthetic_data_type == 'phecodex':
#                 synthetic_data = transform.phecodextophecodexm(synthetic_data)
#             elif config.data.synthetic_data_type == 'phecodexm':
#                 synthetic_data = synthetic_data
                
#         elif config.data.synthetic_data_type == 'snomed':
#             synthetic_data = transform.snomedtophecodexm(config.data.synthetic_data_path)
        
#         elif config.data.synthetic_data_type == 'plasmode':
#             with open('synthEHRella/utils/processed_mimic3.types', 'rb') as f:
#                 types = pickle.load(f)

#             matrix = np.zeros((50000, len(types)))
#             for key, value in types.items():
#                 if os.path.exists(f"{config.data.synthetic_data_path}/ event_{key} .csv"):
#                     data = pd.read_csv(f"{config.data.synthetic_data_path}/ event_{key} .csv")
#                     matrix[:, value] = data.sample(50000, replace=True).values.flatten().astype(int)
#                 else:
#                     print(f"Not exists {key}")
#         return synthetic_data
        