from promptehr import SequencePatient
from promptehr import load_synthetic_data
from promptehr import PromptEHR
import pickle
import numpy as np
import torch
import argparse
import os


if __name__ == "__main__":
    # setup argument
    parser = argparse.ArgumentParser(description='Generate synthetic data using PromptEHR')
    parser.add_argument('--num_gen_samples', type=int, default=50000, help='Number of fake samples to generate')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the generated data')
    
    args = parser.parse_args()

    # init model
    model = PromptEHR()
    model.from_pretrained(input_dir=args.save_path)

    # load input data
    demo = load_synthetic_data(n_sample=args.num_gen_samples) # we have 10,000 samples in total

    # build the standard input data for train or test PromptEHR models
    seqdata = SequencePatient(data={'v':demo['visit'], 'y':demo['y'], 'x':demo['feature'],},
        metadata={
            'visit':{'mode':'dense'},
            'label':{'mode':'tensor'}, 
            'voc':demo['voc'],
            'max_visit':20,
            }
        )
    # you can try to fit on this data by
    # model.fit(seqdata)

    # start generate
    # n: the target total number of samples to generate
    # n_per_sample: based on each sample, how many fake samples will be generated
    # the output will have the same format of `SequencePatient`
    with torch.no_grad():
        # fake_data = model.predict(seqdata, n=50000, n_per_sample=5, verbose=True)
        fake_data = model.predict(seqdata, n=100, verbose=True)

    diags = []
    for patient in fake_data['visit']:
        diag = []
        for visit in patient:
            for individual_diag in visit[0]:
                diag.append(demo['voc']['diag'].idx2word[individual_diag])
        diag = list(set(diag))
        diags.append(diag)

    # save diags
    with open('processed_mimic3.types', 'rb') as f:
        types = pickle.load(f)

    def convert_to_3digit_icd9(dxStr):  # merge into broader category (because the last two digits generally have no use.)
        if dxStr.startswith('E'):
            if len(dxStr) > 4: return dxStr[:4]
            else: return dxStr
        else:
            if len(dxStr) > 3: return dxStr[:3]
            else: return dxStr

    # construct numpy array with shapes (len(diags), len(types))
    data = np.zeros((len(diags), len(types)))
    for i, diag in enumerate(diags):
        for d in diag:
            d = 'D_' + convert_to_3digit_icd9(d)
            if d in types:
                data[i, types[d]] = 1

    # save data
    # np.save(args.save_path, data)
    np.save(os.path.join(args.save_path, "promptehr-synthetic.npy"), data)
    print("Finished.")