# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:56:49 2023

@author: Fanding Xu
"""

import pandas as pd
import os
import yaml
os.environ["BABEL_DATADIR"] = "C:\\Users\\USER\\anaconda3\\envs\\compe\\share\\openbabel"
import pickle
import torch
from tqdm import tqdm
from rdkit import Chem
if __name__ == "__main__":
    from databuild import labeled_data
else:
    from .databuild import labeled_data
    
tasks = {'BBBP': ['p_np'],
         'bace': ['Class'],
         'clintox': ['FDA_APPROVED', 'CT_TOX'],
         'HIV': ['HIV_active'],
         'tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
         'esol': ['measured log solubility in mols per litre'],
         'freesolv': ['expt'],
         'Lipophilicity': ['exp'],
         'muv': ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
            'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
            'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'],
         'sider': ['Hepatobiliary disorders',
                   'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
                   'Investigations', 'Musculoskeletal and connective tissue disorders',
                   'Gastrointestinal disorders', 'Social circumstances',
                   'Immune system disorders', 'Reproductive system and breast disorders',
                   'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                   'General disorders and administration site conditions',
                   'Endocrine disorders', 'Surgical and medical procedures',
                   'Vascular disorders', 'Blood and lymphatic system disorders',
                   'Skin and subcutaneous tissue disorders',
                   'Congenital, familial and genetic disorders',
                   'Infections and infestations',
                   'Respiratory, thoracic and mediastinal disorders',
                   'Psychiatric disorders', 'Renal and urinary disorders',
                   'Pregnancy, puerperium and perinatal conditions',
                   'Ear and labyrinth disorders', 'Cardiac disorders',
                   'Nervous system disorders',
                   'Injury, poisoning and procedural complications'],
         'Mutagenicity': ['Mutagenicity'],
         'PCBA': ['PCBA-1030', 'PCBA-1379', 'PCBA-1452', 'PCBA-1454', 'PCBA-1457',
             'PCBA-1458', 'PCBA-1460', 'PCBA-1461', 'PCBA-1468', 'PCBA-1469',
             'PCBA-1471', 'PCBA-1479', 'PCBA-1631', 'PCBA-1634', 'PCBA-1688',
             'PCBA-1721', 'PCBA-2100', 'PCBA-2101', 'PCBA-2147', 'PCBA-2242',
             'PCBA-2326', 'PCBA-2451', 'PCBA-2517', 'PCBA-2528', 'PCBA-2546',
             'PCBA-2549', 'PCBA-2551', 'PCBA-2662', 'PCBA-2675', 'PCBA-2676', 'PCBA-411',
             'PCBA-463254', 'PCBA-485281', 'PCBA-485290', 'PCBA-485294', 'PCBA-485297',
             'PCBA-485313', 'PCBA-485314', 'PCBA-485341', 'PCBA-485349', 'PCBA-485353',
             'PCBA-485360', 'PCBA-485364', 'PCBA-485367', 'PCBA-492947', 'PCBA-493208',
             'PCBA-504327', 'PCBA-504332', 'PCBA-504333', 'PCBA-504339', 'PCBA-504444',
             'PCBA-504466', 'PCBA-504467', 'PCBA-504706', 'PCBA-504842', 'PCBA-504845',
             'PCBA-504847', 'PCBA-504891', 'PCBA-540276', 'PCBA-540317', 'PCBA-588342',
             'PCBA-588453', 'PCBA-588456', 'PCBA-588579', 'PCBA-588590', 'PCBA-588591',
             'PCBA-588795', 'PCBA-588855', 'PCBA-602179', 'PCBA-602233', 'PCBA-602310',
             'PCBA-602313', 'PCBA-602332', 'PCBA-624170', 'PCBA-624171', 'PCBA-624173',
             'PCBA-624202', 'PCBA-624246', 'PCBA-624287', 'PCBA-624288', 'PCBA-624291',
             'PCBA-624296', 'PCBA-624297', 'PCBA-624417', 'PCBA-651635', 'PCBA-651644',
             'PCBA-651768', 'PCBA-651965', 'PCBA-652025', 'PCBA-652104', 'PCBA-652105',
             'PCBA-652106', 'PCBA-686970', 'PCBA-686978', 'PCBA-686979', 'PCBA-720504',
             'PCBA-720532', 'PCBA-720542', 'PCBA-720551', 'PCBA-720553', 'PCBA-720579',
             'PCBA-720580', 'PCBA-720707', 'PCBA-720708', 'PCBA-720709', 'PCBA-720711',
             'PCBA-743255', 'PCBA-743266', 'PCBA-875', 'PCBA-881', 'PCBA-883',
             'PCBA-884', 'PCBA-885', 'PCBA-887', 'PCBA-891', 'PCBA-899', 'PCBA-902',
             'PCBA-903', 'PCBA-904', 'PCBA-912', 'PCBA-914', 'PCBA-915', 'PCBA-924',
             'PCBA-925', 'PCBA-926', 'PCBA-927', 'PCBA-938', 'PCBA-995'],
         'covid': ['isactive'],
         'clintox_tdc': ['tox'],
         'ld50': ['ld50']
         }

REG = ['esol', 'freesolv', 'Lipophilicity', 'ld50']
CLS = ['BBBP', 'bace', 'clintox', 'HIV', 'tox21', 'muv', 'sider', 'Mutagenicity', 'PCBA', 'covid', 'clintox_tdc']

def save_data(data, filename):
    dirname = 'data/property/processed'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = os.path.join(dirname, filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!\n\n')


if not __name__ == "__main__":
    class DatasetConfig:
        config = yaml.load(open('dataset/tasks_config.yaml', 'r'), Loader=yaml.CLoader)
        
        @staticmethod
        def register_args(args):
            try:
                dataset = args.dataset
            except: raise ValueError("dataset name not defined")
            assert dataset in DatasetConfig.config.keys(), f"wrong dataset name: {dataset}, check in {REG+CLS}"
    
            args.task = DatasetConfig.config[dataset]['task']
            args.monitor = DatasetConfig.config[dataset]['monitor']
    
            args.num_class = len(tasks[dataset])
            args.batch_size = DatasetConfig.config[dataset]['batch_size']
            args.lr = float(DatasetConfig.config[dataset]['lr'])
            args.lr_multiplier = float(DatasetConfig.config[dataset]['lr_multiplier'])
            args.unet_decay = float(DatasetConfig.config[dataset]['unet_decay'])
            args.min_epochs = DatasetConfig.config[dataset]['min_epochs']
            args.patience = DatasetConfig.config[dataset]['patience']
        

def DataPrePop(mols, labels, smiles=None):
    """
    This function is to delete single atom molecules or whose SMILES filed converting to rdkit.Chem.rdchem.Mol
    mols and labels are List here
    """
    for i in range(len(mols))[::-1]:
        if mols[i] is None:
            mols.pop(i)
            labels.pop(i)
            if smiles is not None:
                smiles.pop(i)
        elif mols[i].GetNumAtoms() == 1:
            mols.pop(i)
            labels.pop(i)    
            if smiles is not None:
                smiles.pop(i)
    if smiles is not None:
        return mols, labels, smiles
    return mols, labels

def mol_filter_by_smi(smiles, labels):
    new_smiles, new_labels = [], []
    mols = []
    for smi, y in tqdm(zip(smiles, labels), total=len(smiles)):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                new_smiles.append(smi)
                new_labels.append(y)
        except:
            continue
    return mols, new_smiles, new_labels


def generate_data_list(dataset):
    df = pd.read_csv(f'data/property/{dataset}.csv')
    smiles = df['smiles'].tolist()
    # mols = [Chem.MolFromSmiles(i) for i in tqdm(smiles, desc=f'smiles2mol for {dataset}...')]
    labels = df[tasks[dataset]]
    labels = labels.fillna(-1)
    labels = labels.values.tolist()
    # mols, labels, smiles = DataPrePop(mols, labels, smiles)
    print(f'smiles2mol for {dataset}...')
    mols, smiles, labels = mol_filter_by_smi(smiles, labels)
    data_list = [labeled_data(smi, label, get_fp=True, use_OB=False, ignore_frags=False,
                              with_hydrogen=False, with_coordinate=False, seed=123) \
                 for smi, mol, label in tqdm(zip(smiles, mols, labels), total=len(smiles), desc=f"processing data {dataset}...")]
    return data_list

def get_dataset_info(dataset):
    assert dataset in REG or dataset in CLS, f"wrong dataset name: {dataset}"
    if dataset in REG:
        task = 'reg'
    else:
        task = 'cls'
    num_class = len(tasks[dataset])
    print(f"Running {task} task: {dataset}, with {num_class} labels")
    return num_class, task


#%%
if __name__ == "__main__":
    for dataset in tasks.keys():
        if dataset not in ['PCBA', 'HIV', 'muv', 'Mutagenicity']:
            data_list = generate_data_list(dataset)
            torch.save(data_list, f'data/property/processed/{dataset}.pth')
            
    # data_list = generate_data_list('ld50')
    # torch.save(data_list, 'data/property/processed/ld50.pth')















































































