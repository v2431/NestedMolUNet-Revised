# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:56:49 2023

@author: Fanding Xu
"""

import torch
import os
from typing import Optional
os.environ["BABEL_DATADIR"] = "C:\\Users\\USER\\anaconda3\\envs\\compe\\share\\openbabel"
# replace the path with your openbabel installation path
import pickle
import torch_geometric
from torch_geometric.data import Data

# from torch_geometric.utils.smiles import x_map, e_map
from rdkit import Chem, RDLogger
from rdkit.Chem.rdmolops import GetMolFrags
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from typing import List
try:
    from .descriptors.pubchemfp import GetPubChemFPs
    from .maplight import get_maplight_fingerprint
    from .utils import x_map, e_map
except:
    from descriptors.pubchemfp import GetPubChemFPs
    from maplight import get_maplight_fingerprint
    from utils import x_map, e_map
    
def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def remove_frag(smiles, mol):
    smiles_main = smiles
    if len(GetMolFrags(mol)) > 1:
        smiles_main = max(smiles.split('.'), key=len)
        mol = Chem.MolFromSmiles(smiles_main)
        print(f"\n{smiles}\nomit small frags to\n{smiles_main}")
    return mol, smiles_main

def getTraditionalFingerprintsFeature(mol: Chem.Mol):
    fp2 = []
    fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
    fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
    fp_pubcfp = GetPubChemFPs(mol)
    # print(f'maccs: {torch.tensor(fp_maccs).shape} pubchem: {torch.tensor(fp_pubcfp).shape} phaerg: {torch.tensor(fp_phaErGfp).shape}')
    fp2.extend(fp_maccs)
    fp2.extend(fp_phaErGfp)
    fp2.extend(fp_pubcfp)
    fp2 = torch.tensor([fp2], dtype=torch.float32)
    return fp2

def get_3d_conformer_random(mol: Chem.Mol):
    # mol = Chem.AddHs(mol)
    pos = torch.zeros((mol.GetNumAtoms(), 3)).uniform_(-1, 1)
    return pos

def get_3d_conformer(mol: Chem.Mol, smiles, seed):
    if len(smiles) >= 200:
        AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=42, useRandomCoords=True)
    else:
        AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=42)
    conformers = mol.GetConformers()
    if len(conformers) == 0:
        AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=42, useRandomCoords=True)
    conformers = mol.GetConformers()
    if len(conformers) == 0:
        print(f"\n3D failed, get 2D for {smiles}")
        AllChem.Compute2DCoords(mol, sampleSeed=seed)
    conformers = mol.GetConformers()
    conformer = conformers[0]
    point = conformer.GetPositions()
    return point

def from_smiles(smiles: str, mol: Optional[Chem.rdchem.Mol] = None,
                with_hydrogen: bool = False,
                with_coordinate: bool = False, kekulize: bool = False,
                seed: int = -1,
                get_fp: bool = True, ignore_frags=False) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.
    adapted from torch_geometric.utils.smiles.from_smiles

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        with_coordinate (bool, optional): If set to :obj:`True`, will store
            atom coordinate in the molecule graph as data.z. (default: :obj:`True`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
        seed:int (int, optional): provide a seed for the random number generator 
            so that the same coordinates can be obtained for a molecule on multiple runs.
            (default: -1, the RNG will not seeded)
        get_fp (bool, True): If set to :obj:`True`, will generate 'TraditionalFingerprintsFeature'
    """

    RDLogger.DisableLog('rdApp.*')
    
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        mol = Chem.MolFromSmiles('')
        
    smiles_main = ''
    if ignore_frags:
        if len(GetMolFrags(mol)) > 1:
            smiles_main = max(smiles.split('.'), key=len)
            mol = Chem.MolFromSmiles(smiles_main)
            print(f"\n{smiles}\nomit small frags to\n{smiles_main}")
        
    NumHeavyAtoms = mol.GetNumAtoms()
    
    if kekulize:
        Chem.Kekulize(mol)
        
    if get_fp:
        fp = getTraditionalFingerprintsFeature(mol)
    
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    
    if with_coordinate:
        
        pos = get_3d_conformer(mol, smiles, seed)
            # mol2 = Chem.AddHs(mol)
            # conf_num = AllChem.EmbedMolecule(mol2, randomSeed=seed)
            # if conf_num < 0:
            #     print(f"\n3D failed, get 2D for {smiles}")
            #     conf_num = AllChem.Compute2DCoords(mol2, sampleSeed=seed)
            # conf = mol2.GetConformer()
            # pos = conf.GetPositions()
            
        if not with_hydrogen:
            pos = pos[:NumHeavyAtoms]

        pos = torch.tensor(pos, dtype=torch.float32)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():
        row: List[int] = []
        row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        row.append(x_map['degree'].index(atom.GetTotalDegree()))
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        row.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    
    if with_coordinate:
        data.pos = pos
    if get_fp:
        data.fp = fp
    return data

def labeled_data(smiles, label, **args):
    """
    Args:
        smiles (str): The SMILES string.
        label: molecular property label
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        with_coordinate (bool, optional): If set to :obj:`True`, will store
            atom coordinate in the molecule graph as data.z. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
        seed:int (int, optional): provide a seed for the random number generator 
            so that the same coordinates can be obtained for a molecule on multiple runs.
            (default: -1, the RNG will not seeded)

    """
    drug_data = from_smiles(smiles, **args)
    label = torch.tensor([label], dtype=torch.float32)
    drug_data.y = label
    return drug_data



















































































