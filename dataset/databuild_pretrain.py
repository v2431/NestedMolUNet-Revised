# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:56:49 2023

@author: Fanding Xu
"""

import pandas as pd
import numpy as np
import torch
import os
os.environ["BABEL_DATADIR"] = "C:\\Users\\USER\\anaconda3\\envs\\compe\\share\\openbabel"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from joblib import Parallel, delayed
import argparse
from itertools import chain
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig, BRICS, FragmentCatalog, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
if __name__ == "__main__":
    from descriptors.rdNormalizedDescriptors import RDKit2DNormalized
    from descriptors.pubchemfp import GetPubChemFPs
    from databuild import from_smiles, read_pickle
    from utils import MaskAtom, MaskAtomMotif
else:
    from .descriptors.rdNormalizedDescriptors import RDKit2DNormalized
    from .descriptors.pubchemfp import GetPubChemFPs
    from .databuild import from_smiles, read_pickle
    from .utils import MaskAtom, MaskAtomMotif
    
fg_without_ca_smart = ['[N;D2]-[C;D3](=O)-[C;D1;H3]', 'C(=O)[O;D1]', 'C(=O)[O;D2]-[C;D1;H3]',
                       'C(=O)-[H]', 'C(=O)-[N;D1]', 'C(=O)-[C;D1;H3]', '[N;D2]=[C;D2]=[O;D1]',
                       '[N;D2]=[C;D2]=[S;D1]', '[N;D3](=[O;D1])[O;D1]', '[N;R0]=[O;D1]', '[N;R0]-[O;D1]',
                       '[N;R0]-[C;D1;H3]', '[N;R0]=[C;D1;H2]', '[N;D2]=[N;D2]-[C;D1;H3]', '[N;D2]=[N;D1]',
                       '[N;D2]#[N;D1]', '[C;D2]#[N;D1]', '[S;D4](=[O;D1])(=[O;D1])-[N;D1]',
                       '[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]', '[S;D4](=O)(=O)-[O;D1]',
                       '[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]', '[S;D4](=O)(=O)-[C;D1;H3]', '[S;D4](=O)(=O)-[Cl]',
                       '[S;D3](=O)-[C;D1]', '[S;D2]-[C;D1;H3]', '[S;D1]', '[S;D1]', '[#9,#17,#35,#53]',
                       '[C;D4]([C;D1])([C;D1])-[C;D1]',
                       '[C;D4](F)(F)F', '[C;D2]#[C;D1;H]', '[C;D3]1-[C;D2]-[C;D2]1', '[O;D2]-[C;D2]-[C;D1;H3]',
                       '[O;D2]-[C;D1;H3]', '[O;D1]', '[O;D1]', '[N;D1]', '[N;D1]', '[N;D1]']

fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
fparams = FragmentCatalog.FragCatParams(1, 6, fName)
fg_without_ca_list = [Chem.MolFromSmarts(smarts) for smarts in fg_without_ca_smart]
fg_with_ca_list = [fparams.GetFuncGroup(i) for i in range(39)]


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


def make_graph_data(smiles, mol=None,
                    get_desc=False, get_comp=False,
                    **args):
    """
    Args:
        smiles (str): The SMILES string.

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
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    num_at = mol.GetNumAtoms()
    drug_data = from_smiles(smiles, mol, **args)
    if get_comp:
        # fingerprint
        label_fg = return_fg_hit_atom(smiles, fg_with_ca_list, fg_without_ca_list)
        label_fg = torch.tensor(label_fg, dtype=torch.float32)
        # brics
        label_brics = torch.tensor(return_brics_leaf_structure(mol),
                                   dtype=torch.float32)
        # scaffold
        label_scaffold = np.zeros([num_at, 1], dtype=np.float32)
        core = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_index = list(mol.GetSubstructMatch(core))
        label_scaffold[scaffold_index] = 1
        label_scaffold = torch.tensor(label_scaffold, dtype=torch.float32)
        
        drug_data.label_fg = label_fg
        drug_data.label_brics = label_brics
        drug_data.label_scaffold = label_scaffold
    if get_desc:
        generator = RDKit2DNormalized()
        desc = generator.processMol(mol, smiles, internalParsing=False)
        desc = np.where(np.isnan(desc), 0, desc)
        drug_data.desc = torch.tensor([desc[1:]], dtype=torch.float32)
    return drug_data

    
def mol_filter_by_smi(smiles, labels=None):
    new_smiles, mols = [], []
    if labels is not None:
        assert len(labels) == len(smiles)
        new_labels = []
        
    for i, smi in tqdm(enumerate(smiles), total=len(smiles)):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                new_smiles.append(smi)
                if labels is not None:
                    new_labels.append(labels[i])
        except:
            continue
    if labels is not None:
        return mols, new_smiles, new_labels
    return mols, new_smiles



class easy_data:
    def __init__(self, get_desc=False, get_fp=False, get_comp=False,
                 with_hydrogen=False, with_coordinate=False,
                 seed=123):
        self.comp_labeled_data = make_graph_data
        self.get_desc = get_desc
        self.get_fp = get_fp
        self.get_comp = get_comp
        self.with_hydrogen = with_hydrogen
        self.with_coordinate = with_coordinate
        self.seed = seed
        self.rdkit_fp = False
        self.mask_attr = False

    def int_rdkit_fpgen(self, args):
        self.rdkit_fp = True
        self.fpSize = args.fpSize
        # self.fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=args.fpSize)
        
    def init_attr_masker(self, args):
        self.mask_attr = True
        self.masker_w = MaskAtom(args.weighted_mask_rate, mask_edge=args.weighted_mask_edge, seed=args.seed)
        self.masker_m = MaskAtomMotif(args.motif_mask_rate, mask_edge=args.motif_mask_edge, seed=args.seed)
        
    def process(self, smi, mol, label=None):
        data = self.comp_labeled_data(smi, mol,
                                      get_desc=self.get_desc, get_fp=self.get_fp, get_comp = self.get_comp,
                                      with_hydrogen=self.with_hydrogen, with_coordinate=self.with_coordinate, seed=self.seed)
        if self.mask_attr:
            self.masker_w(data)
            self.masker_m(data)
        if self.rdkit_fp:
            m = AllChem.RDKFingerprint(mol, fpSize=self.fpSize)
            # m = self.fpgen.GetFingerprint(mol)
            m = np.frombuffer(bytes(m.ToBitString(), 'utf-8'), 'u1') - ord('0')
            data.fpsim = torch.tensor(m, dtype=torch.float32).unsqueeze(0)
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.float32)
        return data
    
    def __repr__(self):
        info = f"easy_data(rdkit_fp={self.rdkit_fp}, fpSize={self.fpSize}, mask_attr={self.mask_attr})"
        return info


def generate_zinc_data_mp(args, path='data/pretrain/smiles.csv', num_processor=8):
    df = pd.read_csv(path)
    smiles = df['smiles'].tolist()
    print(f'smiles2mol for {path}...')
    mols, smiles = mol_filter_by_smi(smiles)

    ed = easy_data(get_desc=True, get_fp=False, get_comp=True,
                   with_hydrogen=False, with_coordinate=False, seed=123)
    ed.init_attr_masker(args)
    data_list = list(Parallel(n_jobs=num_processor)(delayed(ed.process)(smi, mol)
                                         for smi, mol in tqdm(zip(smiles, mols),
                                                              total=len(smiles),
                                                              desc='Generating data...')))
    return data_list
    
def generate_chembl_data_mp(args, path='data/pretrain/chembl.pkl', num_processor=8):
    print(f"Loading {path}")
    smiles, mols, labels = read_pickle(path)
    assert len(smiles) == len(mols) and len(smiles) == len(labels)
    print(f"Data loaded, with {len(mols)} mols.")
    
    # in case the original mol doesn't match Chem.MolFromSmiles(smiles)
    mols = [Chem.MolFromSmiles(s) for s in tqdm(smiles, desc='smiles2mol...')]

    ed = easy_data(get_desc=True, get_fp=False, get_comp=False,
                   with_hydrogen=False, with_coordinate=False, seed=123)
    ed.int_rdkit_fpgen(args)
    data_list = list(Parallel(n_jobs=num_processor)(delayed(ed.process)(smi, mol, label)
                                         for smi, mol, label in tqdm(zip(smiles, mols, labels),
                                                                     total=len(smiles),
                                                                     desc='Generating data...')))
    return data_list

def generate_pcba_data_mp(args, path='data/pretrain/pcba.csv', num_processor=8):
    df = pd.read_csv(path)
    df = df.fillna(-1)
    smiles = df['smiles'].tolist()
    labels = df[df.columns[:128]]
    labels = labels.fillna(-1)
    labels = labels.values.tolist()
    print(f'smiles2mol for {path}...')
    mols, smiles, labels = mol_filter_by_smi(smiles, labels)
    
    ed = easy_data(get_desc=True, get_fp=False, get_comp=False,
                   with_hydrogen=False, with_coordinate=False, seed=123)
    ed.int_rdkit_fpgen(args)
    data_list = list(Parallel(n_jobs=num_processor)(delayed(ed.process)(smi, mol, label)
                                         for smi, mol, label in tqdm(zip(smiles, mols, labels),
                                                                     total=len(smiles),
                                                                     desc='Generating data...')))
    return data_list
# =========================================================================================================================
# The following functions are modified from https://github.com/wzxxxx/Substructure-Mask-Explanation/tree/v1.0.0
# =========================================================================================================================
def return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i):
    # the fragment genereated from smarts would have a redundant carbon, here to remove the redundant carbon
    fg_without_c_i_wash = []
    for fg_with_c in fg_with_c_i:
        for fg_without_c in fg_without_c_i:
            if set(fg_without_c).issubset(set(fg_with_c)):
                fg_without_c_i_wash.append(list(fg_without_c))
    return fg_without_c_i_wash

def return_fg_hit_atom(item, fg_with_ca_list, fg_without_ca_list):
    if type(item) is str:
        mol = Chem.MolFromSmiles(item)
    else:
        mol = item
    num_at = mol.GetNumAtoms()
    hit_at = []
    hit_fg_label = []
    all_hit_fg_at = []
    for i in range(len(fg_with_ca_list)):
        fg_with_c_i = mol.GetSubstructMatches(fg_with_ca_list[i])
        fg_without_c_i = mol.GetSubstructMatches(fg_without_ca_list[i])
        fg_without_c_i_wash = return_fg_without_c_i_wash(fg_with_c_i, fg_without_c_i)
        if len(fg_without_c_i_wash) > 0:
            hit_at.append(fg_without_c_i_wash)
            hit_fg_label.append(i)
            all_hit_fg_at += fg_without_c_i_wash
    # sort function group atom by atom number
    sorted_all_hit_fg_at = sorted(all_hit_fg_at,
                                  key=lambda fg: len(fg),
                                  reverse=True)
    # remove small function group (wrongly matched), they are part of other big function groups
    remain_fg_list = []
    for fg in sorted_all_hit_fg_at:
        if fg not in remain_fg_list:
            if len(remain_fg_list) == 0:
                remain_fg_list.append(fg)
            else:
                i = 0
                for remain_fg in remain_fg_list:
                    if set(fg).issubset(set(remain_fg)):
                        break
                    else:
                        i += 1
                if i == len(remain_fg_list):
                    remain_fg_list.append(fg)
    # wash the hit function group atom by using the remained fg, remove the small wrongly matched fg
    hit_at_wash = []
    hit_fg_label_wash = []
    for j in range(len(hit_at)):
        hit_at_wash_j = []
        for fg in hit_at[j]:
            if fg in remain_fg_list:
                hit_at_wash_j.append(fg)
        if len(hit_at_wash_j) > 0:
            hit_at_wash.append(hit_at_wash_j)
            hit_fg_label_wash.append(hit_fg_label[j])
            
    label = np.zeros([num_at, 39], dtype=np.float32)
    for at, l in zip(hit_at_wash, hit_fg_label_wash):
        idx = list(chain(*at))
        label[idx, l] = 1
    return label


def return_brics_leaf_structure(item):
    if type(item) is str:
        m = Chem.MolFromSmiles(item)
    else:
        m = item
    res = list(BRICS.FindBRICSBonds(m))  # [((1, 2), ('1', '5'))]

    # return brics_bond
    all_brics_bond = [set(res[i][0]) for i in range(len(res))]
    
    num_bonds = m.GetNumBonds()
    bond_label = np.zeros([num_bonds * 2, 1], dtype=np.float32)

    bonds = []
    for bond in m.GetBonds():
        op = bond.GetBeginAtomIdx()
        ed = bond.GetEndAtomIdx()
        bonds.append([op, ed])
        bonds.append([ed, op])
    bonds = sorted(bonds)
    for i, bond in enumerate(bonds):
        if set(bond) in all_brics_bond:
            bond_label[i] = 1
    return bond_label

def find_murcko_link_bond(mol):
    core = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_index = mol.GetSubstructMatch(core)
    link_bond_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        link_score = 0
        if u in scaffold_index:
            link_score += 1
        if v in scaffold_index:
            link_score += 1
        if link_score == 1:
            link_bond_list.append([u, v])
    return link_bond_list

def return_murcko_leaf_structure(smiles):
    m = Chem.MolFromSmiles(smiles)

    # return murcko_link_bond
    all_murcko_bond = find_murcko_link_bond(m)

    # return atom in all_murcko_bond
    all_murcko_atom = []
    for murcko_bond in all_murcko_bond:
        all_murcko_atom = list(set(all_murcko_atom + murcko_bond))

    if len(all_murcko_atom) > 0:
        # return all break atom (the break atoms did'n appear in the same substructure)
        all_break_atom = dict()
        for murcko_atom in all_murcko_atom:
            murcko_break_atom = []
            for murcko_bond in all_murcko_bond:
                if murcko_atom in murcko_bond:
                    murcko_break_atom += list(set(murcko_bond))
            murcko_break_atom = [x for x in murcko_break_atom if x != murcko_atom]
            all_break_atom[murcko_atom] = murcko_break_atom

        substrate_idx = dict()
        used_atom = []
        for initial_atom_idx, break_atoms_idx in all_break_atom.items():
            if initial_atom_idx not in used_atom:
                neighbor_idx = [initial_atom_idx]
                substrate_idx_i = neighbor_idx
                begin_atom_idx_list = [initial_atom_idx]
                while len(neighbor_idx) != 0:
                    for idx in begin_atom_idx_list:
                        initial_atom = m.GetAtomWithIdx(idx)
                        neighbor_idx = neighbor_idx + [neighbor_atom.GetIdx() for neighbor_atom in
                                                       initial_atom.GetNeighbors()]
                        exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i
                        if idx in all_break_atom.keys():
                            exlude_idx = all_break_atom[initial_atom_idx] + substrate_idx_i + all_break_atom[idx]
                        neighbor_idx = [x for x in neighbor_idx if x not in exlude_idx]
                        substrate_idx_i += neighbor_idx
                        begin_atom_idx_list += neighbor_idx
                    begin_atom_idx_list = [x for x in begin_atom_idx_list if x not in substrate_idx_i]
                subname = Chem.rdmolfiles.MolFragmentToSmiles(m, atomsToUse=substrate_idx_i)
                substrate_idx[subname] = substrate_idx_i
                used_atom += substrate_idx_i
            else:
                pass
    else:
        substrate_idx = dict()
        substrate_idx[0] = [x for x in range(m.GetNumAtoms())]

    return substrate_idx
# =========================================================================================================================
# End
# =========================================================================================================================
#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weighted_mask_rate', type=float, default=0.2,
                        help='weighted atom masking rate (default: 0.2)')
    parser.add_argument('--weighted_mask_edge', action="store_false",
                        help='do weighted edge mask (default: True)')
    parser.add_argument('--motif_mask_rate', type=float, default=0.15,
                        help='motif atom masking rate (default: 0.15)')
    parser.add_argument('--motif_mask_edge', action="store_false",
                        help='do motif edge mask (default: True)')
    parser.add_argument('--fpSize', type=int, default=1024,
                        help='size of rdkit fingerprint for graph similarity loss {default: 1024}')
    parser.add_argument('--num_processor', type=float, default=24,
                        help='multi process num processors (default: 16)')
    parser.add_argument('--seed', type=float, default=123,
                        help='Random seed(default: 123)')
    args = parser.parse_args()
    
    
    data_list = generate_zinc_data_mp(args, path='data/pretrain/zinc2m.csv', num_processor=args.num_processor)
    torch.save(data_list, 'data/pretrain/processed/zinc2m.pth')
    data_list = generate_chembl_data_mp(args, num_processor=args.num_processor)
    torch.save(data_list, 'data/pretrain/processed/chembl.pth')

    












































