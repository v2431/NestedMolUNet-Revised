# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:07:03 2024

@author: Fanding Xu
"""

import numpy as np
import torch
import random
import heapq
from tqdm import tqdm
from collections import Counter
from torch_geometric.utils import index_to_mask
from rdkit import Chem
from rdkit.Chem import BRICS
import math
import torch.nn.functional as F
from pathos.multiprocessing import ProcessingPool as Pool
from typing import Any, Dict, List

# copyed from torch_geometric.utils.smiles
# torch_geometric == 2.5.0
x_map: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


class EasyMask:
    def __init__(self, args):
        self.masker_w = MaskAtom(args.weighted_mask_rate, mask_edge=args.weighted_mask_edge, seed=args.seed)
        self.masker_m = MaskAtomMotif(args.motif_mask_rate, mask_edge=args.motif_mask_edge, seed=args.seed)
    
    def process(self, data):
        self.masker_w(data)
        self.masker_m(data)
        return data
    
def mask_data_list(data_list, args, num_processor=8):
    masker = EasyMask(args)
    p = Pool(num_processor)
    data_list = list(tqdm(p.imap(masker.process, data_list), total=len(data_list), desc='masking data...'))
    return data_list


# ----------------------- weighted node masking ------------------------
# adapted form https://github.com/Austin13579/weighted-masking-molecules
def a_res(samples, m):
    heap = [] # [(new_weight, item), ...]
    for sample in samples:
        wi = sample[1]
        ui = random.uniform(0, 1)
        ki = ui ** (1/wi)

        if len(heap) < m:
            heapq.heappush(heap, (ki, sample))
        elif ki > heap[0][0]:
            heapq.heappush(heap, (ki, sample))

            if len(heap) > m:
                heapq.heappop(heap)
    return [item[1][0] for item in heap]


def weight_sample(x, sample_size):
    res=[]
    mm=Counter(x[:,0].tolist())
    pro_dict={}
    for kk in mm.keys():
        pro_dict[kk]=round(np.log(0.9*(mm[kk]+1))/mm[kk], 4)
    #print(pro_dict)
    for i,atom in enumerate(x[:,0].tolist()):
        res.append([i,pro_dict[atom]])
    return a_res(res,sample_size)

# TODO(Bowen): more unittests
class MaskAtom:
    def __init__(self, mask_rate=0.2, mask_edge=True, seed=123):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.seed = seed
    
    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """
        np.random.seed(self.seed)
        
        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            #masked_atom_indices = random.sample(range(num_atoms), sample_size)
            masked_atom_indices=weight_sample(data.x,sample_size)

        # create mask node label by copying atom feature of mask atom
        data.node_mask = index_to_mask(torch.tensor(masked_atom_indices),
                                       size=data.x.size(0))

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)
                        
            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms

                connected_edge_indices = torch.tensor(connected_edge_indices[::2])
            else:
                connected_edge_indices = torch.tensor(connected_edge_indices).to(torch.int64)
                
            data.edge_mask = index_to_mask(connected_edge_indices,
                                           size=data.edge_index.size(1))
            # data.edge_mask = connected_edge_indices

        # return data

    def __repr__(self):
        return '{}(mask_rate={}, mask_edge={}, seed={})'.format(
            self.__class__.__name__,
            self.mask_rate, self.mask_edge, self.seed)



# ----------------------- weighted node masking ------------------------
# adapted form https://github.com/einae-nd/MoAMa-dev
class MaskAtomMotif:
    def __init__(self, mask_rate=0.15, inter_mask_rate=1, mask_edge=False, seed=123):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param mask_rate: % of atoms/motifs to be masked
        :param inter_mask_rate: % of atoms within motif to be masked
        :param mask_strat: node or element-wise masking
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.seed = seed
        self.inter_mask_rate = inter_mask_rate

    def __call__(self, data, mol=None, masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """
        np.random.seed(self.seed)
        if mol is None:
            mol = Chem.MolFromSmiles(data.smiles)
        if mol is None:
            mol = Chem.MolFromSmiles('')
            
        motifs = get_motifs(mol)

        num_atoms = data.x.size()[0]
        sample_size = int(num_atoms * self.mask_rate + 1)

        valid_motifs = []
        if len(motifs) != 1:
            for motif in motifs:
                for atom in mol.GetAtoms():
                    if atom.GetIdx() in motif:
                        if (inter_motif_proximity(motif, [atom], []) > 5):
                            break
                valid_motifs.append(motif)
        
        masked_atom_indices = []

        # Select motifs according to 
        while len(masked_atom_indices) < sample_size:
            if len(valid_motifs) < 1:
                index_list = random.sample(range(num_atoms), sample_size)
                for index in index_list:
                    if index not in masked_atom_indices:
                        masked_atom_indices.append(index)
            else:
                candidate = valid_motifs[random.sample(range(0, len(valid_motifs)), 1)[0]]
                valid_motifs.remove(candidate)
                for atom_idx in candidate:
                    for i, edge in enumerate(data.edge_index[0]):
                        if atom_idx == edge:
                            for motif in valid_motifs:
                                if data.edge_index[1][i].item() in motif:
                                    valid_motifs.remove(motif)
                            
                if len(masked_atom_indices) + len(candidate) > sample_size + 0.1 * num_atoms:
                    continue
                for index in candidate:
                    masked_atom_indices.append(index)

        # random masking
        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        l = math.ceil(len(masked_atom_indices) * self.inter_mask_rate)

        masked_atom_indices_atom = random.sample(masked_atom_indices, l)
        # create mask node label by copying atom feature of mask atom
        # node-wise masking

        data.node_mask_motif = index_to_mask(torch.tensor(masked_atom_indices_atom),
                                             size=data.x.size(0))

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                connected_edge_indices = torch.tensor(connected_edge_indices[::2])
            else:
                connected_edge_indices = torch.tensor(connected_edge_indices).to(torch.int64)
            
            data.edge_mask_motif = index_to_mask(connected_edge_indices,
                                                 size=data.edge_index.size(1))


    def __repr__(self):
        return '{}(mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__,
            self.mask_rate, self.mask_edge)
    
    
def inter_motif_proximity(target_motif, neighbors, checked):
    new_neighbors = []
    for atom in neighbors:
        for nei in atom.GetNeighbors():
            if nei.GetIdx() in checked:
                continue
            new_neighbors.append(nei)
            if nei.GetIdx() not in target_motif:
                return 1
        checked.append(atom.GetIdx())
    return inter_motif_proximity(target_motif, new_neighbors, checked) + 1
    
    
def brics_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)

    # select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if len(c) > 0]

    # edges
    edges = []
    for bond in res:
        for c in range(len(cliques)):
            if bond[0][0] in cliques[c]:
                c1 = c
            if bond[0][1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))

    return cliques, edges

def get_motifs(data):
    Chem.SanitizeMol(data)
    motifs, edges = brics_decomp(data)
    return motifs

def get_motifs_edges(data):

    Chem.SanitizeMol(data)
    motifs, edges = brics_decomp(data)
    return motifs, edges   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    