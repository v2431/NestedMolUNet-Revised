# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:36:56 2023

@author: Fanding Xu
"""

import torch
import os
import numpy as np
import random
import pickle
from torch_geometric.utils import degree
from rdkit.Chem.Draw import rdMolDraw2D, MolsToGridImage
from itertools import chain, combinations, product
from distinctipy import distinctipy
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem, Draw
from prettytable import PrettyTable
from rdkit_heatmaps import mapvalues2mol
from rdkit_heatmaps.utils import transform2png

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(filename, " successfully loaded")
    return obj

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

class GridSearch():
    def __init__(self, search_space):
        self.results = []
        self.grid = list(product(*search_space.values()))
        head = list(search_space.keys())
        head.append('roc')
        self.table = PrettyTable(head)
        self.table.add_rows(np.hstack([np.array(self.grid), np.empty([len(self.grid), 1], dtype=object)]))
        self.idx = 0
        
    def report(self, val):
        mean = val.mean()
        std = val.std()
        self.results.append([mean, std])
        self.table.rows[self.idx][-1] = "{:.3f} +/- {:.3f}".format(mean, std)
        print("\n=====================\nGroup {:d} over\nResults: {:.3f} +/- {:.3f}\n=====================\n\n".format(self.idx, mean, std))
        self.idx += 1
    
    def show(self):
        print(self.table)
    
    
    def conclusion(self, mode='max'):
        assert mode in ['max', 'min']
        results = np.array(self.results)[:,0]
        if mode == 'max':
            idx = results.argmax()
        else:
            idx = results.argmin()
        self.table.rows[idx][-1] = "\033[0;33;40m"+self.table.rows[idx][-1]+"\033[0m"
        print("Results Table")
        self.show()
        
        print("\nThe best hyper_para group is: ")
        print(self.table.get_string(start=idx, end=idx+1))

def concat_group_by(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    modified from
    https://github.com/rusty1s/pytorch_scatter/issues/398
    """
    index_count = torch.bincount(index)
    fill_count = index_count.max() - index_count
    fill_zeros = (torch.ones_like(x[0])*-1).repeat(fill_count.sum(), *([1]*(len(x.shape)-1)))
    fill_index = torch.arange(0, fill_count.shape[0], device=x.device).repeat_interleave(fill_count)
    index_ = torch.cat([index, fill_index], dim=0)
    x_ = torch.cat([x, fill_zeros], dim=0)
    x_ = x_[torch.argsort(index_, stable=True)].view(index_count.shape[0], index_count.max()*x.size(1), x.size(-1))
    return x_

def get_dataset_deg(load_path):
    data_list = torch.load(load_path)
    max_degree = -1
    for data in data_list:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
    
    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in data_list:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg

def get_deg_from_list(data_list):
    max_degree = -1
    
    # 第一次遍历：计算最大度并确定数据所在设备
    device = None
    for data in data_list:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
        if device is None:
            # 从d张量获取设备信息
            device = d.device
    
    # 如果没有找到设备（空列表），默认为CPU
    if device is None:
        device = torch.device('cpu')
    
    # 在正确的设备上创建度分布张量
    deg = torch.zeros(max_degree + 1, dtype=torch.long, device=device)
    
    # 第二次遍历：计算度分布
    for data in data_list:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        # 确保d和deg在同一设备上
        d = d.to(device)
        # 计算bincount并累加到deg
        bincount = torch.bincount(d, minlength=deg.numel())
        deg += bincount
    
    return deg

def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol


def generate_edge_batch(edge_index, batch):
    return batch.index_select(0, edge_index[0])

def unique_edge(edge_index, edge_attr=None):
    mask = edge_index[0] < edge_index[1]
    if edge_attr is not None:
        return edge_index[:, mask], edge_attr[mask]
    return edge_index[:, mask]

random.seed(123)
colors = distinctipy.get_colors(50, pastel_factor=0.4, colorblind_type='Deuteranomaly')

def comps_visualize_multi(mol, comps,
                          tars=None, edge_index=None,
                          size=(500, 500), label='',
                          count_in_node_edges=True, form='png',
                          only_mol_img = True,
                          sub_layer=None):
    if sub_layer is None: sub_layer = list(range(len(comps)))
    if isinstance(sub_layer, int): sub_layer = [sub_layer]
    if tars is not None:
        assert edge_index is not None, "edge_index is needed when using edge target information"
        tars = [x.T.cpu().tolist() for x in tars]
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.T.cpu().tolist()
    
    smiles_list, frgs = [], []
            
    if form == 'png':
        drawer = rdMolDraw2D.MolDraw2DCairo
    else:
        drawer = rdMolDraw2D.MolDraw2DSVG
        
    comps = [x.cpu().numpy().astype(np.int32) for x in comps]
    atoms = mol.GetNumAtoms()
    imgs = []
    s_nodes = []
    groups = []
    tar_last = []
    all_sub_bond = []
    all_sub_atom = []
    for layer, comp in enumerate(comps):
        if tars is not None:
            tar = tars[layer]
        if layer != 0:
            temp = comp
            comp = np.zeros(atoms, dtype=np.int32)
            """
            affine s small size comp (post layers) to atoms_num size
            """
            for i in range(temp.size):
                comp[groups[i]] = temp[i]
            if tars is not None:
                tar = [[groups[i], groups[j]] for i,j in tar]
                tar = [list(product(x[0], x[1])) for x in tar]
                tar = list(map(list, chain.from_iterable(tar)))
                tar = [x for x in tar if x in edge_index]
                
                if count_in_node_edges:
                    tar = tar + tar_last
                    
        value, counts = np.unique(comp, return_counts=True)
        s_nodes = value[counts > 1]
        groups = []
        groups_e = []
        for s_node in s_nodes:
            group = np.where(comp == s_node)[0].tolist()
            groups.append(group)
            group_e = []
            for atom_pair in combinations(group, 2):
                bond = mol.GetBondBetweenAtoms(atom_pair[0], atom_pair[1])
                if bond is not None:
                    if tars is not None:  
                        if list(atom_pair) in tar:
                            group_e.append(bond.GetIdx())
                    else:
                        group_e.append(bond.GetIdx())
            groups_e.append(group_e)
            
            if not group_e in all_sub_bond:
                if layer in sub_layer:
                    all_sub_bond.append(group_e)
                    all_sub_atom.append(group)
                # 创建当前子结构的mol对象
                em = Chem.EditableMol(Chem.Mol())
                atom_mapping = {}
                for atom_idx in group:
                    new_idx = em.AddAtom(mol.GetAtomWithIdx(atom_idx))
                    atom_mapping[atom_idx] = new_idx
                for bond_idx in group_e:
                    bond = mol.GetBondWithIdx(bond_idx)
                    em.AddBond(atom_mapping[bond.GetBeginAtomIdx()], atom_mapping[bond.GetEndAtomIdx()], bond.GetBondType())
                submol = em.GetMol()
                
                # 将子结构添加到frgs列表中
                frgs.append(submol)
                # 获取子结构的SMILES并添加到列表中
                smiles = Chem.MolToSmiles(submol)
                smiles_list.append(smiles)
            
        atom_cols = {}
        bond_cols = {}
        atom_list = []
        bond_list = []
        for i, (hit_atom, hit_bond) in enumerate(zip(groups, groups_e)):
            for at in hit_atom:
                atom_cols[at] = colors[i % len(colors)]
                atom_list.append(at)
            for bd in hit_bond:
                bond_cols[bd] = colors[i % len(colors)]
                bond_list.append(bd)
        d = drawer(size[0], size[1])
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atom_list,
                                            highlightAtomColors=atom_cols,
                                            highlightBonds=bond_list,
                                            highlightBondColors=bond_cols)
        d.FinishDrawing()
        img=d.GetDrawingText()
        imgs.append(img)   
        
        
        """
        pre-process for next layer: set single atoms as independent groups
        and insert in LIST: groups
        """
        # print(atom_list)
        diff = list(set(range(atoms))^set(atom_list))
        for i in diff:
            groups.insert(comp[i], [i])
        tar_last = tar[:] 
    
    if only_mol_img:
        return imgs
    
    # 使用 MolsToGridImage 绘制所有子结构，并在每个子结构下显示其SMARTS
    img = None
    if len(frgs) > 0:
        img = MolsToGridImage(frgs, molsPerRow=3, subImgSize=(200,200), legends=smiles_list, useSVG=True)
        
    
    return imgs, img, all_sub_atom, all_sub_bond


def comps_visualize_single(mol, comps,
                           tars=None, edge_index=None,
                           size=(500, 500), label='',
                           count_in_node_edges=True, form='png'):
    if tars is not None:
        assert edge_index is not None, "edge_index is needed when using edge target information"
        tars = [x.T.cpu().tolist() for x in tars]
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.T.cpu().tolist()
    
    if form == 'png':
        drawer = rdMolDraw2D.MolDraw2DCairo
    else:
        drawer = rdMolDraw2D.MolDraw2DSVG
        
    comps = [x.cpu().numpy().astype(np.int32) for x in comps]
    
    atoms = mol.GetNumAtoms()
    atom_cols = {}
    bond_cols = {}
    
    h_rads = {}
    h_lw_mult = {}
    all_sub_bond = []
    all_sub_atom = []
    s_nodes = []
    groups = []
    tar_last = []
    op_color = 0
    for layer, comp in enumerate(comps):
        if tars is not None:
            tar = tars[layer]
        if layer != 0:
            temp = comp
            comp = np.zeros(atoms, dtype=np.int32)
            """
            affine s small size comp (post layers) to atoms_num size
            """
            for i in range(temp.size):
                comp[groups[i]] = temp[i]
            if tars is not None:
                tar = [[groups[i], groups[j]] for i,j in tar]
                tar = [list(product(x[0], x[1])) for x in tar]
                tar = list(map(list, chain.from_iterable(tar)))
                tar = [x for x in tar if x in edge_index]
                
                if count_in_node_edges:
                    tar = tar + tar_last
                    
        value, counts = np.unique(comp, return_counts=True)
        s_nodes = value[counts > 1]
        groups = []
        groups_e = []
        for s_node in s_nodes:
            group = np.where(comp == s_node)[0].tolist()
            groups.append(group)
            group_e = []
            for atom_pair in combinations(group, 2):
                bond = mol.GetBondBetweenAtoms(atom_pair[0], atom_pair[1])
                if bond is not None:
                    if tars is not None:  
                        if list(atom_pair) in tar:
                            group_e.append(bond.GetIdx())
                    else:
                        group_e.append(bond.GetIdx())
            groups_e.append(group_e)
                
            
        atom_list = []
        for i, (hit_atom, hit_bond) in enumerate(zip(groups, groups_e)):
            # print(hit_bond)
            for at in hit_atom:
                atom_list.append(at)
            if hit_bond not in all_sub_bond:
                all_sub_bond.append(hit_bond)
                all_sub_atom.append(hit_atom)
                for at in hit_atom:
                    if at in atom_cols:
                        atom_cols[at].append(colors[(i + op_color) % len(colors)])
                        # h_rads[at] = 0.3
                    else:
                        atom_cols[at] = [colors[(i + op_color) % len(colors)]]
                        # h_rads[at] = 0.3
                    
                for bd in hit_bond:
                    if bd in bond_cols:
                        bond_cols[bd].append(colors[(i + op_color) % len(colors)])
                    else:
                        bond_cols[bd] = [colors[(i + op_color) % len(colors)]]
        try:
            op_color += i + 1
        except:
            print("Null extraction: block {:d}".format(layer))
        """
        pre-process for next layer: set single atoms as independent groups
        and insert in LIST: groups
        """
        # print(atom_list)
        diff = list(set(range(atoms))^set(atom_list))
        for i in diff:
            groups.insert(comp[i], [i])
        tar_last = tar[:] 
    
    
    d = drawer(size[0], size[1])
    # d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    dos = d.drawOptions()
    dos.atomHighlightsAreCircles = False
    dos.fillHighlights = False
    dos.bondLineWidth = 2
    dos.scaleBondWidth = True
    d.DrawMoleculeWithHighlights(mol, label, atom_cols, bond_cols, h_rads, h_lw_mult)
    d.FinishDrawing()
    img=d.GetDrawingText()
    return img

def visual_sep_subs(mol, all_sub_atom, all_sub_bond,
                    attribution=None, show_smiles=True):
    frgs = []
    smiles_list = []
    for group, group_e in zip(all_sub_atom, all_sub_bond):
        # 创建当前子结构的mol对象
        em = Chem.EditableMol(Chem.Mol())
        atom_mapping = {}
        for atom_idx in group:
            new_idx = em.AddAtom(mol.GetAtomWithIdx(atom_idx))
            atom_mapping[atom_idx] = new_idx
        for bond_idx in group_e:
            bond = mol.GetBondWithIdx(bond_idx)
            em.AddBond(atom_mapping[bond.GetBeginAtomIdx()], atom_mapping[bond.GetEndAtomIdx()], bond.GetBondType())
        submol = em.GetMol()
        
        # 将子结构添加到frgs列表中
        frgs.append(submol)
        # 获取子结构的SMILES并添加到列表中
        smiles = Chem.MolToSmiles(submol)
        smiles_list.append(smiles)
        
    if show_smiles:
        legends = smiles_list
    else:
        legends = [f"Substructure {i+1}" for i in range(len(smiles_list))]
    if attribution is not None:
        legends = [f"Attribution: {attr}\n"+smi for attr, smi in zip(attribution, legends)]
        
    img = None
    if len(frgs) > 0:
        img = MolsToGridImage(frgs, molsPerRow=3, subImgSize=(200,200), legends=legends, useSVG=True)
    return img, smiles_list, frgs

def ModifyConf(mol, pos, path='3Dvisual/mol.sdf'):
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    assert len(pos) == mol.GetNumAtoms(), "atom num error"
    pos = pos.astype(np.float64)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
         x,y,z = pos[i]
         conf.SetAtomPosition(i, Point3D(x,y,z))
    Chem.MolToMolFile(mol, path)
    

def GenerateConf(pos, edge_index, path='3Dvisual/mol_pool.sdf'):
    mol = Chem.RWMol()
    for i in range(pos.size(0)):
        atom = Chem.Atom(6)
        mol.AddAtom(atom)
    edge_index = unique_edge(edge_index)
    edges = [tuple(i) for i in edge_index.t().tolist()]
    for i in range(len(edges)):
            src, dst = edges[i]
            mol.AddBond(src, dst, Chem.BondType.values[0])
    ModifyConf(mol, pos, path)
    # sdf_path = path
    # pdb_path = sdf_path.replace('.sdf', '.pdb')
    # command = f"obabel -isdf {sdf_path} -opdbqt -O {pdb_path} -h"
    # os.system(command)


def search_edge(edge, edge_index):
    a, b = edge
    pos_1 = (edge_index[0]==a).logical_and(edge_index[1]==b).nonzero().item()
    pos_2 = (edge_index[0]==b).logical_and(edge_index[1]==a).nonzero().item()
    return [pos_1, pos_2]
    


def get_mol_heatmap(mol:Chem.Mol, atom_weights, bond_weights=None, **kwargs):
    mol = Draw.PrepareMolForDrawing(mol)
    canvas = mapvalues2mol(mol,
                           atom_weights=atom_weights,
                           bond_weights=bond_weights,
                           **kwargs)
    img = transform2png(canvas.GetDrawingText())
    return img


def get_3d_conformer_random(mol: Chem.Mol):
    # mol = Chem.AddHs(mol)
    pos = torch.zeros((mol.GetNumAtoms(), 3)).uniform_(-1, 1)
    return pos













































































