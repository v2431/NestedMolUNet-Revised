# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:18:37 2023

@author: Fanding Xu
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, mean_squared_error, mean_absolute_error, f1_score, average_precision_score
# from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, Image, SVG

import sys
sys.path.append('../')
from dataset.databuild_pretrain import make_graph_data
from utils import mol_with_atom_index, comps_visualize_multi

def aver_list(x):
    return sum(x) / len(x)

class PretrainTrainer():
    def __init__(self, args, model, device):
        self.args = args
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=args.lr_reduce_rate,
                                                                    patience=args.lr_reduce_patience, min_lr=1e-8)

            
    def load_buffer(self, load_path):
        assert os.path.exists(load_path), "load_path does not exist"
        self.model.load_state_dict(torch.load(load_path), strict=False)
        print(f"********** Model Loaded {load_path} **********")
    
    
    def __call__(self, loader_tr, loader_va, loader_te=None,
                 save_path='checkpoint/pretrain/temp.pt',
                 load_path=None, tensorboard=False,
                 log_path = None):
        dirname = os.path.dirname(save_path)
        basename = os.path.basename(save_path)
        form = basename[basename.index('.'):]
        basename = basename[:basename.index('.')]
        seperate_dir = os.path.join(dirname, basename)
        if not os.path.exists(seperate_dir):
            os.makedirs(seperate_dir)
        if tensorboard:
            tb = SummaryWriter(log_dir='log/tensorboard')
            note = defaultdict(list)
        
        
        self.run_time = time.strftime("RUN-%Y%m%d-%H%M%S", time.localtime())
        if log_path is None:
            self.log_path = f"log/{self.run_time}.txt"
        else:
            self.log_path = log_path
        # save_path = save_path + 'temp.pt'
        
        with open(self.log_path, "w") as f:
            f.write(self.args.config+'\n')
        if load_path is not None:
            self.load_buffer(load_path)
        
        best = None
        for epoch in range(1, self.args.epochs + 1):
            tic = time.time()
            print("Epoch: {:d}/{:d}".format(epoch, self.args.epochs), end='') 
            self.model.train()
            epoch_loss = 0   
            for data in tqdm(loader_tr, desc="Epoch: {:d}/{:d}".format(epoch, self.args.epochs)):
                data = data.to(self.device)
                _, loss = self.model(data)
                # print(_)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                torch.cuda.empty_cache()
                            
            loss_tr = epoch_loss / (len(loader_tr))
            plr = self.optimizer.param_groups[0]['lr']
            print(f'lr：{round(plr, 7)}')
            print("Training loss = {:.4f}".format(loss_tr))
            
            info_dict = self.val_all(loader_va)
            info_dict['epoch'] = epoch
            #************************* tensorboard ****************************
            if tensorboard:
                # for key, value in info_dict.items():
                #     tb.add_scalars(key, {'val': value}, epoch)
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        note[name+'_param'].append(param.mean())
                        note[name+'_grad'].append(param.grad.mean())
                        tb.add_histogram(name + '_param', param, epoch)
                        tb.add_histogram(name + '_grad', param.grad, epoch)
                    else:
                        print(name)
            #******************************************************************
            
            
            self.log_info(info_dict)
            loss_va = info_dict['loss']
            
            self.scheduler.step(loss_va)
            
            epoch_path = os.path.join(seperate_dir, f"{basename}_{epoch}{form}")
            
            torch.save(self.model.state_dict(), epoch_path)
            if epoch > self.args.min_epochs:
                judge = (best - loss_va) > 1e-4
                if judge:
                    best = loss_va
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), save_path)
                    times = 0
                    print("New best model saved, epoch {:d}, best validation loss: {:.4f}".format(best_epoch, best))
                    self.check_pool()
                else:
                    times += 1
                    print("Not improved for {:d}/{:d} times, current best is epoch {:d}: {:.4f}".format(times, self.args.patience, best_epoch, best))
                if times >= self.args.patience:
                    break
                
            else:
                best = loss_va
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_path)
                times = 0
                print("Model saved, epoch {:d}, current validation loss: {:.4f}".format(epoch, loss_va))
                self.check_pool()
                
            toc = time.time()
            print("Time costs: {:.3f}\n".format(toc-tic))
               
        self.load_buffer(save_path)
        # torch.save(self.model.unet.state_dict(), 'checkpoint/unet.pt')
        if tensorboard:
            tb.close()
        if loader_te is not None:
            info_dict = self.val_all(loader_te)
            self.log_info(info_dict, desc="Test info: ", test=True)
            
            
            return info_dict
    
    @torch.no_grad()
    def val_all(self, loader):
        self.model.eval()
        loss_val = 0
        losses = []
        for data in tqdm(loader, desc="validating..."):
            data = data.to(self.device)
            separate_loss, loss = self.model(data)
            loss_val += loss.item()
            losses.append(separate_loss)
            torch.cuda.empty_cache()
            
        losses = np.array(losses).mean(axis=0).tolist()
        loss_val /= len(loader)
        info_dict = {'epoch':-1,
                     'loss':loss_val}
        info_dict['separate_loss'] = losses   
        return info_dict
    
    

    def print_info(self, info_dict, desc="Validation info: "):
        info = ""
        for key, value in info_dict.items():
            if key=='epoch':
                info += f'{key}: {value}\t' 
            elif isinstance(value, dict):
                info += f'{key}: '
                for k, v in value.items():
                    info += f'{k}: {v:.4f}  '
                info += '\t'
            elif isinstance(value, torch.Tensor):
                info += f'{key}: {value}\t'
            elif isinstance(value, list):
                value = [round(x, 4) for x in value]
                info += f'{key}: {value}\t'
            else:
                info += f'{key}: {value:.4f}\t'
        print(desc, end='') 
        print(info)
        return info

    def log_info(self, info_dict, desc="Validation info: ", test=False):
        info = self.print_info(info_dict, desc)
        info += "\n"
        f = open(self.log_path,"a")
        if test:
            f.write('======== Test set result ========\n')
        f.write(info)
        f.close()
        
    def check_pool(self):
        # smiles = 'COC(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)[C@@H](O)CN(Cc1ccc(-c2ccccn2)cc1)NC(=O)[C@@H](NC(=O)OC)C(C)(C)C)C(C)(C)C'
        smiles = 'C1=CC=C(C=C1)C2=CC(=O)C3=C(C(=C(C=C3O2)O[C@H]4[C@@H]([C@H]([C@@H]([C@H](O4)C(=O)O)O)O)O)O)O'
        data = make_graph_data(smiles, 
                               get_desc=False, get_fp=True, get_comp=False, 
                               with_hydrogen=False, with_coordinate=True,
                               seed=123, use_OB=True)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        data = data.to(self.device)
        edge_index = data.edge_index
        self.model.eval()
        with torch.no_grad():
            _ = self.model.unet(data)
           
        pool_info = self.model.unet.pool_info
        comps = pool_info['comp']
        tars = pool_info['edge_target']
        mol = mol_with_atom_index(Chem.MolFromSmiles(smiles))
        imgs = comps_visualize_multi(mol, comps, tars, edge_index)
        for png in imgs:
            display(Image(png))
