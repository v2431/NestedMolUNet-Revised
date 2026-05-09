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
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, f1_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from scipy.stats import spearmanr
# from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, Image, SVG

import sys
sys.path.append('../')
from dataset.databuild import from_smiles
from utils import mol_with_atom_index, comps_visualize_multi
def aver_list(x):
    return sum(x) / len(x)

class PropertyTrainer():
    def __init__(self, args, model, device, pretrained=False):
        self.args = args
        if args.patience == 0:
            args.epochs = args.min_epochs
        self.model = model.to(device)
        self.device = device
        if args.task == 'reg':
            self.loss_function = torch.nn.MSELoss() 
            self.val_all = self._val_reg
            self.scheduler_mode = 'min'
            if args.monitor in ['spearman', 'r2']: self.scheduler_mode = 'max'
        else:
            self.loss_function = torch.nn.BCEWithLogitsLoss()
            self.val_all = self._val_cls
            self.scheduler_mode = 'max'
        
        self.pretrained = pretrained
        if pretrained:
            optimizer_grouped_parameters = [
                {'params': model.unet.parameters(),
                  'weight_decay': 0.0, 'lr': args.lr * args.lr_multiplier},
                {'params': [p for n, p in model.named_parameters() if not n.startswith('unet')],
                  'weight_decay': 0.0, 'lr': args.lr}
                ]
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters)     
        
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
            
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        print(self.optimizer)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=self.scheduler_mode, factor=args.lr_reduce_rate,
                                                                    patience=args.lr_reduce_patience, min_lr=1e-8)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=15)
            
    def load_buffer(self, load_path):
        assert os.path.exists(load_path), "load_path does not exist"
        self.model.load_state_dict(torch.load(load_path), strict=False)
        print(f"********** Model Loaded {load_path} **********")
    
        
    
    def __call__(self, loader_tr, loader_va, loader_te=None,
                 save_path='checkpoint/property/temp.pt',
                 log_root = 'log/temp',
                 load_path=None, tensorboard=False):
        
        if tensorboard:
            tb = SummaryWriter(log_dir='log/tensorboard')
            note = defaultdict(list)
        
        self.run_time = time.strftime(f"{self.args.dataset}-%Y%m%d-%H%M%S", time.localtime())
        if not os.path.exists(log_root): os.makedirs(log_root)
        self.log_path = os.path.join(log_root, f"{self.run_time}.txt")
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
                _, pred = self.model(data)
                y = data.y
                if self.args.task == 'cls':
                    mask = y >= 0
                    pred = pred[mask]
                    y = y[mask]
                loss = self.loss_function(pred, y)
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
                for key, value in info_dict.items():
                    tb.add_scalars(key, {'val': value}, epoch)
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
            # state = loss_va
            state = info_dict[self.args.monitor]
            self.scheduler.step(state)
            # self.scheduler.step()
            
            if epoch > self.args.min_epochs:
                if self.scheduler_mode == 'min':
                    judge = (best - state) > 1e-6
                else:
                    judge = (state - best) > 1e-6
                    
                # if best is None or (best - loss_va) > 1e-6:
                if judge:
                    best = state
                    best_epoch = epoch
                    torch.save(self.model.state_dict(), save_path)
                    times = 0
                    print("New best model saved, epoch {:d}, best metric: {:.4f}".format(best_epoch, best))
                    
                    # self.check_pool()
                else:
                    times += 1
                    print("Not improved for {:d}/{:d} times, current best is epoch {:d}: {:.4f}".format(times, self.args.patience, best_epoch, best))
                if times >= self.args.patience:
                    break
                
            else:
                # self.check_pool()
                best = state
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_path)
                times = 0
                print("Model saved, epoch {:d}, current metric: {:.4f}".format(epoch, state))
                
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
        return None
        
    
    def _val_cls(self, loader):
        self.model.eval()
        y_true = []
        y_scores = []
        loss_val = 0
        for data in tqdm(loader, desc="validating..."):
            data = data.to(self.device)
            with torch.no_grad():
                _, pred = self.model(data)
                y = data.y
                mask = y >= 0
                loss = self.loss_function(pred[mask], y[mask])
                loss_val += loss.item()
                y_true.append(y)
                y_scores.append(pred.sigmoid())
                
        loss_val /= len(loader)
        
        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
        roc_list = []
        prc_list = []
        f1_list = []
        ap_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_valid = y_true[:,i] >= 0
                cur_roc = roc_auc_score(y_true[is_valid,i], y_scores[is_valid,i])
                pred = (y_scores[is_valid,i] >= 0.5).astype(int)
                precision, recall, thresholds = precision_recall_curve(y_true[is_valid,i], y_scores[is_valid,i])
                cur_prc = auc(recall, precision)
                cur_f1 = f1_score(y_true[is_valid,i], pred)
                cur_ap = average_precision_score(y_true[is_valid,i], pred)
                roc_list.append(cur_roc)
                prc_list.append(cur_prc)
                f1_list.append(cur_f1)
                ap_list.append(cur_ap)
                
        info_dict = {'epoch':-1,
                     'loss':loss_val}
        info_dict['roc'] = aver_list(roc_list)
        info_dict['prc'] = aver_list(prc_list)
        info_dict['f1'] = aver_list(f1_list)
        info_dict['ap'] = aver_list(ap_list)
        
        return info_dict
    
    def _val_reg(self, loader):
        self.model.eval()
        y_true = []
        y_scores = []
        loss_val = 0
        for data in tqdm(loader, desc="validating..."):
            data = data.to(self.device)
            with torch.no_grad():
                _, pred = self.model(data)
                y = data.y
                loss = self.loss_function(pred, y)
                loss_val += loss.item()
                y_true.append(y)
                y_scores.append(pred)
                
        loss_val /= len(loader)
        
        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
        
        rmse_list = []
        # mse_list = []
        r2_list = []
        mae_list = []
        spearman_list = []
        
        for i in range(y_true.shape[1]):
            rmse_list.append(root_mean_squared_error(y_true[:,i], y_scores[:,i]))
            r2_list.append(r2_score(y_true[:,i], y_scores[:,i]))
            mae_list.append(mean_absolute_error(y_true[:,i], y_scores[:,i]))
            spearman_list.append(spearmanr(y_true[:,i], y_scores[:,i]).statistic)
            
        info_dict = {'epoch':-1,
                     'loss':loss_val}
        info_dict['rmse'] = aver_list(rmse_list)
        info_dict['mae'] = aver_list(mae_list)
        info_dict['r2'] = aver_list(r2_list)
        info_dict['spearman'] = aver_list(spearman_list)
        return info_dict
    
    @torch.no_grad()
    def predict(self, loader):
        preds = []
        for data in tqdm(loader, desc="predicting..."):
            data = data.to(self.device)
            _, pred = self.model(data)
            preds.append(pred)
        preds = torch.cat(preds, dim=0).cpu().numpy()
        return preds
    
    def print_info(self, info_dict, desc="Validation info: "):
        info = ""
        for key, value in info_dict.items():
            if key=='epoch':
                info += f'{key}: {value}\t'
            elif type(value) is dict:
                info += f'{key}: '
                for k, v in value.items():
                        info += f'{k}: {v:.4f}  '
                info += '\t'
            else:
                info += f'{key}: {value:.4f}\t'
        print(desc, end='') 
        print(info)
        return info

    def log_info(self, info_dict, desc="Validation info: ", test=False):
        info = self.print_info(info_dict, desc)
        info += "\n"
        f = open(self.log_path, "a")
        if test:
            f.write('======== Test set result ========\n')
        f.write(info)
        f.close()
        
    def check_pool(self):
        # smiles = 'COC(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)[C@@H](O)CN(Cc1ccc(-c2ccccn2)cc1)NC(=O)[C@@H](NC(=O)OC)C(C)(C)C)C(C)(C)C'
        smiles = 'C1=CC=C(C=C1)C2=CC(=O)C3=C(C(=C(C=C3O2)O[C@H]4[C@@H]([C@H]([C@@H]([C@H](O4)C(=O)O)O)O)O)O)O'
        data = from_smiles(smiles, 
                               get_fp=True, 
                               with_hydrogen=False, with_coordinate=True,
                               seed=123, use_OB=True)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        data = data.to(self.device)
        edge_index = data.edge_index
        self.model.eval()
        with torch.no_grad():
            _ = self.model(data)
           
        pool_info = self.model.unet.pool_info
        comps = pool_info['comp']
        tars = pool_info['edge_target']
        mol = mol_with_atom_index(Chem.MolFromSmiles(smiles))
        imgs = comps_visualize_multi(mol, comps, tars, edge_index)
        for png in imgs:
            display(Image(png))


    def __repr__(self):
        info = f"device: {self.device}; pretrained: {self.pretrained}"
        return info
