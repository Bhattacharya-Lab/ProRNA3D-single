import os
import dgl
import torch as trc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
import json
'''
from opt_potential import update_potential_values
from utils.utils_data import update_relative_positions
'''
import random
def random_rotation(x):
        M = np.random.randn(3,3)
        Q, __ = np.linalg.qr(M)
        return x @ Q
class buildGraph(DGLDataset):
    
    def __init__(self):
        super().__init__(name='buildgraph')


    def process(self):
        self.data_and_label = []
        #self.label = []
        trainlist = 'inputs/inputs.list' #'trainsimple100.list'#'AlphaFold2models_train_BP.list'
        
        
        f = open(trainlist, 'r')
        flines = f.readlines()
        f.close()
        for line in flines:
            tgt = line.strip()
            #print(tgt)
            prot_tgt = tgt.split('_')[0] + tgt[-2]
            rna_tgt = tgt.split('_')[0] + tgt[-1] 
        

            prot1df2 = np.load('inputs/' + prot_tgt + '.rep_1280.npy')

            prot1df = prot1df2[1:-1,:]#discarding <start> and <end>
            
            rna1df = np.load('inputs/' + rna_tgt + '_RNA.npy')

            protlen = len(prot1df)
            rnalen = len(rna1df)

            #### Create graph features for protein ####
            nodesLeft_prot = []
            nodesRight_prot = []
            simple_edge_prot = []
            w_prot = []
            rrfile = open('prot_dist/' + prot_tgt + '_prot.dist', 'r')
            rrlines = rrfile.readlines()
            ### Sanity check: if no contact found, skip the target
            if(len(rrlines[1:]) == 0):
                print('No contact/distance found! Skipping the target ... !')
                continue
            for rline in rrlines[1:]:

                ni = int(rline.split()[0])-1
                nj = int(rline.split()[1])-1

                #sanity check
                if((ni >= protlen) or (nj >= protlen)):
                    continue
                d = float(rline.split()[4])

                #making bi-directional edge and 1 edge feature
                if(d < 14):
                    weight = np.log(abs(ni-nj))/d
                    w_prot.append([weight])
                    w_prot.append([weight])
                    nodesLeft_prot.append(ni)
                    nodesRight_prot.append(nj)
                    nodesLeft_prot.append(nj)
                    nodesRight_prot.append(ni)
            rrfile.close()

            xyz_f_prot = open('inputs/' + prot_tgt + '.pdb')

            xyz_ca = [[0,0,0] for _ in range(protlen)]
            xyz_flines = xyz_f_prot.readlines()
            for xyzline in xyz_flines:
                if(xyzline[:4] == "ATOM" and xyzline[12:16].strip() == "CA"):
                    x = float(xyzline[30:38].strip())
                    y = float(xyzline[38:46].strip())
                    z = float(xyzline[46:54].strip())

                    res_no = int(xyzline[22:(22+4)]) - 1
                    if(res_no >= len(xyz_ca)):
                        continue
                    xyz_ca[res_no] = [x, y, z]
            xyz_f_prot.close()
            xyz_ca = np.array((xyz_ca))
            xyz_ca = random_rotation(xyz_ca) #introducing random rotation to discard positional bias


            edge_prot = [nodesLeft_prot, nodesRight_prot]
            self.edge_prot = [trc.LongTensor(edge_prot[0]), trc.LongTensor(edge_prot[1])] 
            w_prot = np.array(w_prot) 
            self.edge_att_prot = trc.LongTensor(w_prot)
            xyz_feats_prot = xyz_ca.astype(np.float32)
            self.xyz_feats_prot = trc.Tensor(xyz_feats_prot)

            #### Create graph features for RNA ####
            nodesLeft_rna = []
            nodesRight_rna = []
            simple_edge_rna = []
            w_rna = []
            rrfile = open('rna_dist/' + rna_tgt + '_RNA.c4p.dist', 'r')
            rrlines = rrfile.readlines()
            ### Sanity check: if no contact found, skip the target
            if(len(rrlines[1:]) == 0):
                print('No contact/distance found! Skipping the target ... !')
                continue
            for rline in rrlines:

                ni = int(rline.split()[0])-1
                nj = int(rline.split()[1])-1

                #sanity check
                if((ni >= rnalen) or (nj >= rnalen)):
                    continue
                d = float(rline.split()[4])
                #making bi-directional edge
                if(d < 20):
                    simple_edge_rna.append([1/d])
                    simple_edge_rna.append([1/d])
                    weight = np.log(abs(ni-nj))/d
                    w_rna.append([weight])
                    w_rna.append([weight])

                    nodesLeft_rna.append(ni)
                    nodesRight_rna.append(nj)
                    nodesLeft_rna.append(nj)
                    nodesRight_rna.append(ni)
            rrfile.close()

            xyz_f_rna = open('inputs/' + rna_tgt + '.pdb')

            xyz_c4p = [[0,0,0] for _ in range(rnalen)]
            xyz_flines = xyz_f_rna.readlines()
            for xyzline in xyz_flines:
                if(xyzline[:4] == "ATOM" and xyzline[12:16].strip() == "C4'"):
                    x = float(xyzline[30:38].strip())
                    y = float(xyzline[38:46].strip())
                    z = float(xyzline[46:54].strip())

                    res_no = int(xyzline[22:(22+4)]) - 1
                    if(res_no >= len(xyz_c4p)):
                        continue
                    xyz_c4p[res_no] = [x, y, z]
            xyz_f_rna.close()
            xyz_c4p = np.array((xyz_c4p))
            #print('prev xyz for rna', xyz_c4p)
            xyz_c4p = random_rotation(xyz_c4p)
            #print('after rotation xyz for rna', xyz_c4p)
                                         
            xyz_f_rna.close()

            edge_rna = [nodesLeft_rna, nodesRight_rna]
            self.edge_rna = [trc.LongTensor(edge_rna[0]), trc.LongTensor(edge_rna[1])] 

            w_rna = np.array(w_rna)
            self.edge_att_rna = trc.LongTensor(w_rna)
            xyz_feats_rna = xyz_c4p.astype(np.float32)
            self.xyz_feats_rna = trc.Tensor(xyz_feats_rna)


            self.prot1df = trc.Tensor(prot1df)
            self.rna1df = trc.Tensor(rna1df)

            self.data_and_label.append((tgt, self.prot1df, self.rna1df, self.edge_prot, self.edge_rna, self.edge_att_prot, self.edge_att_rna, self.xyz_feats_prot, self.xyz_feats_rna))

            

    def __getitem__(self, i):
        return self.data_and_label[i]
        #return self.nodeFeats, self.xyz_feats, self.edges, self.edge_att, self.labels

    def __len__(self):
        return len(self.data_and_label)

