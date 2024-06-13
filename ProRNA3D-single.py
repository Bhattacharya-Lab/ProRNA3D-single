#!/usr/bin/python

#  ProRNA3D-single
#
#  Copyright (C) Bhattacharya Laboratory 2024
#
#  ProRNA3D-single is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ProRNA3D-single is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with ProRNA3D-single.  If not, see <http://www.gnu.org/licenses/>.
#
############################################################################

from ProRNA3D_net import *
import argparse
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from dgl.dataloading import GraphDataLoader
import operator
import dgl
import math
import numpy as np
import torch
from dataloader import buildGraph
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

def to_np(x):
    return x.cpu().detach().numpy()

def test_epoch(epoch, model, dataloader, PARS):
    model.eval()

    rloss = 0
    for i, (data_feats) in enumerate(dataloader):
        (tgtname, nodeFeats_prot, nodeFeats_rna, edges_prot, edges_rna, edge_att_prot, edge_att_rna, xyz_prot, xyz_rna) = data_feats
        print(tgtname[0])
        name = tgtname[0]
        #print(edges[0].shape)
        n_nodes_prot = len(nodeFeats_prot[0])
        n_nodes_rna = len(nodeFeats_rna[0])
        n_e_prot = len(edges_prot[0])
        n_e_rna = len(edges_rna[0])
        nodeFeats_prot = nodeFeats_prot.to(PARS.device)
        nodeFeats_rna = nodeFeats_rna.to(PARS.device)
        #print('node feat', nodeFeats_rna.shape)
        xyz_prot = xyz_prot.to(PARS.device)
        xyz_rna = xyz_rna.to(PARS.device)
        edges_prot[0] = edges_prot[0].to(PARS.device)
        edges_prot[1] = edges_prot[1].to(PARS.device)
        edges_rna[0] = edges_rna[0].to(PARS.device)
        edges_rna[1] = edges_rna[1].to(PARS.device)
        edge_att_prot = edge_att_prot.to(PARS.device)
        edge_att_rna = edge_att_rna.to(PARS.device)


        nodeFeats_prot = nodeFeats_prot.squeeze()
        nodeFeats_rna = nodeFeats_rna.squeeze()
        xyz_prot = xyz_prot.squeeze()
        xyz_rna = xyz_rna.squeeze()
        edges_prot[0] = edges_prot[0].squeeze()
        edges_prot[1] = edges_prot[1].squeeze()

        edges_rna[0] = edges_rna[0].squeeze()
        edges_rna[1] = edges_rna[1].squeeze()

        edge_att_prot = edge_att_prot.squeeze()
        edge_att_prot = edge_att_prot.unsqueeze(dim=1)

        edge_att_rna = edge_att_rna.squeeze()
        edge_att_rna = edge_att_rna.unsqueeze(dim=1)

        pred = model(nodeFeats_prot, xyz_prot, edges_prot, edge_att_prot, nodeFeats_rna, xyz_rna, edges_rna, edge_att_rna)

        pred = torch.nn.Softmax(1)(pred)
        #pred = pred[:, :, :prot_feat2d.shape[-1]:, prot_feat2d.shape[-1]:]
        pred = pred.detach().numpy()
        #print(pred.shape)
        frr = open(PARS.outdir + '/' + str(name) + '.10bins.out', 'w')
        rrList = []
        for protres in range(pred.shape[-2]):
            for rnares in range(pred.shape[-1]):
                #print(pred[0, : , protres, rnares])
                rrList.append([str(protres+1), str(rnares+1), sum(pred[0, 0:1,protres,rnares]), sum(pred[0, :4, protres, rnares]), sum(pred[0, :8, protres, rnares]), sum(pred[0, :12, protres, rnares]), sum(pred[0, :16, protres, rnares]), sum(pred[0, :20, protres, rnares]), sum(pred[0, :24, protres, rnares]), sum(pred[0, :28, protres, rnares]), sum(pred[0, :32, protres, rnares]), sum(pred[0, :-1, protres, rnares])])
                #fo.write(str(protres+1) + ' ' +  str(rnares+1) +  ' ' + str(pred[0,:,protres,rnares][-1]) + '\n')

        dict1 = {}
        for x in rrList:
	        dict1[(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10])] = x[11]
        sorted_rr = sorted(dict1.items(), key=operator.itemgetter(1))
        sorted_rr.reverse()
        count = 0
        for sr in sorted_rr:
	        (i, j, p1, p2, p3, p4, p5, p6, p7, p8, p9), p10 = sr[0], sr[1]

	        frr.write(str(i))
	        frr.write(' ')
	        frr.write(str(j))
	        frr.write(' ' + str(p1) + ' ' + str(p2) + ' ' + str(p3) + ' ' + str(p4) + ' ' + str(p5) + ' ' + str(p6) + ' ' + str(p7) + ' ' + str(p8) + ' ' + str(p9) + ' ' + str(p10))
	        frr.write('\n')
	        count += 1
        frr.close()

        print('done!')


def print_usage():
    print("\nUsage: ProRNA3D-single [options]\n")

    print("Options:")
    print("  -h, --help            show this help message and exit")
    print("  --model_state_dict MODEL_STATE_DICT")
    print("                        Saved model")
    print("  --indir INDIR         Path to input data containing distance maps and input features (default 'datasets/DNA_test_129_Preprocessing_AlphaFold2/')")
    print("  --outdir OUTDIR       Prediction output directory")
    print("  --num_workers NUM_WORKERS")
    print("                        Number of data loader workers")


def main(PARS):
    dataset = buildGraph()
    inference_loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    # Get Model
    model = EGNN_TriangularAttnNet(prot=1280,prot_egnn_hidden=128, prot_e=1, prot_egnn_layers=4, rna=640, rna_egnn_hidden=128, rna_e=1, rna_egnn_layers=4, mid_c=256, out_c=37, device=PARS.device, attention=True)

    if not PARS.model_state_dict:
        print("PARS.model_state_dict file must be set")
    model.load_state_dict(torch.load(PARS.model_state_dict))
    model.to(PARS.device)

    test_epoch(0, model, inference_loader, PARS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_state_dict', type=str, default=None,
            help="Saved model")
    parser.add_argument('--outdir', type=str, default='test/',
            help="Prediction output directory")
    parser.add_argument('--num_workers', type=int, default=4,
            help="Number of data loader workers")
    PARS, _ = parser.parse_known_args()

    #basic input check
    if not PARS.model_state_dict: 
        print('Error! Trained model must be provided. Exiting ...')
        print_usage()
        sys.exit()
    if (PARS.outdir == ''):
        print('Error! Path to the output directory must be provided. Exiting ...')
        print_usage()
        sys.exit()

    #existance check
    if not os.path.exists(PARS.model_state_dict):
        print('Error! No such trained model exists. Exiting ...')   
        print_usage()
        sys.exit()
    if not os.path.exists(PARS.outdir):
        print('Error! No such output directory exists. Exiting ...')
        print_usage()
        sys.exit()

    #header

    print("\n********************************************************************************")
    print("*                            ProRNA3D-single                                   *")
    print("*           Single-sequence protein-RNA complex structure prediction           *")
    print("*            by geometric triangle-aware pairing of language models            *")
    print("*             For comments, please email to dbhattacharya@vt.edu               *")
    print("********************************************************************************\n")

    print('Inter protein-RNA hybrid interaction predictions for each target is being saved at ' + PARS.outdir + '/\n')

    seed = 1992 
    torch.manual_seed(seed)
    np.random.seed(seed)
    PARS.device = torch.device('cpu') #cuda can be used if that is available
    main(PARS)
