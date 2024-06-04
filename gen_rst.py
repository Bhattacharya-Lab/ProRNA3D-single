import argparse
import os, shutil, math
import pickle
import numpy as np
import sys
import random
random.seed(10)

def gen_rst_file_u(rstfile, chainprot, chainRNA, a_atom, b_atom, native_rr, udist, ldist):
    sd = 1.0# binwidth/2
    array = []
    fn = open(native_rr, 'r')
    flines = fn.readlines()

    for line in flines:
        line = line.split()
        a = int(line[0])-1
        b = int(line[1])-1
        prob = 0
        if(float(line[-1]) < 0.5):
            break
        if(float(line[2]) >= 0.5):
                lb = 2.5
                prob = float(line[2])
        elif(float(line[3]) >= 0.5):
                lb = 4
                prob = float(line[3])
        elif(float(line[4]) >= 0.5):
                lb = 6
                prob = float(line[4])
        elif(float(line[5]) >= 0.5):
                lb = 8
                prob = float(line[5])
        elif(float(line[6]) >= 0.5):
                lb = 10
                prob = float(line[6])
        elif(float(line[7]) >= 0.5):
                lb = 12 
                prob = float(line[7])
        elif(float(line[8]) >= 0.5):
                lb = 14 
                prob = float(line[8])
        elif(float(line[9]) >= 0.5):
                lb = 16 
                prob = float(line[9])
        elif(float(line[10]) >= 0.5):
                lb = 18 
                prob = float(line[10])
        elif(float(line[11]) >= 0.5):
                lb = 20
                prob = float(line[11]) 
        ub = 2
        rst_line = 'AtomPair %s %d%s %s %d%s SCALARWEIGHTEDFUNC %.2f BOUNDED %.2f %.2f %.1f %.1f %s'%(a_atom, a+1, chainprot, b_atom, b+1, chainRNA, prob, ub, lb, sd, 0.5, 'tag')
        array.append(rst_line)
    with open(rstfile,'w') as f:
        f.write('\n'.join(array)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rrfile', type=str, default = '', help="rr file to create restraints")
    parser.add_argument('--prot_chain', type=str, default = 'A', help="protein chain")
    parser.add_argument('--rna_chain', type=str, default = 'B', help="RNA chain")
    parser.add_argument('--rstfile', type=str, default = '', help="restraint file")
    parser.add_argument('--upper', type=float, default = 20, help="upper limit in contact")
    parser.add_argument('--lower', type=float, default = 2, help="lower limit in contact")

    args, _ = parser.parse_known_args()

    gen_rst_file_u(args.rstfile, args.prot_chain, args.rna_chain, 'CA', 'C4\'', args.rrfile, args.upper, args.lower)


if __name__ == '__main__':
    main()
