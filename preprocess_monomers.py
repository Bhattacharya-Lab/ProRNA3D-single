import argparse

import numpy as np
import sys,os, math
import decimal

import random
def pdb_to_coord(pdbfile): #return a set of coordinates from all atom of a given pdb file
    pos = []
    with open(pdbfile, 'r', encoding='UTF-8') as pdb:
        while (line := pdb.readline()):
            if(line[:4] == "ATOM" or line[:6] == "HETATM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                pos.append([x, y, z])
    return np.array(pos)
def reformat_xyz(x):
    x = "{:.3f}".format(x)

    d = decimal.Decimal(x)
    digit = d.as_tuple().exponent
    rem = 3 + digit
    for i in range(rem):
        x += "0"
    return x


def addchain(inputfile, outfile, chain): 
    if(chain.isnumeric()):
        number = int(chain)
        chain = chr(100+number)

    with open(inputfile, 'r', encoding='UTF-8') as pdb, open(outfile, 'w', encoding='UTF-8') as outfile:
        counter = 0
        while (line := pdb.readline()):
            if(line[:4] == "ATOM" or line[:6] == "HETATM"):
                newstring = line[:21]
                newstring += chain
                newstring += line[22:54]
                
                newstring += '{:>6}'.format("1.00")
                newstring += '{:>6}'.format("0.00")
                outfile.write(newstring + "\n")
                #outfile.write(newstring)
                counter += 1



def combine_prot_rna(newprotfile, rnafile, combinedoutfile):
    fprot = open(newprotfile)
    fprotlines = fprot.readlines()
    frna = open(rnafile)
    frnalines = frna.readlines()
    outf = open(combinedoutfile, 'w')
    for line in fprotlines:
        outf.write(line)
    for line in frnalines:
        outf.write(line)
    fprot.close()
    frna.close()
    outf.close()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prot', type=str, default = '', help="input protein monomer")
    parser.add_argument('--prot_chain', type=str, default = 'A', help="input protein chain")
    parser.add_argument('--rna', type=str, default = '', help="input RNA monomer")
    parser.add_argument('--rna_chain', type=str, default = 'B', help="input RNA chain")
    parser.add_argument('--combined', type=str, default = '', help="output combined complex")
    


    args, _ = parser.parse_known_args()
    initprot = args.prot
    initrna = args.rna
    combined = args.combined
    rechainprot = combined + 'prottmp'
    rechainrna = combined + 'rnatmp'
    addchain(initprot, rechainprot, args.prot_chain)
    addchain(initrna, rechainrna, args.rna_chain)
    combine_prot_rna(rechainprot, rechainrna, combined)



if __name__ == '__main__':
    main()




