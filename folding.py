#credit: pyrosetta tutorials: https://rosettacommons.github.io/PyRosetta.notebooks/
import argparse
import os, shutil, math, time

from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.core.pose.rna import *
from pyrosetta.rosetta.core.scoring import ScoreFunction, ScoreType
from pyrosetta.rosetta.protocols.constraint_movers import ConstraintSetMover
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.pose import *


def apply_rst(pose, rst_file):

	constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
	constraints.constraint_file(rst_file)
	constraints.add_constraints(True)
	constraints.apply(pose)


def run_min(cstfile, pose, mover1, mover2=None):
	apply_rst(pose, cstfile)
	mover1.apply(pose)
	if mover2 is not None:
		mover2.apply(pose)
def fold_Comp(start_pdb, rst_file, refined_pdb):#, method, rst_weight=10, max_iter=500, ):
    #print(start_pdb, refined_pdb)
    rst_weight=10
    max_iter=500
    # init PyRosetta
    init('-hb_cen_soft -constant_seed -relax:default_repeats 5 -default_max_cycles 200 -out:level 100')

    # initialize pose
    pose = pose_from_pdb(start_pdb)
    # full-atom score_function
    sf_fa = create_score_function('ref2015')
    sf_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, rst_weight)
    switch = SwitchResidueTypeSetMover("fa_standard")
    switch.apply(pose)
    # MoveMap
    mmap = MoveMap()
    mmap.set_bb(False)
    mmap.set_chi(False)
    mmap.set_jump(True)
    # FastRelax
    print('fast relax run')
    relax = rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf_fa)
    relax.max_iter(max_iter)
    relax.dualspace(True)
    relax.set_movemap(mmap)
    # appy the rst to pose
    apply_rst(pose, rst_file)#, 1, len(seq), PCUT, refined_pdb, seq, nogly=False)
    # relax
    print('folding the complex ...')
    relax.apply(pose)
    refined_energy = sf_fa(pose)
    # save model
    pose.dump_pdb(refined_pdb)
    print(refined_pdb, 'done complex.')
    return start_pdb, refined_pdb, refined_energy


def folding(rst_file, start_pdb_filename, refined_pdb_filename):

		
    start_pdb, refined_pdb, refined_energy = fold_Comp(start_pdb_filename, rst_file, refined_pdb_filename)
	

def run_batch_folding():
    parser = argparse.ArgumentParser(description='Folding')
    parser.add_argument('--rst_file', type=str, default='',
            help='restraint file')
    parser.add_argument('--start_pdb', type=str, default='',
            help='starting (complex) pdb file')
    parser.add_argument('--refined_pdb', type=str, default='',
            help='final pdb')
    args = parser.parse_args()
    folding(args.rst_file, args.start_pdb, args.refined_pdb)


if __name__ == '__main__':
	
	run_batch_folding()
	

