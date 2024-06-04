import os, sys
os.system('python ProRNA3D-single.py --model_state_dict ProRNA3D_model/model.pt --outdir out_inter_rr/')
print('All interactions generated! Now preprocessing monomers and generating restraints ...')
f = open('inputs/inputs.list')
flines = f.readlines()
for line in flines:
    tgt = line.strip()
    tgtid = tgt.split('_')[0]
    prot_chain = tgt[-2]
    rna_chain = tgt[-1]
    os.system('python preprocess_monomers.py --prot inputs/' + tgtid + prot_chain + '.pdb --rna inputs/' + tgtid + rna_chain + '.pdb --combined inputs/tmp/' + tgt + '_init.pdb --prot_chain ' + prot_chain + ' --rna_chain  ' + rna_chain)

    os.system('python gen_rst.py --rrfile out_inter_rr/' + tgt + '.10bins.out --prot_chain ' + prot_chain + ' --rna_chain ' + rna_chain + ' --rstfile inputs/tmp/' + tgt + '.rst')
    print('Done! Starting folding ...')
    os.system('python folding.py --rst_file inputs/tmp/' + tgt + '.rst --start_pdb inputs/tmp/' + tgtid + '_' + prot_chain + rna_chain + '_init.pdb --refined_pdb predictions/' + tgt + '.pdb')
f.close()
