# ProRNA3D-single

## Single-sequence protein-RNA complex structure prediction by geometric triangle-aware pairing of language models

by Rahmatullah Roche, Sumit Tarafder, Bernard Moussad, and Debswapna Bhattacharya

Codebase for our protein-RNA complex structure prediction method, ProRNA3D-single.

![Workflow](./workflow.png)

## Installation

1.) We recommend conda virtual environment to install dependencies for ProRNA3D-single. The following command will create a virtual environment named 'ProRNA3D-single'

`conda env create -f ProRNA3D-single_environment.yml`

2.) Then activate the virtual environment 

`conda activate ProRNA3D-single`

3.) Download model from [here](https://zenodo.org/records/11477127), extract and place inside `ProRNA3D_model/`

That's it! ProRNA3D-single is ready to be used.

## Usage

1.) Place the protein and RNA monomers (pdbs) inside `inputs/` 

2.) Place the [esm2](https://github.com/facebookresearch/esm) embeddings inside `inputs/`. See an example here `inputs/7ZLQB.rep_1280.npy` 

3.) Place the [RNA-FM](https://github.com/ml4bio/RNA-FM) embeddings inside `inputs/`. See an example here `inputs/7ZLQC_RNA.npy`

4.) Put the list of targets in the file `inputs.list` inside `inputs/`.

5.) Run

```
python run_predictions.py
```

This script will run inference and generate inter-protein-RNA interactions inside `out_inter_rr/`. Then it will transform the predictions into folded 3D protein-RNA complex structures inside `predictions/`

## Datasets
- The train, validation, and test set lists are available inside `Datasets/`.
- All the data are curated from [PDB](https://www.rcsb.org).
