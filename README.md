# Final project for the 'Algorithms in Molecular Biology' Course

This is my attempt at replicating the [Dali algorithm](https://en.wikipedia.org/wiki/Structural_alignment#DALI) for the final project of the course.

## How to replicate

First I downloaded the SCOPe (Structural Classification Of Proteins, extended) data from SCOP-Astral, [here](https://scop.berkeley.edu/astral/ver=2.08). As of today, the latest version is __2.08__. You don't have to download it again, it lives in the [meta/](meta/) directory. To process it and extract the relevant data, run the `scripts/parse_astral_fasta.py` script.

Then run the `random_sample.ipynb` notebook which will download a random subset of structures to perform the analysis. SCOPe classifies proteins in Classes, Folds, Superfamilies, and Families. Structures in the same Family are closer than structures in the same Superfamily and different Family, etc. The notebook will select `n` random structures as the "queries". For each of query::

- `m` structures that are in the same family
- `m` structures that are in the same superfamily but different family
- `m` structures that are in the same fold but different superfamily (and consicuently different family)
- `m` structures that are in the same class but different fold (and superfamily, etc)
- `m` structures that are in different class.

So, for each query we will have several structures "references" of varying degree of similarity. The function `select_domain` performs that selection and the function `download_and_process_domains` downloads them and processes them (select atom, residues, etc from the pdb file).

The structures will be organized as such:

```text
data
├── query1
|   ├── references
|   |   ├── ref1.pdb
⋮    ⋮   ⋮
|   |   └── refx.pdb
|   └── query1.pdb
├── query2
|   ├── references
|   |   ├── ref1.pdb
⋮    ⋮   ⋮
|   |   └── refx.pdb
|   └── query2.pdb
⋮
├── queryz
|   ├── references
|   |   ├── ref1.pdb
⋮    ⋮   ⋮
|   |   └── refx.pdb
|   └── queryz.pdb
⋮
```

Then run the `create_reduced_matrices.ipynb` notebook which will reduce the 3D protein structures into lists of Contact Patterns and store them as `pkl.gz` files in `data_reduced/`.

Then run the `monte_carlo.ipynb` notebook to perform the alignment.

Finally run the `result_analysis.ipynb` notebook to analyze the results.

Also check out the `tests/d3jbra3_showcase.ipynb` notebook.
