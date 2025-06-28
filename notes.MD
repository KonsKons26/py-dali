- Downloaded a [sequence](meta/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa) file from [SCOP](https://scop.berkeley.edu/astral/ver=2.08).
- Then run:
```bash
./scripts/parse_astral_fasta.py -i meta/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa -o meta/astral.csv > logs/parse_astral_fasta.log 2>&1
```
- Then run:
```bash
./scripts/random_sample.py -n 5 -m 5 -d meta/astral.csv -b data/db/ -s 66 > logs/random_sample.log 2>&1
```
- Then run the `create_reduced_matrices.ipynb` notebook.