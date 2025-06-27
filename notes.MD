- Downloaded a [sequence](meta/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa) file from [SCOP](https://scop.berkeley.edu/astral/ver=2.08).
- Ran: 
    ```bash
    ./scripts/parse_astral_fasta.py meta/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa meta/scop_data.csv > logs/parse_astral_fasta.log 2>&1
    ```
    and then:
    ```bash
    ./scripts/download_and_extract_chains.py meta/scop_data.csv data/db tmp > logs/download_and_extract_chains.log 2>&1
    ```
    and finally for clean-up:
    ```bash
    rm -rf tmp/
    ```