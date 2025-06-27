#!/usr/bin/env python3
import os
import pandas as pd


help_text = """Parse ASTRAL FASTA files and save the parsed data to a CSV file.

Usage:
    python parse_astral_fasta.py -i <input_file> -o <output_file>
    -i, --input      Path to the input FASTA file.
    -o, --output     Path to the output CSV file for the parsed data.

Example:
    python parse_astral_fasta.py -i astral_sequences.fa -o parsed_astral.csv
"""


def parse_astral_fasta(sequence_file):
    parsed = []
    counter = 0
    with open(sequence_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                # Remove ">"
                header = line.strip()[1:]

                # domain_id
                domain_id = header.split(" ")[0]
                header = header.replace(domain_id, "").strip()

                # scop_code
                scop_code = header.split(" (")[0]
                header = header.replace(scop_code, "").strip()
                class_, fold, superfamily, family = scop_code.split(".")

                # pdb_id
                pdb_id = domain_id[1:5]

                # chain, residues
                chain_residues = header.split(") ")[0].replace("(", "")
                header = header.replace(f"({chain_residues})", "").strip()
                chain_residues_split = chain_residues.split(":")
                if len(chain_residues_split) == 2:
                    chain, residues = chain_residues_split
                else:
                    chain, residues = chain_residues_split[0], None

                # protein_name
                protein_name = header.split(" {")[0]
                header = header.replace(protein_name, "").strip()

                # organism
                organism = header.replace("}", "").replace("{", "")

                parsed.append([
                    domain_id,
                    scop_code,
                    class_,
                    fold,
                    superfamily,
                    family,
                    pdb_id,
                    chain,
                    residues,
                    protein_name,
                    organism,
                ])
                counter += 1

    print(f"Parsed {counter} entries from {sequence_file}")

    return parsed


def save_parsed_data(parsed, output_file):
    try:
        field_names = [
            "domain_id", "scop_code", "class", "fold", "superfamily", "family",
            "pdb_id", "chain", "residues", "protein_name", "organism"
        ]
        df = pd.DataFrame(parsed, columns=field_names)
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error saving parsed data: {e}")
        return False
    return True


def main(input_file, output_file):
    parsed = parse_astral_fasta(input_file)
    success = save_parsed_data(parsed, output_file)
    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=help_text)
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input FASTA file."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to the output .csv for the parsed data."
    )
    args = parser.parse_args()
    input = args.input
    output = args.output
    success = main(input, output)
    if success:
        print(f"Parsed data saved to {output}")
    else:
        print("Failed to parse data.")