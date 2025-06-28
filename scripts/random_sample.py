#!/usr/bin/env python3
import pandas as pd
import random
import os
from src.utils import select_domains, download_and_process_domains, move_query_files


help_text = """Select random samples from the astral dataset and download their
domains. This script allows you to specify the number of queries and
samples per query, and it will randomly select domains from the provided
DataFrame. After downloading, query files can be moved to a separate folder.

Usage:
    python random_sample.py -n <num_queries> -m <num_samples> -d <dataframe> -b <db_path> -s <seed> [-q <query_folder>]

Arguments:
    -n, --num_queries: Number of queries to select from the dataset (default: 5).
    -m, --num_samples: Number of samples to select for each query (default: 10).
    -d, --dataframe: Path to the input DataFrame CSV file containing domain information.
    -b, --db_path: Path to the directory where the downloaded domains will be saved.
    -s, --seed: Random seed for reproducibility (default: 42).
    -q, --query_folder: Path to the folder where query files should be moved after downloading (optional).
"""

def main(n, m, df_path, db_path, seed, query_folder=None):
    os.makedirs(db_path, exist_ok=True)
    os.makedirs(query_folder, exist_ok=True) if query_folder else None
    rng = random.Random(seed)
    df = pd.read_csv(df_path)
    queries, to_download = select_domains(df, n, m, rng)
    to_download = queries + to_download
    to_download_df = df.loc[
        df["domain_id"].isin(to_download),
        [
            "domain_id", "pdb_id", "chain", "residues"
        ]
    ]
    print("\n---Downloading---\n")
    download_and_process_domains(to_download_df, db_path)
    if query_folder:
        print("\n---Moving Query Files---\n")
        move_query_files(queries, db_path, query_folder)
    print("\n---Done---\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=help_text)
    parser.add_argument(
        "-n", "--num_queries",
        type=int,
        default=5,
        required=False,
        help="Number of queries to select from the dataset."
    )
    parser.add_argument(
        "-m", "--num_samples",
        type=int,
        default=10,
        required=False,
        help="Number of samples to select for each query."
    )
    parser.add_argument(
        "-d", "--dataframe",
        type=str,
        required=True,
        help="Path to the input DataFrame CSV file."
    )
    parser.add_argument(
        "-b", "--db_path",
        type=str,
        required=True,
        help="Path to the directory where the downloaded domains will be saved."
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        required=False,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "-q", "--query_folder",
        type=str,
        required=False,
        help="Path to the folder where query files should be moved after downloading."
    )
    args = parser.parse_args()
    n = args.num_queries
    m = args.num_samples
    df_path = args.dataframe
    db_path = args.db_path
    seed = args.seed
    query_folder = args.query_folder
    main(n, m, df_path, db_path, seed, query_folder)
