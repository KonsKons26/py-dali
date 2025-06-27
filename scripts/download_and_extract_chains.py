#!/usr/bin/env python3
"""
Script to download PDB files and extract specific chains based on CSV data
"""

import csv
import os
import requests
import time
from pathlib import Path
from collections import defaultdict

def download_pdb(pdb_id, output_dir="pdb_files"):
    """Download PDB file from RCSB"""
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
    output_path = Path(output_dir) / f"{pdb_id.lower()}.pdb"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'w') as f:
            f.write(response.text)
        
        return True
    except requests.RequestException as e:
        print(f"Error downloading {pdb_id}: {e}")
        return False

def extract_chain_from_pdb(pdb_file, chain_id, output_file):
    """Extract specific chain from PDB file"""
    
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    chain_lines = []
    
    for line in lines:
        # Keep HEADER, TITLE, COMPND, SOURCE, REMARK lines
        if line.startswith(('HEADER', 'TITLE', 'COMPND', 'SOURCE', 'REMARK')):
            chain_lines.append(line)
        # Keep ATOM/HETATM lines for the specific chain
        elif line.startswith(('ATOM', 'HETATM')):
            if len(line) > 21 and line[21] == chain_id:
                chain_lines.append(line)
        # Keep connectivity info for the chain
        elif line.startswith('CONECT'):
            chain_lines.append(line)
        # Keep END record
        elif line.startswith('END'):
            chain_lines.append(line)
    
    # Write extracted chain
    with open(output_file, 'w') as f:
        f.writelines(chain_lines)
    
    return len([l for l in chain_lines if l.startswith('ATOM')])

def process_csv_and_download(csv_file, output_dir="extracted_chains", download_dir="temp_pdb"):
    """Process CSV file and download/extract chains"""
    
    # Create directories
    Path(output_dir).mkdir(exist_ok=True)
    Path(download_dir).mkdir(exist_ok=True)
    
    # Read CSV and group by PDB ID
    pdb_chains = defaultdict(list)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row['pdb_id'].upper()
            chain = row['chain']
            domain_id = row['domain_id']
            pdb_chains[pdb_id].append((chain, domain_id, row))
    
    print(f"Found {len(pdb_chains)} unique PDB IDs to process")
    
    downloaded = 0
    extracted = 0
    failed = 0
    
    for i, (pdb_id, chain_info) in enumerate(pdb_chains.items(), 1):
        print(f"Processing {i}/{len(pdb_chains)}: {pdb_id}")
        
        # Download PDB file
        pdb_file = Path(download_dir) / f"{pdb_id.lower()}.pdb"
        
        if not pdb_file.exists():
            if not download_pdb(pdb_id, download_dir):
                failed += 1
                continue
            downloaded += 1
            # Small delay to be nice to PDB servers
            time.sleep(0.5)
        
        # Extract each chain for this PDB
        for chain_id, domain_id, row_data in chain_info:
            if not chain_id:  # Skip if no chain specified
                continue
                
            # Create output filename: domain_id.pdb
            output_file = Path(output_dir) / f"{domain_id}.pdb"
            
            try:
                atom_count = extract_chain_from_pdb(pdb_file, chain_id, output_file)
                if atom_count > 0:
                    extracted += 1
                    print(f"  Extracted chain {chain_id} -> {domain_id}.pdb ({atom_count} atoms)")
                else:
                    print(f"  Warning: No atoms found for chain {chain_id} in {pdb_id}")
                    failed += 1
            except Exception as e:
                print(f"  Error extracting chain {chain_id} from {pdb_id}: {e}")
                failed += 1
    
    print(f"\nSummary:")
    print(f"PDB files downloaded: {downloaded}")
    print(f"Chains extracted: {extracted}")
    print(f"Failed extractions: {failed}")
    print(f"Output directory: {output_dir}")
    
    # Optionally clean up downloaded PDB files
    cleanup = input("\nDelete temporary PDB files? (y/n): ").lower().strip()
    if cleanup == 'y':
        import shutil
        shutil.rmtree(download_dir)
        print("Temporary files cleaned up")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python download_and_extract_chains.py <csv_file> [output_dir] [temp_dir]")
        print("Example: python download_and_extract_chains.py scop_data.csv extracted_chains temp_pdb")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "extracted_chains"
    temp_dir = sys.argv[3] if len(sys.argv) > 3 else "temp_pdb"
    
    if not Path(csv_file).exists():
        print(f"Error: CSV file {csv_file} not found")
        sys.exit(1)
    
    process_csv_and_download(csv_file, output_dir, temp_dir)
