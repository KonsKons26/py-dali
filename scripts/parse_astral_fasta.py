#!/usr/bin/env python3
"""
Script to parse ASTRAL SCOP FASTA file and extract classification info to CSV
"""

import csv
import re
from pathlib import Path

def parse_astral_fasta(fasta_file, output_csv):
    """
    Parse ASTRAL FASTA file and extract SCOP classification info
    
    Example header:
    >d1a0pa_ a.1.1.1 (A:) Hemoglobin, alpha chain {Human (Homo sapiens)}
    """
    
    results = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Parse the header line
                header = line.strip()[1:]  # Remove '>'
                
                # Split by spaces to get components
                parts = header.split()
                
                if len(parts) < 3:
                    print(f"Warning: Skipping malformed header: {header}")
                    continue
                
                # Extract domain ID (e.g., d1a0pa_)
                domain_id = parts[0]
                
                # Extract SCOP classification (e.g., a.1.1.1)
                scop_class = parts[1]
                
                # Extract chain info (e.g., (A:))
                chain_match = re.search(r'\(([A-Z0-9]+):', header)
                chain = chain_match.group(1) if chain_match else ''
                
                # Extract PDB ID from domain ID (first 4 chars after 'd')
                pdb_match = re.match(r'd([a-z0-9]{4})', domain_id)
                if not pdb_match:
                    print(f"Warning: Could not extract PDB ID from {domain_id}")
                    continue
                
                pdb_id = pdb_match.group(1).upper()
                
                # Parse SCOP classification
                scop_parts = scop_class.split('.')
                if len(scop_parts) != 4:
                    print(f"Warning: Unexpected SCOP format: {scop_class}")
                    continue
                
                class_id = scop_parts[0]
                fold_id = scop_parts[1]
                superfamily_id = scop_parts[2]
                family_id = scop_parts[3]
                
                results.append({
                    'Class': class_id,
                    'Fold': fold_id,
                    'Superfamily': superfamily_id,
                    'Family': family_id,
                    'pdb_id': pdb_id,
                    'chain': chain,
                    'domain_id': domain_id,
                    'full_scop': scop_class
                })
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Class', 'Fold', 'Superfamily', 'Family', 'pdb_id', 'chain', 'domain_id', 'full_scop']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"Processed {len(results)} entries")
    print(f"Output written to: {output_csv}")
    
    # Print some statistics
    unique_pdbs = len(set(row['pdb_id'] for row in results))
    unique_folds = len(set(row['full_scop'] for row in results))
    print(f"Unique PDB IDs: {unique_pdbs}")
    print(f"Unique SCOP classifications: {unique_folds}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python parse_astral_fasta.py <input_fasta> <output_csv>")
        print("Example: python parse_astral_fasta.py astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa scop_data.csv")
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    output_csv = sys.argv[2]
    
    if not Path(fasta_file).exists():
        print(f"Error: Input file {fasta_file} not found")
        sys.exit(1)
    
    parse_astral_fasta(fasta_file, output_csv)
