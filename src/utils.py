import os
import numpy as np
import pickle
import gzip
import numpy as np
import pandas as pd
import subprocess
import random
import time
from typing import List, Tuple
from Bio.PDB import PDBParser
from scipy.spatial.distance import pdist, squareform



def get_db_pdb_paths_and_names(pdb_dir: str) -> tuple[list[str], list[str]]:
    """
    Get the paths and names of PDB files in a directory.

    This function scans a specified directory for PDB files and returns
    their full paths and names (without the .pdb extension).

    Parameters
    ----------
    pdb_dir : str
        The path to the directory containing PDB files.

    Returns
    -------
    pdb_paths : list[str]
        A list of paths to the PDB files.
    pdb_names : list[str]
        A list of names of the PDB files (without the .pdb extension).
    """

    pdb_paths = []
    pdb_names = []

    for filename in os.listdir(pdb_dir):
        if filename.endswith(".pdb"):
            pdb_paths.append(os.path.join(pdb_dir, filename))
            pdb_names.append(filename.split(".")[0])

    return pdb_paths, pdb_names


def get_coords(pdb_file: str, pid: str, Calpha_only: bool = True) -> np.ndarray:
    """
    Get the 3D coordinates of a PDB file.

    This function parses a PDB file and extracts the coordinates of the atoms.
    If `Calpha_only` is set to True, it returns only the coordinates of the
    alpha carbons (CA) of the residues. The coordinates are returned as a
    NumPy array.

    Parameters
    ----------
    pdb_file : str
        The path to the PDB file.
    pid : str
        The PDB ID of the protein.
    Calpha_only : bool = True
        Whether to return only the coordinates of the alpha carbons.

    Returns
    -------
    coords : np.ndarray
        The coordinates of the atoms in the PDB file.
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pid, pdb_file)

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if Calpha_only and atom.get_id() != "CA":
                        continue
                    coords.append(atom.get_coord())

    coords = np.array(coords)

    return coords


def pairwise_dist(
        arr: np.ndarray,
        distance_type: str = "euclidean",
        get_squareform: bool = True,
        **kwargs
    ) -> np.ndarray:
    """
    Calculate the pairwise distance of coordinates in an array.

    This function computes the pairwise distance between the coordinates in the
    input array using the specified distance metric. The result can be returned
    as a condensed distance matrix or a square form distance matrix.

    Parameters
    ----------
    arr : np.ndarray
        The array of coordinates.
    distance_type : str = "euclidean"
        The type of distance to calculate.
    **kwargs
        Additional keyword arguments to pass to the pdist() function.

    Returns
    -------
    dist : np.ndarray
        The pairwise distance matrix. If `get_squareform` is True, it returns
        a square form distance matrix; otherwise, it returns a condensed
        distance matrix.

    Raises
    ------
    ValueError
        If the `distance_type` is not a valid distance metric.
    """

    valid_distance_types = [
        "braycurtis", "canberra", "chebyshev", "cityblock", "correlation',",
        "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon",
        "kulczynski1", "mahalanobis", "matching", "minkowski", "rogerstanimoto",
        "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
        "sqeuclidean", "yule"
    ]

    if distance_type not in valid_distance_types:
        raise ValueError(
            f"Invalid distance type. Choose from: {valid_distance_types}"
        )

    if get_squareform:
        return squareform(pdist(arr, distance_type, **kwargs))
    else:
        return pdist(arr, distance_type, **kwargs)


def save_data(
        data: List[Tuple],
        filename: str,
        compression_level: int = 6,
        optimize_dtypes: bool = True,
        verbose: bool = False
    ) -> dict:
    """
    Save data using compressed pickle format.

    Parameters
    ----------
    data: List
        List of tuples containing data to save.
    filename: str
        Output filename (will add .pkl.gz if not present)
    compression_level: int
        Compression level 1-9 (6 is good balance of speed/size)
    optimize_dtypes: bool
        Whether to optimize data types for smaller file size
    verbose: bool
        Whether to print detailed information about the save operation

    Returns
    -------
    dict
        Statistics about the save operation
    """

    # Ensure filename has correct extension
    if not filename.endswith('.pkl.gz'):
        filename = filename + '.pkl.gz'

    start_time = time.time()

    # Optionally optimize data types
    if optimize_dtypes:
        data_to_save = _optimize_data_types(data)
    else:
        data_to_save = data

    # Save with compression
    with gzip.open(filename, 'wb', compresslevel=compression_level) as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_time = time.time() - start_time

    # Get file info
    file_size = os.path.getsize(filename)

    stats = {
        'filename': filename,
        'file_size_mb': file_size / (1024 * 1024),
        'save_time': save_time,
        'num_items': len(data),
        'compression_level': compression_level,
        'optimized': optimize_dtypes
    }

    if verbose:
        print(f"Saved {len(data)} items to {filename}")
        print(f"File size: {stats['file_size_mb']:.2f} MB")
        print(f"Save time: {save_time:.3f} seconds")

    return stats


def load_data(filename: str, verbose: bool = False) -> List[Tuple]:
    """
    Load data from compressed pickle format.

    Parameters
    ----------
    filename: str
        Input filename (will add .pkl.gz if not present)
    verbose: bool
        Whether to print detailed information about the load operation.

    Returns
    -------
    List[Tuple]
        The loaded data

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    pickle.UnpicklingError
        If file is corrupted
    """

    # Ensure filename has correct extension
    if not filename.endswith('.pkl.gz'):
        filename = filename + '.pkl.gz'

    start_time = time.time()

    try:
        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f)

        load_time = time.time() - start_time

        if verbose:
            print(f"Loaded {len(data)} items from {filename}")
            print(f"Load time: {load_time:.3f} seconds")

        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found")
    except (pickle.UnpicklingError, gzip.BadGzipFile) as e:
        raise pickle.UnpicklingError(f"Error loading '{filename}': {e}")


def _optimize_data_types(data: List[Tuple]) -> List[Tuple]:
    """
    Optimize data types to reduce file size while preserving data integrity.

    Parameters
    ----------
    data: List[Tuple]
        Original data list

    Returns
    -------
    List[Tuple]
        Data with optimized types
    """
    optimized_data = []

    for item in data:
        id_val, coords, value, matrix = item

        # Optimize ID - use smallest int type that fits
        if isinstance(id_val, (int, np.integer)):
            if -128 <= id_val <= 127:
                id_optimized = np.int8(id_val)
            elif -32768 <= id_val <= 32767:
                id_optimized = np.int16(id_val)
            elif -2147483648 <= id_val <= 2147483647:
                id_optimized = np.int32(id_val)
            else:
                id_optimized = id_val
        else:
            id_optimized = id_val

        # Convert coordinates to numpy array with appropriate dtype
        if isinstance(coords, list):
            coords_array = np.array(coords, dtype=np.int32)
            # Try smaller type if values fit
            coords_flat = np.array(coords).flatten()
            if np.all(coords_flat >= -32768) and np.all(coords_flat <= 32767):
                coords_array = coords_array.astype(np.int16)
        else:
            coords_array = coords

        # Optimize float precision if loss is acceptable
        if isinstance(value, (float, np.floating)):
            # Use float32 if the value can be represented accurately
            float32_val = np.float32(value)
            if abs(float32_val - value) / abs(value) < 1e-6:  # Less than 0.0001% error
                value_optimized = float32_val
            else:
                value_optimized = value
        else:
            value_optimized = value

        # Optimize matrix dtype if possible
        if isinstance(matrix, np.ndarray):
            if matrix.dtype == np.float64:
                # Try float32 if precision loss is minimal
                matrix_f32 = matrix.astype(np.float32)
                if np.allclose(matrix, matrix_f32, rtol=1e-6):
                    matrix_optimized = matrix_f32
                else:
                    matrix_optimized = matrix
            else:
                matrix_optimized = matrix
        else:
            matrix_optimized = matrix

        optimized_data.append((
            id_optimized,
            coords_array,
            value_optimized,
            matrix_optimized
        ))

    return optimized_data


# Convenience functions for quick save/load
def quick_save(data: List[Tuple], filename: str):
    """Quick save with default settings."""
    return save_data(data, filename)


def quick_load(filename: str) -> List[Tuple]:
    """Quick load with default settings."""
    return load_data(filename)


def select_domains(
        df: pd.DataFrame, n: int, m: int, rng: random.Random
    ) -> dict:
    """
    Selects `N` queries from the DataFrame and retrieves `m` domains
    for each query based on the specified criteria.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing protein domain information with columns:
        - 'domain_id': Unique identifier for each protein domain.
        - 'class': Class of the protein domain.
        - 'fold': Fold of the protein domain.
        - 'superfamily': Superfamily of the protein domain.
        - 'family': Family of the protein domain.
    n : int
        Number of distinct classes to select queries from.
    m : int
        Number of domains to select for each query based on the criteria.
    rng : random.Random
        Random number generator for reproducibility.

    Returns
    -------
    dict
        A dictionary where keys are the selected query domain IDs and values
        are dictionaries containing the query information and lists of selected
        domains based on the specified criteria:
        - 'query_info': Information about the query domain (class, fold,
          superfamily, family).
        - 'different_class': List of `m` domains from different classes.
        - 'same_class_different_fold': List of `m` domains from the same class
          but different folds.
        - 'same_fold_different_superfamily': List of `m` domains from the same
          fold but different superfamilies.
        - 'same_superfamily_different_family': List of `m` domains from the same
          superfamily but different families.
        - 'same_family': List of `m` domains from the same family.
    """
    all_classes = df["class"].unique()

    if len(all_classes) < n:
        print(
            f"Warning: Not enough unique classes ({len(all_classes)}) to "
            f"select {n} distinct classes. Adjusting N to {len(all_classes)}."
        )
        n = len(all_classes)

    # Random selection of 'N' queries from different classes
    selected_classes = rng.sample(list(all_classes), n)
    queries = []
    for clss in selected_classes:
        # Get all proteins in the class
        proteins_in_class = df[df["class"] == clss]["domain_id"].tolist()
        if proteins_in_class:
            # Randomly select one protein as the query
            query = rng.choice(proteins_in_class)
            queries.append(query)
        else:
            print(
                f"Warning: No proteins found for class '{clss}'. "
                "Skipping selection for this class."
            )

    results = {}

    for query in queries:
        query_data = df.loc[df["domain_id"] == query].iloc[0]
        query_class = query_data["class"]
        query_fold = query_data["fold"]
        query_superfamily = query_data["superfamily"]
        query_family = query_data["family"]

        results[query] = {
            "query_info": {
                "class": query_class,
                "fold": query_fold,
                "superfamily": query_superfamily,
                "family": query_family
            },
            "different_class": [],
            "same_class_different_fold": [],
            "same_fold_different_superfamily": [],
            "same_superfamily_different_family": [],
            "same_family": []
        }

        # --- m domains of different class ---
        potential_different_class = df.loc[
            (df["class"] != query_class),
            "domain_id"
        ].tolist()
        results[query]["different_class"] = rng.sample(
            potential_different_class,
            min(m, len(potential_different_class))
        )

        # --- m domains of the same class ---
        potential_same_class = df.loc[
            (df["class"] == query_class) &
            (df["fold"] != query_fold) &
            (df["domain_id"] != query),
            "domain_id"
        ].tolist()
        results[query]["same_class_different_fold"] = rng.sample(
            potential_same_class,
            min(m, len(potential_same_class))
        )

        # --- m domains of the same fold ---
        potential_same_fold = df.loc[
            (df["class"] == query_class) &
            (df["fold"] == query_fold) &
            (df["superfamily"] != query_superfamily) &
            (df["domain_id"] != query),
            "domain_id"
        ].tolist()
        results[query]["same_fold_different_superfamily"] = rng.sample(
            potential_same_fold,
            min(m, len(potential_same_fold))
        )

        # --- m domains of the same superfamily ---
        potential_same_superfamily = df.loc[
            (df["class"] == query_class) &
            (df["fold"] == query_fold) &
            (df["superfamily"] == query_superfamily) &
            (df["family"] != query_family) &
            (df["domain_id"] != query),
            "domain_id"
        ].tolist()
        results[query]["same_superfamily_different_family"] = rng.sample(
            potential_same_superfamily,
            min(m, len(potential_same_superfamily))
        )

        # --- m domains of the same family ---
        potential_same_family = df.loc[
            (df["class"] == query_class) &
            (df["fold"] == query_fold) &
            (df["superfamily"] == query_superfamily) &
            (df["family"] == query_family) &
            (df["domain_id"] != query),
            "domain_id"
        ].tolist()
        results[query]["same_family"] = rng.sample(
            potential_same_family,
            min(m, len(potential_same_family))
        )

    to_download = []

    for query, data in results.items():
        to_download.extend(data[
            "different_class"] + data[
                "same_class_different_fold"] +  data[
                    "same_fold_different_superfamily"] +  data[
                        "same_superfamily_different_family"] + data[
                            "same_family"])

        print(
            f"\nQuery: {query} (Class: {data['query_info']['class']}, "
            f"Fold: {data['query_info']['fold']}, Superfamily: "
            f"{data['query_info']['superfamily']}, "
            f"Family: {data['query_info']['family']})"
        )
        print(
            f"  - Different class: {data['different_class']}"
        )
        print(
            "  - Same class, different fold: "
            f"{data['same_class_different_fold']}"
        )
        print(
            f"  - Same fold, different superfamily: "
            f"{data['same_fold_different_superfamily']}"
        )
        print(
            f"  - Same superfamily, different family: "
            f"{data['same_superfamily_different_family']}"
        )
        print(
            f"  - Same family: {data['same_family']}"
        )

    return queries, to_download


def download_and_process_domains(df: pd.DataFrame, output_folder: str):
    """
    Downloads PDB files, selects specific chains and residues, and saves them
    as new PDB files named after their domain_id.

    This function relies on the command-line tools 'wget', 'pdb_selchain', and
    'pdb_selres'. Ensure these are installed and accessible in your system's PATH.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the following columns:
        - 'domain_id': Unique identifier for each protein domain.
        - 'pdb_id': PDB ID of the protein.
        - 'chain': Chain identifier to select from the PDB file.
        - 'residues': Optional string specifying residues to select
        (e.g., "1-10, 20-30").
    output_folder : str
        Directory where the processed PDB files will be saved. If it does not
        exist, it will be created.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Base URL for PDB file downloads
    PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/"

    print(
        f"Starting domain processing. Output will be saved in '{output_folder}'"
    )

    # Iterate over each row of the DataFrame
    for index, row in df.iterrows():
        domain_id = row["domain_id"]
        pdb_id = row["pdb_id"]
        chain = row["chain"]
        # pd.isna() correctly handles None, np.nan, etc.
        residues = row["residues"] if pd.notna(row["residues"]) and row["residues"] else None

        output_filename = os.path.join(output_folder, f"{domain_id}.pdb")
        
        print(
            f"\nProcessing domain: {domain_id} "
            f"(PDB: {pdb_id}, Chain: {chain})..."
        )

        # --- Construct the processing pipeline ---
        
        # 1. Download the PDB file to standard output
        #    -q is for quiet mode, -O- sends the file to stdout
        download_cmd = f"wget -qO- {PDB_DOWNLOAD_URL}{pdb_id}.pdb"
        
        # 2. Select the specified chain
        select_chain_cmd = f"pdb_selchain -{chain}"
        
        # 3. Build the full command pipeline
        pipeline_cmd = f"{download_cmd} | {select_chain_cmd}"
        
        # 4. Add residue selection if 'residues' are specified
        if residues:
            residues = residues.split("-")
            residues = ":".join(residues)  # join with ':' for pdb_selres
            print(f"  -> Selecting residues: '{residues}'")
            # Enclose residues in quotes to handle complex selections safely
            select_residues_cmd = f"pdb_selres -'{residues}'"
            pipeline_cmd += f" | {select_residues_cmd}"
        else:
            print("  -> Selecting entire chain (no residues specified).")
            
        # 5. Redirect the final output to the destination file
        pipeline_cmd += f" > {output_filename}"
        
        # --- Execute the command ---
        try:
            # Using shell=True to interpret the pipeline correctly
            process = subprocess.run(
                pipeline_cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"  -> SUCCESS: Saved to {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"  -> ERROR processing {domain_id}.")
            print(f"     Command failed with exit code {e.returncode}.")
            print(f"     Stderr: {e.stderr.strip()}")
            # Clean up the empty or partially created file on error
            if os.path.exists(output_filename):
                os.remove(output_filename)