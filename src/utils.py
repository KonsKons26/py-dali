import os
import numpy as np
import pickle
import gzip
import numpy as np
import pandas as pd
import subprocess
import random
import shutil
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
    Select 'n' queries from the DataFrame, ensuring each query is from a
    distinct class, and for each query, select 'm' domains from different
    categories (different class, same class, same fold, same superfamily,
    same family).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing protein domain metadata with columns:
        - 'domain_id': Unique identifier for each protein domain.
        - 'class': Class of the protein domain.
        - 'fold': Fold of the protein domain.
        - 'superfamily': Superfamily of the protein domain.
        - 'family': Family of the protein domain.
    n : int
        Number of distinct classes to select queries from.
    m : int
        Number of domains to select for each query from different categories.
    rng : random.Random
        Random number generator instance for reproducibility.

    Returns
    -------
    dict
        A dictionary where keys are the selected query domain IDs and values
        are DataFrames containing the selected domains for each query.
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

    new_dfs = {}

    for query in queries:
        query_data = df.loc[df["domain_id"] == query].iloc[0]
        query_class = query_data["class"]
        query_fold = query_data["fold"]
        query_superfamily = query_data["superfamily"]
        query_family = query_data["family"]

        new_dfs[query] = pd.DataFrame(
            columns=["domain_id", "class", "fold", "superfamily", "family"]
        )

        # self
        new_dfs[query] = pd.concat(
            [new_dfs[query], df.loc[df["domain_id"] == query]],
            ignore_index=True
        )

        # --- m domains of different class ---
        potential_different_class = df.loc[
            (df["class"] != query_class),
            "domain_id"
        ].tolist()
        selected_different_class = rng.sample(
            potential_different_class,
            min(m, len(potential_different_class))
        )
        new_dfs[query] = pd.concat(
            [new_dfs[query], df.loc[df["domain_id"].isin(
                selected_different_class
            )]], ignore_index=True
        )

        # --- m domains of the same class ---
        potential_same_class = df.loc[
            (df["class"] == query_class) &
            (df["fold"] != query_fold) &
            (df["domain_id"] != query),
            "domain_id"
        ].tolist()
        selected_same_class = rng.sample(
            potential_same_class,
            min(m, len(potential_same_class))
        )
        new_dfs[query] = pd.concat(
            [new_dfs[query], df.loc[df["domain_id"].isin(
                selected_same_class
            )]], ignore_index=True
        )

        # --- m domains of the same fold ---
        potential_same_fold = df.loc[
            (df["class"] == query_class) &
            (df["fold"] == query_fold) &
            (df["superfamily"] != query_superfamily) &
            (df["domain_id"] != query),
            "domain_id"
        ].tolist()
        selected_same_fold = rng.sample(
            potential_same_fold,
            min(m, len(potential_same_fold))
        )
        new_dfs[query] = pd.concat(
            [new_dfs[query], df.loc[df["domain_id"].isin(
                selected_same_fold
            )]], ignore_index=True
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
        selected_same_superfamily = rng.sample(
            potential_same_superfamily,
            min(m, len(potential_same_superfamily))
        )
        new_dfs[query] = pd.concat(
            [new_dfs[query], df.loc[df["domain_id"].isin(
                selected_same_superfamily
            )]], ignore_index=True
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
        selected_same_family = rng.sample(
            potential_same_family,
            min(m, len(potential_same_family))
        )
        new_dfs[query] = pd.concat(
            [new_dfs[query], df.loc[df["domain_id"].isin(
                selected_same_family
            )]], ignore_index=True
        )

    return new_dfs


def download_and_process_domains(
        d: dict[str, pd.DataFrame], output_folder: str
    ) -> None:
    """
    Downloads and processes PDB domain files based on a dictionary of DataFrames,
    organizing them into a specific directory structure.

    For each key-value pair in the input dictionary 'd':
    1. A directory named after the key (domain_id) is created in the
        `output_folder`.
    2. The PDB file corresponding to the key's domain_id is processed and saved
        in this new directory.
    3. A 'references' subdirectory is created within the key's directory.
    4. All other PDB files listed in the DataFrame (the value of the pair) are
        processed and saved in the 'references' subdirectory.

    Parammeters
    ----------
    d : dict[str, pd.DataFrame]
        A dictionary where each key is a domain_id (string) and each value is a
        DataFrame containing PDB metadata. The DataFrame should have columns
        like 'domain_id', 'pdb_id', 'chain', and optionally 'residues'.
    output_folder : str
        The path to the main output directory where the domain-specific folders
        will be created.
    """
    # Create the main output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Base URL for PDB file downloads
    PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/"

    print(
        f"Starting domain processing. Output will be saved in '{output_folder}'"
    )

    def process_pdb(domain_id, pdb_id, chain, residues, save_path):
        """Helper function to process a single PDB entry."""
        print(
            f"\nProcessing domain: {domain_id} "
            f"(PDB: {pdb_id}, Chain: {chain})..."
        )

        # --- Construct the processing pipeline ---
        download_cmd = f"wget -qO- {PDB_DOWNLOAD_URL}{pdb_id}.pdb"
        select_chain_cmd = f"pdb_selchain -{chain}"
        select_model_cmd = "pdb_selmodel -1"
        pipeline_cmd = f"{download_cmd} | {select_chain_cmd} | {select_model_cmd}"

        if residues and pd.notna(residues):
            processed_residues = str(residues).replace("-", ":")
            print(f"  -> Selecting residues: '{processed_residues}'")
            select_residues_cmd = f"pdb_selres -'{processed_residues}'"
            pipeline_cmd += f" | {select_residues_cmd}"
        else:
            print("  -> Selecting entire chain (no residues specified).")

        pipeline_cmd += f" > {save_path}"

        # --- Execute the command ---
        try:
            process = subprocess.run(
                pipeline_cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"  -> SUCCESS: Saved to {save_path}")
        except subprocess.CalledProcessError as e:
            print(f"  -> ERROR processing {domain_id}.")
            print(f"     Command failed with exit code {e.returncode}.")
            print(f"     Stderr: {e.stderr.strip()}")
            if os.path.exists(save_path):
                os.remove(save_path)

    for key_domain_id, df in d.items():
        print(f"\nProcessing main query: {key_domain_id}...")
        
        # Create a directory for the key_domain_id
        key_specific_folder = os.path.join(output_folder, key_domain_id)
        os.makedirs(key_specific_folder, exist_ok=True)
        
        # Create a 'references' subdirectory
        references_folder = os.path.join(key_specific_folder, 'references')
        os.makedirs(references_folder, exist_ok=True)

        # Find the row corresponding to the key_domain_id to process it first
        key_row = df[df['domain_id'] == key_domain_id]
        if not key_row.empty:
            row = key_row.iloc[0]
            output_filename = os.path.join(
                key_specific_folder, f"{key_domain_id}.pdb"
            )
            process_pdb(
                row["domain_id"],
                row["pdb_id"],
                row["chain"],
                row.get("residues"),  # Use .get() for safe access
                output_filename
            )
        else:
            print(
                f"  -> WARNING: Main domain {key_domain_id} not found in its "
                "own DataFrame."
            )

        # Process the rest of the PDBs in the DataFrame and save them in the
        # 'references' folder
        print(f"\nProcessing associated references for {key_domain_id}...")
        for index, row in df.iterrows():
            # Skip the key_domain_id itself as it's already processed
            if row["domain_id"] == key_domain_id:
                continue

            output_filename = os.path.join(
                references_folder, f"{row['domain_id']}.pdb"
            )
            process_pdb(
                row["domain_id"],
                row["pdb_id"],
                row["chain"],
                row.get("residues"),
                output_filename
            )