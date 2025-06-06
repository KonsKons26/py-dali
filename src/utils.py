import os
import numpy as np
import pickle
import gzip
import numpy as np
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
            pdb_names.append(filename[:-4])

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


