import os
import numpy as np
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
