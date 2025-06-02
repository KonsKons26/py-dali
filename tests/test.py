import os
from src.utils import get_db_pdb_paths_and_names, get_coords, pairwise_dist
from src.reduction import reduce_distance_matrix


k = 6
contact_pattern_size = 12
contact_pattern_min_size = 2
max_contact_patterns = 1000
pdb_path = os.path.join(os.getcwd(), "queries")
pdb_files, pdb_names = get_db_pdb_paths_and_names(pdb_path)
A = get_coords(pdb_files[1], pdb_names[1])
DA = pairwise_dist(A)

reduced = reduce_distance_matrix(
    DA,
    k=k,
    start=k,
    overlap_contact_patterns=False,
    contact_pattern_max_size=contact_pattern_size,
    contact_pattern_min_size=contact_pattern_min_size,
    max_contact_patterns=max_contact_patterns
)

for cp in reduced:
    print(cp)
