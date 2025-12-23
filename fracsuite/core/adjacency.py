import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import multiprocessing as mp
from typing import Dict, Set, List, Tuple

def process_splinter_chunk(chunk_data):
    """Process a chunk of splinters to find adjacencies"""
    splinters, all_points_tree, point_to_splinter_map, distance_threshold = chunk_data
    results = {}
    
    for splinter in splinters:
        try:
            # Get points for current splinter
            contour_points = splinter.contour[:, 0, :]
            
            # Query KD-tree for nearby points
            nearby_indices = all_points_tree.query_ball_point(
                contour_points, 
                distance_threshold
            )
            
            # Flatten and get unique splinter IDs
            nearby_splinters = set()
            for indices in nearby_indices:
                for idx in indices:
                    nearby_id = point_to_splinter_map[idx]
                    if nearby_id != splinter.ID:
                        nearby_splinters.add(nearby_id)
            
            results[splinter.ID] = len(nearby_splinters)
            
        except Exception as e:
            print(f"Error processing splinter {splinter.ID}: {str(e)}")
            results[splinter.ID] = 0
            
    return results

def find_adjacent_splinters(specimen, distance_threshold: float = 1.0) -> Dict[int, int]:
    """
    Find the number of adjacent splinters for each splinter.
    
    Args:
        distance_threshold: Maximum distance between points to consider splinters adjacent
        
    Returns:
        Dictionary mapping splinter ID to number of adjacent splinters
    """
    print("Building spatial index...")
    
    # Step 1: Collect all points and build mapping
    all_points = []
    point_to_splinter_map = []
    
    # Pre-allocate based on total points
    total_points = sum(len(splinter.contour) for splinter in specimen.splinters)
    all_points = np.empty((total_points, 2), dtype=np.float32)
    point_to_splinter_map = np.empty(total_points, dtype=np.int32)
    
    # Fill arrays
    idx = 0
    for splinter in specimen.splinters:
        points = splinter.contour[:, 0, :]
        n_points = len(points)
        all_points[idx:idx + n_points] = points
        point_to_splinter_map[idx:idx + n_points] = splinter.ID
        idx += n_points
    
    # Build KD-tree for efficient spatial queries
    print("Building KD-tree...")
    tree = cKDTree(all_points)
    
    # Prepare chunks for parallel processing
    n_cores = 10
    splinters = np.array(specimen.splinters)
    chunks = np.array_split(splinters, n_cores)
    
    # Prepare chunk data
    chunk_data = [
        (chunk, tree, point_to_splinter_map, distance_threshold)
        for chunk in chunks if len(chunk) > 0
    ]
    
    # Process chunks in parallel
    print("Finding adjacent splinters...")
    with mp.Pool(n_cores) as pool:
        results = {}
        with tqdm(total=len(specimen.splinters)) as pbar:
            async_results = [
                pool.apply_async(
                    process_splinter_chunk,
                    args=(chunk,),
                    callback=lambda x: pbar.update(len(x))
                )
                for chunk in chunk_data
            ]
            
            # Collect results
            for async_result in async_results:
                chunk_results = async_result.get()
                results.update(chunk_results)
    
    # Analyze results
    total_edges = sum(results.values())
    avg_edges = total_edges / len(results) if results else 0
    max_edges = max(results.values()) if results else 0
    min_edges = min(results.values()) if results else 0
    
    print(f"\nAdjacency Analysis:")
    print(f"Total connections: {total_edges}")
    print(f"Average edges per splinter: {avg_edges:.2f}")
    print(f"Max edges: {max_edges}")
    print(f"Min edges: {min_edges}")
    
    return [x for _,x in results.items()]