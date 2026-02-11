import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from typing import Dict, Tuple, List
import numpy.typing as npt

def validate_contour(contour: np.ndarray) -> np.ndarray:
    """Validate and fix contour for OpenCV compatibility"""
    if contour is None or len(contour) < 3:
        return None
        
    try:
        # Ensure proper shape (N, 1, 2) and dtype
        if len(contour.shape) == 2:
            contour = contour.reshape(-1, 1, 2)
        elif len(contour.shape) == 3 and contour.shape[1] != 1:
            contour = contour.reshape(-1, 1, 2)
            
        # Ensure int32 dtype
        contour = contour.astype(np.int32)
        
        # Verify final shape is correct
        if len(contour.shape) != 3 or contour.shape[1] != 1 or contour.shape[2] != 2:
            return None
            
        # Check for invalid values
        if np.any(np.isnan(contour)) or np.any(np.isinf(contour)):
            return None
            
        return contour
    except:
        return None

def process_contour_chunk(chunk_data):
    """Process a chunk of contours in parallel"""
    contours, point_mapping = chunk_data
    results = []
    
    for contour in contours:
        try:
            # Create new contour with proper formatting
            points = []
            for point in contour:
                try:
                    mapped_point = point_mapping[tuple(map(float, point[0]))]
                    points.append([[mapped_point[0], mapped_point[1]]])
                except:
                    continue
                    
            if len(points) >= 3:  # Only process if we have enough points
                new_contour = np.array(points, dtype=np.int32)
                # Validate the contour
                valid_contour = validate_contour(new_contour)
                if valid_contour is not None:
                    results.append(valid_contour)
                else:
                    results.append(None)
            else:
                results.append(None)
        except:
            results.append(None)
    return results

def simplify_contours(splinters: list, distance_threshold: float = 1.0) -> None:
    """
    Highly optimized contour simplification using vectorized operations and parallel processing.
    
    Args:
        distance_threshold: Maximum distance between points to be considered the same
    """
    # Step 1: Vectorized point collection
    print("Collecting points...")
    # Pre-calculate total points for better memory allocation
    total_points = sum(len(item.contour) for item in splinters)
    all_points = np.empty((total_points, 2), dtype=np.float32)
    point_to_splinter_map = np.empty(total_points, dtype=np.int32)
    
    # Vectorized point extraction
    idx = 0
    for splinter_idx, item in enumerate(splinters):
        batch_size = len(item.contour)
        all_points[idx:idx + batch_size] = item.contour[:, 0, :]
        point_to_splinter_map[idx:idx + batch_size] = splinter_idx
        idx += batch_size

    # Step 2: Efficient KD-tree search
    print("Building KD-tree...")
    tree = cKDTree(all_points)
    
    # Step 3: Vectorized group finding
    print("Finding point groups...")
    pairs = tree.query_pairs(distance_threshold, output_type='ndarray')
    
    # Initialize disjoint set data structure using numpy arrays
    parent = np.arange(len(all_points), dtype=np.int32)
    rank = np.zeros(len(all_points), dtype=np.int32)
    
    def find(x):
        """Path compression find"""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        """Union by rank"""
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
    
    # Vectorized union operations
    if len(pairs) > 0:
        np.vectorize(union)(pairs[:, 0], pairs[:, 1])
    
    # Vectorized find operations for all points
    groups = np.vectorize(find)(np.arange(len(all_points)))
    
    # Step 4: Vectorized representative point calculation
    print("Calculating representative points...")
    unique_groups, inverse_indices = np.unique(groups, return_inverse=True)
    
    # Calculate means for each group using vectorized operations
    representatives = np.zeros((len(unique_groups), 2))
    np.add.at(representatives, inverse_indices, all_points)
    counts = np.bincount(inverse_indices)
    representatives /= counts[:, None]
    representatives = np.round(representatives).astype(np.int32)
    
    # Step 5: Create point mapping using dictionary comprehension
    point_mapping = {
        tuple(map(float, point)): tuple(rep)
        for point, rep in zip(all_points, representatives[inverse_indices])
    }
    
    # Step 6: Optimized parallel contour updating
    print("Updating contours...")
    n_cores = 10
    
    # Create optimal chunk size based on number of contours and cores
    total_contours = len(splinters)
    chunk_size = max(100, total_contours // (n_cores * 4))  # Ensure chunks aren't too small
    
    # Create chunks of contours
    chunks = []
    current_chunk = []
    current_size = 0
    
    for splinter in splinters:
        # Validate input contour before processing
        if validate_contour(splinter.contour) is not None:
            current_chunk.append(splinter.contour)
            current_size += 1
            if current_size >= chunk_size:
                chunks.append((current_chunk, point_mapping))
                current_chunk = []
                current_size = 0
    
    if current_chunk:  # Add remaining contours
        chunks.append((current_chunk, point_mapping))
    
    # Process chunks in parallel using starmap
    with mp.Pool(n_cores) as pool:
        results = []
        with tqdm(total=len(splinters)) as pbar:
            for chunk_results in pool.imap_unordered(process_contour_chunk, chunks):
                results.extend(chunk_results)
                pbar.update(len(chunk_results))
    
    # Update splinters with new contours
    valid_count = 0
    for splinter, new_contour in zip(splinters, results):
        if new_contour is not None:
            validated_contour = validate_contour(new_contour)
            if validated_contour is not None and len(validated_contour) >= 3:
                splinter.contour = validated_contour
                valid_count += 1
    
    print(f"Successfully processed {valid_count}/{len(splinters)} contours")