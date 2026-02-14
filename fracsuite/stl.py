# sub app to analyze stl files
# TAKEN FROM MASTERTHESIS MARINA BACHMANN, TU DARMSTADT
"""STL fragment analysis module for fracture-suite.

This module provides functions to analyze STL files containing scanned glass fragments,
computing properties like volume, surface area, fracture surface roughness, etc.
"""
from __future__ import annotations

import gc
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyvista as pv
import typer
from scipy.spatial import ConvexHull, KDTree

from fracsuite.callbacks import main_callback

stl_app = typer.Typer(help=__doc__, callback=main_callback)

SLICE_COUNT = 40
THEORETICAL_PERIMETER_SLICES = 5  # Number of slices for mean perimeter calculation

# Outlier filtering parameters
THICKNESS_FILTER_ENABLED = True   # If True, filter out bodies with thickness deviation > threshold
THICKNESS_FILTER_THRESHOLD = 0.10  # Maximum allowed deviation from expected thickness (30%)

# Box-counting fractal dimension parameters
FRACTAL_MIN_POINTS = 4      # Minimum data points for regression
FRACTAL_MAX_POINTS = 20     # Maximum box size iterations (only used if FRACTAL_FIXED_STEPS is None)
FRACTAL_SIZE_FACTOR = 1.5   # Box size reduction factor
FRACTAL_LMIN_DIVISOR = 16.0 # L_min = edge_length / FRACTAL_LMIN_DIVISOR (only used if FRACTAL_FIXED_STEPS is None)
FRACTAL_LMIN_USE_MIN = True  # If True, use min edge length for L_min; if False, use mean edge length
FRACTAL_FIXED_STEPS = None  # If set (e.g., 15), use exactly this many steps; overrides L_min-based stopping

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Point3D:
    """A point in 3D space."""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Point3D:
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


@dataclass
class BodyAnalysisResult:
    """Results from analyzing a single body/fragment."""
    body_index: int
    volume_total: float
    area_total: float
    perimeter_slice1: float
    perimeter_slice2: float
    area_slice1: float
    area_slice2: float
    volume_theoretical: float
    fracture_surface_area: float  # directly measured from mesh clipping (excluding caps)
    area_theoretical: float  # theoretical fracture surface area (smooth prism)
    fsr_a: float  # fracture surface roughness (area-based): A_real / A_theoretical
    fsr_v: float  # fracture surface roughness (volume-based): V_real / V_theoretical
    pr: float  # perimeter ratio: actual perimeter / convex hull perimeter (multi-slice average)
    rad: float  # Ra deviation: arithmetic mean radial deviation from smooth reference [mm]
    rf: float  # roughness factor: mean deviation from linearly interpolated ideal [mm]
    rsd: float  # radial std deviation: mean std dev of radial distances (surface bumpiness) [mm]
    fractal_dim: float  # box-counting fractal dimension (2.0=smooth, >2.0=rough)
    thickness: float
    distance_between_slices: float
    max_distance_z: float
    z_lower: float = 0.0  # lower z bound used for fracture surface extraction
    z_upper: float = 0.0  # upper z bound used for fracture surface extraction


@dataclass
class SpecimenData:
    """Data retrieved from the specimen database."""
    t_measured: Optional[float] = None  # measured thickness from scalp [mm]
    sig_h: Optional[float] = None  # pre-stress [MPa]
    U: Optional[float] = None  # strain energy [J/m²]
    U_d: Optional[float] = None  # strain energy density [J/m³]
    N50: Optional[float] = None  # fragment count


@dataclass
class STLAnalysisResult:
    """Results from analyzing a single STL file."""
    specimen_name: str
    input_file: Path
    output_folder: Path
    body_results: list[BodyAnalysisResult] = field(default_factory=list)
    specimen_data: Optional[SpecimenData] = None  # Data from specimen database

    @property
    def mean_thickness(self) -> Optional[float]:
        """Mean calculated thickness across all bodies."""
        if not self.body_results:
            return None
        return np.mean([b.thickness for b in self.body_results])

    @property
    def mean_fsr_a(self) -> Optional[float]:
        """Mean FSR (area-based) across all bodies."""
        if not self.body_results:
            return None
        return np.mean([b.fsr_a for b in self.body_results])

    @property
    def mean_fsr_v(self) -> Optional[float]:
        """Mean FSR (volume-based) across all bodies."""
        if not self.body_results:
            return None
        return np.mean([b.fsr_v for b in self.body_results])

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        data = []
        for r in self.body_results:
            data.append({
                'Specimen': self.specimen_name,
                'Body': r.body_index,
                'Volume_total [mm³]': r.volume_total,
                'Volume_theoretical [mm³]': r.volume_theoretical,
                'Area_total [mm²]': r.area_total,
                'Perimeter1 [mm]': r.perimeter_slice1,
                'Perimeter2 [mm]': r.perimeter_slice2,
                'Area_slice1 [mm²]': r.area_slice1,
                'Area_slice2 [mm²]': r.area_slice2,
                'Fracture_surface_area [mm²]': r.fracture_surface_area,
                'Area_theoretical [mm²]': r.area_theoretical,
                'FSR_A': r.fsr_a,
                'FSR_V': r.fsr_v,
                'PR': r.pr,
                'RAD [mm]': r.rad,
                'RF [mm]': r.rf,
                'RSD [mm]': r.rsd,
                'Fractal_Dim': r.fractal_dim,
                'Thickness [mm]': r.thickness,
                'Slice_distance [mm]': r.distance_between_slices,
                'Max_Z_distance [mm]': r.max_distance_z,
            })
        return pd.DataFrame(data)


# =============================================================================
# Optimized Utility Functions (using numpy arrays)
# =============================================================================

def sort_points_by_nearest_neighbor_fast(points: np.ndarray) -> np.ndarray:
    """Sort points by nearest neighbor traversal using KDTree.

    This is O(n log n) instead of O(n²) for the naive approach.

    Args:
        points: Nx3 numpy array of points.

    Returns:
        Sorted Nx3 numpy array of points.
    """
    if len(points) < 2:
        return points

    n = len(points)

    # Build KDTree for fast nearest neighbor queries
    tree = KDTree(points)

    # Start from the point closest to centroid
    centroid = points.mean(axis=0)
    _, start_idx = tree.query(centroid)

    visited = np.zeros(n, dtype=bool)
    sorted_indices = np.empty(n, dtype=int)
    sorted_indices[0] = start_idx
    visited[start_idx] = True

    current_idx = start_idx
    for i in range(1, n):
        # Query k nearest neighbors (need k > 1 to skip visited)
        # Start with small k and increase if needed
        k = min(10, n)
        found = False

        while not found and k <= n:
            distances, indices = tree.query(points[current_idx], k=k)

            for idx in indices:
                if not visited[idx]:
                    sorted_indices[i] = idx
                    visited[idx] = True
                    current_idx = idx
                    found = True
                    break

            if not found:
                k = min(k * 2, n)

        if not found:
            # Fallback: find any unvisited point
            for idx in range(n):
                if not visited[idx]:
                    sorted_indices[i] = idx
                    visited[idx] = True
                    current_idx = idx
                    break

    return points[sorted_indices]


def calculate_perimeter_fast(points: np.ndarray) -> float:
    """Calculate the perimeter of a polygon from sorted points using numpy.

    Args:
        points: Nx3 numpy array of points sorted in order around the polygon.

    Returns:
        The perimeter length.
    """
    if len(points) < 2:
        return 0.0

    # Calculate distances between consecutive points
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)

    # Add distance from last to first point to close the loop
    closing_distance = np.linalg.norm(points[-1] - points[0])

    return float(distances.sum() + closing_distance)


def polygon_area_xy_fast(points: np.ndarray) -> float:
    """Calculate the area of a polygon in the XY plane using the shoelace formula.

    Args:
        points: Nx3 numpy array of points defining the polygon vertices.

    Returns:
        The area of the polygon.
    """
    if len(points) < 3:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]

    # Shoelace formula vectorized
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def calculate_perimeter_ratio(
    body: pv.PolyData,
    z_lower: float,
    z_upper: float,
    n_slices: int = SLICE_COUNT,
) -> float:
    """Calculate perimeter ratio (PR) across multiple horizontal slices.

    PR measures how jagged/rough the perimeter is by comparing the actual
    perimeter length to the convex hull perimeter at each Z-level.

    Args:
        body: The PyVista mesh of the body.
        z_lower: Lower Z bound for slicing.
        z_upper: Upper Z bound for slicing.
        n_slices: Number of horizontal slices to take.

    Returns:
        Mean ratio of actual perimeter / convex hull perimeter.
        Values > 1.0 indicate rougher perimeters.
    """
    z_levels = np.linspace(z_lower, z_upper, n_slices)
    ratios = []

    for z in z_levels:
        try:
            slice_mesh = body.slice(normal='z', origin=(0, 0, z))
            if slice_mesh.n_cells < 1 or slice_mesh.n_points < 4:
                continue

            # Calculate perimeter directly from edge lengths (no sorting needed)
            actual_perimeter = slice_mesh.compute_cell_sizes(length=True)['Length'].sum()

            if actual_perimeter <= 0:
                continue

            # Convex hull perimeter (in 2D, ConvexHull.area is the perimeter)
            points_2d = slice_mesh.points[:, :2]
            try:
                hull = ConvexHull(points_2d)
                hull_perimeter = hull.area  # In 2D, 'area' is actually perimeter
            except Exception:
                continue

            if hull_perimeter > 0:
                ratios.append(actual_perimeter / hull_perimeter)
        except Exception:
            continue

    return float(np.mean(ratios)) if ratios else 1.0


def calculate_ra_deviation(
    body: pv.PolyData,
    z_lower: float,
    z_upper: float,
    n_slices: int = SLICE_COUNT,
) -> float:
    """Calculate Ra (arithmetic average) deviation of the lateral surface.

    Ra measures how much points on the lateral surface deviate from a smooth
    reference surface. For each Z-level slice, we calculate the centroid and
    mean radius, then measure how much each point deviates from this mean radius.

    Args:
        body: The PyVista mesh of the body.
        z_lower: Lower Z bound for analysis.
        z_upper: Upper Z bound for analysis.
        n_slices: Number of horizontal slices to sample.

    Returns:
        Arithmetic mean of radial deviations in mm.
        Higher values indicate rougher surfaces.
    """
    z_levels = np.linspace(z_lower, z_upper, n_slices)
    all_deviations = []

    for z in z_levels:
        try:
            slice_mesh = body.slice(normal='z', origin=(0, 0, z))
            if slice_mesh.n_points < 3:
                continue

            points_2d = slice_mesh.points[:, :2]  # X, Y only

            # Calculate centroid
            centroid = points_2d.mean(axis=0)

            # Calculate distances from centroid
            distances = np.linalg.norm(points_2d - centroid, axis=1)

            # Mean radius is the "smooth" reference
            mean_radius = distances.mean()

            # Deviation from mean radius
            deviations = np.abs(distances - mean_radius)
            all_deviations.extend(deviations)
        except Exception:
            continue

    return float(np.mean(all_deviations)) if all_deviations else 0.0


def calculate_radial_std_deviation(
    body: pv.PolyData,
    z_lower: float,
    z_upper: float,
    n_slices: int = SLICE_COUNT,
) -> float:
    """Calculate mean standard deviation of radial distances (RSD).

    RSD measures how irregular/bumpy the perimeter is at each slice by computing
    the standard deviation of radial distances from the centroid. A perfectly
    smooth circular slice would have RSD=0. Bumps and irregularities increase RSD.

    This metric is more sensitive to local surface roughness than RAD because
    std dev captures the spread of deviations, not just their average magnitude.

    Args:
        body: The PyVista mesh of the body.
        z_lower: Lower Z bound for analysis.
        z_upper: Upper Z bound for analysis.
        n_slices: Number of horizontal slices to sample.

    Returns:
        Mean standard deviation of radial distances across all slices in mm.
        Higher values indicate bumpier/more irregular perimeters.
    """
    z_levels = np.linspace(z_lower, z_upper, n_slices)
    slice_stds = []

    for z in z_levels:
        try:
            slice_mesh = body.slice(normal='z', origin=(0, 0, z))
            if slice_mesh.n_points < 3:
                continue

            points_2d = slice_mesh.points[:, :2]

            # Calculate centroid and radial distances
            centroid = points_2d.mean(axis=0)
            distances = np.linalg.norm(points_2d - centroid, axis=1)

            # Standard deviation of radial distances for this slice
            slice_stds.append(distances.std())

        except Exception:
            continue

    return float(np.mean(slice_stds)) if slice_stds else 0.0


def calculate_roughness_factor(
    body: pv.PolyData,
    z_lower: float,
    z_upper: float,
    n_slices: int = SLICE_COUNT,
) -> float:
    """Calculate Roughness Factor (RF) based on deviation from linearly interpolated ideal.

    RF measures the mean absolute deviation of each slice's mean radius from
    the linearly interpolated ideal radius (between top and bottom slices).
    This captures how much the fragment "bulges" or "necks" compared to a
    smooth linear transition.

    The result is normalized by the number of slices, making it independent
    of n_slices and directly comparable across different analyses.

    Args:
        body: The PyVista mesh of the body.
        z_lower: Lower Z bound for analysis (bottom slice).
        z_upper: Upper Z bound for analysis (top slice).
        n_slices: Number of horizontal slices to sample.

    Returns:
        Mean absolute deviation from ideal radius in mm.
        Higher values indicate rougher/more irregular perimeters.
    """
    # Get reference slices (bottom and top)
    try:
        bottom_slice = body.slice(normal='z', origin=(0, 0, z_lower))
        top_slice = body.slice(normal='z', origin=(0, 0, z_upper))

        if bottom_slice.n_points < 3 or top_slice.n_points < 3:
            return 0.0

        # Calculate reference properties for bottom and top slices
        bottom_2d = bottom_slice.points[:, :2]
        top_2d = top_slice.points[:, :2]

        bottom_centroid = bottom_2d.mean(axis=0)
        top_centroid = top_2d.mean(axis=0)

        # Mean radius for each reference slice
        bottom_radii = np.linalg.norm(bottom_2d - bottom_centroid, axis=1)
        top_radii = np.linalg.norm(top_2d - top_centroid, axis=1)

        bottom_mean_radius = bottom_radii.mean()
        top_mean_radius = top_radii.mean()

    except Exception:
        return 0.0

    # Calculate total height for interpolation
    total_height = z_upper - z_lower
    if total_height <= 0:
        return 0.0

    # Sample intermediate slices and collect per-slice deviations
    z_levels = np.linspace(z_lower, z_upper, n_slices)
    slice_deviations = []

    for z in z_levels:
        try:
            slice_mesh = body.slice(normal='z', origin=(0, 0, z))
            if slice_mesh.n_points < 3:
                continue

            points_2d = slice_mesh.points[:, :2]

            # Interpolation factor (0 at bottom, 1 at top)
            t = (z - z_lower) / total_height

            # Interpolated ideal centroid and radius
            ideal_centroid = (1 - t) * bottom_centroid + t * top_centroid
            ideal_radius = (1 - t) * bottom_mean_radius + t * top_mean_radius

            # Calculate actual mean radius from interpolated centroid
            actual_radii = np.linalg.norm(points_2d - ideal_centroid, axis=1)
            actual_mean_radius = actual_radii.mean()

            # Deviation of this slice's mean radius from ideal
            slice_deviations.append(abs(actual_mean_radius - ideal_radius))

        except Exception:
            continue

    # Return mean deviation across all slices (normalized)
    return float(np.mean(slice_deviations)) if slice_deviations else 0.0


def cut_and_calculate_fracture_surface_area(
    mesh: pv.PolyData,
    z_values: list[float],
) -> float:
    """Cut a mesh between two Z planes and calculate the fracture surface area.

    The mesh is clipped between the two Z planes. PyVista's clip() does NOT
    add cap faces at the clipping planes, so the resulting area is directly
    the fracture surface (the "mantel" or lateral surface).

    Args:
        mesh: The PyVista mesh to cut.
        z_values: Two Z values defining the cutting planes [z_lower, z_upper].

    Returns:
        The fracture surface area (lateral surface between the two Z planes).
    """
    clipped_mesh = mesh.clip(normal='z', origin=(0, 0, z_values[0]), invert=False)
    clipped_mesh = clipped_mesh.clip(normal='z', origin=(0, 0, z_values[1]), invert=True)

    # Handle case where clip returns UnstructuredGrid instead of PolyData
    if hasattr(clipped_mesh, 'extract_surface'):
        clipped_mesh = clipped_mesh.extract_surface()

    if hasattr(clipped_mesh, 'is_all_triangles') and not clipped_mesh.is_all_triangles:
        clipped_mesh = clipped_mesh.triangulate()

    # PyVista clip() does not add cap faces, so area is directly the fracture surface
    return clipped_mesh.area


def clip_fracture_surface(
    mesh: pv.PolyData,
    z_lower: float,
    z_upper: float,
) -> pv.PolyData:
    """Extract the fracture surface (lateral surface) between two Z planes.

    Clips the mesh between z_lower and z_upper to isolate only the fracture
    surface (excluding the top and bottom glass faces).

    Args:
        mesh: The PyVista mesh of the body.
        z_lower: Lower Z bound for clipping.
        z_upper: Upper Z bound for clipping.

    Returns:
        The clipped mesh containing only the fracture surface.
    """
    clipped = mesh.clip(normal='z', origin=(0, 0, z_lower), invert=False)
    clipped = clipped.clip(normal='z', origin=(0, 0, z_upper), invert=True)

    # Handle case where clip returns UnstructuredGrid instead of PolyData
    if hasattr(clipped, 'extract_surface'):
        clipped = clipped.extract_surface()

    if hasattr(clipped, 'is_all_triangles') and not clipped.is_all_triangles:
        clipped = clipped.triangulate()

    return clipped


def count_occupied_boxes(
    points: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
    box_size: float,
) -> int:
    """Count boxes of a given size containing at least one point.

    Uses vertex-based counting: for each mesh vertex, determine which box
    it falls into, then count the total number of unique occupied boxes.

    Args:
        points: Nx3 numpy array of point coordinates.
        bounds: Bounding box as (xmin, xmax, ymin, ymax, zmin, zmax).
        box_size: Size of each cubic box.

    Returns:
        Number of occupied boxes.
    """
    occupied = get_occupied_box_indices(points, bounds, box_size)

    return len(occupied)


def get_occupied_box_indices(
    points: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
    box_size: float,
) -> set[tuple[int, int, int]]:
    """Get the set of occupied box indices (vectorized).

    Args:
        points: Nx3 numpy array of point coordinates.
        bounds: Bounding box as (xmin, xmax, ymin, ymax, zmin, zmax).
        box_size: Size of each cubic box.

    Returns:
        Set of (ix, iy, iz) tuples for occupied boxes.
    """
    xmin, _, ymin, _, zmin, _ = bounds

    # Vectorized: compute all box indices at once
    origin = np.array([xmin, ymin, zmin])
    indices = ((points - origin) / box_size).astype(np.int32)

    # Get unique box indices efficiently
    unique_indices = np.unique(indices, axis=0)

    # Convert to set of tuples for compatibility with existing code
    return set(map(tuple, unique_indices))


def create_box_mesh(
    box_indices: set[tuple[int, int, int]],
    bounds: tuple[float, float, float, float, float, float],
    box_size: float,
) -> pv.PolyData:
    """Create a PyVista mesh of the occupied boxes for visualization.

    Args:
        box_indices: Set of (ix, iy, iz) tuples for occupied boxes.
        bounds: Bounding box as (xmin, xmax, ymin, ymax, zmin, zmax).
        box_size: Size of each cubic box.

    Returns:
        PyVista mesh containing all occupied boxes as wireframe cubes.
    """
    xmin, ymin, zmin = bounds[0], bounds[2], bounds[4]

    boxes = []
    for ix, iy, iz in box_indices:
        # Calculate box corner
        x0 = xmin + ix * box_size
        y0 = ymin + iy * box_size
        z0 = zmin + iz * box_size

        # Create box
        box = pv.Box(bounds=(x0, x0 + box_size, y0, y0 + box_size, z0, z0 + box_size))
        boxes.append(box)

    if boxes:
        return pv.MultiBlock(boxes).combine()
    else:
        return pv.PolyData()


@dataclass
class EdgeStats:
    """Edge length statistics for a mesh."""
    mean: float
    min: float
    max: float
    L_min: float  # Computed L_min based on edge stats


def _compute_edge_stats(fracture_mesh: pv.PolyData, L_max: float) -> EdgeStats:
    """Compute edge length statistics and L_min for box-counting.

    Args:
        fracture_mesh: The fracture surface mesh.
        L_max: Maximum dimension of the bounding box.

    Returns:
        EdgeStats with mean, min, max edge lengths and computed L_min.
    """
    try:
        edge_lengths = fracture_mesh.compute_cell_sizes(length=True)['Length']
        if len(edge_lengths) > 0:
            mean_edge = edge_lengths.mean()
            min_edge = edge_lengths.min()
            max_edge = edge_lengths.max()
            if FRACTAL_LMIN_USE_MIN:
                # L_min = min_edge / SIZE_FACTOR so that box counting includes
                # a step at approximately min_edge (the mesh resolution limit)
                L_min = min_edge / FRACTAL_SIZE_FACTOR
            else:
                L_min = mean_edge / FRACTAL_LMIN_DIVISOR
        else:
            mean_edge = min_edge = max_edge = 0
            L_min = L_max / 100
    except Exception:
        mean_edge = min_edge = max_edge = 0
        L_min = L_max / 100

    # Ensure L_min is valid
    if L_min <= 0 or L_min >= L_max:
        L_min = L_max / 100

    return EdgeStats(mean=mean_edge, min=min_edge, max=max_edge, L_min=L_min)


def _generate_box_sizes(L_max: float, L_min: float) -> list[float]:
    """Generate box sizes for box-counting algorithm.

    Args:
        L_max: Starting box size (largest dimension).
        L_min: Minimum box size (only used if FRACTAL_FIXED_STEPS is None).

    Returns:
        List of box sizes in decreasing order.
    """
    box_sizes = []
    L = L_max

    if FRACTAL_FIXED_STEPS is not None and FRACTAL_FIXED_STEPS > 0:
        # Use fixed number of steps
        for _ in range(FRACTAL_FIXED_STEPS):
            box_sizes.append(L)
            L = L / FRACTAL_SIZE_FACTOR
    else:
        # Use L_min-based stopping
        while len(box_sizes) < FRACTAL_MAX_POINTS:
            box_sizes.append(L)
            if L < L_min:
                break  # Include this step, then stop
            L = L / FRACTAL_SIZE_FACTOR

    return box_sizes


def _perform_fractal_regression(
    n_l_data: list[tuple[float, int]]
) -> tuple[float, float, float, float]:
    """Perform linear regression on log-log data for fractal dimension.

    Args:
        n_l_data: List of (L, N) pairs from box counting.

    Returns:
        Tuple of (D, D_raw, c, r_squared) where:
        - D: clamped fractal dimension [2.0, 3.0]
        - D_raw: raw (unclamped) slope from regression
        - c: intercept
        - r_squared: coefficient of determination
        Returns (2.0, 2.0, 0.0, 0.0) on failure.
    """
    if len(n_l_data) < FRACTAL_MIN_POINTS:
        return 2.0, 2.0, 0.0, 0.0

    log_inv_L = np.array([np.log(1.0 / L) for L, N in n_l_data])
    log_N = np.array([np.log(N) for L, N in n_l_data])

    try:
        D_raw, c = np.polyfit(log_inv_L, log_N, 1)

        # Calculate R² value
        y_pred = D_raw * log_inv_L + c
        ss_res = np.sum((log_N - y_pred) ** 2)
        ss_tot = np.sum((log_N - np.mean(log_N)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Clamp D to physically meaningful range [2.0, 3.0]
        D = max(2.0, min(3.0, D_raw))

        return float(D), float(D_raw), float(c), float(r_squared)
    except Exception:
        return 2.0, 2.0, 0.0, 0.0


def compute_fractal_dimension(
    body: pv.PolyData,
    z_lower: float,
    z_upper: float,
) -> tuple[float, list[tuple[float, int]]]:
    """Compute box-counting fractal dimension of the fracture surface.

    The fractal dimension D is computed using the box-counting method:
    D = lim(L→0) [log(N(L)) / log(1/L)]

    In practice, we cover the surface with boxes of decreasing sizes and
    fit a line to log(N) vs log(1/L). The slope is the fractal dimension.

    - D = 2.0 for a perfectly smooth 2D surface
    - D > 2.0 for rough/fractal surfaces (up to 3.0 for volume-filling)
    - Higher energy fractures produce rougher surfaces with higher D

    Args:
        body: The PyVista mesh of the body.
        z_lower: Lower Z bound for fracture surface extraction.
        z_upper: Upper Z bound for fracture surface extraction.

    Returns:
        Tuple of (D, n_l_data) where:
        - D: fractal dimension (2.0 = smooth, >2.0 = rough)
        - n_l_data: list of (L, N) pairs for plotting/diagnostics
    """
    # 1. Extract fracture surface (clip between z planes)
    try:
        fracture_mesh = clip_fracture_surface(body, z_lower, z_upper)
    except Exception:
        return 2.0, []

    if fracture_mesh.n_points < 10:
        return 2.0, []

    # 2. Get bounding box of fracture surface
    bounds = fracture_mesh.bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    L_max = max(xmax - xmin, ymax - ymin, zmax - zmin)

    if L_max <= 0:
        return 2.0, []

    # 3. Compute edge stats and generate box sizes
    edge_stats = _compute_edge_stats(fracture_mesh, L_max)
    box_sizes = _generate_box_sizes(L_max, edge_stats.L_min)

    if len(box_sizes) < FRACTAL_MIN_POINTS:
        return 2.0, []

    # 4. Count occupied boxes for each L
    points = fracture_mesh.points
    n_l_data = []
    for L in box_sizes:
        N = count_occupied_boxes(points, bounds, L)
        if N > 0:  # Only include valid counts
            n_l_data.append((L, N))

    # 5. Perform regression
    D, _, _, _ = _perform_fractal_regression(n_l_data)

    return D, n_l_data


def compute_fractal_dimension_with_debug(
    body: pv.PolyData,
    z_lower: float,
    z_upper: float,
    debug_dir: Path,
    body_name: str = "body",
) -> tuple[float, list[tuple[float, int]]]:
    """Compute fractal dimension with comprehensive debug output.

    Creates debug output including:
    - CSV file with raw L, N, log(1/L), log(N) data
    - Log-log plot with regression line and R² value
    - 3D visualizations of boxes at different scales
    - Fracture surface mesh visualization

    Args:
        body: The PyVista mesh of the body.
        z_lower: Lower Z bound for fracture surface extraction.
        z_upper: Upper Z bound for fracture surface extraction.
        debug_dir: Directory to save debug output.
        body_name: Name prefix for debug files.

    Returns:
        Tuple of (D, n_l_data) same as compute_fractal_dimension.
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Helper to save body render on error
    def save_error_body_render(error_msg: str):
        with open(debug_dir / f"{body_name}_error.txt", 'w') as f:
            f.write(f"{error_msg}\n")
            f.write(f"\nBody info:\n")
            f.write(f"  z_lower: {z_lower}\n")
            f.write(f"  z_upper: {z_upper}\n")
            f.write(f"  n_points: {body.n_points}\n")
            f.write(f"  n_cells: {body.n_cells}\n")
            f.write(f"  bounds: {body.bounds}\n")
        try:
            p = pv.Plotter(off_screen=True, window_size=[1200, 1000])
            p.add_mesh(body, color='lightblue', show_edges=True, edge_color='gray', opacity=0.8)
            p.add_mesh(body.outline(), color='black', line_width=2)
            # Add z-plane indicators
            bounds = body.bounds
            z_plane_lower = pv.Plane(center=(0, 0, z_lower), direction=(0, 0, 1),
                                     i_size=bounds[1]-bounds[0]+2, j_size=bounds[3]-bounds[2]+2)
            z_plane_upper = pv.Plane(center=(0, 0, z_upper), direction=(0, 0, 1),
                                     i_size=bounds[1]-bounds[0]+2, j_size=bounds[3]-bounds[2]+2)
            p.add_mesh(z_plane_lower, color='red', opacity=0.3)
            p.add_mesh(z_plane_upper, color='green', opacity=0.3)
            p.camera_position = 'iso'
            p.add_axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
            p.camera.zoom(0.8)
            p.screenshot(str(debug_dir / f"{body_name}_error_body.png"))
            p.close()
        except Exception as render_e:
            with open(debug_dir / f"{body_name}_error.txt", 'a') as f:
                f.write(f"\nCould not render body: {render_e}\n")

    # 1. Extract fracture surface
    try:
        fracture_mesh = clip_fracture_surface(body, z_lower, z_upper)
    except Exception as e:
        save_error_body_render(f"Error clipping fracture surface: {e}")
        return 2.0, []

    if fracture_mesh.n_points < 10:
        save_error_body_render(f"Too few points in fracture mesh: {fracture_mesh.n_points}")
        return 2.0, []

    # Get mesh info and compute edge stats
    bounds = fracture_mesh.bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    L_max = max(xmax - xmin, ymax - ymin, zmax - zmin)

    edge_stats = _compute_edge_stats(fracture_mesh, L_max)

    # Save mesh info to text file
    with open(debug_dir / f"{body_name}_mesh_info.txt", 'w') as f:
        f.write(f"Fracture Surface Mesh Information\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Number of points: {fracture_mesh.n_points}\n")
        f.write(f"Number of cells: {fracture_mesh.n_cells}\n")
        f.write(f"Surface area: {fracture_mesh.area:.4f} mm²\n\n")
        f.write(f"Bounding Box:\n")
        f.write(f"  X: [{xmin:.4f}, {xmax:.4f}] (range: {xmax-xmin:.4f})\n")
        f.write(f"  Y: [{ymin:.4f}, {ymax:.4f}] (range: {ymax-ymin:.4f})\n")
        f.write(f"  Z: [{zmin:.4f}, {zmax:.4f}] (range: {zmax-zmin:.4f})\n\n")
        f.write(f"L_max (largest dimension): {L_max:.4f}\n")
        if FRACTAL_LMIN_USE_MIN:
            f.write(f"L_min (min_edge / {FRACTAL_SIZE_FACTOR}): {edge_stats.L_min:.4f}\n")
            f.write(f"  -> Ensures last box size min_edge (mesh resolution limit)\n\n")
        else:
            f.write(f"L_min (mean_edge / {FRACTAL_LMIN_DIVISOR}): {edge_stats.L_min:.4f}\n\n")
        f.write(f"Edge Length Statistics:\n")
        f.write(f"  Mean: {edge_stats.mean:.4f}\n")
        f.write(f"  Min: {edge_stats.min:.4f}\n")
        f.write(f"  Max: {edge_stats.max:.4f}\n\n")
        f.write(f"Box-Counting Parameters:\n")
        f.write(f"  FRACTAL_SIZE_FACTOR: {FRACTAL_SIZE_FACTOR}\n")
        f.write(f"  FRACTAL_LMIN_USE_MIN: {FRACTAL_LMIN_USE_MIN}\n")
        f.write(f"  FRACTAL_FIXED_STEPS: {FRACTAL_FIXED_STEPS}\n")
        if FRACTAL_FIXED_STEPS is not None and FRACTAL_FIXED_STEPS > 0:
            f.write(f"  Mode: Fixed {FRACTAL_FIXED_STEPS} steps\n")
        else:
            f.write(f"  Mode: L_min-based (stop when L < {edge_stats.L_min:.4f})\n")

    # Save fracture mesh visualization
    try:
        p = pv.Plotter(off_screen=True)
        p.add_mesh(fracture_mesh, color='lightblue', show_edges=True, edge_color='gray', opacity=0.8)
        p.add_mesh(fracture_mesh.outline(), color='black', line_width=2)
        p.camera_position = 'iso'
        p.add_axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
        p.screenshot(str(debug_dir / f"{body_name}_fracture_surface.png"))
        # Tilted view
        p.camera_position = 'xz'
        p.camera.elevation = 30
        p.camera.azimuth = 45
        p.screenshot(str(debug_dir / f"{body_name}_fracture_surface_tilted.png"))
        p.close()
    except Exception as e:
        print(f"Warning: Could not save fracture surface image: {e}")

    # Generate box sizes
    box_sizes = _generate_box_sizes(L_max, edge_stats.L_min)

    if len(box_sizes) < FRACTAL_MIN_POINTS:
        with open(debug_dir / f"{body_name}_error.txt", 'w') as f:
            f.write(f"Too few box sizes: {len(box_sizes)} (need {FRACTAL_MIN_POINTS})\n")
        return 2.0, []

    # Count boxes and collect debug data
    points = fracture_mesh.points
    n_l_data = []
    debug_rows = []

    # Calculate expected box counts at L_max for each dimension
    nx_max = math.ceil((xmax - xmin) / L_max)
    ny_max = math.ceil((ymax - ymin) / L_max)
    nz_max = math.ceil((zmax - zmin) / L_max)

    # Append expected counts info to mesh_info.txt
    with open(debug_dir / f"{body_name}_mesh_info.txt", 'a') as f:
        f.write(f"\nExpected Box Counts at L_max={L_max:.4f}:\n")
        f.write(f"  X direction: ceil({xmax-xmin:.4f}/{L_max:.4f}) = {nx_max}\n")
        f.write(f"  Y direction: ceil({ymax-ymin:.4f}/{L_max:.4f}) = {ny_max}\n")
        f.write(f"  Z direction: ceil({zmax-zmin:.4f}/{L_max:.4f}) = {nz_max}\n")
        f.write(f"  Max possible boxes: {nx_max * ny_max * nz_max}\n")
        f.write(f"\nNote: At large L, few boxes are expected. This is correct.\n")
        f.write(f"The fractal dimension is computed from how N scales as L decreases.\n")

    for i, L in enumerate(box_sizes):
        occupied_indices = get_occupied_box_indices(points, bounds, L)
        N = len(occupied_indices)

        if N > 0:
            n_l_data.append((L, N))
            log_inv_L = np.log(1.0 / L)
            log_N = np.log(N)
            debug_rows.append({
                'step': i + 1,
                'L': L,
                'N': N,
                'log(1/L)': log_inv_L,
                'log(N)': log_N,
            })

            # Save box visualization for ALL scales
            try:
                box_mesh = create_box_mesh(occupied_indices, bounds, L)
                p = pv.Plotter(off_screen=True)
                p.add_mesh(fracture_mesh, color='lightblue', opacity=0.3)
                if box_mesh.n_points > 0:
                    p.add_mesh(box_mesh, style='wireframe', color='red', line_width=1)
                p.camera_position = 'iso'
                p.add_axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
                p.screenshot(str(debug_dir / f"{body_name}_boxes_step{i+1:02d}_L{L:.4f}.png"))

                # 2D cross-section view for small box sizes (last 3 steps)
                if i >= len(box_sizes) - 3:
                    try:
                        # Slice mesh at Y=center to get cross-section
                        y_center = (ymin + ymax) / 2
                        slice_mesh = fracture_mesh.slice(normal='y', origin=(0, y_center, 0))

                        if slice_mesh.n_points > 0:
                            # Get slice points (X, Z coordinates)
                            slice_points = slice_mesh.points

                            # Create 2D plot
                            fig, ax = plt.subplots(figsize=(12, 8))

                            # Plot the cross-section contour
                            ax.scatter(slice_points[:, 0], slice_points[:, 2],
                                      s=1, c='blue', label='Fracture surface')

                            # Draw boxes that intersect this Y-slice
                            for ix, iy, iz in occupied_indices:
                                box_x0 = xmin + ix * L
                                box_y0 = ymin + iy * L
                                box_z0 = zmin + iz * L

                                # Check if box intersects the Y-slice
                                if box_y0 <= y_center < box_y0 + L:
                                    rect = plt.Rectangle((box_x0, box_z0), L, L,
                                                        fill=False, edgecolor='red', linewidth=1)
                                    ax.add_patch(rect)

                            ax.set_xlabel('X (mm)', fontsize=12)
                            ax.set_ylabel('Z (mm)', fontsize=12)
                            ax.set_aspect('equal')
                            ax.grid(True, alpha=0.3)

                            plt.tight_layout()
                            plt.savefig(debug_dir / f"{body_name}_boxes_step{i+1:02d}_L{L:.4f}_section.png", dpi=150)
                            plt.close()
                    except Exception as e:
                        print(f"Warning: Could not create cross-section for L={L}: {e}")

                p.close()
            except Exception as e:
                print(f"Warning: Could not save box visualization for L={L}: {e}")

    # Save raw data to CSV
    if debug_rows:
        df = pd.DataFrame(debug_rows)
        df.to_csv(debug_dir / f"{body_name}_box_counting_data.csv", index=False)

    # Perform regression using helper
    D, D_raw, c, r_squared = _perform_fractal_regression(n_l_data)

    if len(n_l_data) < FRACTAL_MIN_POINTS:
        with open(debug_dir / f"{body_name}_error.txt", 'a') as f:
            f.write(f"Too few valid data points: {len(n_l_data)}\n")
        return 2.0, n_l_data

    # Compute log arrays for plotting
    log_inv_L_arr = np.array([np.log(1.0 / L) for L, N in n_l_data])
    log_N_arr = np.array([np.log(N) for L, N in n_l_data])

    # Save regression results
    with open(debug_dir / f"{body_name}_results.txt", 'w') as f:
        f.write(f"Box-Counting Fractal Dimension Results\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Raw slope (D_raw): {D_raw:.6f}\n")
        f.write(f"Clamped D [2.0, 3.0]: {D:.6f}\n")
        f.write(f"Intercept (c): {c:.6f}\n")
        f.write(f"R² value: {r_squared:.6f}\n\n")
        f.write(f"Number of data points: {len(n_l_data)}\n")
        f.write(f"L range: [{n_l_data[-1][0]:.4f}, {n_l_data[0][0]:.4f}]\n")
        f.write(f"N range: [{n_l_data[0][1]}, {n_l_data[-1][1]}]\n\n")
        f.write(f"Interpretation:\n")
        f.write(f"  D = 2.0: Perfectly smooth 2D surface\n")
        f.write(f"  D > 2.0: Rough/fractal surface\n")
        f.write(f"  D = 3.0: Volume-filling surface\n")

    # Create log-log plot
    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot data points
        ax.scatter(log_inv_L_arr, log_N_arr, s=100, c='blue', marker='o', label='Data points', zorder=5)

        # Plot regression line
        x_line = np.linspace(log_inv_L_arr.min() - 0.1, log_inv_L_arr.max() + 0.1, 100)
        y_line = D_raw * x_line + c
        ax.plot(x_line, y_line, 'r-', linewidth=2, label='Linear fit')

        # Add reference lines for D=2 and D=3
        y_d2 = 2.0 * x_line + (log_N_arr[0] - 2.0 * log_inv_L_arr[0])
        y_d3 = 3.0 * x_line + (log_N_arr[0] - 3.0 * log_inv_L_arr[0])
        ax.plot(x_line, y_d2, 'g--', alpha=0.5, label='D=2 (smooth)')
        ax.plot(x_line, y_d3, 'm--', alpha=0.5, label='D=3 (volume-filling)')

        # Add vertical lines for mesh size limits
        if edge_stats.min > 0:
            log_inv_min_edge = np.log(1.0 / edge_stats.min)
            ax.axvline(x=log_inv_min_edge, color='orange', linestyle=':', linewidth=2,
                       label='min edge')
        log_inv_L_min = np.log(1.0 / edge_stats.L_min)
        ax.axvline(x=log_inv_L_min, color='brown', linestyle='--', linewidth=2,
                   label='L_min')

        ax.set_xlabel('log(1/L)', fontsize=12)
        ax.set_ylabel('log(N)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Fractal Dimension Fit: D={D_raw:.4f}, R²={r_squared:.4f}", fontsize=14)

        plt.tight_layout()
        plt.savefig(debug_dir / f"{body_name}_loglog_plot.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create log-log plot: {e}")

    # Create summary plot with multiple views
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: N vs L (linear scale)
        L_vals = [L for L, N in n_l_data]
        N_vals = [N for L, N in n_l_data]
        axes[0].plot(L_vals, N_vals, 'bo-', markersize=8)
        axes[0].set_xlabel('Box size L', fontsize=12)
        axes[0].set_ylabel('Number of occupied boxes N', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].invert_xaxis()  # Larger L on left

        # Right: N vs L (log-log scale)
        axes[1].loglog(L_vals, N_vals, 'bo-', markersize=8)
        axes[1].set_xlabel('Box size L (log scale)', fontsize=12)
        axes[1].set_ylabel('Number of boxes N (log scale)', fontsize=12)
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].invert_xaxis()

        plt.tight_layout()
        plt.savefig(debug_dir / f"{body_name}_scaling_plots.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create scaling plots: {e}")

    print(f"  Debug output saved to: {debug_dir}")
    print(f"  Fractal dimension D = {D:.4f} (R² = {r_squared:.4f})")

    return float(D), n_l_data


# =============================================================================
# Specimen Name Extraction
# =============================================================================

def extract_specimen_name(filename: str) -> Optional[str]:
    """Extract specimen name from STL filename.

    Expects filenames like "8.100.B.04.stl" or "8.100.B.04_something.stl".
    The specimen name pattern is: thickness.stress.boundary.number

    Args:
        filename: The STL filename (with or without path).

    Returns:
        The specimen name (e.g., "8.100.B.04") or None if not found.
    """
    import re

    # Get just the filename without path
    name = Path(filename).stem

    # Pattern: digit(s).digit(s).letter.digit(s) optionally followed by more stuff
    # Examples: 8.100.B.04, 12.120.A.01, 4.80.Z.03
    pattern = r'^(\d+\.\d+\.[A-Z]\.\d+)'
    match = re.match(pattern, name, re.IGNORECASE)

    if match:
        return match.group(1)

    return None


def fetch_specimen_data(specimen_name: str) -> Optional[SpecimenData]:
    """Fetch specimen data from the database.

    Args:
        specimen_name: Name of the specimen (e.g., "8.100.B.04").

    Returns:
        SpecimenData object or None if specimen not found.
    """
    try:
        from fracsuite.core.specimen import Specimen

        spec = Specimen.get(specimen_name, load=True, panic=False, printout=False)
        if spec is None:
            return None

        data = SpecimenData(
            t_measured=spec.measured_thickness,
            sig_h=float(spec.sig_h) if hasattr(spec.sig_h, '__float__') else spec.sig_h,
            U=spec.U,
            U_d=spec.U_d,
        )

        # Try to get N50
        try:
            data.N50 = spec.calculate_nfifty_in_windows()
        except Exception:
            data.N50 = None

        return data

    except Exception as e:
        print(f"Warning: Could not load specimen '{specimen_name}': {e}")
        return None


# =============================================================================
# Color Map
# =============================================================================

DEFAULT_COLOR_MAP = {
    1: 'green',
    2: 'blue',
    3: 'yellow',
    4: 'purple',
    5: 'orange',
    6: 'cyan',
    7: 'magenta',
    8: 'lime',
    9: 'teal',
    10: 'red',
    11: '#FF33A1',  # Bright Pink
    12: '#FFD700',  # Gold
    13: '#FFA07A',  # Light Salmon
}


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_body(
    body: pv.PolyData,
    body_index: int,
    z_offset_lower: float = 0.1,
    z_offset_upper: float = 0.05,
    min_volume: float = 1.0,
) -> Optional[tuple[BodyAnalysisResult, dict]]:
    """Analyze a single body/fragment from an STL mesh.

    Args:
        body: The PyVista mesh of the body to analyze.
        body_index: Index of this body in the parent mesh.
        z_offset_lower: Offset from lower intersection point for slicing (mm).
        z_offset_upper: Offset from upper intersection point for slicing (mm).
        min_volume: Minimum volume threshold to process (mm³).

    Returns:
        Tuple of (BodyAnalysisResult, visualization_data dict) or None if body is too small.
        The visualization_data contains meshes needed for plotting.
    """
    # Ensure the mesh is triangulated
    if not np.all(body.celltypes == 5):
        body = body.triangulate()

    # Clean and prepare mesh
    body = body.extract_surface().triangulate() #.clean()

    # Calculate total volume
    volume_total = body.volume
    if volume_total == 0:
        volume_total = body.compute_cell_sizes(length=False, area=False, volume=True).volume.sum()

    # Filter by minimum volume
    if volume_total < min_volume:
        return None

    # Total surface area
    area_total = body.area

    # Calculate centroid
    centroid = np.mean(body.points, axis=0)

    # Cast rays from centroid in Z direction to find thickness
    directions = np.array([[0, 0, 1], [0, 0, -1]])
    points_up, _ = body.ray_trace(centroid, centroid + directions[0] * 1000)
    points_down, _ = body.ray_trace(centroid, centroid + directions[1] * 1000)

    intersection_points = np.vstack((points_up, points_down))
    intersection_points = intersection_points[np.argsort(intersection_points[:, 2])]

    if len(intersection_points) != 2:
        print(f"Could not find two intersection points in Z direction for Body {body_index}.")
        return None

    # Thickness from intersection points
    thickness = np.linalg.norm(intersection_points[0] - intersection_points[1])

    # Define slice Z values
    z1 = intersection_points[0][2] + z_offset_lower
    z2 = intersection_points[1][2] - z_offset_upper
    z_values = [z1, z2]

    # Theoretical thickness (distance between slices)
    t_theo = abs(z2 - z1)

    # Create slices
    single_slice1 = body.slice(normal=[0, 0, 1], origin=[0, 0, z1])
    single_slice2 = body.slice(normal=[0, 0, 1], origin=[0, 0, z2])

    # Max Z distance
    min_z = np.min(body.points[:, 2])
    max_z = np.max(body.points[:, 2])
    max_distance_z = max_z - min_z

    # Calculate perimeters directly from slice edge lengths (faster than sorting)
    perimeter1 = single_slice1.compute_cell_sizes(length=True)['Length'].sum() if single_slice1.n_cells > 0 else 0.0
    perimeter2 = single_slice2.compute_cell_sizes(length=True)['Length'].sum() if single_slice2.n_cells > 0 else 0.0

    # Sort points for area calculation (shoelace formula requires ordered points)
    points1 = np.asarray(single_slice1.points)
    points2 = np.asarray(single_slice2.points)
    sorted_points1 = sort_points_by_nearest_neighbor_fast(points1) if len(points1) > 0 else points1
    sorted_points2 = sort_points_by_nearest_neighbor_fast(points2) if len(points2) > 0 else points2

    # Calculate slice areas using vectorized shoelace formula
    area_slice1 = polygon_area_xy_fast(sorted_points1) if len(sorted_points1) > 0 else 0.0
    area_slice2 = polygon_area_xy_fast(sorted_points2) if len(sorted_points2) > 0 else 0.0

    # Calculate fracture surface area (lateral surface between slices)
    fracture_surface_area = cut_and_calculate_fracture_surface_area(body, z_values)

    # Calculate mean perimeter using multiple slices for better theoretical estimate
    z_levels = np.linspace(z1, z2, THEORETICAL_PERIMETER_SLICES)
    perimeters = []
    for z in z_levels:
        try:
            slice_mesh = body.slice(normal='z', origin=(0, 0, z))
            if slice_mesh.n_cells < 1:
                continue
            # Compute perimeter directly from edge lengths (no sorting needed)
            perimeter = slice_mesh.compute_cell_sizes(length=True)['Length'].sum()
            perimeters.append(perimeter)
        except Exception:
            continue

    mean_perimeter = np.mean(perimeters) if perimeters else (perimeter1 + perimeter2) / 2

    # Theoretical values for an ideal smooth prism
    v_theo = ((area_slice1 + area_slice2) / 2) * t_theo  # volume of ideal prism
    area_theoretical = mean_perimeter * t_theo  # lateral surface using mean perimeter

    # FSR_A: Area-based fracture surface roughness
    # Ratio of actual fracture surface area to theoretical smooth surface area
    fsr_a = fracture_surface_area / area_theoretical if area_theoretical > 0 else 0.0

    # FSR_V: Volume-based fracture surface roughness
    # Ratio of actual volume to theoretical prism volume
    fsr_v = volume_total / v_theo if v_theo > 0 else 0.0

    # PR: Perimeter Ratio - measures perimeter jaggedness across multiple slices
    pr = calculate_perimeter_ratio(body, z1, z2, n_slices=SLICE_COUNT)

    # RAD: Ra Deviation - arithmetic mean radial deviation from smooth reference
    rad = calculate_ra_deviation(body, z1, z2, n_slices=SLICE_COUNT)

    # RF: Roughness Factor - mean deviation from linearly interpolated ideal
    rf = calculate_roughness_factor(body, z1, z2, n_slices=SLICE_COUNT)

    # RSD: Radial Standard Deviation - measures surface bumpiness
    rsd = calculate_radial_std_deviation(body, z1, z2, n_slices=SLICE_COUNT)

    # Fractal Dimension: Box-counting fractal dimension of fracture surface
    # D = 2.0 for smooth, D > 2.0 for rough/fractal surfaces
    fractal_dim, _ = compute_fractal_dimension(body, z1, z2)

    result = BodyAnalysisResult(
        body_index=body_index,
        volume_total=volume_total,
        area_total=area_total,
        perimeter_slice1=perimeter1,
        perimeter_slice2=perimeter2,
        area_slice1=area_slice1,
        area_slice2=area_slice2,
        volume_theoretical=v_theo,
        fracture_surface_area=fracture_surface_area,
        area_theoretical=area_theoretical,
        fsr_a=fsr_a,
        fsr_v=fsr_v,
        pr=pr,
        rad=rad,
        rf=rf,
        rsd=rsd,
        fractal_dim=fractal_dim,
        thickness=thickness,
        distance_between_slices=t_theo,
        max_distance_z=max_distance_z,
        z_lower=z1,
        z_upper=z2,
    )

    vis_data = {
        'body': body,
        'slice1': single_slice1,
        'slice2': single_slice2,
        'intersection_points': intersection_points,
        'centroid': centroid,
        'vector_thickness': pv.Line(intersection_points[0], intersection_points[1]),
    }

    return result, vis_data


def analyze_stl_file(
    input_file: Path | str,
    output_dir: Optional[Path | str] = None,
    save_plots: bool = True,
    save_html: bool = True,
    save_excel: bool = True,
    off_screen: bool = True,
    z_offset_lower: float = 0.1,
    z_offset_upper: float = 0.05,
    min_volume: float = 1.0,
    data_only: bool = False,
) -> STLAnalysisResult:
    """Analyze an STL file containing multiple bodies/fragments.

    Args:
        input_file: Path to the input STL file.
        output_dir: Directory for output files. Defaults to a subfolder next to input.
        save_plots: Whether to save PNG plots.
        save_html: Whether to save interactive HTML visualization.
        save_excel: Whether to save Excel results file.
        off_screen: Whether to run visualization off-screen.
        z_offset_lower: Offset from lower intersection point for slicing (mm).
        z_offset_upper: Offset from upper intersection point for slicing (mm).
        min_volume: Minimum volume threshold to process (mm³).
        data_only: If True, skip all visualization (fastest mode).

    Returns:
        STLAnalysisResult containing all body analysis results.
    """
    input_file = Path(input_file)
    specimen_name = input_file.stem

    if output_dir is None:
        output_dir = input_file.parent / f"{specimen_name}_output"
    else:
        output_dir = Path(output_dir)

    # Only create output dir if we're saving something
    if save_plots or save_html or save_excel:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Read mesh and split into bodies
    mesh = pv.read(str(input_file))
    bodies = mesh.split_bodies()

    result = STLAnalysisResult(
        specimen_name=specimen_name,
        input_file=input_file,
        output_folder=output_dir,
    )

    # Skip visualization setup if data_only mode
    if data_only:
        save_plots = False
        save_html = False

    # Collect visualization data for batch plotting
    vis_data_list = []

    for i, body in enumerate(bodies, start=1):
        body_result = analyze_body(
            body,
            body_index=i,
            z_offset_lower=z_offset_lower,
            z_offset_upper=z_offset_upper,
            min_volume=min_volume,
        )

        if body_result is None:
            continue

        analysis, vis_data = body_result
        result.body_results.append(analysis)

        if not data_only:
            vis_data_list.append((i, analysis, vis_data))

    # Batch visualization (only if needed)
    if not data_only and (save_plots or save_html):
        _create_visualizations(
            specimen_name=specimen_name,
            vis_data_list=vis_data_list,
            output_dir=output_dir,
            save_plots=save_plots,
            save_html=save_html,
            off_screen=off_screen,
        )

    # Save CSV
    if save_excel and result.body_results:
        df = result.to_dataframe()
        csv_path = output_dir / f'{specimen_name}_body_data.csv'
        df.to_csv(csv_path, index=False)

    print(f'Analysis of specimen {specimen_name} is finished ({len(result.body_results)} bodies)')

    return result


def _create_visualizations(
    specimen_name: str,
    vis_data_list: list[tuple[int, BodyAnalysisResult, dict]],
    output_dir: Path,
    save_plots: bool,
    save_html: bool,
    off_screen: bool = True,
) -> None:
    """Create all visualizations in a batch (more efficient than one-by-one).

    Args:
        specimen_name: Name of the specimen.
        vis_data_list: List of (body_index, analysis, vis_data) tuples.
        output_dir: Directory to save outputs.
        save_plots: Whether to save PNG plots.
        save_html: Whether to save HTML visualization.
        off_screen: Whether to run off-screen.
    """
    if not vis_data_list:
        return

    # Combined 3D plotter
    tp3D = pv.Plotter(off_screen=off_screen, window_size=[1200, 1000])
    tpslice = pv.Plotter(off_screen=off_screen, window_size=[1200, 1000])

    for body_index, analysis, vis_data in vis_data_list:
        color = DEFAULT_COLOR_MAP.get(body_index, 'grey')
        body = vis_data['body']
        slice1 = vis_data['slice1']
        slice2 = vis_data['slice2']

        # Save individual body plots
        if save_plots:
            _save_body_plots_fast(
                specimen_name=specimen_name,
                body_index=body_index,
                analysis=analysis,
                vis_data=vis_data,
                color=color,
                output_dir=output_dir,
                off_screen=off_screen,
            )

        # Add to combined plotters
        tp3D.add_mesh(
            body,
            color=color,
            label=f'Body {body_index} (Volume: {analysis.volume_total:.2f})'
        )
        tp3D.add_point_labels(
            vis_data['intersection_points'][1:2],
            [f'{body_index}'],
            font_size=20,
            point_color='red',
            text_color='black'
        )
        tp3D.add_mesh(slice1)
        tp3D.add_mesh(slice2)

        tpslice.add_mesh(body, color=color, opacity=0.01)
        tpslice.add_mesh(slice1, color="red")
        tpslice.add_mesh(slice2, color="blue")

    # Finalize combined 3D plot
    tp3D.add_legend()
    tp3D.add_axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
    tp3D.show_grid(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
    tp3D.camera_position = 'xy'
    tp3D.enable_parallel_projection()
    tp3D.camera.zoom(0.8)  # Zoom out to prevent axis clipping

    if save_plots:
        tp3D.screenshot(str(output_dir / f"{specimen_name}_Body_total.png"))
        # Tilted view - use isometric as base for proper 3D perspective
        tp3D.camera_position = 'iso'
        tp3D.camera.zoom(0.8)
        tp3D.screenshot(str(output_dir / f"{specimen_name}_Body_total_tilted.png"))

    if save_html:
        try:
            import panel as pn
            pane = pn.pane.VTK(tp3D.ren_win, width=1000, height=750)
            html_path = output_dir / f"{specimen_name}_Body_total.html"
            pn.panel(pane).save(str(html_path))
        except ImportError:
            print("Panel not available, skipping HTML export")

    tp3D.close()

    # Finalize slice plot
    tpslice.camera_position = 'xy'
    tpslice.enable_parallel_projection()
    tpslice.show_grid(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
    tpslice.camera.zoom(0.8)  # Zoom out to prevent axis clipping

    if save_plots:
        tpslice.screenshot(str(output_dir / f"{specimen_name}_Total_DifferenceSlices.png"))

    tpslice.close()


def _save_body_plots_fast(
    specimen_name: str,
    body_index: int,
    analysis: BodyAnalysisResult,
    vis_data: dict,
    color: str,
    output_dir: Path,
    off_screen: bool = True,
) -> None:
    """Save individual body visualization plots efficiently.

    Uses a single plotter instance reused across plots where possible.
    """
    body = vis_data['body']
    slice1 = vis_data['slice1']
    slice2 = vis_data['slice2']
    vector_thickness = vis_data['vector_thickness']

    # # Plot 1: Difference of slices (top view)
    # p = pv.Plotter(off_screen=off_screen)
    # p.add_mesh(body.outline(), color="k")
    # p.add_mesh(slice1, color="red")
    # p.add_mesh(slice2, color="blue")
    # p.camera_position = 'xy'
    # p.enable_parallel_projection()
    # p.screenshot(str(output_dir / f"{specimen_name}_Body{body_index}_DifferenceSlices.png"))
    # p.close()

    # Plot 2: Vector thickness
    p = pv.Plotter(off_screen=off_screen, window_size=[1200, 1000])
    p.add_mesh(body, color='white', opacity=1.0)
    p.show_grid(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
    p.camera.zoom(0.8)  # Zoom out to prevent axis clipping
    p.screenshot(str(output_dir / f"{specimen_name}_Body{body_index}_VectorThickness.png"))
    # Tilted view
    p.camera_position = 'xz'
    p.camera.elevation = 30
    p.camera.azimuth = 45
    p.camera.zoom(0.8)
    p.screenshot(str(output_dir / f"{specimen_name}_Body{body_index}_VectorThickness_tilted.png"))
    p.close()

    # # Plot 3: Vector and slice (lateral view)
    # p = pv.Plotter(off_screen=off_screen, window_size=[1200, 1000])
    # p.add_mesh(body, color='white', opacity=0.5)
    # p.add_mesh(vector_thickness, color='red', line_width=3)
    # p.add_mesh(slice1, color="red")
    # p.add_mesh(slice2, color="blue")
    # p.camera_position = 'xz'
    # p.enable_parallel_projection()
    # p.show_grid()
    # p.camera.zoom(0.8)  # Zoom out to prevent axis clipping
    # p.screenshot(str(output_dir / f"{specimen_name}_Body{body_index}_VectorandSlice.png"))
    # p.close()

    # Plot 4: Colored body
    p = pv.Plotter(off_screen=off_screen, window_size=[1200, 1000])
    p.add_mesh(body, color=color, label=f'Body {body_index} (Volume: {analysis.volume_total:.2f})')
    p.add_mesh(slice1)
    p.add_mesh(slice2)
    p.camera_position = 'iso'
    p.camera.zoom(0.8)
    p.screenshot(str(output_dir / f"{specimen_name}_Body{body_index}_coloured.png"))
    # Tilted view
    p.camera_position = 'xz'
    p.camera.elevation = 30
    p.camera.azimuth = 45
    p.camera.zoom(0.8)
    p.screenshot(str(output_dir / f"{specimen_name}_Body{body_index}_coloured_tilted.png"))
    p.close()


def _analyze_stl_file_worker(args: tuple) -> Optional[STLAnalysisResult]:
    """Worker function for parallel processing."""
    (
        stl_file,
        output_dir,
        save_plots,
        save_html,
        z_offset_lower,
        z_offset_upper,
        min_volume,
        data_only,
    ) = args

    try:
        return analyze_stl_file(
            input_file=stl_file,
            output_dir=output_dir,
            save_plots=save_plots,
            save_html=save_html,
            save_excel=True,
            off_screen=True,
            z_offset_lower=z_offset_lower,
            z_offset_upper=z_offset_upper,
            min_volume=min_volume,
            data_only=data_only,
        )
    except Exception as e:
        print(f"Error analyzing {stl_file}: {e}")
        return None


def analyze_folder(
    input_dir: Path | str,
    output_dir: Optional[Path | str] = None,
    save_plots: bool = True,
    save_html: bool = True,
    save_combined_excel: bool = True,
    off_screen: bool = True,
    z_offset_lower: float = 0.1,
    z_offset_upper: float = 0.05,
    min_volume: float = 1.0,
    data_only: bool = False,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    fetch_specimens: bool = True,
) -> list[STLAnalysisResult]:
    """Analyze all STL files in a folder.

    Args:
        input_dir: Directory containing STL files.
        output_dir: Base directory for outputs. Defaults to input_dir.
        save_plots: Whether to save PNG plots.
        save_html: Whether to save interactive HTML visualizations.
        save_combined_excel: Whether to save a combined Excel file with all results.
        off_screen: Whether to run visualization off-screen.
        z_offset_lower: Offset from lower intersection point for slicing (mm).
        z_offset_upper: Offset from upper intersection point for slicing (mm).
        min_volume: Minimum volume threshold to process (mm³).
        data_only: If True, skip all visualization (fastest mode).
        parallel: If True, process files in parallel (best with data_only=True).
        max_workers: Maximum number of parallel workers. Defaults to CPU count.
        fetch_specimens: If True, fetch specimen data from database (default).

    Returns:
        List of STLAnalysisResult for each analyzed file.
    """
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)

    stl_files = list(input_dir.glob('*.stl'))
    if not stl_files:
        print(f"No STL files found in {input_dir}")
        return []

    print(f"Found {len(stl_files)} STL files to analyze")

    results = []

    if parallel and len(stl_files) > 1:
        # Parallel processing (best for data_only mode)
        if not data_only:
            print("Warning: Parallel mode with visualization may cause issues. Consider using --data-only.")

        # Prepare arguments for worker function
        worker_args = [
            (
                stl_file,
                output_dir / f"{stl_file.stem}_output",
                save_plots and not data_only,
                save_html and not data_only,
                z_offset_lower,
                z_offset_upper,
                min_volume,
                data_only,
            )
            for stl_file in stl_files
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_analyze_stl_file_worker, args): args[0] for args in worker_args}

            for future in as_completed(futures):
                stl_file = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing {stl_file}: {e}")
    else:
        # Sequential processing
        for stl_file in stl_files:
            print(f"\nAnalyzing: {stl_file.name}")
            try:
                result = analyze_stl_file(
                    input_file=stl_file,
                    output_dir=output_dir / f"{stl_file.stem}_output",
                    save_plots=save_plots and not data_only,
                    save_html=save_html and not data_only,
                    save_excel=True,
                    off_screen=off_screen,
                    z_offset_lower=z_offset_lower,
                    z_offset_upper=z_offset_upper,
                    min_volume=min_volume,
                    data_only=data_only,
                )
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {stl_file.name}: {e}")
                continue

            gc.collect()

    # Fetch specimen data for each result
    if fetch_specimens:
        print("\nFetching specimen data...")
        for result in results:
            specimen_name = extract_specimen_name(result.input_file.name)
            if specimen_name:
                result.specimen_data = fetch_specimen_data(specimen_name)
                if result.specimen_data:
                    print(f"  {specimen_name}: t_m={result.specimen_data.t_measured:.2f}mm, σ_h={result.specimen_data.sig_h:.1f}MPa")
                else:
                    print(f"  {specimen_name}: not found in database")

    # Filter outlier bodies based on thickness deviation
    if THICKNESS_FILTER_ENABLED:
        print("\nFiltering outlier bodies...")
        for result in results:
            original_count = len(result.body_results)
            result.body_results = get_filtered_bodies(result)
            filtered_count = original_count - len(result.body_results)
            if filtered_count > 0:
                print(f"  {result.input_file.stem}: removed {filtered_count} outlier(s)")

    # Debug output for D=2 bodies - single diagnostic image per body
    print("\nGenerating debug output for D=2 bodies...")
    d2_debug_dir = output_dir / f"{input_dir.name}_csv" / "d2_debug"
    d2_count = 0

    for result in results:
        # Check if this result has any D=2 bodies
        d2_bodies = [b for b in result.body_results if b.fractal_dim == 2.0]
        if not d2_bodies:
            continue

        # Re-load STL file to get body meshes
        try:
            mesh = pv.read(result.input_file)
            bodies = mesh.split_bodies()

            for body_result in d2_bodies:
                body_idx = body_result.body_index - 1  # Convert to 0-indexed
                if body_idx < len(bodies):
                    body = bodies[body_idx]
                    specimen_name = result.input_file.stem

                    # Create simple diagnostic image
                    d2_debug_dir.mkdir(parents=True, exist_ok=True)
                    img_path = d2_debug_dir / f"{specimen_name}_body{body_result.body_index:03d}.png"

                    try:
                        # Extract fracture surface for analysis
                        z_lower = body_result.z_lower
                        z_upper = body_result.z_upper
                        fracture_mesh = clip_fracture_surface(body, z_lower, z_upper)

                        # Compute diagnostic info
                        bounds = fracture_mesh.bounds
                        L_max = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
                        edge_stats = _compute_edge_stats(fracture_mesh, L_max)
                        box_sizes = _generate_box_sizes(L_max, edge_stats.L_min)

                        # Create diagnostic plot
                        p = pv.Plotter(off_screen=True, window_size=[1200, 900])
                        p.add_mesh(fracture_mesh, color='lightblue', show_edges=True,
                                   edge_color='gray', opacity=0.9)
                        p.add_mesh(fracture_mesh.outline(), color='black', line_width=2)

                        # Add diagnostic text
                        info_text = (
                            f"Specimen: {specimen_name}\n"
                            f"Body: {body_result.body_index}\n"
                            f"D = 2.0 (clamped)\n"
                            f"---\n"
                            f"n_points: {fracture_mesh.n_points}\n"
                            f"L_max: {L_max:.3f} mm\n"
                            f"L_min: {edge_stats.L_min:.4f} mm\n"
                            f"Box steps: {len(box_sizes)}\n"
                            f"Edge mean: {edge_stats.mean:.4f} mm\n"
                            f"Edge min: {edge_stats.min:.4f} mm"
                        )
                        p.add_text(info_text, position='upper_left', font_size=10)

                        p.camera_position = 'iso'
                        p.add_axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
                        p.camera.zoom(0.85)
                        p.screenshot(str(img_path))

                        # Second image: tilted view (top-down angle)
                        img_path_tilted = d2_debug_dir / f"{specimen_name}_body{body_result.body_index:03d}_tilted.png"
                        p.camera_position = 'xz'  # Top-down view
                        p.camera.elevation = 30   # Tilt 30 degrees
                        p.camera.azimuth = 45     # Rotate 45 degrees
                        p.camera.zoom(0.85)
                        p.screenshot(str(img_path_tilted))
                        p.close()

                        # Write success text file with diagnostic info
                        txt_path = d2_debug_dir / f"{specimen_name}_body{body_result.body_index:03d}.txt"
                        with open(txt_path, 'w') as f:
                            f.write("STATUS: SUCCESS (D=2.0 computed without errors)\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(f"Specimen: {specimen_name}\n")
                            f.write(f"Body: {body_result.body_index}\n")
                            f.write(f"Fractal Dimension: 2.0 (clamped to [2.0, 3.0])\n\n")
                            f.write("Diagnostic Info:\n")
                            f.write(f"  n_points: {fracture_mesh.n_points}\n")
                            f.write(f"  L_max: {L_max:.4f} mm\n")
                            f.write(f"  L_min: {edge_stats.L_min:.4f} mm\n")
                            f.write(f"  L_max/L_min ratio: {L_max/edge_stats.L_min:.2f}\n")
                            f.write(f"  Box steps: {len(box_sizes)}\n")
                            f.write(f"  Edge mean: {edge_stats.mean:.4f} mm\n")
                            f.write(f"  Edge min: {edge_stats.min:.4f} mm\n\n")
                            if len(box_sizes) < 5:
                                f.write("LIKELY CAUSE: Too few box steps (<5) for reliable regression.\n")
                                f.write("  -> Fragment size too close to mesh resolution.\n")
                            elif fracture_mesh.n_points < 50:
                                f.write("LIKELY CAUSE: Very few mesh points in fracture surface.\n")
                                f.write("  -> Fragment may be too small for fractal analysis.\n")
                            else:
                                f.write("LIKELY CAUSE: Surface is genuinely smooth (D_raw <= 2.0).\n")
                                f.write("  -> Fracture surface has low roughness/complexity.\n")

                        d2_count += 1

                    except Exception as e:
                        # If fracture surface extraction fails, show the body with z-planes
                        p = pv.Plotter(off_screen=True, window_size=[1200, 900])
                        p.add_mesh(body, color='lightblue', opacity=0.7)
                        bounds = body.bounds
                        z_lower = body_result.z_lower
                        z_upper = body_result.z_upper
                        # Add z-plane indicators
                        plane_size = max(bounds[1]-bounds[0], bounds[3]-bounds[2]) + 2
                        p.add_mesh(pv.Plane(center=(0, 0, z_lower), direction=(0, 0, 1),
                                           i_size=plane_size, j_size=plane_size),
                                   color='red', opacity=0.3, label='z_lower')
                        p.add_mesh(pv.Plane(center=(0, 0, z_upper), direction=(0, 0, 1),
                                           i_size=plane_size, j_size=plane_size),
                                   color='green', opacity=0.3, label='z_upper')
                        p.add_text(f"{specimen_name} body{body_result.body_index}\nError: {e}",
                                   position='upper_left', font_size=10)
                        p.camera_position = 'iso'
                        p.add_axes(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
                        p.camera.zoom(0.85)
                        p.screenshot(str(img_path))

                        # Second image: tilted view
                        img_path_tilted = d2_debug_dir / f"{specimen_name}_body{body_result.body_index:03d}_tilted.png"
                        p.camera_position = 'xz'
                        p.camera.elevation = 30
                        p.camera.azimuth = 45
                        p.camera.zoom(0.85)
                        p.screenshot(str(img_path_tilted))
                        p.close()

                        # Write error text file
                        txt_path = d2_debug_dir / f"{specimen_name}_body{body_result.body_index:03d}.txt"
                        with open(txt_path, 'w') as f:
                            f.write("STATUS: ERROR\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(f"Specimen: {specimen_name}\n")
                            f.write(f"Body: {body_result.body_index}\n")
                            f.write(f"z_lower: {body_result.z_lower}\n")
                            f.write(f"z_upper: {body_result.z_upper}\n\n")
                            f.write(f"Error: {e}\n\n")
                            f.write("Body bounds:\n")
                            f.write(f"  X: [{bounds[0]:.4f}, {bounds[1]:.4f}]\n")
                            f.write(f"  Y: [{bounds[2]:.4f}, {bounds[3]:.4f}]\n")
                            f.write(f"  Z: [{bounds[4]:.4f}, {bounds[5]:.4f}]\n")

                        d2_count += 1

        except Exception as e:
            print(f"  Error processing {result.input_file.name}: {e}")

    if d2_count > 0:
        print(f"  Generated {d2_count} diagnostic images in {d2_debug_dir}")
    else:
        print("  No D=2 bodies found")

    # Save combined CSV files (detailed per-body data)
    if save_combined_excel and results:
        all_dataframes = [r.to_dataframe() for r in results]
        all_dataframes = [df for df in all_dataframes if not df.empty]  # Remove empty dataframes

        csv_dir = output_dir / f"{input_dir.name}_csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            combined_df.to_csv(csv_dir / "combined_results.csv", index=False)

        # Save the summary table (one row per specimen)
        summary_df = create_specimen_summary_table(results)
        summary_df.to_csv(csv_dir / "specimen_summary.csv", index=False)

        # Save the raw body table (one row per body with specimen data)
        raw_body_df = create_raw_body_table(results)
        raw_body_df.to_csv(csv_dir / "raw_body_data.csv", index=False)

        # Save per-thickness CSV files (bodies-4mm.csv, bodies-8mm.csv, etc.)
        print(f"\nPer-thickness CSV files:")
        create_per_thickness_tables(results, csv_dir)

        # Save per-thickness-position CSV files (bodies-8mm-b.csv, etc.)
        print(f"\nPer-thickness-position CSV files:")
        create_per_thickness_position_tables(results, csv_dir)

        # Save per-specimen CSV files
        print(f"\nPer-specimen CSV files:")
        create_per_specimen_csvs(results, csv_dir)

        print(f"\nCSV files saved to: {csv_dir}")
        print(f"  - combined_results.csv (detailed STL analysis per body)")
        print(f"  - specimen_summary.csv (one row per specimen)")
        print(f"  - raw_body_data.csv (one row per body with specimen data)")
        print(f"  - bodies-Xmm.csv (grouped by thickness)")
        print(f"  - bodies-Xmm-Y.csv (grouped by thickness + position)")
        print(f"  - specimens/<specimen_name>.csv (per-specimen body data)")

    total_bodies = sum(len(r.body_results) for r in results)
    print(f'\nAnalysis complete: {len(results)} files, {total_bodies} bodies')

    # Perform ANOVA analysis
    perform_anova_analysis(results)

    return results


def perform_anova_analysis(results: list[STLAnalysisResult]) -> None:
    """Perform one-way ANOVA on fractal dimension grouped by thickness."""
    from scipy.stats import f_oneway

    # Group fractal dimensions by thickness
    thickness_groups: dict[str, list[float]] = {}

    for r in results:
        specimen_name = extract_specimen_name(r.input_file.name)
        if not specimen_name:
            continue

        parts = specimen_name.split('.')
        if parts and parts[0].isdigit():
            thickness = f"{parts[0]}mm"
        else:
            continue

        if thickness not in thickness_groups:
            thickness_groups[thickness] = []

        # Bodies are already filtered at source
        for body in r.body_results:
            thickness_groups[thickness].append(body.fractal_dim)

    if len(thickness_groups) < 2:
        print("ANOVA requires at least 2 groups")
        return

    # Print statistics
    print("\n" + "=" * 60)
    print("Fractal Dimension Statistics by Thickness")
    print("=" * 60)

    for thickness in sorted(thickness_groups.keys()):
        values = thickness_groups[thickness]
        print(f"{thickness}: D = {np.mean(values):.4f} ± {np.std(values):.4f} (n={len(values)})")

    # ANOVA test
    groups = [np.array(v) for v in thickness_groups.values()]
    f_stat, p_value = f_oneway(*groups)

    print("\n" + "-" * 60)
    print("One-Way ANOVA Results")
    print("-" * 60)
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Result: {'Significant' if p_value < 0.05 else 'No significant'} difference (p {'<' if p_value < 0.05 else '>='} 0.05)")


def filter_body_by_thickness(
    body: BodyAnalysisResult,
    expected_thickness: Optional[float],
    threshold: float = THICKNESS_FILTER_THRESHOLD,
) -> bool:
    """Check if a body passes the thickness filter.

    Args:
        body: The body analysis result to check.
        expected_thickness: Expected thickness (from specimen data or nominal).
        threshold: Maximum allowed relative deviation (default 30%).

    Returns:
        True if body passes filter (should be included), False if outlier.
    """
    if not THICKNESS_FILTER_ENABLED:
        return True

    if expected_thickness is None or expected_thickness <= 0:
        return True  # Can't filter without expected thickness

    deviation = abs(body.thickness - expected_thickness) / expected_thickness
    return deviation <= threshold


def get_filtered_bodies(
    result: STLAnalysisResult,
    threshold: float = THICKNESS_FILTER_THRESHOLD,
) -> list[BodyAnalysisResult]:
    """Get filtered list of bodies that pass the thickness filter.

    Args:
        result: The STL analysis result containing bodies.
        threshold: Maximum allowed relative deviation (default 30%).

    Returns:
        List of bodies that pass the filter.
    """
    if not THICKNESS_FILTER_ENABLED:
        return result.body_results

    # Get expected thickness from specimen data or extract from name
    expected_thickness = None
    if result.specimen_data and result.specimen_data.t_measured:
        expected_thickness = result.specimen_data.t_measured
    else:
        # Try to extract nominal thickness from specimen name (e.g., "8" from "8.100.B.04")
        base_specimen = extract_specimen_name(result.input_file.name)
        if base_specimen:
            parts = base_specimen.split('.')
            if parts and parts[0].isdigit():
                expected_thickness = float(parts[0])

    if expected_thickness is None:
        return result.body_results  # Can't filter without expected thickness

    filtered = [b for b in result.body_results
                if filter_body_by_thickness(b, expected_thickness, threshold)]

    if len(filtered) < len(result.body_results):
        removed = len(result.body_results) - len(filtered)
        print(f"  Filtered {removed} outlier(s) from {result.input_file.stem} "
              f"(thickness deviation > {threshold*100:.0f}%)")

    return filtered


def create_specimen_summary_table(results: list[STLAnalysisResult]) -> pd.DataFrame:
    """Create a summary table with one row per specimen.

    Creates a table with:
    - Specimen identification
    - Measured thickness (t_m) from SCALP
    - Calculated thickness (t_calc) from STL analysis
    - Pre-stress (sig_h), strain energy (U, U_d), fragment count (N50)
    - Individual FSR values for each body found in the STL file

    Args:
        results: List of STLAnalysisResult objects.

    Returns:
        DataFrame with one row per specimen.
    """
    if not results:
        return pd.DataFrame()

    # Find max number of bodies across all specimens
    max_bodies = max((len(r.body_results) for r in results), default=0)

    rows = []
    for r in results:
        # Use full filename (without extension) as specimen name
        specimen_name = r.input_file.stem

        row = {
            'Specimen': specimen_name,
            'STL_File': r.input_file.name,
        }

        # Specimen data (from database)
        if r.specimen_data:
            row['t_m [mm]'] = r.specimen_data.t_measured
            row['sig_h [MPa]'] = r.specimen_data.sig_h
            row['U [J/m²]'] = r.specimen_data.U
            row['U_d [J/m³]'] = r.specimen_data.U_d
            row['N50'] = r.specimen_data.N50
        else:
            row['t_m [mm]'] = None
            row['sig_h [MPa]'] = None
            row['U [J/m²]'] = None
            row['U_d [J/m³]'] = None
            row['N50'] = None

        # STL analysis data
        row['t_calc [mm]'] = r.mean_thickness
        row['Num_Bodies'] = len(r.body_results)

        # Mean values across bodies
        if r.body_results:
            row['Mean_Volume [mm³]'] = np.mean([b.volume_total for b in r.body_results])
            row['Mean_Area [mm²]'] = np.mean([b.area_total for b in r.body_results])
            row['Mean_FSR_A'] = np.mean([b.fsr_a for b in r.body_results])
            row['Mean_FSR_V'] = np.mean([b.fsr_v for b in r.body_results])
            row['Mean_PR'] = np.mean([b.pr for b in r.body_results])
            row['Mean_RAD [mm]'] = np.mean([b.rad for b in r.body_results])
            row['Mean_RF [mm]'] = np.mean([b.rf for b in r.body_results])
            row['Mean_RSD [mm]'] = np.mean([b.rsd for b in r.body_results])
            row['Mean_Fractal_Dim'] = np.mean([b.fractal_dim for b in r.body_results])
        else:
            row['Mean_Volume [mm³]'] = None
            row['Mean_Area [mm²]'] = None
            row['Mean_FSR_A'] = None
            row['Mean_FSR_V'] = None
            row['Mean_PR'] = None
            row['Mean_RAD [mm]'] = None
            row['Mean_RF [mm]'] = None
            row['Mean_RSD [mm]'] = None
            row['Mean_Fractal_Dim'] = None

        # Add individual body columns (FSR_A, FSR_V, PR, RAD, RF, RSD, Fractal_Dim)
        for i in range(max_bodies):
            col_name_a = f'FSR_A_Body_{i+1}'
            col_name_v = f'FSR_V_Body_{i+1}'
            col_name_pr = f'PR_Body_{i+1}'
            col_name_rad = f'RAD_Body_{i+1}'
            col_name_rf = f'RF_Body_{i+1}'
            col_name_rsd = f'RSD_Body_{i+1}'
            col_name_fd = f'Fractal_Dim_Body_{i+1}'
            if i < len(r.body_results):
                row[col_name_a] = r.body_results[i].fsr_a
                row[col_name_v] = r.body_results[i].fsr_v
                row[col_name_pr] = r.body_results[i].pr
                row[col_name_rad] = r.body_results[i].rad
                row[col_name_rf] = r.body_results[i].rf
                row[col_name_rsd] = r.body_results[i].rsd
                row[col_name_fd] = r.body_results[i].fractal_dim
            else:
                row[col_name_a] = None
                row[col_name_v] = None
                row[col_name_pr] = None
                row[col_name_rad] = None
                row[col_name_rf] = None
                row[col_name_rsd] = None
                row[col_name_fd] = None

        rows.append(row)

    return pd.DataFrame(rows)


def create_raw_body_table(results: list[STLAnalysisResult]) -> pd.DataFrame:
    """Create a raw data table with one row per body.

    Creates a comprehensive table with all body analysis data including
    specimen info, geometric properties, and roughness metrics.

    Args:
        results: List of STLAnalysisResult objects.

    Returns:
        DataFrame with one row per body across all specimens.
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        # Use full filename (without extension) as specimen name
        specimen_name = r.input_file.stem

        # Get specimen data
        t_measured = r.specimen_data.t_measured if r.specimen_data else None
        sig_h = r.specimen_data.sig_h if r.specimen_data else None
        U = r.specimen_data.U if r.specimen_data else None
        U_d = r.specimen_data.U_d if r.specimen_data else None
        N50 = r.specimen_data.N50 if r.specimen_data else None

        # Create one row per body
        for body in r.body_results:
            rows.append({
                'Specimen': specimen_name,
                'Body_ID': body.body_index,
                't [mm]': t_measured,
                'sig_h [MPa]': sig_h,
                'U [J/m²]': U,
                'U_d [J/m³]': U_d,
                'N50': N50,
                't_calc [mm]': body.thickness,
                'Volume_total [mm³]': body.volume_total,
                'Volume_theoretical [mm³]': body.volume_theoretical,
                'Area_total [mm²]': body.area_total,
                'Area_slice1 [mm²]': body.area_slice1,
                'Area_slice2 [mm²]': body.area_slice2,
                'Perimeter1 [mm]': body.perimeter_slice1,
                'Perimeter2 [mm]': body.perimeter_slice2,
                'Fracture_surface_area [mm²]': body.fracture_surface_area,
                'Area_theoretical [mm²]': body.area_theoretical,
                'FSR_A': body.fsr_a,
                'FSR_V': body.fsr_v,
                'PR': body.pr,
                'RAD [mm]': body.rad,
                'RF [mm]': body.rf,
                'RSD [mm]': body.rsd,
                'Fractal_Dim': body.fractal_dim,
                'Slice_distance [mm]': body.distance_between_slices,
                'Max_Z_distance [mm]': body.max_distance_z,
            })

    return pd.DataFrame(rows)


def create_per_thickness_tables(
    results: list[STLAnalysisResult],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Create separate CSV files for each glass thickness.

    Groups bodies by their nominal thickness (extracted from specimen name)
    and creates one CSV file per thickness with all body data.

    Args:
        results: List of STLAnalysisResult objects.
        output_dir: Directory to save the CSV files.

    Returns:
        Dictionary mapping thickness strings to DataFrames.
    """
    if not results:
        return {}

    # Group results by thickness (first number in specimen name like "8.100.B.04")
    thickness_groups: dict[str, list[dict]] = {}

    for r in results:
        # Use full filename (without extension) for display
        full_name = r.input_file.stem

        # Extract thickness from pattern (e.g., "8" from "8.100.B.04")
        base_specimen = extract_specimen_name(r.input_file.name)
        parts = base_specimen.split('.') if base_specimen else []
        if parts and parts[0].isdigit():
            thickness_key = f"{parts[0]}mm"
        else:
            thickness_key = "unknown"

        if thickness_key not in thickness_groups:
            thickness_groups[thickness_key] = []

        # Get specimen data
        t_measured = r.specimen_data.t_measured if r.specimen_data else None
        sig_h = r.specimen_data.sig_h if r.specimen_data else None
        U = r.specimen_data.U if r.specimen_data else None
        U_d = r.specimen_data.U_d if r.specimen_data else None
        N50 = r.specimen_data.N50 if r.specimen_data else None

        # Add each body as a row
        for body in r.body_results:
            thickness_groups[thickness_key].append({
                'Specimen': full_name,
                'Body_ID': body.body_index,
                't [mm]': t_measured,
                'sig_h [MPa]': sig_h,
                'U [J/m²]': U,
                'U_d [J/m³]': U_d,
                'N50': N50,
                't_calc [mm]': body.thickness,
                'Volume_total [mm³]': body.volume_total,
                'Volume_theoretical [mm³]': body.volume_theoretical,
                'Area_total [mm²]': body.area_total,
                'Area_slice1 [mm²]': body.area_slice1,
                'Area_slice2 [mm²]': body.area_slice2,
                'Perimeter1 [mm]': body.perimeter_slice1,
                'Perimeter2 [mm]': body.perimeter_slice2,
                'Fracture_surface_area [mm²]': body.fracture_surface_area,
                'Area_theoretical [mm²]': body.area_theoretical,
                'FSR_A': body.fsr_a,
                'FSR_V': body.fsr_v,
                'PR': body.pr,
                'RAD [mm]': body.rad,
                'RF [mm]': body.rf,
                'RSD [mm]': body.rsd,
                'Fractal_Dim': body.fractal_dim,
                'Slice_distance [mm]': body.distance_between_slices,
                'Max_Z_distance [mm]': body.max_distance_z,
            })

    # Create DataFrames and save CSV files
    dataframes = {}
    for thickness_key, rows in thickness_groups.items():
        df = pd.DataFrame(rows)
        dataframes[thickness_key] = df

        # Save CSV file
        csv_path = output_dir / f"bodies-{thickness_key}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  - bodies-{thickness_key}.csv ({len(rows)} bodies)")

    return dataframes


def extract_position_from_filename(filename: str) -> Optional[str]:
    """Extract position code from STL filename.

    Expects filenames ending with _[position].stl where position is m, ol, or ur.
    Examples: "8.100.B.04_m.stl" -> "m", "8.100.B.04_ol.stl" -> "ol"

    Args:
        filename: The STL filename.

    Returns:
        Position code (m, ol, ur) or None if not found.
    """
    import re
    name = Path(filename).stem
    # Match _position at the end (m, ol, ur)
    match = re.search(r'_([a-z]+)$', name, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def create_per_thickness_position_tables(
    results: list[STLAnalysisResult],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Create separate CSV files grouped by thickness and position.

    Groups bodies by their nominal thickness (from specimen name) and position
    (from filename suffix like _m, _ol, _ur).

    Creates files like bodies-8mm-m.csv, bodies-8mm-ol.csv

    Args:
        results: List of STLAnalysisResult objects.
        output_dir: Directory to save the CSV files.

    Returns:
        Dictionary mapping group keys to DataFrames.
    """
    if not results:
        return {}

    # Group results by thickness + position
    groups: dict[str, list[dict]] = {}

    for r in results:
        # Use full filename (without extension) for display
        full_name = r.input_file.stem

        # Extract thickness from pattern (e.g., "8" from "8.100.B.04")
        base_specimen = extract_specimen_name(r.input_file.name)
        parts = base_specimen.split('.') if base_specimen else []
        thickness = parts[0] if parts and parts[0].isdigit() else "unknown"

        # Extract position from filename suffix (e.g., "m" from "8.100.B.04_m.stl")
        position = extract_position_from_filename(r.input_file.name)
        if position:
            group_key = f"{thickness}mm-{position}"
        else:
            group_key = f"{thickness}mm-unknown"

        if group_key not in groups:
            groups[group_key] = []

        # Get specimen data
        t_measured = r.specimen_data.t_measured if r.specimen_data else None
        sig_h = r.specimen_data.sig_h if r.specimen_data else None
        U = r.specimen_data.U if r.specimen_data else None
        U_d = r.specimen_data.U_d if r.specimen_data else None
        N50 = r.specimen_data.N50 if r.specimen_data else None

        # Add each body as a row
        for body in r.body_results:
            groups[group_key].append({
                'Specimen': full_name,
                'Body_ID': body.body_index,
                't [mm]': t_measured,
                'sig_h [MPa]': sig_h,
                'U [J/m²]': U,
                'U_d [J/m³]': U_d,
                'N50': N50,
                't_calc [mm]': body.thickness,
                'Volume_total [mm³]': body.volume_total,
                'Volume_theoretical [mm³]': body.volume_theoretical,
                'Area_total [mm²]': body.area_total,
                'Area_slice1 [mm²]': body.area_slice1,
                'Area_slice2 [mm²]': body.area_slice2,
                'Perimeter1 [mm]': body.perimeter_slice1,
                'Perimeter2 [mm]': body.perimeter_slice2,
                'Fracture_surface_area [mm²]': body.fracture_surface_area,
                'Area_theoretical [mm²]': body.area_theoretical,
                'FSR_A': body.fsr_a,
                'FSR_V': body.fsr_v,
                'PR': body.pr,
                'RAD [mm]': body.rad,
                'RF [mm]': body.rf,
                'RSD [mm]': body.rsd,
                'Fractal_Dim': body.fractal_dim,
                'Slice_distance [mm]': body.distance_between_slices,
                'Max_Z_distance [mm]': body.max_distance_z,
            })

    # Create DataFrames and save CSV files
    dataframes = {}
    for group_key, rows in sorted(groups.items()):
        df = pd.DataFrame(rows)
        dataframes[group_key] = df

        # Save CSV file
        csv_path = output_dir / f"bodies-{group_key}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  - bodies-{group_key}.csv ({len(rows)} bodies)")

    return dataframes


def create_per_specimen_csvs(
    results: list[STLAnalysisResult],
    output_dir: Path,
) -> None:
    """Create individual CSV files for each specimen.

    Each specimen gets its own CSV file with all body data, using the same
    format as raw_body_data but grouped by specimen.

    Args:
        results: List of STLAnalysisResult objects.
        output_dir: Directory to save the CSV files.
    """
    if not results:
        return

    # Create specimens subdirectory
    specimens_dir = output_dir / "specimens"
    specimens_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        # Use full filename (without extension) as specimen name
        specimen_name = r.input_file.stem

        # Bodies are already filtered at source
        if not r.body_results:
            continue

        # Get specimen data
        t_measured = r.specimen_data.t_measured if r.specimen_data else None
        sig_h = r.specimen_data.sig_h if r.specimen_data else None
        U = r.specimen_data.U if r.specimen_data else None
        U_d = r.specimen_data.U_d if r.specimen_data else None
        N50 = r.specimen_data.N50 if r.specimen_data else None

        rows = []
        for body in r.body_results:
            rows.append({
                'Specimen': specimen_name,
                'Body_ID': body.body_index,
                't [mm]': t_measured,
                'sig_h [MPa]': sig_h,
                'U [J/m²]': U,
                'U_d [J/m³]': U_d,
                'N50': N50,
                't_calc [mm]': body.thickness,
                'Volume_total [mm³]': body.volume_total,
                'Volume_theoretical [mm³]': body.volume_theoretical,
                'Area_total [mm²]': body.area_total,
                'Area_slice1 [mm²]': body.area_slice1,
                'Area_slice2 [mm²]': body.area_slice2,
                'Perimeter1 [mm]': body.perimeter_slice1,
                'Perimeter2 [mm]': body.perimeter_slice2,
                'Fracture_surface_area [mm²]': body.fracture_surface_area,
                'Area_theoretical [mm²]': body.area_theoretical,
                'FSR_A': body.fsr_a,
                'FSR_V': body.fsr_v,
                'PR': body.pr,
                'RAD [mm]': body.rad,
                'RF [mm]': body.rf,
                'RSD [mm]': body.rsd,
                'Fractal_Dim': body.fractal_dim,
                'Slice_distance [mm]': body.distance_between_slices,
                'Max_Z_distance [mm]': body.max_distance_z,
            })

        df = pd.DataFrame(rows)
        csv_path = specimens_dir / f"{specimen_name}.csv"
        df.to_csv(csv_path, index=False)

    print(f"  - specimens/ ({len(results)} specimen files)")


# =============================================================================
# CLI Commands
# =============================================================================

@stl_app.command("analyze-file")
def cmd_analyze_file(
    input_file: Path = typer.Argument(..., help="Path to the STL file to analyze."),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory."),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip saving PNG plots."),
    no_html: bool = typer.Option(False, "--no-html", help="Skip saving HTML visualization."),
    no_csv: bool = typer.Option(False, "--no-csv", help="Skip saving CSV results."),
    data_only: bool = typer.Option(False, "--data-only", "-d", help="Data only mode (skip all visualization, fastest)."),
    z_offset_lower: float = typer.Option(0.1, "--z-lower", help="Z offset from lower intersection (mm)."),
    z_offset_upper: float = typer.Option(0.05, "--z-upper", help="Z offset from upper intersection (mm)."),
    min_volume: float = typer.Option(1.0, "--min-volume", help="Minimum volume threshold (mm³)."),
) -> None:
    """Analyze a single STL file containing fragment meshes."""
    result = analyze_stl_file(
        input_file=input_file,
        output_dir=output_dir,
        save_plots=not no_plots,
        save_html=not no_html,
        save_excel=not no_csv,
        off_screen=True,
        z_offset_lower=z_offset_lower,
        z_offset_upper=z_offset_upper,
        min_volume=min_volume,
        data_only=data_only,
    )
    print(f"\nAnalyzed {len(result.body_results)} bodies.")


@stl_app.command("analyze-folder")
def cmd_analyze_folder(
    input_dir: Path = typer.Argument(..., help="Directory containing STL files."),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory."),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip saving PNG plots."),
    no_html: bool = typer.Option(False, "--no-html", help="Skip saving HTML visualizations."),
    no_combined_excel: bool = typer.Option(False, "--no-csv", help="Skip combined CSV files."),
    data_only: bool = typer.Option(False, "--data-only", "-d", help="Data only mode (skip all visualization, fastest)."),
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Process files in parallel (best with --data-only)."),
    max_workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Max parallel workers (default: CPU count)."),
    no_specimens: bool = typer.Option(False, "--no-specimens", help="Skip fetching specimen data from database."),
    z_offset_lower: float = typer.Option(0.1, "--z-lower", help="Z offset from lower intersection (mm)."),
    z_offset_upper: float = typer.Option(0.05, "--z-upper", help="Z offset from upper intersection (mm)."),
    min_volume: float = typer.Option(1.0, "--min-volume", help="Minimum volume threshold (mm³)."),
) -> None:
    """Analyze all STL files in a folder.

    Extracts specimen names from filenames (e.g., '8.100.B.04.stl') and fetches
    corresponding specimen data (t_m, sig_h, U, U_d, N50) from the database.

    Creates CSV files:
    - combined_results.csv: Detailed per-body data
    - specimen_summary.csv: One row per specimen with t_m, t_calc, sig_h, U, U_d, N50, FSR, PR, RAD
    - raw_body_data.csv: One row per body with all data
    - bodies-Xmm.csv: Per-thickness files (e.g., bodies-4mm.csv, bodies-8mm.csv)
    """
    results = analyze_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        save_plots=not no_plots,
        save_html=not no_html,
        save_combined_excel=not no_combined_excel,
        off_screen=True,
        z_offset_lower=z_offset_lower,
        z_offset_upper=z_offset_upper,
        min_volume=min_volume,
        data_only=data_only,
        parallel=parallel,
        max_workers=max_workers,
        fetch_specimens=not no_specimens,
    )
    total_bodies = sum(len(r.body_results) for r in results)
    print(f"\nAnalyzed {len(results)} files with {total_bodies} total bodies.")

    # Print summary table
    if results and any(r.specimen_data for r in results):
        print("\nSpecimen Summary:")
        print("-" * 80)
        print(f"{'Specimen':<15} {'t_m [mm]':>10} {'t_calc [mm]':>12} {'sig_h [MPa]':>12} {'N50':>8} {'Bodies':>8}")
        print("-" * 80)
        for r in results:
            spec_name = extract_specimen_name(r.input_file.name) or r.specimen_name
            t_m = f"{r.specimen_data.t_measured:.2f}" if r.specimen_data and r.specimen_data.t_measured else "N/A"
            t_calc = f"{r.mean_thickness:.2f}" if r.mean_thickness else "N/A"
            sig_h = f"{r.specimen_data.sig_h:.1f}" if r.specimen_data and r.specimen_data.sig_h else "N/A"
            n50 = f"{r.specimen_data.N50:.0f}" if r.specimen_data and r.specimen_data.N50 else "N/A"
            print(f"{spec_name:<15} {t_m:>10} {t_calc:>12} {sig_h:>12} {n50:>8} {len(r.body_results):>8}")


@stl_app.command("test-clip-caps")
def cmd_test_clip_caps() -> None:
    """Test whether PyVista adds cap faces when clipping meshes.

    Creates a 1x1x1 cube and a 2x1x1 box clipped in half.
    If PyVista adds caps, both should have the same surface area (6.0).
    """
    print("Testing PyVista clip behavior...")
    print("=" * 60)

    # Test 1: Create a unit cube (1x1x1)
    cube = pv.Box(bounds=(0, 1, 0, 1, 0, 1))
    cube_area = cube.area
    cube_volume = cube.volume
    print(f"\n1. Unit Cube (1x1x1):")
    print(f"   Surface area: {cube_area:.6f} (expected: 6.0)")
    print(f"   Volume:       {cube_volume:.6f} (expected: 1.0)")

    # Test 2: Create a 2x1x1 box and clip it in the middle along X
    box_2x1x1 = pv.Box(bounds=(0, 2, 0, 1, 0, 1))
    print(f"\n2. Box (2x1x1) before clipping:")
    print(f"   Surface area: {box_2x1x1.area:.6f} (expected: 10.0)")
    print(f"   Volume:       {box_2x1x1.volume:.6f} (expected: 2.0)")

    # Clip at x=1 (middle of the box)
    clipped_box = box_2x1x1.clip(normal='x', origin=(1, 0, 0), invert=False)
    clipped_area = clipped_box.area
    clipped_volume = clipped_box.volume
    print(f"\n3. Box (2x1x1) clipped at x=1:")
    print(f"   Surface area: {clipped_area:.6f}")
    print(f"   Volume:       {clipped_volume:.6f} (expected: 1.0)")

    # Test 3: Clip in Z direction (more similar to actual usage)
    box_z = pv.Box(bounds=(0, 1, 0, 1, 0, 2))
    print(f"\n4. Box (1x1x2) before clipping:")
    print(f"   Surface area: {box_z.area:.6f} (expected: 10.0)")
    print(f"   Volume:       {box_z.volume:.6f} (expected: 2.0)")

    clipped_z = box_z.clip(normal='z', origin=(0, 0, 1), invert=False)
    print(f"\n5. Box (1x1x2) clipped at z=1 (invert=False, keeping z>1):")
    print(f"   Surface area: {clipped_z.area:.6f}")
    print(f"   Volume:       {clipped_z.volume:.6f}")

    clipped_z_inv = box_z.clip(normal='z', origin=(0, 0, 1), invert=True)
    print(f"\n6. Box (1x1x2) clipped at z=1 (invert=True, keeping z<1):")
    print(f"   Surface area: {clipped_z_inv.area:.6f}")
    print(f"   Volume:       {clipped_z_inv.volume:.6f}")

    # Test 4: Double clip (like in our actual code)
    box_double = pv.Box(bounds=(0, 1, 0, 1, 0, 3))
    print(f"\n7. Box (1x1x3) before clipping:")
    print(f"   Surface area: {box_double.area:.6f} (expected: 14.0)")
    print(f"   Volume:       {box_double.volume:.6f} (expected: 3.0)")

    # Clip between z=1 and z=2 (keeping middle section)
    double_clipped = box_double.clip(normal='z', origin=(0, 0, 1), invert=False)
    double_clipped = double_clipped.clip(normal='z', origin=(0, 0, 2), invert=True)
    print(f"\n8. Box (1x1x3) clipped between z=1 and z=2:")
    print(f"   Surface area: {double_clipped.area:.6f}")
    print(f"   Volume:       {double_clipped.volume:.6f} (expected: 1.0)")
    print(f"   Expected if caps added: 6.0 (4 sides + 2 caps)")
    print(f"   Expected if no caps:    4.0 (4 sides only)")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    caps_added = abs(clipped_area - cube_area) < 0.001
    if caps_added:
        print("  PyVista DOES add cap faces when clipping.")
        print("  -> Clipped mesh area includes caps (would need subtraction).")
    else:
        print("  PyVista does NOT add cap faces when clipping.")
        print(f"  Clipped area ({clipped_area:.2f}) != Cube area ({cube_area:.2f})")
        print("  -> Clipped mesh area IS the fracture surface directly.")

    print("\n  Current implementation: Using clipped area directly (no subtraction).")


@stl_app.command("fractal-debug")
def cmd_fractal_debug(
    input_file: Path = typer.Argument(..., help="Path to the STL file to analyze."),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for debug files."),
    z_offset_lower: float = typer.Option(0.1, "--z-lower", help="Z offset from lower intersection (mm)."),
    z_offset_upper: float = typer.Option(0.05, "--z-upper", help="Z offset from upper intersection (mm)."),
    min_volume: float = typer.Option(1.0, "--min-volume", help="Minimum volume threshold (mm³)."),
    body_index: Optional[int] = typer.Option(None, "--body", "-b", help="Analyze only this body index (1-based). If not specified, analyzes all bodies."),
) -> None:
    """Run fractal dimension analysis with comprehensive debug output.

    Creates a 'fractal-debug' folder containing:
    - mesh_info.txt: Fracture surface mesh statistics
    - box_counting_data.csv: Raw L, N, log values for each step
    - results.txt: Fractal dimension D, R² value, interpretation
    - loglog_plot.png: Log-log plot with regression line
    - scaling_plots.png: N vs L in linear and log scales
    - fracture_surface.png: 3D view of the clipped fracture surface
    - boxes_stepXX_L*.png: 3D visualizations of boxes at different scales
    """
    input_file = Path(input_file)

    if output_dir is None:
        output_dir = input_file.parent / "fractal-debug"
    else:
        output_dir = Path.joinpath(Path(output_dir), input_file.stem + "-fractal-debug")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fractal Dimension Debug Analysis")
    print(f"=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output dir: {output_dir}")
    print(f"Z offsets: lower={z_offset_lower}, upper={z_offset_upper}")
    print(f"Min volume: {min_volume}")
    print()

    # Read mesh and split into bodies
    mesh = pv.read(str(input_file))
    bodies = mesh.split_bodies()
    print(f"Found {len(bodies)} bodies in STL file")

    # Determine which bodies to analyze
    if body_index is not None:
        if body_index < 1 or body_index > len(bodies):
            print(f"Error: Body index {body_index} out of range (1-{len(bodies)})")
            raise typer.Exit(1)
        body_indices = [body_index]
    else:
        body_indices = list(range(1, len(bodies) + 1))

    results_summary = []

    for idx in body_indices:
        body = bodies[idx - 1]  # 0-based indexing
        print(f"\n--- Body {idx} ---")

        # Prepare mesh
        if not np.all(body.celltypes == 5):
            body = body.triangulate()
        body = body.extract_surface().triangulate()

        # Check volume
        volume = body.volume
        if volume < min_volume:
            print(f"  Skipping: volume {volume:.2f} < min_volume {min_volume}")
            continue

        # Get intersection points for Z bounds
        centroid = np.mean(body.points, axis=0)
        points_up, _ = body.ray_trace(centroid, centroid + np.array([0, 0, 1000]))
        points_down, _ = body.ray_trace(centroid, centroid + np.array([0, 0, -1000]))

        if len(points_up) == 0 or len(points_down) == 0:
            print(f"  Skipping: Could not find Z intersection points")
            continue

        intersection_points = np.vstack((points_up, points_down))
        intersection_points = intersection_points[np.argsort(intersection_points[:, 2])]

        if len(intersection_points) < 2:
            print(f"  Skipping: Need 2 intersection points, found {len(intersection_points)}")
            continue

        z_lower = intersection_points[0][2] + z_offset_lower
        z_upper = intersection_points[1][2] - z_offset_upper

        print(f"  Volume: {volume:.2f} mm³")
        print(f"  Z bounds: [{z_lower:.4f}, {z_upper:.4f}]")

        # Create body-specific debug directory
        body_debug_dir = output_dir / f"body_{idx:02d}"
        body_name = f"body_{idx:02d}"

        # Run debug analysis
        D, n_l_data = compute_fractal_dimension_with_debug(
            body=body,
            z_lower=z_lower,
            z_upper=z_upper,
            debug_dir=body_debug_dir,
            body_name=body_name,
        )

        results_summary.append({
            'body_index': idx,
            'volume': volume,
            'z_lower': z_lower,
            'z_upper': z_upper,
            'fractal_dim': D,
            'n_data_points': len(n_l_data),
        })

    # Save summary
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(output_dir / "summary.csv", index=False)

        print(f"\n" + "=" * 60)
        print(f"Summary")
        print(f"=" * 60)
        for r in results_summary:
            print(f"Body {r['body_index']}: D = {r['fractal_dim']:.4f} ({r['n_data_points']} data points)")

        if len(results_summary) > 1:
            mean_D = np.mean([r['fractal_dim'] for r in results_summary])
            std_D = np.std([r['fractal_dim'] for r in results_summary])
            print(f"\nMean D = {mean_D:.4f} ± {std_D:.4f}")

        print(f"\nDebug output saved to: {output_dir}")


if __name__ == "__main__":
    stl_app()
