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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import panel as pn
import pyvista as pv
import typer

from fracsuite.callbacks import main_callback

stl_app = typer.Typer(help=__doc__, callback=main_callback)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Point3D:
    """A point in 3D space."""
    x: float
    y: float
    z: float

    def to_dict(self) -> dict:
        return {'x': self.x, 'y': self.y, 'z': self.z}

    @classmethod
    def from_dict(cls, d: dict) -> Point3D:
        return cls(x=d['x'], y=d['y'], z=d['z'])

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
    ratio_of_volume: float
    area_mantel_calc: float
    area_mantel: float
    area_theoretical_old: float
    area_theoretical_new: float
    fsr_old: float
    fsr_new: float
    thickness: float
    distance_between_slices: float
    max_distance_z: float


@dataclass
class STLAnalysisResult:
    """Results from analyzing a single STL file."""
    specimen_name: str
    input_file: Path
    output_folder: Path
    body_results: list[BodyAnalysisResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        data = []
        for r in self.body_results:
            data.append({
                'Specimen': self.specimen_name,
                'Body': r.body_index,
                'Volume_total_fragment': r.volume_total,
                'Area_total_fragment': r.area_total,
                'Perimeter1': r.perimeter_slice1,
                'Perimeter2': r.perimeter_slice2,
                'Area_slice1': r.area_slice1,
                'Area_slice2': r.area_slice2,
                'Volume_theoretical': r.volume_theoretical,
                'ratio_of_volume': r.ratio_of_volume,
                'Calculated Area Mantel': r.area_mantel_calc,
                'Area Mantel': r.area_mantel,
                'Area_theoretical_old': r.area_theoretical_old,
                'Area_theoretical_new': r.area_theoretical_new,
                'Fracture Surface Roughness old': r.fsr_old,
                'Fracture Surface Roughness new': r.fsr_new,
                'thickness (vector)': r.thickness,
                'distance between slices': r.distance_between_slices,
                'distance highest - lowest point': r.max_distance_z,
            })
        return pd.DataFrame(data)


# =============================================================================
# Utility Functions
# =============================================================================

def distance_3d(p1: dict | Point3D, p2: dict | Point3D) -> float:
    """Calculate the Euclidean distance between two 3D points.

    Args:
        p1: First point (dict with x, y, z keys or Point3D).
        p2: Second point (dict with x, y, z keys or Point3D).

    Returns:
        The distance between the two points, or infinity if either is None.
    """
    if p1 is None or p2 is None:
        return float('inf')

    if isinstance(p1, Point3D):
        p1 = p1.to_dict()
    if isinstance(p2, Point3D):
        p2 = p2.to_dict()

    return math.sqrt(
        (p1['x'] - p2['x']) ** 2 +
        (p1['y'] - p2['y']) ** 2 +
        (p1['z'] - p2['z']) ** 2
    )


def compute_center_point(points: list[dict]) -> dict:
    """Compute the center point (centroid) of a polygon.

    Args:
        points: List of points, each a dict with 'x', 'y', 'z' keys.

    Returns:
        A dict with the center point coordinates.

    Raises:
        ValueError: If any point is missing required coordinates.
    """
    if any('x' not in point or 'y' not in point or 'z' not in point for point in points):
        raise ValueError("Points should have 'x', 'y' and 'z' coordinates")

    n = len(points)
    center_x = sum(point['x'] for point in points) / n
    center_y = sum(point['y'] for point in points) / n
    center_z = sum(point['z'] for point in points) / n
    return {'x': center_x, 'y': center_y, 'z': center_z}


def find_k_nearest_neighbors(point: dict, points: list[dict], k: int) -> list[dict]:
    """Find the k nearest neighbors of a point.

    Args:
        point: The reference point.
        points: List of candidate points.
        k: Number of neighbors to find.

    Returns:
        List of the k nearest points.
    """
    distances = [(distance_3d(point, p), p) for p in points]
    distances.sort(key=lambda x: x[0])
    return [p for _, p in distances[:k]]


def polygon_area_xy(points: list[dict]) -> float:
    """Calculate the area of a polygon in the XY plane using the shoelace formula.

    Note: The polygon is assumed to be in the XY plane (Z coordinate ignored).

    Args:
        points: List of points defining the polygon vertices.

    Returns:
        The area of the polygon.
    """
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i]['x'] * points[j]['y']
        area -= points[j]['x'] * points[i]['y']
    return abs(area) / 2.0


def slice_mesh_at_z(mesh: pv.PolyData, z_values: list[float]) -> list[pv.PolyData]:
    """Create slices of a mesh at specific Z values.

    Args:
        mesh: The PyVista mesh to slice.
        z_values: List of Z coordinates where slices should be created.

    Returns:
        List of sliced meshes.
    """
    slices = []
    for z in z_values:
        slice_mesh = mesh.slice(normal='z', origin=(0, 0, z))
        if not slice_mesh.is_all_triangles:
            slice_mesh = slice_mesh.triangulate()
        slices.append(slice_mesh)
    return slices


def cut_and_calculate_surface_area(mesh: pv.PolyData, z_values: list[float]) -> float:
    """Cut a mesh between two Z planes and calculate the surface area.

    Args:
        mesh: The PyVista mesh to cut.
        z_values: Two Z values defining the cutting planes [z_lower, z_upper].

    Returns:
        The surface area of the clipped mesh.
    """
    clipped_mesh = mesh.clip(normal='z', origin=(0, 0, z_values[0]), invert=False)
    clipped_mesh = clipped_mesh.clip(normal='z', origin=(0, 0, z_values[1]), invert=True)

    if not clipped_mesh.is_all_triangles:
        clipped_mesh = clipped_mesh.triangulate()

    return clipped_mesh.area


def sort_points_by_nearest_neighbor(points_list: list[dict]) -> list[dict]:
    """Sort points by nearest neighbor traversal starting from center.

    Args:
        points_list: List of points to sort.

    Returns:
        Sorted list of points.
    """
    if not points_list:
        return []

    center_point = compute_center_point(points_list)
    sorted_points = []
    remaining_points = points_list.copy()

    start_point = min(remaining_points, key=lambda p: distance_3d(p, center_point))
    sorted_points.append(start_point)
    remaining_points.remove(start_point)

    while remaining_points:
        nearest_neighbors = find_k_nearest_neighbors(sorted_points[-1], remaining_points, 1)
        if nearest_neighbors:
            nearest_neighbor = nearest_neighbors[0]
            sorted_points.append(nearest_neighbor)
            remaining_points.remove(nearest_neighbor)
        else:
            break

    return sorted_points


def calculate_perimeter(sorted_points: list[dict]) -> float:
    """Calculate the perimeter of a polygon from sorted points.

    Args:
        sorted_points: List of points sorted in order around the polygon.

    Returns:
        The perimeter length.
    """
    if len(sorted_points) < 2:
        return 0.0

    perimeter = sum(
        distance_3d(sorted_points[i], sorted_points[i + 1])
        for i in range(len(sorted_points) - 1)
    )
    perimeter += distance_3d(sorted_points[-1], sorted_points[0])
    return perimeter


def polydata_to_point_list(polydata: pv.PolyData) -> list[dict]:
    """Convert PyVista PolyData points to a list of dicts.

    Args:
        polydata: PyVista PolyData object.

    Returns:
        List of point dicts with 'x', 'y', 'z' keys.
    """
    points = polydata.points
    return [{'x': p[0], 'y': p[1], 'z': p[2]} for p in points]


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
    body = body.extract_surface().triangulate().clean()

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

    # Calculate fracture surface area
    surface_area_cutted = cut_and_calculate_surface_area(body, z_values)

    # Create slices
    single_slice1 = body.slice(normal=[0, 0, 1], origin=[0, 0, z1])
    single_slice2 = body.slice(normal=[0, 0, 1], origin=[0, 0, z2])

    # Max Z distance
    min_z = np.min(body.points[:, 2])
    max_z = np.max(body.points[:, 2])
    max_distance_z = max_z - min_z

    # Convert slices to point lists and sort
    points_list1 = polydata_to_point_list(pv.PolyData(single_slice1))
    points_list2 = polydata_to_point_list(pv.PolyData(single_slice2))

    sorted_points1 = sort_points_by_nearest_neighbor(points_list1)
    sorted_points2 = sort_points_by_nearest_neighbor(points_list2)

    # Calculate perimeters
    perimeter1 = calculate_perimeter(sorted_points1)
    perimeter2 = calculate_perimeter(sorted_points2)

    # Calculate slice areas
    area_slice1 = polygon_area_xy(sorted_points1) if sorted_points1 else 0.0
    area_slice2 = polygon_area_xy(sorted_points2) if sorted_points2 else 0.0

    # Characteristic values
    v_theo = ((area_slice1 + area_slice2) / 2) * thickness
    ratio_of_volume = volume_total / v_theo if v_theo > 0 else 0.0
    area_mantel_calc = area_total - (area_slice1 + area_slice2)
    area_theo_old = ((perimeter1 + perimeter2) / 2) * thickness
    area_theo_new = ((perimeter1 + perimeter2) / 2) * t_theo
    fsr_old = area_mantel_calc / area_theo_old if area_theo_old > 0 else 0.0
    fsr_new = surface_area_cutted / area_theo_new if area_theo_new > 0 else 0.0

    result = BodyAnalysisResult(
        body_index=body_index,
        volume_total=volume_total,
        area_total=area_total,
        perimeter_slice1=perimeter1,
        perimeter_slice2=perimeter2,
        area_slice1=area_slice1,
        area_slice2=area_slice2,
        volume_theoretical=v_theo,
        ratio_of_volume=ratio_of_volume,
        area_mantel_calc=area_mantel_calc,
        area_mantel=surface_area_cutted,
        area_theoretical_old=area_theo_old,
        area_theoretical_new=area_theo_new,
        fsr_old=fsr_old,
        fsr_new=fsr_new,
        thickness=thickness,
        distance_between_slices=t_theo,
        max_distance_z=max_distance_z,
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

    Returns:
        STLAnalysisResult containing all body analysis results.
    """
    input_file = Path(input_file)
    specimen_name = input_file.stem

    if output_dir is None:
        output_dir = input_file.parent / f"{specimen_name}_output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read mesh and split into bodies
    mesh = pv.read(str(input_file))
    bodies = mesh.split_bodies()

    result = STLAnalysisResult(
        specimen_name=specimen_name,
        input_file=input_file,
        output_folder=output_dir,
    )

    # Plotters for combined visualizations
    tp3D = pv.Plotter(off_screen=off_screen)
    tpslice = pv.Plotter(off_screen=off_screen)

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

        color = DEFAULT_COLOR_MAP.get(i, 'grey')

        if save_plots:
            _save_body_plots(
                specimen_name=specimen_name,
                body_index=i,
                analysis=analysis,
                vis_data=vis_data,
                color=color,
                output_dir=output_dir,
                off_screen=off_screen,
            )

        # Add to combined plotters
        tp3D.add_mesh(
            vis_data['body'],
            color=color,
            label=f'Body {i} (Volume: {analysis.volume_total:.2f})'
        )
        tp3D.add_point_labels(
            vis_data['intersection_points'][1:2],
            [f'{i}'],
            font_size=20,
            point_color='red',
            text_color='black'
        )
        tp3D.add_mesh(vis_data['slice1'])
        tp3D.add_mesh(vis_data['slice2'])

        tpslice.add_mesh(
            vis_data['body'],
            color=color,
            opacity=0.01,
            label=f'Body {i} (Volume: {analysis.volume_total:.2f})'
        )
        tpslice.add_mesh(vis_data['slice1'], color="red")
        tpslice.add_mesh(vis_data['slice2'], color="blue")

        print(f'Body {i} is finished')

    # Finalize combined 3D plot
    tp3D.add_legend()
    tp3D.add_axes()
    tp3D.show_grid()
    tp3D.add_text(specimen_name, position='upper_left', font_size=20)
    tp3D.camera_position = 'xy'
    tp3D.enable_parallel_projection()

    if save_plots:
        screenshot_path = output_dir / f"{specimen_name}_Body_total.png"
        tp3D.screenshot(str(screenshot_path))

    if save_html:
        pane = pn.pane.VTK(tp3D.ren_win, width=1000, height=750)
        html_path = output_dir / f"{specimen_name}_Body_total.html"
        pn.panel(pane).save(str(html_path))

    tp3D.show()

    # Finalize slice plot
    tpslice.camera_position = 'xy'
    tpslice.enable_parallel_projection()
    tpslice.show_grid()
    tpslice.add_text(specimen_name, position='upper_left', font_size=20)

    if save_plots:
        screenshot_path = output_dir / f"{specimen_name}_Total_DifferenceSlices.png"
        tpslice.screenshot(str(screenshot_path))

    tpslice.show()

    # Save Excel
    if save_excel:
        df = result.to_dataframe()
        excel_path = output_dir / f'{specimen_name}_body_data.xlsx'
        df.to_excel(str(excel_path), index=False)

    print(f'Analysis of specimen {specimen_name} is finished')

    return result


def _save_body_plots(
    specimen_name: str,
    body_index: int,
    analysis: BodyAnalysisResult,
    vis_data: dict,
    color: str,
    output_dir: Path,
    off_screen: bool = True,
) -> None:
    """Save individual body visualization plots.

    Args:
        specimen_name: Name of the specimen.
        body_index: Index of the body.
        analysis: Analysis results for this body.
        vis_data: Visualization data dict from analyze_body.
        color: Color for this body.
        output_dir: Directory to save plots.
        off_screen: Whether to run off-screen.
    """
    body = vis_data['body']
    slice1 = vis_data['slice1']
    slice2 = vis_data['slice2']
    vector_thickness = vis_data['vector_thickness']

    # Plot 1: Difference of slices
    p = pv.Plotter(off_screen=off_screen)
    p.add_mesh(body.outline(), color="k")
    p.add_mesh(slice1, color="red")
    p.add_mesh(slice2, color="blue")
    p.add_text(f'{specimen_name}_Body{body_index}_Slices', position='upper_left', font_size=15)
    p.camera_position = 'xy'
    p.enable_parallel_projection()
    p.show()
    p.screenshot(str(output_dir / f"{specimen_name}_Body{body_index}_DifferenceSlices.png"))

    # Plot 2: Vector thickness
    p = pv.Plotter(off_screen=off_screen)
    p.add_mesh(body, color='white', opacity=0.5)
    p.add_mesh(vector_thickness, color='red', line_width=3)
    p.add_text(f'{specimen_name}_Body{body_index}_t={analysis.thickness:.2f}', position='upper_left', font_size=15)
    p.show_grid()
    p.screenshot(str(output_dir / f"{specimen_name}_Body{body_index}_VectorThickness.png"))

    # Plot 3: Vector and slice (lateral view)
    p = pv.Plotter(off_screen=off_screen)
    p.add_mesh(body, color='white', opacity=0.5)
    p.add_mesh(vector_thickness, color='red', line_width=3)
    p.add_mesh(slice1, color="red")
    p.add_mesh(slice2, color="blue")
    p.add_text(f'{specimen_name}_Body{body_index}_t(theo)={analysis.distance_between_slices:.2f}', position='upper_left', font_size=15)
    p.camera_position = 'xz'
    p.enable_parallel_projection()
    p.show_grid()
    p.screenshot(str(output_dir / f"{specimen_name}_Body{body_index}_VectorandSlice.png"))

    # Plot 4: Colored body
    p = pv.Plotter(off_screen=off_screen)
    p.add_mesh(body, color=color, label=f'Body {body_index} (Volume: {analysis.volume_total:.2f})')
    p.add_mesh(slice1)
    p.add_mesh(slice2)
    p.screenshot(str(output_dir / f"{specimen_name}_Body{body_index}_coloured.png"))


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

    results = []
    all_dataframes = []

    for stl_file in stl_files:
        print(f"\nAnalyzing: {stl_file.name}")
        try:
            result = analyze_stl_file(
                input_file=stl_file,
                output_dir=output_dir / f"{stl_file.stem}_output",
                save_plots=save_plots,
                save_html=save_html,
                save_excel=True,
                off_screen=off_screen,
                z_offset_lower=z_offset_lower,
                z_offset_upper=z_offset_upper,
                min_volume=min_volume,
            )
            results.append(result)
            all_dataframes.append(result.to_dataframe())
        except Exception as e:
            print(f"Error analyzing {stl_file.name}: {e}")
            continue

        gc.collect()

    # Save combined Excel
    if save_combined_excel and all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        excel_dir = output_dir / f"{input_dir.name}_excel"
        excel_dir.mkdir(parents=True, exist_ok=True)
        combined_df.to_excel(str(excel_dir / "combined_results.xlsx"), index=False)

    print('\nAnalysis of folder is finished')
    return results


# =============================================================================
# CLI Commands
# =============================================================================

@stl_app.command("analyze-file")
def cmd_analyze_file(
    input_file: Path = typer.Argument(..., help="Path to the STL file to analyze."),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory."),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip saving PNG plots."),
    no_html: bool = typer.Option(False, "--no-html", help="Skip saving HTML visualization."),
    no_excel: bool = typer.Option(False, "--no-excel", help="Skip saving Excel results."),
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
        save_excel=not no_excel,
        off_screen=True,
        z_offset_lower=z_offset_lower,
        z_offset_upper=z_offset_upper,
        min_volume=min_volume,
    )
    print(f"\nAnalyzed {len(result.body_results)} bodies.")


@stl_app.command("analyze-folder")
def cmd_analyze_folder(
    input_dir: Path = typer.Argument(..., help="Directory containing STL files."),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory."),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip saving PNG plots."),
    no_html: bool = typer.Option(False, "--no-html", help="Skip saving HTML visualizations."),
    no_combined_excel: bool = typer.Option(False, "--no-combined-excel", help="Skip combined Excel file."),
    z_offset_lower: float = typer.Option(0.1, "--z-lower", help="Z offset from lower intersection (mm)."),
    z_offset_upper: float = typer.Option(0.05, "--z-upper", help="Z offset from upper intersection (mm)."),
    min_volume: float = typer.Option(1.0, "--min-volume", help="Minimum volume threshold (mm³)."),
) -> None:
    """Analyze all STL files in a folder."""
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
    )
    total_bodies = sum(len(r.body_results) for r in results)
    print(f"\nAnalyzed {len(results)} files with {total_bodies} total bodies.")


if __name__ == "__main__":
    stl_app()
