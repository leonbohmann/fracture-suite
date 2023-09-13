import os
from typing import Annotated
import cv2
import numpy as np
import typer
from fracsuite.splinters.splinter import Splinter

from fracsuite.tools.general import GeneralSettings
from fracsuite.tools.helpers import get_color, annotate_image_cbar, write_image
from fracsuite.tools.specimen import fetch_specimens

from rich import print
from rich.progress import Progress, SpinnerColumn, Spinner, track

app = typer.Typer()

general = GeneralSettings()

@app.command()
def roughness_f(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
                regionsize: Annotated[int, typer.Option(help='Size of the region to calculate the roughness on.')] = 200,):
    """Create a contour plot of the roughness on the specimen.

    Args:
        specimen_name (str, optional): Name of the specimens to load.
    """
    def roughness_function(splinters: list[Splinter]):
        return np.mean([splinter.calculate_roughness() for splinter in splinters])
            
    # create contour plot of roughness
    specimen = fetch_specimens([specimen_name], general.base_path)[0]
    
    fig = specimen.splinters.plot_intensity(regionsize, roughness_function, clr_label='Mean roughness')
    
    # with Progress(SpinnerColumn("arc", ), transient=False, ) as progress:
    #     task = progress.add_task("Create intensity plots", total=1, )
    #     # Start your operation here
    #     # Mark the task as complete
    #     progress.update(task, completed=1)
    
    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"fig_roughintensity.{general.plot_extension}")
    fig.savefig(out_path, dpi=500)
    del fig
    os.system(f"start {out_path}")
    print(f"Saved roughness image to '{out_path}'.")
    
@app.command()
def roundness_f(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
                regionsize: Annotated[int, typer.Option(help='Size of the region to calculate the roughness on.')] = 200,):
    """Create a contour plot of the roundness on the specimen.

    Args:
        specimen_name (str, optional): Name of the specimens to load.
    """
    def roundness_function(splinters: list[Splinter]):
        return np.mean([splinter.calculate_roundness() for splinter in splinters])
            
    # create contour plot of roughness
    specimen = fetch_specimens([specimen_name], general.base_path)[0]
    
    fig = specimen.splinters.plot_intensity(regionsize, roundness_function, clr_label='', fig_title='Mean roundness ')
    
    # with Progress(SpinnerColumn("arc", ), transient=False, ) as progress:
    #     task = progress.add_task("Create intensity plots", total=1, )
    #     # Start your operation here
    #     # Mark the task as complete
    #     progress.update(task, completed=1)
    
    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"fig_roundintensity.{general.plot_extension}")
    fig.savefig(out_path, dpi=500)
    del fig
    os.system(f"start {out_path}")
    print(f"Saved roughness image to '{out_path}'.")
    
@app.command()
def intensity(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')],
              regionsize: Annotated[int, typer.Option(help='Size of the region to calculate the roughness on.')] = 200,):
    specimen = fetch_specimens([specimen_name], general.base_path)[0]
    
    with Progress(SpinnerColumn("arc", ), transient=False, ) as progress:
        task = progress.add_task("Create intensity plots", total=1, )
        # Start your operation here
        fig = specimen.splinters.create_intensity_plot(regionsize)
        # Mark the task as complete
        progress.update(task, completed=1)
        
    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"fig_fracintensity.{general.plot_extension}")
    fig.savefig(out_path, dpi=500)
    del fig
    os.system(f"start {out_path}")
    print(f"Saved roughness image to '{out_path}'.")

    

@app.command()
def roughness(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')]):    
    """Plot the roughness of a specimen."""
       
    specimen = fetch_specimens([specimen_name], general.base_path)[0]
    
    
    out_img = specimen.splinters.original_image.copy()
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    
    rough = [splinter.calculate_roughness() for splinter in specimen.splinters.splinters]    
    max_r = np.max(rough)
    min_r = np.min(rough)
    
    print(f"Max roughness: {max_r}")
    print(f"Min roughness: {min_r}")
    print(f"Mean roughness: {np.mean(rough)}")
    print(f"Median roughness: {np.median(rough)}")
    
    # d = np.median(rough) - min_r
    # max_r = np.median(rough) + d
    
    for splinter in track(specimen.splinters.splinters, description="Calculating roughness", transient=True):
        roughness = splinter.calculate_roughness()
        clr = get_color(roughness, min_r, max_r)
        
        cv2.drawContours(out_img, [splinter.contour], 0, clr, -1)
        
    # plot an overlay of the colorbar into the image on the right side
    # colorbar = np.zeros((out_img.shape[0], 50, 3), dtype=np.uint8)
    # for i in range(out_img.shape[0]):
    #     clr = get_color(i, 0, out_img.shape[0])
    #     colorbar[i] = clr
    # out_img = np.concatenate((out_img, colorbar), axis=1)
    out_img = annotate_image_cbar(out_img, "Roughness", min_value=min_r, max_value=max_r)    
    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"roughness.{general.plot_extension}")
    
    write_image(out_img, out_path)
    
    os.system(f"start {out_path}")
    print(f"Saved roughness image to '{out_path}'.")
    
@app.command()
def roundness(specimen_name: Annotated[str, typer.Argument(help='Name of specimens to load')]):    
    """Plot the roundness of a specimen."""
       
    specimen = fetch_specimens([specimen_name], general.base_path)[0]
    
    
    out_img = specimen.splinters.original_image.copy()
    
    rounds = [splinter.calculate_roundness() for splinter in specimen.splinters.splinters]    
    max_r = np.max(rounds)
    min_r = np.min(rounds)
    
    print(f"Max roundness: {max_r}")
    print(f"Min roundness: {min_r}")
    print(f"Mean roundness: {np.mean(rounds)}")
    
    # scale max and min roundness to +- 60% around mean
    max_r = np.mean(rounds) + np.mean(rounds) * 0.6
    min_r = np.mean(rounds) - np.mean(rounds) * 0.6
    
    
    for splinter in track(specimen.splinters.splinters):
        r = splinter.calculate_roundness()
        clr = get_color(r, min_r, max_r)
        
        cv2.drawContours(out_img, [splinter.contour], 0, clr, -1)
        
    out_img = annotate_image_cbar(out_img, "Roundness", min_value=min_r, max_value=max_r)
    out_path = os.path.join(general.base_path, specimen_name, "fracture", "splinter", f"roundness.{general.plot_extension}")
    cv2.imwrite(out_path, out_img)
    
    os.system(f"start {out_path}")
    typer.echo(f"Saved roughness image to '{out_path}'.")