from __future__ import annotations

import argparse
import glob
import os
from typing import TypeVar
from rich import print
import numpy as np
from fracsuite.scalper.scalpSpecimen import ScalpProject, ScalpSpecimen


descr = """
███████╗ ██████╗ █████╗ ██╗     ██████╗ ███████╗██████╗
██╔════╝██╔════╝██╔══██╗██║     ██╔══██╗██╔════╝██╔══██╗
███████╗██║     ███████║██║     ██████╔╝█████╗  ██████╔╝
╚════██║██║     ██╔══██║██║     ██╔═══╝ ██╔══╝  ██╔══██╗
███████║╚██████╗██║  ██║███████╗██║     ███████╗██║  ██║
╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝
Leon Bohmann     TUD - ISMD - GCC        www.tu-darmstadt.de/glass-cc


ScalpProject:           Create with project file. This will split all measurements into
                        specimens and then into locations.
Specimen:               All measurements on a single specimen.
MeasurementLocation:    One point on the ply that has been measured (3+ Measurements)
Measurement:            A measurement at a location in one specific direction.


This script iterates either one single project or recursively searches for projects in a folder.
For every .scp File, a ScapProject is created, which will handle the reading of data
and also the grouping into Specimens and MeasurementLocations and Measurement respectively.

At the end of the script, all specimens are extracted for better organisation, so that
individual specimen tasks can be performed. For example the calculation of prinicipal stresses
and the mean value of all measurements.

"""

T = TypeVar("T")
def flatten_and_sort(l: list[list[T]], sort_mtd) -> list[T]:
    ret = [item for sublist in l for item in sublist]
    ret.sort(key = sort_mtd)
    return ret

def remove_duplicates(my_list, my_lambda):
    new_list = []
    for item in my_list:
        exists = False
        for ex_item in new_list:
            if my_lambda(ex_item) == my_lambda(item):
                exists = True
                print(f'Found and removed duplicate specimen: {item}')

        if exists:
            continue

        new_list.append(item)

    return new_list


def get_specimens_from_projects(projects: list[ScalpProject]) -> list[ScalpSpecimen]:
    """Extract all specimens in a distinct list from a list of projects.

    Args:
        projects (list[ScalpProject]): List of scalp projects.

    Returns:
        list[Specimen]: Ordered and distinct list of all specimens.
    """
    specimens = flatten_and_sort([x.specimens for x in projects], lambda s: s.name)
    specimens = [x for x in specimens if not x.invalid]
    specimens = remove_duplicates(specimens, lambda t: t.name)

    return specimens

########################################################################################
#   SCRIPT AREA
#
#


# implement parse to make this script callable from outside
parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-project', nargs="?", help='The project to be processed.', default=None)
parser.add_argument('-output', nargs="?", help='Output directory.')
parser.add_argument('-folder', nargs="?", help='If this is passed, analyzes all scalp projects in a folder.', default=None)
parser.add_argument('--mohr', action='store_true', help='Display mohr"sche circle for stress states.')
args = parser.parse_args()

# overwrite global vars
SWITCH_DISPLAY_MOHR_CIRCLE = args.mohr

# input vars
project_file = args.project
output_directory = args.output

output_extension = "txt"
open_output_directory = True

# default project file
if project_file is None and args.folder is None:
    raise Exception("No project file or folder supplied!")
# default project file
if output_directory is None:
    if project_file is not None:
        output_directory = os.path.join(os.path.dirname(project_file), "out")
    elif args.folder is not None:
        output_directory = os.path.join(os.path.dirname(args.folder), "out")

# create the output folder if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

projects: list[ScalpProject] = []

# analyze all projects in the folder
if args.folder is not None:
    project_folder = args.folder
    extension = '.scp'
    files = glob.glob(os.path.join(project_folder, '**', f'*{extension}'), recursive=True)

    for file in files:
        print(f'[green]Analyze[/green] file: {file.replace(project_folder, "")}')
        project = ScalpProject(file)
        projects.append(project)
        for spec in project.specimens:
            print(f'\t{"!!!" if spec.invalid else ""}\t{spec.name:10}: {len(spec.measurementlocations)} Locations ({[f"{x.location_name} ({len(x.measurements)})" for x in spec.measurementlocations]})')
# analyze one project
else:
    project = ScalpProject(project_file)
    projects.append(project)


# write a summary
project_len = len(projects)
measurements_len = np.sum([len(x.measurements) for x in projects])

print(f'Projects       : {project_len}')
print(f'Measurements   : {measurements_len}')

# output all specimens independently from their project
specimens = get_specimens_from_projects(projects)
for specimen in specimens:

    pre = ""
    if not all(not t.invalid for t in specimen.measurementlocations):
        pre = "!!!"

    print(f'\t{pre}\t{specimen.name:10}: {len(specimen.measurementlocations)} Locations ({[(x.location_name, [l.orientation for l in x.measurements]) for x in specimen.measurementlocations]})')


# calculate stresses for specimens
for specimen in specimens:

    print(f'\t{specimen.name:10}: sig_h={specimen.sig_h:.2f} (+- {specimen.sig_h.deviation:.2f})MPa')
    print('\t\t', end="")

    for loc in specimen.measurementlocations:
        print(f'\t{loc.location_name}={loc.stress[0]:.2f}/{loc.stress[1]:.2f}', end = "")

    print()

# write specimen
for specimen in specimens:
    specimen.write_measurements(output_directory, output_extension)