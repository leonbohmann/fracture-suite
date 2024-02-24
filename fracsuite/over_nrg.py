"""
Subapp to plot some things over the energy.
"""

from matplotlib import pyplot as plt
import numpy as np
import typer

from fracsuite.callbacks import main_callback
from fracsuite.core.specimen import Specimen
from fracsuite.splinters import create_filter_function

over_nrg = typer.Typer(callback=main_callback)


@over_nrg.command()
def mean_edges(

):
    """
    Plot the mean edge count over the energy of all specimens.
    """
    filter = create_filter_function("*", needs_scalp=False, needs_splinters=True)
    specimens = Specimen.get_all_by(filter)


    energies = []
    mean_edges = []
    for s in specimens:
        s.load()

        energies.append(s.U_d)
        mean_edge = np.mean([len(s.touching_splinters) for s in s.splinters])
        mean_edges.append(mean_edge)

    energies = np.array(energies)
    mean_edges = np.array(mean_edges)

    fig, axs = plt.subplots()
    axs.plot(energies, mean_edges, 'o')
    axs.set_xlabel('Energy')
    axs.set_ylabel('Mean edge count')
    plt.show()
