"""
Nominal comparisons between real and nominal values.
"""

import os
from matplotlib import pyplot as plt
import numpy as np
import typer
from fracsuite.core.plotting import FigureSize, get_fig_width
from fracsuite.state import State
from fracsuite.general import GeneralSettings
from fracsuite.core.specimen import Specimen

nominals_app = typer.Typer(help=__doc__)
general = GeneralSettings.get()

@nominals_app.command()
def stress():
    """Compares all nominal stresses with real scalped stresses."""
    def has_stress(specimen: Specimen):
        return specimen.has_scalp and specimen.nom_stress != 0
    def get_spec(specimen: Specimen) -> Specimen:
        return specimen

    specimens: list[Specimen] = Specimen.get_all_by(has_stress, get_spec, load=True)


    thicknesses = {
            4: {}, 8: {}, 12: {}
        }

    for t in thicknesses:
        for i in range(40, 150, 10):
            thicknesses[t][i] = []

    for spec in specimens:
        thicknesses[spec.thickness][spec.nom_stress].append(np.abs(spec.sig_h))

    fig, axs = plt.subplots(figsize=get_fig_width(FigureSize.ROW1HL))
    # axs.scatter(nominal_4, scalped_4, marker='x', color='orange', label="4mm")
    # axs.scatter(nominal_8+5, scalped_8, marker='o', color='blue', label="8mm")
    # axs.scatter(nominal_12+10, scalped_12, marker='v', color='green', label="12mm")
    lbs = ["4mm", "8mm", "12mm"]
    bars = []
    for it, nom_thick in enumerate(thicknesses):
        bars.append(None)
        for i,nom_sig in enumerate(thicknesses[nom_thick]):
            real_sig = thicknesses[nom_thick][nom_sig]
            nom_sig = np.array(nom_sig)
            bar = plt.errorbar(nom_sig-5+it*5, np.mean(real_sig), yerr=np.std(real_sig), fmt='ovx'[it], color='bgm'[it])
            axs.scatter([nom_sig-5+it*5]*len(real_sig), real_sig, marker='x', color='gray', linewidths=0.5, alpha=0.5)
            if bars[it] is None:
                bars[it] = bar

    axs.set_xlabel("Nominal surface stress [MPa]")
    axs.set_ylabel("Measured surface stress $\sigma_S$ [MPa]")

    axs.axline((0, 0), slope=1, color="black", linestyle="-")

    axs.set_xlim((0,200))
    axs.set_ylim((0,200))
    fig.tight_layout()
    axs.legend(bars, lbs, loc='lower right')

    State.output(fig, 'compare_nominal_stress_to_real_stress', figwidth=FigureSize.ROW1HL)

@nominals_app.command()
def thickness():
    """Compares all nominal stresses with real scalped stresses."""
    def has_stress(specimen: Specimen):
        return specimen.has_scalp and specimen.measured_thickness != 0 \
            and specimen.thickness != 0
    def get_spec(specimen: Specimen) -> Specimen:
        return specimen

    specimens: list[Specimen] = Specimen.get_all_by(has_stress, get_spec, load=True)

    thicknesses = {
            4: [], 8: [], 12: []
        }

    for spec in specimens:
        thicknesses[spec.thickness].append(np.abs(spec.measured_thickness))


    print(thicknesses)


    fig, axs = plt.subplots(1,3, figsize=general.figure_size)
    # axs.scatter(nominal_4, scalped_4, marker='x', color='orange', label="4mm")
    # axs.scatter(nominal_8+5, scalped_8, marker='o', color='blue', label="8mm")
    # axs.scatter(nominal_12+10, scalped_12, marker='v', color='green', label="12mm")
    xspace = 0.3
    yspace = 0.3
    lbs = ["4mm", "8mm", "12mm"]
    bars = []
    for it, nom_thick in enumerate(thicknesses):
        ax = axs[it]
        ax.axline((nom_thick-1, nom_thick-1), slope=1, color="black", linestyle="-")
        ax.set_xlim((nom_thick-xspace, nom_thick+xspace))
        ax.set_ylim((nom_thick-yspace, nom_thick+yspace))
        ax.set_title(f'{nom_thick}mm')

        # show only the nom_thick x label
        ax.set_xticks([nom_thick])

        print(f'{nom_thick}mm: {thicknesses[nom_thick]}')
        real_thick = thicknesses[nom_thick]
        bar = ax.errorbar(nom_thick, np.mean(real_thick), yerr=np.std(real_thick), fmt='ovx'[it], color='bgm'[it])
        ax.scatter(np.full(len(real_thick), nom_thick), real_thick, marker='x', color='gray', linewidths=0.5, alpha=0.5)
        bars.append(bar)


    axs[1].set_xlabel("Nominal glass thickness [mm]")
    axs[0].set_ylabel("Measured glass thickness [mm]")



    fig.legend(bars, lbs)

    fig.tight_layout()


    State.output(fig, 'compare_nominal_to_real_thickness', figwidth=FigureSize.ROW1)