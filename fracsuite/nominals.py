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
        if specimen.thickness == 12:
            return False

        return specimen.has_scalp and specimen.nom_stress != 0
    def get_spec(specimen: Specimen) -> Specimen:
        return specimen

    specimens: list[Specimen] = Specimen.get_all_by(has_stress, get_spec, load=True)


    thicknesses = {
            4: {}, 8: {}
        }

    for t in thicknesses:
        for i in range(40, 150, 10):
            thicknesses[t][i] = []

    for spec in specimens:
        thicknesses[spec.thickness][spec.nom_stress].append(np.abs(spec.sig_h))

    sz = FigureSize.ROW1HL
    fig, axs = plt.subplots(figsize=get_fig_width(sz))
    # axs.scatter(nominal_4, scalped_4, marker='x', color='orange', label="4mm")
    # axs.scatter(nominal_8+5, scalped_8, marker='o', color='blue', label="8mm")
    # axs.scatter(nominal_12+10, scalped_12, marker='v', color='green', label="12mm")
    lbs = ["4mm", "8mm"]
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

    axs.set_xlabel("Nenn-Oberflächenvorspannung $\sigma_\\text{S}$ (MPa)")
    axs.set_ylabel("Gemessene $\sigma_\\text{S,meas}$ (MPa)")

    axs.axline((0, 0), slope=1, color="black", linestyle="-")

    axs.set_xlim((50,160))
    axs.set_ylim((50,160))
    axs.legend(bars, lbs, loc='lower right')

    State.output(fig, 'compare_nominal_stress_to_real_stress', figwidth=sz)

@nominals_app.command()
def thickness(exclude_names: str = None):
    """Compares all nominal stresses with real scalped stresses."""
    def has_stress(specimen: Specimen):
        if specimen.thickness == 12:
            return False

        if exclude_names is not None:
            if specimen.name.startswith(exclude_names):
                return False

        return specimen.has_scalp and specimen.measured_thickness != 0 \
            and specimen.thickness != 0
    def get_spec(specimen: Specimen) -> Specimen:
        return specimen

    specimens: list[Specimen] = Specimen.get_all_by(has_stress, get_spec, load=True)

    thicknesses = {
            4: [], 8: []
        }

    for spec in specimens:
        thicknesses[spec.thickness].append(np.abs(spec.measured_thickness))


    print(thicknesses)

    sz = FigureSize.ROW1HL

    fig,axs=plt.subplots(1,len(thicknesses),figsize=get_fig_width(sz))

    for it, nom_thick in enumerate(thicknesses):
        real_thickness = thicknesses[nom_thick] # list of real thicknesses

        ax = axs[it]
        ax.set_title(f"{nom_thick}mm")
        ax.hist(real_thickness, bins=10, label=f'{nom_thick}mm', alpha=0.9)

        # tick format as integer and only integers
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # plot hline at thickness
        ax.axvline(x=nom_thick, color='black', linestyle='--')

    fig.supxlabel("Gemessene Glasdicke (mm)")
    fig.supylabel("Anzahl von Probekörpern")
    # hide x ranges from 4.5 to 7.5 and 8.5 to 11.5
    State.output(fig, 'thickness_distribution', figwidth=sz)
