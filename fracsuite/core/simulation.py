from __future__ import annotations

import os
import pickle
import shutil
import numpy as np

import typer
from fracsuite.core.mechanics import U, Ud
from fracsuite.core.specimen import Specimen
from fracsuite.core.specimenprops import SpecimenBoundary
from fracsuite.core.splinter import Splinter
from fracsuite.general import GeneralSettings

general = GeneralSettings.get()

class SimulationException(Exception):
    """Exception for Simulation class."""
    pass

class Simulation:
    """Same as Specimen but in another folder for simulations."""

    @staticmethod
    def gen_name(name):
        counter = 1

        path = os.path.join(general.simulation_path, name)

        while os.path.exists(path):
            path = path.split("_")[0] + "_" + str(counter)
            counter += 1

        return path


    @staticmethod
    def get(name: str | Simulation, load: bool = True, panic: bool = True) -> Simulation:
        """Gets a specimen by name. Raises exception, if not found."""
        if isinstance(name, Simulation):
            return name

        path = os.path.join(general.simulation_path, name)
        if not os.path.isdir(path):
            if panic:
                raise SimulationException(f"Simulation '{name}' not found.")
            else:
                return None

        simu = Simulation(path)
        simu.print_loaded()
        return simu

    @classmethod
    def create(cls, thickness:int, sigma_s:float, boundary: str, splinters: list[Splinter]):
        name = f"{thickness:.0f}-{sigma_s:.0f}-{boundary}"

        simpath = os.path.join(general.simulation_path, name)
        simpath = cls.gen_name(simpath)
        os.makedirs(simpath)

        # create simulation which is effectively a specimen
        simu = cls(simpath)

        # put splinters into the simulation folder
        simsplinterpath = simu.splinter_file
        with open(simsplinterpath, "wb") as f:
            pickle.dump(splinters, f)


        return simu

    def put_splinters(self, splinters):
        # put splinters into the simulation folder
        simsplinterpath = self.splinter_file
        with open(simsplinterpath, "wb") as f:
            pickle.dump(splinters, f)

    @property
    def splinters(self) -> list[Splinter]:
        return self.__splinters

    def get_file(self, path):
        return os.path.join(self.path, path)

    @property
    def splinter_file(self):
        return self.get_file("splinters.pkl")

    def print_loaded(self):

        print(f"Loaded {self.name:>15}"
                    f': t={self.thickness:>5.2f}mm, U={U(self.nom_stress, self.thickness):>7.2f}J/mm², U_d={Ud(self.nom_stress):>9.2f}J/mm³, σ_s={self.nom_stress:>7.2f}MPa')

    def __init__(self, path: str, realsize = (500,500)):
        self.path = path
        self.name = os.path.basename(path)

        if "_" in self.name:
            self.nbr = int(self.name.split("_")[-1])
            self.name = self.name.split("_")[0]
            self.fullname = self.name + "_" + str(self.nbr)
        else:
            self.nbr = int(0)
            self.fullname = self.name

        # load splinters
        self.__splinters: list[Splinter] = []
        if os.path.exists(self.splinter_file):
            with open(self.splinter_file, "rb") as f:
                self.__splinters = pickle.load(f)

        # remove all splinters whose centroid is closer than 1 cm to the edge
        delta_edge = 10
        self.__splinters = [s for s in self.__splinters
                            if  delta_edge < s.centroid_mm[0] < realsize[0] - delta_edge
                            and delta_edge < s.centroid_mm[1] < realsize[1] - delta_edge]

        # or within a 2cm radius to the impact point
        delta_impact = 20
        self.__splinters = [s for s in self.__splinters if np.linalg.norm(np.array(s.centroid_mm) - np.array((50,50))) > delta_impact]


        print(self.name)
        thickness, sigma_s, boundary = self.name.split("-")

        self.thickness = int(thickness)
        self.nom_stress = int(sigma_s)
        self.boundary = SpecimenBoundary(boundary)
        self.comment = ""

        if "_" in self.name:
            self.nbr = int(self.name.split("_")[-1])