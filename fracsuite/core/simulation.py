from __future__ import annotations

import os
import pickle
import shutil

import typer
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
            path = path + "_" + str(counter)
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

        simu = Simulation(path, load=load)
        return simu

    @classmethod
    def create(cls, thickness:int, sigma_s:float, boundary: str, splinters: list[Splinter]):
        name = f"{thickness:.0f}-{sigma_s:.0f}-{boundary}"

        simpath = os.path.join(general.simulation_path, name)
        simpath = cls.gen_name(simpath)
        os.makedirs(simpath)

        # create simulation which is effectively a specimen
        simu = cls(simpath, thickness, sigma_s, boundary)

        # put splinters into the simulation folder
        simsplinterpath = simu.splinter_file
        with open(simsplinterpath, "wb") as f:
            pickle.dump(splinters, f)


        return simu

    @property
    def splinters(self) -> list[Splinter]:
        return self.__splinters

    def get_file(self, path):
        return os.path.join(self.path, path)

    @property
    def splinter_file(self):
        return self.get_file("splinters.pkl")


    def __init__(self, path: str, thickness, sigma_s, boundary):
        self.path = path
        self.name = os.path.basename(path)

        # load splinters
        self.__splinters = []
        if os.path.exists(self.splinter_file):
            with open(self.splinter_file, "rb") as f:
                self.__splinters = pickle.load(f)

        self.thickness = int(thickness)
        self.nom_stress = int(sigma_s)
        self.boundary = SpecimenBoundary(boundary)
        self.nbr = int(0)
        self.comment = ""

        if "_" in self.name:
            self.nbr = int(self.name.split("_")[-1])