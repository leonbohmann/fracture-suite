from __future__ import annotations

import json
import os
import cv2
import numpy as np

from rich import print
from rich.pretty import pretty_repr


class GeneralSettings:
    __create_key = object()

    sub_outpath: str = ""
    sub_specimen: str = ""

    def get() -> GeneralSettings:
        """Creates a new instance of GeneralSettings or returns the existing one and returns it."""
        global __global_setting
        if 'general' in globals():
            global general
        else:
            global general
            general = GeneralSettings(GeneralSettings.__create_key)

        return general

    def __init__(self, key) -> None:
        assert key == self.__create_key, \
            "GeneralSettings must be created using GeneralSettings.get()."

        self.base_path: str = ""
        self.out_path: str = ""
        self.plot_extension: str = "pdf"
        self.image_extension: str = "png"
        self.default_image_size_px: tuple[int,int] = (4000, 4000)
        self.interest_region: str = (250,250,50,50)
        "x,y,w,h of the interest region in mm"
        self.figure_size: tuple[int,int] = (6,4)
        self.output_image_maxsize: int = 2000

        self.output_paths: dict[str,str] = {}

        GeneralSettings.sub_outpath: str = ""

        # print("Loading general settings...", end="")
        self.load()
        # print(f"[green]OK[/green] (Thread: {threading.get_ident()})")

        if not os.path.exists(self.base_path):
            print(f"Base path {self.base_path} does not exist. Creating it...")
            os.makedirs(self.base_path)
        if not os.path.exists(self.out_path):
            print(f"Output path {self.out_path} does not exist. Creating it...")
            os.makedirs(self.out_path)

        if self.plot_extension.startswith("."):
            self.plot_extension = self.plot_extension[1:]
        if self.image_extension.startswith("."):
            self.image_extension = self.image_extension[1:]


    def __str__(self):
        return "General Settings"

    def load(self):
        cfg_path = self.__get_cfg_file()
        # read config file from ~/.config
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                # load json and save members to self
                conf = json.load(f)
                for key in conf:
                    setattr(self, key, conf[key])
        else:
            # create config file
            self.save()

    def save(self) -> None:
        # save members to json
        cfg_path = self.__get_cfg_file()
        with open(cfg_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def save_image(self, out_name, image):
        f = np.max(image.shape[:2]) / general.output_image_maxsize

        h,w = image.shape[:2] / f
        w = int(w)
        h = int(h)

        image = cv2.resize(image, (w,h))
        cv2.imwrite(out_name, image)

    def update_setting(self, key: str, value: str) -> None:
        setattr(self, key, value)
        self.save()

    def print(self):
        print(pretty_repr(self.__dict__))

    def __get_cfg_file(self) -> str:
        return os.path.join(__file__, "..", "..", "..", "config.json")

    def clear(self):
        # delete cfg file
        cfg_path = self.__get_cfg_file()
        p = self.base_path
        pe = self.plot_extension
        if os.path.exists(cfg_path):
            os.remove(cfg_path)

        keys = [x for x in self.__dict__]
        # reset members
        for key in keys:
            delattr(self, key)

        self.base_path = p
        self.plot_extension = pe
        self.save()

    def get_output_file(self, *name, **kwargs):
        """Gets an output file path.

        Kwargs:
            is_plot (bool): If true, the plot extension is appended.
            is_image (bool): If true, the image extension is appended.
        Returns:
            str: path
        """
        name = list(name)
        if 'is_plot' in kwargs and kwargs['is_plot']:
            name[-1] = f'{GeneralSettings.sub_specimen}{name[-1]}.{general.plot_extension}'
        if 'is_image' in kwargs and kwargs['is_image']:
            name[-1] = f'{GeneralSettings.sub_specimen}{name[-1]}.{general.image_extension}'


        return os.path.join(self.out_path, GeneralSettings.sub_outpath, *name)