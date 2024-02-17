import cv2
import os

class AnisotropyImages():

    @property
    def all(self):
        return [self.transmission, self.green, self.blue]

    @property
    def all_paths(self):
        return [self.transmission_path, self.green_path, self.blue_path]

    @property
    def transmission(self):
        if self.transmission_path is None:
            return None

        return cv2.imread(self.transmission_path)

    @property
    def green(self):
        if self.green_path is None:
            return None

        return cv2.imread(self.green_path)

    @property
    def blue(self):
        if self.blue_path is None:
            return None

        return cv2.imread(self.blue_path)

    def __init__(self, path: str):
        self.path = path

        self.green_path = None
        self.blue_path = None
        self.transmission_path = None

        # find files in path
        self.files = []
        for file in os.listdir(path):
            if file.endswith(".bmp"):
                self.files.append(file)

                if "BrightPolarizedGreen" in file:
                    self.green_path = os.path.join(path, file)
                elif "BrightPolarizedBlue" in file:
                    self.blue_path = os.path.join(path, file)
                elif "Transmission" in file:
                    self.transmission_path = os.path.join(path, file)

        self.available = len(self.files) == 3
