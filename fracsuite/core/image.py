"""
Image conversion functions.
"""

from typing import Any
import cv2
import numpy as np

def is_gray(img):
    return len(img.shape) == 2

def is_rgb(img):
    return len(img.shape) == 3

def is_rgba(img):
    return len(img.shape) == 4


def to_gray(img):
    if is_gray(img):
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_rgb(img, rgb = False):
    if is_rgb(img):
        return img

    if rgb:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def to_rgba(img):
    if is_rgba(img):
        return img
    img = to_rgb(img)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)


class SplitImage:
    grid_size: int
    "Size of the grid"
    cols: int
    "Columns of the grid"
    rows: int
    "Rows of the grid"
    parts: list[Any]
    "Parts of the image"

    def __init__(self, parts, grid_size, cols, rows):
        self.parts = parts
        self.grid_size = grid_size
        self.cols = cols
        self.rows = rows

    def build(self, parts):
        # Die Größe des resultierenden Bildes berechnen
        new_width = self.cols * self.grid_size
        new_height = self.rows * self.grid_size
        # Eine leere Leinwand für das resultierende Bild erstellen
        result = np.zeros((new_height, new_width), dtype=np.uint8)

        # Schleife, um die Teile wieder zusammenzusetzen
        for i in range(self.rows):
            for j in range(self.cols):
                part = parts[i * self.cols + j]
                result[self.grid_size * i:self.grid_size * (i + 1),
                       self.grid_size * j:self.grid_size * (j + 1)] = part[0]

        return result

    def get_part(self, i, j) -> tuple[Any, tuple[int, int]]:
        return self.parts[i * self.cols + j]

def split_image(img, grid_size) -> SplitImage:
    gray = is_gray(img)
    if is_gray(img):
        shp = img.shape
    else:
        shp = img.shape[:2]
    # Höhe und Breite des Bildes erhalten
    height, width = shp

    # Anzahl der Zeilen und Spalten im Raster berechnen
    rows = height // grid_size
    cols = width // grid_size

    print(f"Splitting image into {rows} rows and {cols} columns.")

    # Eine leere Liste für die Teile des Bildes erstellen
    parts = []

    for i in range(rows):
        for j in range(cols):
            left = j * grid_size
            upper = i * grid_size
            right = (j + 1) * grid_size
            lower = (i + 1) * grid_size

            # Ausschnitt des Bildes erstellen
            if gray:
                part = img[upper:lower, left:right]
            else:
                part = img[upper:lower, left:right, :]

            location = (j * grid_size, i * grid_size)

            # Modifizierten Teil zur Liste hinzufügen
            parts.append((part, location))

    # Das resultierende Bild zurückgeben
    return SplitImage(parts, grid_size, cols, rows)