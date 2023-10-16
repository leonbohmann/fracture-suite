import typer
import numpy as np
import random
import cv2
from matplotlib import pyplot as plt

from fracsuite.core.imageprocessing import dilateImg
from fracsuite.core.splinter import Splinter


spatial_app = typer.Typer()

def strauss_process(region_size, num_points, interaction_radius, interaction_strength):
    points = set()
    for _ in range(num_points):
        while True:
            x, y = np.random.uniform(0, region_size, 2)
            new_point = (x, y)

            # Check for interaction effects
            can_add = True
            for point in points:
                distance = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
                if distance < interaction_radius:
                    if random.random() > interaction_strength:
                        can_add = False
                        break

            if can_add:
                points.add(new_point)
                break
    return points

@spatial_app.command()
def create_spatial():
    image = np.zeros((100,100,3), dtype=np.uint8)

    points = strauss_process(100, 100, 10, 0.5)
    x,y = zip(*points)

    plt.figure()
    plt.scatter(x,y)
    plt.show()

    # perform watershed on the points
    markers = np.zeros((100,100), dtype=np.uint8)
    for point in points:
        markers[int(point[0]), int(point[1])] = 255

    markers = dilateImg(markers, 5)
    markers = cv2.connectedComponents(np.uint8(markers))[1]

    markers = cv2.watershed(np.zeros_like(image), markers)

    m_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    m_img[markers == -1] = 255

    splinters = Splinter.analyze_contour_image(m_img)

    out_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for s in splinters:
        clr = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv2.drawContours(out_img, [s.contour], -1, clr, -1)

    plt.imshow(out_img)
    plt.show()