import typer
import numpy as np
import random
import cv2
from matplotlib import pyplot as plt
from fracsuite.core.image import to_rgb

from fracsuite.core.imageprocessing import dilateImg
from fracsuite.core.region import RectRegion
from fracsuite.core.specimen import Specimen
from fracsuite.core.splinter import Splinter


gen_app = typer.Typer()

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

@gen_app.command()
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

@gen_app.command()
def compare_spatial(specimen_name):
    specimen = Specimen.get(specimen_name)


    size = 300
    x1 = 500
    x2 = x1 + size
    y1 = 500
    y2 = y1 + size

    # choose a region to replicate
    region = RectRegion(x1,y1,x2,y2)

    # get region from specimen fracture image
    splinters_in_region = specimen.get_splinters_in_region(region)
    frac_img = specimen.get_fracture_image()

    image = np.zeros((size,size,3), dtype=np.uint8)

    orig_contours = to_rgb(np.zeros_like(frac_img, dtype=np.uint8))

    # perform watershed on the points
    markers = np.zeros((size,size), dtype=np.uint8)
    for s in splinters_in_region:
        point = s.centroid_px
        ix = np.max([np.min([int(point[0])-x1, size-1]), 0])
        iy = np.max([np.min([int(point[1])-y1, size-1]), 0])
        markers[ix,iy] = 255

        cv2.drawContours(orig_contours, [s.contour], -1, (0,0,255), 2)

    orig_contours = orig_contours[x1:x2, y1:y2,:]

    markers = dilateImg(markers, 3, it=3)
    plt.imshow( markers)
    plt.show()

    markers = cv2.connectedComponents(np.uint8(markers))[1]

    markers = cv2.watershed(np.zeros_like(image), markers)

    m_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    m_img[markers == -1] = 255

    splinters = Splinter.analyze_contour_image(m_img)

    gen_contours = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for s in splinters:
        clr = (255,0,0)
        cv2.drawContours(gen_contours, [s.contour], -1, clr, 1)

    print(orig_contours.shape)
    print(gen_contours.shape)
    comparison = cv2.addWeighted(orig_contours, 0.5, gen_contours, 0.5, 0)


    plt.imshow(comparison)
    plt.show()