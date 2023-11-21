import random
import numpy as np


def gibbs_strauss_process(
    n_points,
    hardcore_radius,
    acceptance_possibility,
    area=(100, 100),
    max_iterations=10000
):
    # make sure npoints is an integer
    n_points = int(n_points)

    points = []
    width, height = area
    hardcore_radius_squared = hardcore_radius ** 2
    max_iterations = n_points * 100

    for _ in range(max_iterations):
        x, y = np.random.uniform(0, width), np.random.uniform(0, height)
        if np.random.uniform(0, 1) < acceptance_possibility:
            if all((x - px) ** 2 + (y - py) ** 2 >= hardcore_radius_squared for px, py in points):
                points.append((x, y))
                if len(points) >= n_points:
                    break

    return points

def strauss_process(region_size, num_points, hardcore_radius, acceptance_possibility):
    points = set()
    for _ in range(num_points):
        while True:
            x, y = np.random.uniform(0, region_size, 2)
            new_point = (x, y)

            # Check for interaction effects
            can_add = True
            for point in points:
                distance = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
                if distance < hardcore_radius:
                    if random.random() > acceptance_possibility:
                        can_add = False
                        break

            if can_add:
                points.add(new_point)
                break
    return points


def CSR_process(r0, sz):
    """Spatial Poisson process with constant intensity r0."""

    num_points = np.random.poisson(r0)
    points = np.random.uniform(0, np.sqrt(sz), (num_points, 2))

    return points