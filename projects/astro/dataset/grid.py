import numpy as np

"""
 __Grid__

To perform strong lensing, we need a grid of (x,y) coordinates which we map throughout the Universe as if their path
is deflected. For this, we create a simple 2D grid of coordinates below where the origin is (0.0, 0.0) and the size of
a pixel is 0.05, which corresponds to the resolution of our `image`.

(The `grid_from` function below simply creates a uniform 2D grid of coordinates, don't worry too much abouit whether 
you understand the code itself).
"""


def grid_from(shape: Tuple[int, int], pixel_scale: float) -> np.ndarray:
    grid = np.zeros(shape=(shape[0], shape[1], 2))

    centre = (float(shape[0] - 1) / 2, float(shape[1] - 1) / 2)

    for y in range(shape[0]):
        for x in range(shape[1]):
            grid[x, y, 0] = (x - centre[0]) * pixel_scale
            grid[x, y, 1] = (y - centre[1]) * pixel_scale

    return grid


grid = grid_from(shape=(151, 151), pixel_scale=0.05)

np.save(file="grid.npy", arr=grid)
