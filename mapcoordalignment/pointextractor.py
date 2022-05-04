# Licensing agreement:
# https://raw.githubusercontent.com/B-Roux/Map-Coord-Alignment/main/LICENSE
# =============================================================================
# This file contains the logic used for extracting points from LED lights. For
# modularity purposes, all configuration details should be specified
# in ./__init__.py instead of in this file.


import cv2 as cv
import numpy as np
from mapcoordalignment.helpers import \
    format_color, im_resize


def extract_point(
        im,
        color_lb,
        color_ub,
        fuzz_amount,
        erosion_radius,
        accuracy_tradeoff):

    #Gets the coordinates of a colored LED bulb in the image:
    #│
    #├───Parameters:
    #│   ├───im: The starting image
    #│   ├───color_lb: The lower bound of the LED color
    #│   ├───color_ub: The upper bound of the LED color
    #│   ├───fuzz_amount: Quick way of broadening the bounds
    #│   ├───erosion_radius: Morph. erosion to reduce artifacts
    #│   └───accuracy_tradeoff: run faster [=0] or  
    #│       more accurately [=1]
    #│
    #└───Returns:
    #    └───LED coordinate (x,y)

    # mask off the selected colors
    clb = format_color(color_lb - fuzz_amount)
    cub = format_color(color_ub + fuzz_amount)
    mask = cv.inRange(im, clb, cub)

    # erode the image to reduce masking artifacts (optional)
    if erosion_radius != 0:
        strel = np.ones([erosion_radius, erosion_radius])
        mask = cv.erode(mask, strel)

    # reduce the mask size for performance
    fast_mask = im_resize(mask, accuracy_tradeoff)

    # take a weighted sum - sped up with some numpy magic
    xi = np.linspace(0, fast_mask.shape[0], fast_mask.shape[0], dtype=int)
    yi = np.linspace(0, fast_mask.shape[1], fast_mask.shape[1], dtype=int)
    mesh_grid = np.array(np.meshgrid(yi, xi))
    approx_y = round(np.mean(mesh_grid[1][fast_mask >= 255/2]) *
                     (1/accuracy_tradeoff))
    approx_x = round(np.mean(mesh_grid[0][fast_mask >= 255/2]) *
                     (1/accuracy_tradeoff))

    return (approx_x, approx_y)
