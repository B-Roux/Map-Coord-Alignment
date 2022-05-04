# Licensing agreement:
# https://raw.githubusercontent.com/B-Roux/Map-Coord-Alignment/main/LICENSE
# =============================================================================
# This file is a common area for all of the small functions that other parts of
# the project make use of. The purpose is to collect all sorts of miscellaneous
# functionality to reduce the duplication of code.


import cv2 as cv
import numpy as np


def cut0(n):
    # simple function for making any negative numbers 0 -
    # more readable down the line
    return max(n, 0)


def format_color(c):
    # correctly formats a 3x1 np array as a color (GBR)
    r = np.zeros(3)
    for i in range(3):
        if c[i] > 255:
            r[i] = 255
        if c[i] < 0:
            r[i] = 0
        r[i] = int(c[i])
    return r


def im_resize(original, scalefactor):
    # return an image scaled by the scalefactor
    if 0.995 < scalefactor < 1.005:
        return original
    dim = (int(original.shape[1] * scalefactor),
           int(original.shape[0] * scalefactor))
    return cv.resize(original, dim, interpolation=cv.INTER_AREA)


def apply_homography(h, point):
    # multiply a point by a homography and rescale automatically
    pt1 = np.array([[point[0]],
                   [point[1]],
                   [1]])
    pt2 = np.matmul(h, pt1)
    return (int(pt2[0]/pt2[2]), int(pt2[1]/pt2[2]))
