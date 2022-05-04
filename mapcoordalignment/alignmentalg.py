# Licensing agreement:
# https://raw.githubusercontent.com/B-Roux/Map-Coord-Alignment/main/LICENSE
# =============================================================================
# This file implements the main functionality of this project. For modularity
# purposes, all configuration details should be specified in ./__init__.py
# instead of in this file.


import cv2 as cv
import numpy as np
from mapcoordalignment.helpers import \
    im_resize, apply_homography, cut0


def align_images(
        im,
        im_ref,
        max_features,
        good_match_percent):

    # Transforms im to the coordinate system of imRef:
    # │
    # ├───Parameters:
    # │   ├───im: The image to be transformed
    # │   ├───im_ref: The reference image
    # │   ├───max_features: Alignment samples
    # │   └───good_match_percent: % of alignment samples to use
    # │
    # └───Returns:
    #     ├───im in the coordinate system of imRef
    #     └───the homography obtained

    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im_ref_gray = cv.cvtColor(im_ref, cv.COLOR_BGR2GRAY)

    # find features
    orb = cv.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im_ref_gray, None)
    matcher = cv.DescriptorMatcher_create(
        cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = list(matcher.match(descriptors1, descriptors2, None))

    # keep only good matches
    matches.sort(key=lambda x: x.distance, reverse=False)
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    # point extraction
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # homography
    h, _ = cv.findHomography(points1, points2, cv.RANSAC)
    height, width, _ = im_ref.shape
    im_reg = cv.warpPerspective(im, h, (width, height))

    return im_reg, h


def iterative_localize(
        im,
        im_ref,
        point,
        samples,
        perc):
        
    # Transforms a point in im to a point in imReference:
    # │
    # ├───Parameters:
    # │   ├───im: The starting image
    # │   ├───imReference: The reference image
    # │   ├───point: The point (x,y) to convert
    # │   ├───samples: Alignment samples
    # │   ├───perc: % of alignment samples to use
    # │   └───scale_fac: rescale images before processing (more performance)
    # │
    # └───Returns:
    #     └───Transformed point (x,y)

    im_ref_scale = (im.shape[1]/im_ref.shape[1])

    # the amount to scale im_ref down to,
    # Gives better results and runs faster.
    im_ref = im_resize(im_ref, im_ref_scale)

    im_loc, h1 = align_images(im, im_ref, samples, perc)

    # step 1 - compute the point mapping - refer to sec 2.5 in the paper
    pnt_1 = apply_homography(h1, point)
    # at this point, the two images should be roughly in the same location.

    # carve the local regions for closer alignment
    # local area size - should be 'relative' to imRef size.
    # the divisor should be small enough to define a local area,
    # but big enough to contain enough features
    local_area_radius = int(im_ref.shape[0]/3)
    im_locl = im_loc[cut0(pnt_1[1]-local_area_radius):pnt_1[1] +
                     local_area_radius,
                     cut0(pnt_1[0]-local_area_radius):pnt_1[0] +
                     local_area_radius, :]
    im_refl = im_ref[cut0(pnt_1[1]-local_area_radius):pnt_1[1] +
                     local_area_radius,
                     cut0(pnt_1[0]-local_area_radius):pnt_1[0] +
                     local_area_radius, :]

    # step 2 - compute the point mapping - refer to sec 2.5 in the paper
    pnt_2 = (pnt_1[0] - cut0(pnt_1[0]-local_area_radius),
             pnt_1[1] - cut0(pnt_1[1]-local_area_radius))
    # NOTE: perc is being divided by 2 because it is assumed that the images
    # are going to be much closer to perfectly aligned.
    imReg, h2 = align_images(im_locl, im_refl, samples, perc/2)

    # step 3 - compute the point mapping - refer to sec 2.5 in the paper
    pnt_3 = apply_homography(h2, pnt_2)

    # step 4 - compute the point mapping - refer to sec 2.5 in the paper
    off_x_back = cut0(pnt_1[0]-local_area_radius)
    off_y_back = cut0(pnt_1[1]-local_area_radius)
    pnt_4 = (int((pnt_3[0] + off_x_back)/im_ref_scale),
             int((pnt_3[1] + off_y_back)/im_ref_scale))

    return pnt_4
