from time import sleep
import cv2 as cv
import numpy as np

# simple function for making any negative numbers 0
def cut0 (n):
    """
    Makes negative numbers 0:
    │
    ├───Parameters:
    │   └───n: The number to be converted
    │
    └───Returns:
        └───max(n, 0)
    """
    
    return max(n, 0)

# correctly formats a 3x1 np array as a color
def format_color(c):
    """
    Gets the coordinates of a colored LED bulb in the image:
    │
    ├───Parameters:
    │   ├───c: An np.array[]: len=3
    │
    └───Returns:
        └───c, formatted as a color (bounded integers)
    """
    r = np.zeros(3)
    for i in range(3):
        if c[i] > 255:
            r[i] = 255
        if c[i] < 0:
            r[i] = 0
        r[i] = int(c[i])
    return r

# image resizer
def imresize(original, scalefactor):
    """
    Resizes the image:
    │
    ├───Parameters:
    │   ├───original: The image being resized
    │   └───scalefactor: The resizing factor
    │
    └───Returns:
        └───A scalefactor * ??? image
    """

    width = int(original.shape[1] * scalefactor)
    height = int(original.shape[0] * scalefactor)
    dim = (width, height)
    return cv.resize(original, dim, interpolation = cv.INTER_AREA)

#multiply a point by a homography
def apply_homography(h, point):
    """
    Applies a homography
    │
    ├───Parameters:
    │   ├───h: The homography
    │   └───point: The (x,y) tuple being transformed
    │
    └───Returns:
        └───transformed (x,y)
    """

    pt1 = np.array([[point[0]],
                   [point[1]],
                   [1]])
    pt2 = np.matmul(h, pt1)
    return (int(pt2[0]/pt2[2]), int(pt2[1]/pt2[2]))

# perform image alignment
def align_images(im, imRef, max_features, good_match_percent):
    """
    Transforms im to the coordinate system of imRef:
    │
    ├───Parameters:
    │   ├───im: The image to be transformed
    │   ├───imRef: The reference image
    │   ├───max_features: Alignment samples
    │   ├───good_match_percent: % of alignement samples to use
    │   ├───display: display images and information (optional, False)
    │   └───thumb_resolution: The display resolution (optional, 1K)
    │
    └───Returns:
        ├───im in the coordinate system of imRef
        └───the homography obtained
    """

    imGray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    imRefGray = cv.cvtColor(imRef, cv.COLOR_BGR2GRAY)

    # find features
    orb = cv.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(imGray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(imRefGray, None)
    matcher = cv.DescriptorMatcher_create(
        cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # keep only good matches
    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    # point extraction
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # homography
    h, mask = cv.findHomography(points1, points2, cv.RANSAC)
    height, width, channels = imRef.shape
    imReg = cv.warpPerspective(im, h, (width, height))

    return imReg, h

# performs the point transformation
def iterative_localize (im, imReference, point, samples, perc):
    """
    Transforms a point in im to a point in imReference:
    │
    ├───Parameters:
    │   ├───im: The starting image
    │   ├───imReference: The reference image
    │   ├───point: The point (x,y) to convert
    │   ├───samples: Alignment samples
    │   ├───perc: % of alignement samples to use
    │   ├───display: display images and information (optional, False)
    │   └───thumb_resolution: The display resolution (optional, 1K)
    │
    └───Returns:
        └───Transformed point (x,y)
    """

    # the amount to scale imRef down to,
    # for some reason this gives better results.
    imRef_scale = (im.shape[1]/imReference.shape[1])
    imRef = imresize(imReference, imRef_scale)

    imLoc, h1 = align_images(im, imRef, samples, perc)

    #step 1 - compute the point mapping - refer to sec 2.5 in the paper
    pnt_1 = apply_homography(h1, point)

    # at this point, the two images should be roughly in the same location.

    # local area size - should be 'relative' to imRef size.
    # the divisor should be small enough to define a local area,
    # but big enough to contain enough features
    local_area_radius = int(imRef.shape[0]/3)

    # carve the local regions
    imLoc_l = imLoc[cut0(pnt_1[1]-local_area_radius):pnt_1[1]+local_area_radius,
                    cut0(pnt_1[0]-local_area_radius):pnt_1[0]+local_area_radius,
                    :]

    imRef_l = imRef[cut0(pnt_1[1]-local_area_radius):pnt_1[1]+local_area_radius,
                    cut0(pnt_1[0]-local_area_radius):pnt_1[0]+local_area_radius,
                    :]

    #step 2 - compute the point mapping - refer to sec 2.5 in the paper
    pnt_2 = (pnt_1[0] - cut0(pnt_1[0]-local_area_radius),
             pnt_1[1] - cut0(pnt_1[1]-local_area_radius))

    #NOTE: perc is being divided by 2 because it is assumed that the images
    #are going to be much closer to perfectly aligned.
    #THIS MAY NOT ACTUALLY WORK. More testing is required.

    imReg, h2 = align_images(imLoc_l, imRef_l, samples, perc/2)

    #step 3 - compute the point mapping - refer to sec 2.5 in the paper
    pnt_3 = apply_homography(h2, pnt_2)

    #step 4 - compute the point mapping - refer to sec 2.5 in the paper
    off_x_back = cut0(pnt_1[0]-local_area_radius)
    off_y_back = cut0(pnt_1[1]-local_area_radius)

    pnt_4 = (int((pnt_3[0] + off_x_back)/imRef_scale), 
             int((pnt_3[1] + off_y_back)/imRef_scale))

    return pnt_4

# gets the coordinates of a colored LED
def extract_point(im, 
                  color_lb, 
                  color_ub,
                  fuzz_amount = 0,
                  erosion_radius = 0,
                  accuracy_tradeoff = 0.5):
    """
    Gets the coordinates of a colored LED bulb in the image:
    │
    ├───Parameters:
    │   ├───im: The starting image
    │   ├───color_lb: The lower bound of the LED color
    │   ├───color_ub: The upper bound of the LED color
    │   ├───fuzz_amount: Quick way of broadening the bounds (optional, 0)
    │   ├───erosion_radius: Morph. erosion to reduce artifacts (optional, 0)
    │   ├───accuracy_tradeoff: run faster [=0] or  
    │   │   more accurately [=1] (optional, 0.5)
    │   ├───display: display images and information (optional, False)
    │   └───thumb_resolution: The display resolution (optional, 1K)
    │
    └───Returns:
        └───LED coordinate (x,y)
    """

    clb = format_color(color_lb - fuzz_amount)
    cub = format_color(color_ub + fuzz_amount)

    mask = cv.inRange(im, clb, cub)

    #erode the image to reduce masking artifacts (optional)
    strel = np.ones([erosion_radius, erosion_radius])
    mask = cv.erode(mask, strel)

    #take sum of all coordinates weighted by mask
    fast_mask = imresize(mask, accuracy_tradeoff)

    points_x = []
    points_y = []
    for i in range(fast_mask.shape[0]):
        for j in range(fast_mask.shape[1]):
            if fast_mask[i, j] >= 255/2:
                points_x.append(j)
                points_y.append(i)

    approx_x = round(sum(points_x)/len(points_x)*(1/accuracy_tradeoff))
    approx_y = round(sum(points_y)/len(points_y)*(1/accuracy_tradeoff))

    return (approx_x, approx_y)

if __name__ == '__main__':

    import time

    start = time.perf_counter_ns()
    imReference = cv.imread('imRef.png', cv.IMREAD_COLOR)
    im = cv.imread('im.png', cv.IMREAD_COLOR)
    end = time.perf_counter_ns()

    print('File reading took {} nanoseconds (~{:.2f} seconds)'
        .format((end-start), (end-start)/1000000000))

    start = time.perf_counter_ns()
    led_location = extract_point(im, 
                                color_lb = np.array([223, 194, 197]), #dark blue
                                color_ub = np.array([251, 235, 242]), #light blue
                                fuzz_amount = 15,
                                erosion_radius = 3,
                                accuracy_tradeoff = 0.3)

    pt_transform = iterative_localize(im, imReference,
                                    point = led_location,
                                    samples = 1000,
                                    perc = 0.8)
    end = time.perf_counter_ns()

    print('Processing took {} nanoseconds (~{:.2f} seconds)'
        .format((end-start), (end-start)/1000000000))

    print()
    print('Transformed Point:', pt_transform)
    print()
    print('Press \'Enter\' to exit.')
    input()