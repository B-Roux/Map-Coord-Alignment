# Licensing agreement:
# https://raw.githubusercontent.com/B-Roux/Map-Coord-Alignment/main/LICENSE
# =============================================================================
# This file is deprecated and should never be used. It is purely kept because
# it may contain useful information for reference. Feel free to delete it.


from alignmentalg import iterative_localize
from pointextractor import extract_point
import cv2 as cv
import numpy as np
import time

# Throw an exception if this file is ever used.
raise BaseException("Deprecated File Usage: aligndriver.py")

data_path = './data/'

start = time.perf_counter_ns()
reference_image = cv.imread(data_path + 'im_4ka.png', cv.IMREAD_COLOR)
photograph = cv.imread(data_path + 'im_twl.png', cv.IMREAD_COLOR)
end = time.perf_counter_ns()

print('File reading took {} nanoseconds (~{:.2f} seconds)'
      .format((end-start), (end-start)/1000000000))

start = time.perf_counter_ns()
led_location = extract_point(photograph,
                             color_lb=np.array([223, 194, 197]),  # d white
                             color_ub=np.array([251, 235, 242]),  # l white
                             fuzz_amount=15,
                             erosion_radius=3,
                             accuracy_tradeoff=0.3)

pt_transform = iterative_localize(photograph, reference_image,
                                  point=led_location,
                                  samples=1000,
                                  perc=0.8)
end = time.perf_counter_ns()

print('Processing took {} ns (~{:.2f} seconds)'
      .format((end-start), (end-start)/1000000000))

print('\nTransformed Point:', pt_transform)
print('\nPress \'Enter\' to exit.')
input()
