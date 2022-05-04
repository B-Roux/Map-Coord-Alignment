# This is an *example* driver for the mapcoordalignment module. All areas that
# are *required* to be changed are marked with "TODO" comments - grep or search
# for these and assure that they are all noted before attempting to run the
# software, it will not work otherwise!
#
# Since this is only an example, it is meant to illustrate one way in which
# this module is meant to be used. Please feel free to disregard this file and
# develop your own driver as necessary!

import mapcoordalignment as mca
import numpy as np
import cv2 as cv
import time


# =============================================================================
# SECTION 1: Processors
# This section contains all of the processors that the 3rd stage requires to
# get/set or handle information.
# -----------------------------------------------------------------------------

def turn_on_next_led():
    # this is called every iteration, it should turn on the next LED.
    # TODO: implement this
    return


def take_photo_of_map():
    # this is called every iteration, it should return an OpenCV-format image
    # of the map.
    # NOTE: The OpenCV image format is a numpy array in BGR format, NOT RGB!
    # TODO: implement this
    return None


def process_ref_im_coords(ref_im_coords):
    # this is called every iteration, it should take the coordinates in the
    # reference image space and process them into an airport code.
    # TODO: implement this
    return


# TODO: change the path here to the actual path needed
ref_image = cv.imread('path/to/reference/image', cv.IMREAD_COLOR)


# =============================================================================
# SECTION 2: Debug
# This section handles some light debug information and processing logic
# -----------------------------------------------------------------------------

def format_time(start, end):
    # format ns time as seconds for readability
    return '~{:.2f} seconds'.format((end-start)/1000000000)


# control and debug variables
iteration_counter = 0
program_start = time.perf_counter_ns()
end = 0


# =============================================================================
# SECTION 3: Main Driver
# This section contains the main code used to drive the software.
# Please review it and make sure you are familiar with it!
# -----------------------------------------------------------------------------

do_next_iteration = True

while (do_next_iteration):

    # debug
    iteration_counter += 1
    print(f'Starting iteration {iteration_counter}...')
    iteration_start = time.perf_counter_ns()

    # run the algorithm iteration
    turn_on_next_led()
    photograph = take_photo_of_map()
    led_location = mca.get_point(photograph)
    coords_in_ref_image = mca.align(photograph, ref_image, led_location)
    process_ref_im_coords(coords_in_ref_image)

    # is the software done?
    # TODO: Change this to the required condition
    do_next_iteration = False

    # print some debug information
    end = time.perf_counter_ns()
    print(f'Done.')
    print(f'├─────────Iteration took: {format_time(iteration_start, end)}')
    print(f'└─Total runtime (so far): {format_time(program_start, end)}')
    print()

# print final debug information
print(
    'Transformed', iteration_counter, 'points in',
    format_time(program_start, end),
    'with a mean time of ~{:.2f} seconds per iteration.'
    .format((end-program_start)/(iteration_counter*1000000000))
)
