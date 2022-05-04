# Licensing agreement:
# https://raw.githubusercontent.com/B-Roux/Map-Coord-Alignment/main/LICENSE
# =============================================================================
# This file is the "last step" before exposing functionality. It should
# contain all configuration details for *specific use cases*. This is THE ONLY
# place where those configuration details should appear.

__author__ = "github.com/B-Roux"
__version__ = "2"
__license__ = "https://github.com/B-Roux/Map-Coord-Alignment/blob/main/LICENSE"
__all__ = ['align', 'get_point']

import numpy as np
from mapcoordalignment.alignmentalg import \
    iterative_localize as _align_internal
from mapcoordalignment.pointextractor import \
    extract_point as _extract_point_internal

def get_point(photograph,
              color_lb=np.array([223, 194, 197]),  # d white
              color_ub=np.array([251, 235, 242]),  # l white
              fuzz_amount=15,
              erosion_radius=3,
              accuracy_tradeoff=0.3):
    """
    Gets the coordinates of a colored LED bulb in the image:
    │
    ├───Parameters:
    │   ├───im: The starting image
    │   ├───color_lb: The lower bound of the LED color
    │   ├───color_ub: The upper bound of the LED color
    │   ├───fuzz_amount: Quick way of broadening the bounds
    │   ├───erosion_radius: Morph. erosion to reduce artifacts
    │   └───accuracy_tradeoff: run faster [=0] or  
    │       more accurately [=1]
    │
    └───Returns:
        └───LED coordinate (x,y)
    """
    return _extract_point_internal(
        im=photograph,
        color_lb=color_lb,
        color_ub=color_ub,
        fuzz_amount=fuzz_amount,
        erosion_radius=erosion_radius,
        accuracy_tradeoff=accuracy_tradeoff
    )

def align(photograph, reference_image, point,
          samples=1000,
          perc=0.8):
    """
    Transforms a point in im to a point in imReference:
    │
    ├───Parameters:
    │   ├───im: The starting image
    │   ├───imReference: The reference image
    │   ├───point: The point (x,y) to convert
    │   ├───samples: Alignment samples
    │   ├───perc: % of alignment samples to use
    │   └───scale_fac: rescale images before processing (more performance)
    │
    └───Returns:
        └───Transformed point (x,y)
    """
    return _align_internal(
        im=photograph,
        im_ref=reference_image,
        point=point,
        samples=samples,
        perc=perc
    )
