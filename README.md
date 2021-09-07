# Map-Coord-Alignment

```Py
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
```
