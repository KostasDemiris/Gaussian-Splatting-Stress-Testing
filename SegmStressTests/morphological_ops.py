import numpy as np
import cv2
from functools import reduce
import os

from Utility.Loading import load_stress_test_config, load_morph_operation, format_morph_params

# Primary Reference [1]: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
# Useful to understand what's happening: https://en.wikipedia.org/wiki/Mathematical_morphology

config = load_stress_test_config(
    config_directory = os.path.dirname(os.path.abspath(__file__)),
    config_filename  = "config.yaml",
)

# ------------------------ Base Four Morphological Operations -----------------------------
# These are assumed to be binary or grayscale (it doesn't matter really as long as they have a single channel) images being passed in
# It should be applied to segmentation and depth maps of the images

def erosion(img, kernel_size=None, iterations=None):
    kernel_size, iterations = load_morph_operation("erosion", format_morph_params(kernel_size, iterations))
    
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(img, struct_elem, iterations=iterations)

def dilation(img, kernel_size=None, iterations=None):
    kernel_size, iterations = load_morph_operation("dilation", format_morph_params(kernel_size, iterations))

    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(img, struct_elem, iterations=iterations)

def opening(img, kernel_size=None, iterations=None):
    kernel_size, iterations = load_morph_operation("opening", format_morph_params(kernel_size, iterations))
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Since cv2 (as far as I can find in tutorials and on the docs) does not have iterative application, we create an iterator
    return reduce(
        lambda accumulator, i: cv2.morphologyEx(accumulator, cv2.MORPH_OPEN, struct_elem), 
        range(iterations), img
    )

def closing(img, kernel_size=None, iterations=None):
    kernel_size, iterations = load_morph_operation("closing", format_morph_params(kernel_size, iterations))
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Since cv2 (as far as I can find in tutorials and on the docs) does not have iterative application, we create an iterator
    return reduce(
        lambda accumulator, i: cv2.morphologyEx(accumulator, cv2.MORPH_CLOSE, struct_elem), 
        range(iterations), img
    )

# Again, this is a generator function since I already did that in visual dr noise, so I want to stick with that pattern
def random_morph_op(imgs, kernel_size_range=None, iteration_range=None, ops=None):
    params = config.get("random_operations", {})
    for img in imgs:
        morph_ops = [erosion, dilation, opening, closing]
        ops = params.get("oper_number_range", [1, 3]) if ops is None else ops
        operations =  np.random.choice(morph_ops, size=(np.random.randint(*ops)), replace=True) # Can repeat
        morphed_img = img
        for op in operations:
            kernel_size_range = params.get("kernel_size_range", [3, 7]) if kernel_size_range is None else kernel_size_range
            iteration_range = params.get("iters_count_range", [1, 3]) if iteration_range is None else iteration_range
            morphed_img = op(morphed_img, kernel_size=np.random.randint(*kernel_size_range), iterations=np.random.randint(*iteration_range))
        yield morphed_img

# ----------------------- Additional Morphological Operations -------------------------------
# We will probably not use these in the end since they are so extreme, although we keep them here just in case...

# Morphological top-hat subtracts the opening of an image from the image itself, which is useful for picking up small and bright components
def top_hat_transform(img, kernel_size=None):
    kernel_size = config.get("top_hat_transform", {}).get("kernel_size", 5) if kernel_size is None else kernel_size
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    # Since cv2 (as far as I can find in tutorials and on the docs) does not have iterative application, we create an iterator
    return reduce(
        lambda accumulator, i: cv2.morphologyEx(accumulator, cv2.MORPH_TOPHAT, struct_elem), 
        range(iterations), img
    )