import numpy as np
import os

from Utility.Loading import load_stress_test_config, load_transform

config = load_stress_test_config(
    config_directory = os.path.dirname(os.path.abspath(__file__)),
    config_filename  = "config.yaml",
)

# Reasonable noise, occurs if the segmentation model fails and doesn't register part of the tool or something like that!
# Rather than blasting random boxes with noise, we just set random boxes to all 1s or all 0s
def apply_box_corruption(images, likelihood=None, params=None):
    likelihood, params = load_transform("box_corruption", likelihood, params)
    images = np.array(images)

    if np.random.uniform(0, 1) > likelihood:
        return images

    box_number = np.random.randint(*params.get("box_number", [1, 3]))
    box_min, box_max  = params.get("box_dimens", [10, 50])

    for _ in range(box_number):
        height, width = np.random.randint(box_min, box_max, size=(2))
        max_height, max_width = images.shape[0] - height, images.shape[1] - width

        if max_height > 0 and max_width > 0:
            starting_height = np.random.randint(0, max_height)
            starting_width  = np.random.randint(0, max_width)

            # Since the segmentation mask is binary, for each box we either zero out the whole thing or set it to true
            box_size = (height, width) if len(images.shape) == 2 else (height, width, images.shape[2])
            images[starting_height: starting_height+height, starting_width: starting_height+width] = np.random.choice([0, 1])
            
    return images

transformations = [apply_box_corruption]
