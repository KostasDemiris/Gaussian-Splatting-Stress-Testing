import numpy as np
import os
import cv2

from loading import load_sliced_masks
import SegmStressTests.general_noise_ops as gn_ops

def get_transformations():
    return gn_ops.transformations

# The likelihood of each occuring and the different probabilities etc. is handled within each function rather than the pipeline function
def apply_sequential_transforms(data, transform_list=None):
    if transform_list is None:
        transform_list = get_transformations()
    # Data can be a regular form of array, or it can be a generator
    for image in data:
        for transform in transform_list:
            image = transform(image)
        yield image

def run_stress_testing(data_dir, save_name="domain_randomised", visual_save=True):
    output_dir = os.path.join(data_dir, f"seg_mask_{save_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transformations = get_transformations()

    for i, img in enumerate(apply_sequential_transforms(load_sliced_masks(data_dir), transformations)):
        # Masks are binary...
        img_to_save = cv2.cvtColor(np.array(np.clip(img*255, 0, 255), dtype=np.uint8), cv2.COLOR_RGB2BGR)
        if not visual_save:
            file_name = os.path.join(output_dir, f"mod_{str(i)}.npy")
            np.save(file_name, img)
        else:
            file_name = os.path.join(output_dir, f"mod_{str(i)}.png")
            cv2.imwrite(file_name, img_to_save)

        if (i % 10 == 0):
            print(f"Saved {i} images!")

