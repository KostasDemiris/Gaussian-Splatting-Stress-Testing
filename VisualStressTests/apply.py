import numpy as np
import os
import cv2

from loading import load_sliced_image_dataset
import VisualStressTests.general_noise_ops as gn_ops

def run_stress_testing(data_dir, randomisation_lvl, save_name="domain_randomised", visual_save=True):
    output_dir = os.path.join(data_dir, f"{randomisation_lvl}_{save_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transformations = gn_ops.get_transformations(randomisation_lvl)

    for i, img in enumerate(gn_ops.apply_sequential_transforms(load_sliced_image_dataset(data_dir), transformations)):
        img_to_save = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        if not visual_save:
            file_name = os.path.join(output_dir, f"mod_{str(i)}.npy")
            np.save(file_name, img)
        else:
            file_name = os.path.join(output_dir, f"mod_{str(i)}.png")
            cv2.imwrite(file_name, img_to_save)

        if (i % 10 == 0):
            print(f"Saved {i} images!")
