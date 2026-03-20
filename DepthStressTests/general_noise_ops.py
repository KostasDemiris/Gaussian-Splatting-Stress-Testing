# Primary reference: Ibrahim, M.M., Liu, Q., Khan, R., Yang, J., Adeli, E. and Yang, Y. (2020), Depth map artefacts reduction: a review. IET Image Process., 14: 2630-2644. https://doi.org/10.1049/iet-ipr.2019.1622
# also kind of relevant: Benchmarking Robustness of Endoscopic Depth Estimation with Synthetically Corrupted Data

import numpy as np
import cv2
import os

from Utility.Loading import load_stress_test_config, load_transform

config = load_stress_test_config(
    config_directory = os.path.dirname(os.path.abspath(__file__)),
    config_filename  = "config.yaml",
)

# Band-artefacts can result from information loss during image quantisation
def band_quantisation(image, likelihood=None, params=None):
    likelihood, params = load_transform("quantisation", likelihood, params)
    image = np.array(image, dtype=np.float64)

    if np.random.uniform() > likelihood:
        return image
    
    step = (image.max() - image.min()) / params.get("target_bands", 16)
    image = np.floor(image / step) * step

    return np.clip(image, 0, 255)

# Gaussian noise error, which we vary based on the depth level
def depth_varying_gaussian_noise(image, likelihood=None, params=None):
    likelihood, params = load_transform("gaussian_noise", likelihood, params)
    image = np.array(image, dtype=np.float32)

    if np.random.uniform() > likelihood:
        return image
    
    mean, sigma = params.get("mean", 0.0), params.get("sigma", 1.0)
    gaussian_noise = np.random.normal(mean, sigma, size=np.shape(image))

    depth_var = params.get("depth_var", 0.1)
    image += (image * depth_var * gaussian_noise)

    return np.clip(image, 0, 255)

transformations = [band_quantisation, depth_varying_gaussian_noise]

# Box corruption, simulates holes/spikes in the depth map due to specular highlights
def apply_box_corruption(image, likelihood=None, params=None):
    likelihood, params = load_transform("box_corruption", likelihood, params)
    image = np.array(image, dtype=np.float32)

    if np.random.uniform() > likelihood:
        return image

    