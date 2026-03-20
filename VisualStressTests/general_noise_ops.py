import numpy as np
import cv2
import os
import imageio

# Primary reference [1]: Synthetic data accelerates the development of generalizable learning-based algorithms for X-ray image analysis

# Some of these methods have been excluded from usage in our evaluation, due to the complete lack of relevance or plausibility in 
# surgical images. (unlike in typical DR) Unrealistic noise operations on data for 3DGS do not help results since it is an optimisation
# problem so training and evaluation are the same thing.

# The settings for these are controlled via a configuration file stored in the VisualStressTest folder
from Utility.Loading import load_stress_test_config, load_transform

config = load_stress_test_config(
    config_directory = os.path.dirname(os.path.abspath(__file__)),
    config_filename  = "config.yaml",
)

# --------------------- Regular strength transformations -----------------------
def apply_gaussian_noise(image, likelihood=None, params=None):
    likelihood, params = load_transform("gaussian_noise", likelihood, params)
    # Cast it before returning even if transform isn't applied, since either way it's expected in this format!
    image = np.array(image, dtype=np.float32)

    if np.random.uniform() > likelihood:
        return image

    mean = params.get('mean', 0.0)
    # It'll throw an error if the value is None otherwise bc I unpack it later
    sigma_range = params.get('sigma_range') or [0.005, 0.1]
    amp = params.get('amp')

    amp = amp if amp is not None else (np.max(image) - np.min(image)) * np.random.uniform(*sigma_range)
    gaussian_noise = np.random.normal(loc=mean, scale=amp, size=np.shape(image))

    return np.clip(image + gaussian_noise, 0, 255)


# Follows the method described in reference [1], which varies slightly from the typical gamma transform method
def apply_gamma_transform(image, likelihood=None, params=None):
    likelihood, params = load_transform("gamma", likelihood, params)
    image = np.array(image, dtype=np.float32)

    if np.random.uniform() > likelihood:
        return image
    
    random_gamma_range = params.get("random_gamma", [1.0, 1.0])
    gamma = np.random.uniform(*random_gamma_range)
    
    image_max, image_min = np.max(image), np.min(image)
    gamma_transformed_image = np.power(
        (image - image_min) / (1e-5 + image_max - image_min),
        gamma
    )

    return np.clip((gamma_transformed_image * (1e-5 + image_max - image_min)) + image_min, 0, 255)


def apply_crop_transform(image, likelihood=None, params=None):
    likelihood, params = load_transform("crop", likelihood, params)
    image = np.array(image, dtype=np.float32)

    if np.random.uniform() > likelihood:
        return image
    
    crop_size = params.get("crop_size", [0.85, 0.99])

    h, w = np.shape(image)[:2]
    scale = np.random.uniform(*crop_size, 2)
    new_h, new_w = int(h * scale[0]), int(w * scale[1])

    offset_h = np.random.randint(0, h - new_h)
    offset_w = np.random.randint(0, w - new_w)

    cropped_image = image[offset_h: offset_h + new_h, offset_w: offset_w + new_w]

    # OpenCV uses (W, H) ordering for some reason
    resized_image = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized_image

regular_dr_transforms = [apply_gaussian_noise, apply_gamma_transform, apply_crop_transform]

# --------------------- Ex-high strength transformations -----------------------
# These apply a significantly larger distortion effect to the image, and are mostly unrealistic

def apply_pixel_inversion(image, likelihood=None, params=None):
    likelihood, params = load_transform("inversion", likelihood, params)
    image = np.array(image, dtype=np.float32)

    if np.random.uniform() > likelihood:
        return image
    
    return ((image * -1) + np.max(image))

def apply_affine_transform(image, likelihood=None, params=None):
    likelihood, params = load_transform("inversion", likelihood, params)
    image = np.array(image, dtype=np.float32)

    if np.random.uniform() > likelihood:
        return image
    
    h, w = np.shape(image)

    affine_t = np.eye(3)
    theta = params.get("theta", [0.0, 0.0])
    if np.random.uniform() < params.get("rot_chance", 0.0):
        centre_point = (h//2, w//2)
        rot_matrix = cv2.getRotationMatrix2D(centre_point, np.random.uniform(*theta), 1.0)
        affine_t = np.vstack([rot_matrix, [0, 0, 1]]) @ affine_t
    
    scale_factor = params.get("scale_factor", [1.0, 1.0])
    if np.random.uniform() < params.get("scale_chance", 0.0):
        scale = np.random.uniform(*scale_factor)
        scale_matix = [[scale, 0, 0], [0, scale, 0], [0, 0, 1]]
        affine_t = scale_matix @ affine_t
    
    if np.random.uniform() < params.get("shear_chance", 0):
        shear_range = params.get("shear_factor", [0, 0])
        shear = np.random.uniform(*shear_range)
        shear_matrix = np.array([[1, shear, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
        affine_t = shear_matrix @ affine_t

    final_affine_transform = affine_t[:2, :]
    image = cv2.warpAffine(image, final_affine_transform, (w, h), flags=cv2.INTER_LINEAR)
    return image


# Mentioned within the SurgicalGS further work section as an important evaluation factor!
def apply_gaussian_blur(image, likelihood=None, params=None, generator=True):
    likelihood, params = load_transform("gaussian_blur", likelihood, params)
    image = np.array(image, dtype=np.float32)

    if np.random.uniform() > likelihood:
        return image
    
    kernel_size = params.get("kernel_size", 5)
    # Kernel size needs to be odd, and the parameter variation script could accidentally mess this up
    if kernel_size % 2 == 0:
        kernel_size += 1 

    blurred_image = cv2.GaussianBlur(image, ksize=(kernel_size, kernel_size), sigmaX=params.get("sigma", 1))
    return blurred_image
    

# This is actually an addition, related to the fat/oil/water than can stick to the camera
# This should be applied with all frames in the sequence since it remains consistently stuck to the camera
def apply_local_gaussian_blur(images, likelihood=None, params=None):
    likelihood, params = load_transform("local_gaussian_blur", likelihood, params)
    images = np.array(images, dtype=np.float32)

    single = False
    if images.ndim == 3:
        images = images[np.newaxis, ...]
        single = True

    y, x = np.shape(images)[1:3]

    if np.random.uniform() > likelihood:
        return images[0] if singe else images

    # Otherwise we get circular import problems
    from VisualStressTests.blood import initial_blob_generation

    kernel_size = params.get("kernel_size", 5)
    # Kernel size needs to be odd, and the parameter variation script could accidentally mess this up
    if kernel_size % 2 == 0:
        kernel_size += 1    

    blur_number = np.random.randint(*params.get("number", [1, 3]))
    sigma_range = params.get("sigma", [1.0, 1.5])

    for patch in range(blur_number):
        patch_size = np.random.randint(*params.get("patch_size", [15, 30]))
        centroid_y = np.random.randint(patch_size, y - patch_size)
        centroid_x = np.random.randint(patch_size, x - patch_size)

        # Since this is an irregularly shaped patch, we can't directly apply gaussian blurring to it, 
        # so we apply it to a copied image then use alpha compositing to blend it in.
        blob_mask = initial_blob_generation(y, x, patch_size, centroid=(centroid_x, centroid_y))
        blob_mask = cv2.GaussianBlur(blob_mask, ksize=(3, 3), sigmaX=patch_size/4)
        blob_mask = blob_mask[:, :, np.newaxis].astype(np.float32)

        sigma = np.random.uniform(*sigma_range)

        for i, frame in enumerate(images):
            blurred_frame = cv2.GaussianBlur(frame, ksize=(kernel_size, kernel_size), sigmaX=sigma)
            # Alpha compositing
            images[i] = (blurred_frame * blob_mask + frame * (1.0 - blob_mask)).astype(np.uint8)

            # # This results in a harsher edge
            # blurred_image = np.where(blob_mask, blur_region, frame)

    return images[0] if single else images


# This is not necessarily very useful as a visual noise method here, but is more relevant in the segmentation mask testing!
def apply_box_corruption(image, likelihood=None, params=None):
    likelihood, params = load_transform("gaussian_blur", likelihood, params)
    image = np.array(image, dtype=np.float32)

    if np.random.uniform() > likelihood:
        return image

    box_number = np.random.randint(*params.get("box_number", [1, 3]))
    box_min, box_max  = params.get("box_dimens", [10, 50])

    for _ in range(box_number):
        height, width = np.random.randint(box_min, box_max, size=(2))
        max_height, max_width = (image.shape[0] - height, image.shape[1] - width)

        if max_height > 0 and max_width > 0:
            starting_height = np.random.randint(0, max_height)
            starting_width  = np.random.randint(0, max_width)

            noise_strength = params.get("corruption", [-25, 25])
            box_size = (height, width) if len(image.shape) == 2 else (height, width, image.shape[2])
            image[starting_height: starting_height+height, 
                        starting_width: starting_width+width] += np.random.uniform(*noise_strength, size=box_size)

    return np.clip(image, 0, 255)


def apply_dropout(image, likelihood=None, params=None):
    likelihood, params = load_transform("dropout", likelihood, params)
    image = np.array(image, dtype=np.float32)

    if np.random.uniform() > likelihood:
        return image

    dropout_prob = np.random.uniform(*params.get("percentage", [1, 10])) / 100.0
    dropout_mask = np.random.binomial(1, 1 - dropout_prob, size=image.shape)
    image *= dropout_mask

    return image

def apply_sharpening_and_embossing(data, likelihood=None, params=None):
    return data

def apply_pooling_method(data, likelihood=None, params=None):
    return data

def apply_distortion_transform(data, likelihood=None, params=None):
    return data

# Also includes all of the regular strength transforms. 
strong_dr_transforms = [*regular_dr_transforms, apply_pixel_inversion, apply_affine_transform, apply_gaussian_blur, 
    apply_box_corruption, apply_dropout, apply_sharpening_and_embossing, apply_pooling_method, apply_distortion_transform]

def get_transformations(randomisation_lvl):
    if randomisation_lvl == "strong":
        return strong_dr_transforms
    if randomisation_lvl == "gauss_patch":
        return [apply_local_gaussian_blur]
    else:
        return regular_dr_transforms

# The likelihood of each occuring and the different probabilities etc. is handled within each function rather than the pipeline function
def apply_sequential_transforms(data, transform_list=regular_dr_transforms):
    # Data can be a regular form of array, or it can be a generator
    for image in data:
        for transform in transform_list:
            image = transform(image)
        yield image

def save_transformed_gif(frames, output_path="VisualStressTests/ExampleImages/", file_name="general_transform.gif", duration=10):
    frames = np.array(frames)
    frames = np.clip(frames, 0, 255).astype(np.uint8)

    imageio.mimsave(os.path.join(output_path, file_name), frames, duration=duration)
    print(f"Created GIF at {os.path.join(output_path, file_name)}")


if __name__ == "__main__":
    frames = np.random.uniform(0, 255, size=(120, 612, 520, 3))

    output_path = "VisualStressTests/ExampleImages/local_gaussian_blur.gif"
    save_transformed_gif(apply_local_gaussian_blur(frames))