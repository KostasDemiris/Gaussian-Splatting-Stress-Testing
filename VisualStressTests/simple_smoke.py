import numpy as np
from noise import pnoise3
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import label, binary_erosion
import imageio
import glob
import cv2

import os
from Utility.Loading import load_stress_test_config

config = load_stress_test_config(
    config_directory = os.path.dirname(os.path.abspath(__file__)),
    config_filename  = "config.yaml",
)

# Surgical Application Reference: https://ieeexplore.ieee.org/document/8451815
# Perlin Noise is explained here: https://adrianb.io/2014/08/09/perlinnoise.html 
def perlin_smoke(image, seed, scale, octave, temporal_scale=1, t=0, integrate=False, params=None):
    image = np.array(image)
    smoke = np.zeros(image.shape[:2])

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            smoke[i, j] = pnoise3(i/scale, j/scale, t/temporal_scale, octaves=octave, base=seed)

    if integrate:
        smoke = linear_smoke_blending(image, smoke, params=params)

    return smoke

def generate_smoke(images, blend=False, time_frame=None, params=None):
    if params is None:
        params = config.get("perlin_smoke", {})

    seed = np.random.randint(0, 250) if params.get("seed") is None else params.get("seed")
    scale = np.random.uniform(*params.get("scale", [80, 120]))
    octave = np.random.randint(*params.get("octave", [5, 7]))
    temporal_scale = np.random.randint(*params.get("temporal_scale", [100, 110]))

    if time_frame is not None:
        start, end = time_frame
        duration = end - start
        smoke_out = np.zeros((duration, images.shape[1], images.shape[2], 3), dtype=np.float32)

        for i, time_point in enumerate(range(start, end)):
            noise = perlin_smoke(images[time_point], seed, scale, octave, t=time_point, temporal_scale=temporal_scale)
                        
            if not blend:
                mag = params.get("magnitude", 50)
                noise_rgb = np.stack([noise - np.mean(noise)] * 3, axis=-1)
                smoke_out[i] = noise_rgb * mag
            else:
                blended_frame = linear_smoke_blending(images[time_point], noise, params=params)
                smoke_out[i] = blended_frame
            
        return np.clip(smoke_out, 0, 255).astype(np.uint8)

    else:
        noise = perlin_smoke(images[0], seed, scale, octave, t=0, temporal_scale=temporal_scale)
        if blend:
            result = linear_smoke_blending(images[0], noise, params=params)
        else:
            result = np.stack([noise - np.mean(noise)] * 3, axis=-1) * params.get("magnitude", 50)
            
        return np.clip(result, 0, 255).astype(np.uint8)

def apply_whispy_erosion(noise, iterations=1):
    mask = noise > 0.1 
    eroded_mask = binary_erosion(mask, iterations=iterations)
    whispy_noise = noise * eroded_mask
    
    return whispy_noise

def apply_gaussian_blur(smoke, kernel_size):
    blurred_smoke = cv2.GaussianBlur(smoke, (kernel_size, kernel_size), 0)
    return blurred_smoke

def linear_smoke_blending(image, smoke, params=None):
    if params is None:
        params = config.get("perlin_smoke", {})
    
    if smoke.ndim == 2 and image.ndim == 3:
        smoke = np.stack([smoke] * 3, axis=-1)
    
    # smoke = apply_whispy_erosion(smoke)
    # smoke = apply_gaussian_blur(smoke, 15)
    smoke = np.clip(smoke, 0, 255)

    magnitude = params.get("magnitude", 50)
    lin_mixer = params.get("lin_mixer", 0.8)

    # Linear mixing method described in the paper
    blended = image + (lin_mixer * magnitude * (smoke - np.mean(smoke)))
    
    return np.clip(blended, 0, 255).astype(np.uint8)

def get_images(image_dir='/cs/student/projects3/2022/comp0031_grp_1/datasets/EndoNerf/pulling_soft_tissues/images'):
    file_list = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    sequence_list = []
    for file_path in file_list:
        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sequence_list.append(img)

    image_sequence = np.array(sequence_list)
    print(f"Loaded {len(image_sequence)} images.")
    return image_sequence

def perlin_smoke_test(file_name="smoke_test_output.png", use_frames=False):
    frames = np.full((100, 256, 256, 3), [20, 40, 20], dtype=np.uint8)

    if use_frames:
        frames = get_images()
    
    generated_sequence = generate_smoke(frames, blend=True, time_frame=[0, len(frames)])
    save_path = "ExampleImages/SmokeFrames/" + file_name
    if not os.path.exists("ExampleImages/SmokeFrames/"):
        os.makedirs("ExampleImages/SmokeFrames/")

    save_smoke_gif(generated_sequence, output_path="ExampleImages/SmokeFrames/" + "smoke_test.gif", fps=12)
    save_smoke_gif(frames, output_path="ExampleImages/SmokeFrames/" + "unsmoked_test.gif", fps=12)

    for i in range(len(generated_sequence[:5])):
        plt.imshow(generated_sequence[i]) 
        plt.title(f"Perlin Smoke - Frame {i}")
        plt.axis('off')

        save_path = "ExampleImages/SmokeFrames/" + str(i) + "_" + file_name
        if not os.path.exists("ExampleImages/SmokeFrames/"):
            os.makedirs("ExampleImages/SmokeFrames/")
            
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved test image to {save_path}")

def save_smoke_gif(frames, output_path="smoke_animation.gif", dur=10):
    imageio.mimsave(output_path, frames, duration=dur)
    print(f"Created GIF at {output_path}")

perlin_smoke_test(file_name="simple_test", use_frames=True)
