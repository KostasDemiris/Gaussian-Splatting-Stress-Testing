import numpy as np
from noise import pnoise2
import matplotlib.pyplot as plt
import cv2
from functools import reduce
import os

from VisualStressTests.general_noise_ops import apply_gaussian_blur

from Utility.Loading import load_stress_test_config

config = load_stress_test_config(
    config_directory = os.path.dirname(os.path.abspath(__file__)),
    config_filename  = "config.yaml",
)

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

def initial_blob_generation(height, width, radius, noise_scale=2, octaves=4, seed=None, centroid=None):
    if centroid is None:
        centroid = (width/2, height/2)
    centre_x, centre_y = centroid
    coord_y, coord_x = np.mgrid[0:height, 0:width].astype(np.float32)
    displacement = np.sqrt((coord_x-centre_x)**2 + (coord_y-centre_y)**2 + 1e-6)

    if seed is None:
        seed = np.random.randint(0, 1000)

    noise_grid = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            noise_grid[i, j] = radius * (1 + pnoise2((j/noise_scale), (i/noise_scale), octaves=octaves, base=seed))
    
    # Penalise further away from centre of image, to still achieve the centred blob look
    centred_blob = np.clip((noise_grid-displacement), 0, 1.0) 
    return centred_blob

def blood_blob_generation(height, width, radius, noise_scale=2, octaves=4, seed=None, centroid=None, 
            colour=[150, 30, 10], vary_colour=True, blur=False):
    initial_blob = initial_blob_generation(height, width, radius, noise_scale=noise_scale, octaves=octaves, seed=seed, centroid=centroid)
    struct_elem  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blood_blob   = np.clip(cv2.morphologyEx(initial_blob, cv2.MORPH_CLOSE, struct_elem), 0, 1.0)

    colour_array = np.full((np.shape(blood_blob)[0], np.shape(blood_blob)[1], 3), colour, dtype=np.float32)
    if vary_colour:
        colour_array += np.random.uniform(-2, 2, size=(np.shape(blood_blob)[0], np.shape(blood_blob)[1], 3))
    if blur:
        colour_array = apply_gaussian_blur(colour_array)
    
    return (blood_blob[..., np.newaxis] * colour_array)
    

# Interpreting this:
#   Red: Right, Blue: Left, Green: Up, Purple: Down
def optical_flow_tracking(initial_frame, subsequent_frame):
    initial_frame = initial_frame.astype('uint8')
    subsequent_frame = subsequent_frame.astype('uint8')

    prev_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(subsequent_frame, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(initial_frame)
    hsv[..., 1] = 255 # Max out saturation

    # These parameters are just adjusted manually, without too much thought or basis btw
    optical_flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, pyr_scale=0.5, levels=3, winsize=3,
        iterations=5, poly_n=5, poly_sigma=1.1, flags=0
    )

    magnitude, angle = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
    magnitude[magnitude < 0.1] = 0

    hsv[..., 0] = angle * 180 / np.pi / 2  # Direction encoding in the hue channel
    hsv[..., 2] = np.clip(magnitude * 10, 0, 255)  # Magnitude encoding in the value channel

    optical_flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return optical_flow_vis

def surface_normal_calculation(depth_map, focal_x=1, focal_y=1):
    # Collapse RGB to single channel if needed
    if depth_map.ndim == 3:
        depth_map = depth_map[:, :, 0]

    # Depth map is very step like, which results in poor looking blood later on...
    depth_map = depth_map.astype(np.float64)
    depth_map = cv2.bilateralFilter(
        depth_map.astype(np.float32), d=9, sigmaColor=0.1, sigmaSpace=7
    ).astype(np.float64)

    dx, dy = np.gradient(depth_map, axis=1), np.gradient(depth_map, axis=0)
    tangents_x = np.stack([np.ones_like(depth_map)/focal_x, np.zeros_like(depth_map), dx], axis=-1)
    tangents_y = np.stack([np.zeros_like(depth_map), np.ones_like(depth_map)/focal_y, dy], axis=-1)

    normals = np.cross(tangents_x, tangents_y)
    unit_normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6)

    return unit_normals

# Remaps each part of the blood map to the coordinate offset by the optical flow between consecutive frames
def flow_warp_blood_mask(blood_blob, optical_flow):
    grid_x, grid_y = np.meshgrid(np.arange(np.shape(blood_blob)[1]), np.arange(np.shape(blood_blob)[0]))
    flow_disp_x = (grid_x - optical_flow[..., 0]).astype(np.float32)
    flow_disp_y = (grid_y - optical_flow[..., 1]).astype(np.float32)

    warped_blob = cv2.remap(blood_blob.astype(np.float32), flow_disp_x, flow_disp_y, 
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return warped_blob

# Reference Paper: "Blood Harmonisation of Endoscopic Transsphenoidal Surgical Video Frames on Phantom Models"
# https://kclpure.kcl.ac.uk/portal/en/publications/blood-harmonisation-of-endoscopic-transsphenoidal-surgical-video-/
def blend_mask():
    pass

# We initially use the Blinn-Phong reflection model, but will try PBR if we have time (https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model)
# Assumptions: Blood is occluding, the light source is coaxial with the camera, no ambient lighting due to it being pinhole surgery.
def blinn_phong_rendering(surg_image, blood_mask, depth_map, camera_position, lighting_position, params=None):
    params = params if params is not None else config.get("blinn_phong_rendering", {})

    material_properties = params.get("material", {})
    material_diffuse_coeff = np.array(material_properties.get("diffuse_coefficient", [0.4, 0.0, 0.0]))
    material_specular_coeff = np.array(material_properties.get("specular_coefficient", [0.9, 0.9, 0.9]))
    material_ambient_coeff = np.array(material_properties.get("ambient_coefficient", [0.0, 0.0, 0.0]))
    specular_exponent = material_properties.get("specular_exponent", 100)

    camera_params = params.get("camera", {})
    focal_length, baseline = camera_params.get("focal_length", 856), camera_params.get("baseline", 0.0055)
    light_intensity = params.get("light", [200, 200, 200])

    # depth_map = cv2.GaussianBlur(depth_map.astype(np.float32), (3, 3), sigmaX=1.5)
    surface_normals = surface_normal_calculation(depth_map)

    dim_y, dim_x = np.shape(blood_mask)[:2]
    rendered_mask = np.zeros(np.shape(blood_mask))
    for i in range(dim_y):
        for j in range(dim_x):
            surface_norm = surface_normals[i, j]
            point_depth = np.max((np.mean(depth_map[i, j]), 1e-6))

            point_position = np.array([j, i, point_depth])

            light_dir = lighting_position - point_position
            light_vector = light_dir / (np.linalg.norm(light_dir) + 1e-6)

            view_dir = camera_position - point_position
            view_vector = view_dir / (np.linalg.norm(view_dir) + 1e-6)
            half_dir = light_vector + view_vector
            half_vector = half_dir / (np.linalg.norm(half_dir) + 1e-6)


            n_dot_l = np.maximum(np.dot(surface_norm, light_vector), 0.0)
            n_dot_h = np.maximum(np.dot(surface_norm, half_vector), 0.0)

            # Diffuse component: K_d * I_d * (N · L)
            diffuse_component = material_diffuse_coeff * light_intensity * n_dot_l

            # Specular component: K_s * I_s * (N · H)^n
            specular_component = material_specular_coeff * light_intensity * np.power(n_dot_h, specular_exponent)

            # By default this is zero, Ambient component: K_a * I_a
            ambient_component = material_ambient_coeff * light_intensity

            rendered_mask[i, j] = diffuse_component + specular_component + ambient_component

    # # Just an experiment... It makes it incredibly dark so I might leave it off for a bit...
    # h, w = blood_mask.shape[:2]
    # ys, xs = np.mgrid[0:h, 0:w]
    # depth_vals = np.mean(depth_map, axis=-1) if depth_map.ndim == 3 else depth_map
    # depth_vals = np.maximum(depth_vals, 1e-6)
    # points = np.stack([xs, ys, depth_vals], axis=-1)
    # dist = np.linalg.norm(lighting_position[None, None, :] - points, axis=-1, keepdims=True)
    # attenuation = 1.0 / (1.0 + 0.01 * dist + 0.001 * dist**2)
    # rendered_mask *= attenuation

    # if blood_mask.ndim == 2:
    #     mask = blood_mask[..., np.newaxis]
    # else:
    #     mask = blood_mask

    # rendered_mask = (rendered_mask) * (mask > 0)

    return rendered_mask


# -------------------- Usage and testing section --------------------


def save_blood_gif(frames, output_path="VisualStressTests/ExampleImages/blood_animation.gif", fps=10):
    imageio.mimsave(output_path, frames, duration=fps)
    print(f"Created GIF at {output_path}")

def blood_blob_tests(mask, file_name= "blood blob.png", title="Blood Blob Example"):
    random_background_img = np.full((256, 256, 3), [20, 40, 20], dtype=np.uint8)

    # Because I'm program this on the school computers through ssh, there's no backend for 
    # display so we need to save them instead to view...
    plt.imshow(mask.astype(np.uint8))
    plt.title(title)
    plt.axis('off')

    plt.savefig("ExampleImages/"+file_name, bbox_inches='tight')

if __name__ == "__main__":
    seed = np.random.randint(0, 100)

    initi_blob = initial_blob_generation(512, 640, 50, seed=seed)
    initi_blob = initi_blob[..., np.newaxis] * np.array([150, 30, 10])

    blood_blob = blood_blob_generation(512, 640, 50, vary_colour=False, blur=False, seed=seed, centroid=(150, 100))

    blood_blob_tests(initi_blob, file_name="Initial_blood_blob", title="Pre-Opening Blood Blob")
    blood_blob_tests(blood_blob, file_name="Example_blood_blob.png", title="Example Blood Blob")

    flow_img = optical_flow_tracking(initi_blob, blood_blob)
    blood_blob_tests(flow_img, file_name="optical_flow_Example", title="Optical Flow between initial and example blood blobs")

    test_depth_map_0 = cv2.imread(os.path.join("depth", "frame-000000.depth.png"), cv2.IMREAD_ANYDEPTH)
    test_depth_map_1 = cv2.imread(os.path.join("depth", "frame-000001.depth.png"), cv2.IMREAD_ANYDEPTH)

    surface_normals_0 = surface_normal_calculation(test_depth_map_0)
    surface_normals_1 = surface_normal_calculation(test_depth_map_1)
    surface_optical_flow = optical_flow_tracking(((surface_normals_0 + 1.0) / 2.0 * 255), ((surface_normals_1 + 1.0) / 2.0 * 255))
    blood_blob_tests(surface_optical_flow, file_name="depth_optical_flow", title="optical flow between depth map frames")


    test_image_0 = cv2.imread(os.path.join("/cs/student/projects3/2022/comp0031_grp_1/datasets/EndoNerf/pulling_soft_tissues/images", "frame-000000.color.png"), cv2.IMREAD_COLOR)
    test_image_1 = cv2.imread(os.path.join("/cs/student/projects3/2022/comp0031_grp_1/datasets/EndoNerf/pulling_soft_tissues/images", "frame-000001.color.png"), cv2.IMREAD_COLOR)
    image_optical_flow = optical_flow_tracking(test_image_0, test_image_1)
    blood_blob_tests(image_optical_flow, file_name="image_optical_flow", title="optical flow between image frames")


    warped_blood = flow_warp_blood_mask(blood_blob, image_optical_flow)
    blood_blob_tests(warped_blood, file_name="warped_blood_splatter", title="blood splatter warped by optical flow between depth map frames")

    blood_blob_tests(((surface_normals_0 + 1.0) / 2.0 * 255).astype(np.uint8), file_name="depth_surface_normals", title="surface normals in depth map")

    camera_params = config.get("blinn_phong_rendering", {}).get("camera", {})

    render = blinn_phong_rendering(test_image_0, blood_blob, test_depth_map_0, np.array([512/2, 640/2, 100]), np.array([512/2, 640/2, 100]))
    
    # render = np.clip(render / np.max(render + 1e-6), 0, 1)
    render = np.clip(render, 0, 255)
    blood_blob_tests(render, file_name="render_blood_blob", title="Blinn Phong Blood Rendering")


