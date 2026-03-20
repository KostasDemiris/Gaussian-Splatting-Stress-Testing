import os
import cv2
import numpy as np

# This takes in the EndoNerf Dataset in particular, we can later modify it to include other datasets but given our limited storage quota this is not currently feasible
# This is also a generator rather than an array, so cannot directly be indexed, but must rather be iterated through.
def load_sliced_image_dataset(data_dir, image_dir_name="images_right"):
    img_dir = os.path.join(data_dir, image_dir_name)
    assert os.path.exists(img_dir), "Image directory not found in data directory"

    images = sorted(os.listdir(img_dir))
    assert (len(images) > 0)

    for i, image_path in enumerate(images):
        # Height, Width, Channels
        image = np.array(cv2.cvtColor(cv2.imread(os.path.join(img_dir, image_path)), cv2.COLOR_BGR2RGB), dtype=np.float32)
        yield image

# The same as above, but with the binary segmentation masks from the EndoNerf datasets
def load_sliced_masks(data_dir, mask_dir_name = "gt_masks"):
    mask_dir = os.path.join(data_dir, mask_dir_name)
    assert os.path.exists(mask_dir), "Mask directory not found in the data directory"

    masks = sorted([mask for mask in os.listdir(mask_dir) if mask.endswith(".png")])
    assert (len(masks) > 0), "No masks found in mask directory"

    for i, mask_path in enumerate(masks):
        mask = np.array(cv2.cvtColor(cv2.imread(os.path.join(mask_dir, mask_path)), cv2.COLOR_BGR2GRAY) > 0, dtype=np.uint8)
        yield mask

# Same as both of the above, but with the depth maps
def load_sliced_depth_maps(data_dir, depth_dir_name="depth"):
    depth_dir = os.path.join(data_dir, depth_dir_name)
    assert os.path.exists(depth_dir), "Depth directory not found in the data directory"

    depth_maps = sorted([depth for depth in os.listdir(depth_dir) if depth.endswith(".png")])
    assert (len(depth_maps) > 0), "No depth maps found in depth map directory"

    for i, depth_map_path in enumerate(depth_maps):
        depth_map = np.array(cv2.cvtColor(cv2.imread(os.path.join(depth_dir, depth_map_path)), cv2.COLOR_BGR2GRAY), dtype=np.float32)
        yield depth_map

def load_image_data(image_dir):
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

if __name__ == "__main__":
    parse_args
