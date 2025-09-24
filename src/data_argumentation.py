from PIL import Image
import math
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

def load_images(path):
    path = Path(path)
    image_list = sorted(path.iterdir())
    image_hash = {}
    for image_name in image_list:
        if image_name.suffix == ".jpg":
            im = Image.open(image_name)
            image_hash[image_name.name] = im

    return image_hash

def rotate_if_needed(image_hash):
    for k, v in image_hash.items():
        if (v.size[0] < v.size[1]):
            v = v.rotate(90,expand=True)
            image_hash[k] = v

    return image_hash

def add_flipped_images(image_hash):
    print("Adding flipped images...")
    new_image_hash = {}
    for k, v in tqdm(image_hash.items()):
        k_new = k.split(".")[0]
        ext = ".jpg" #feel free to change the extension
        new_image_hash[k_new + "_0" + ext] = v
        new_image_hash[k_new + "_1" + ext] = v.transpose(Image.FLIP_TOP_BOTTOM)

    return new_image_hash

def rotate_and_crop(im, angle):
    rotated = im.rotate(angle, Image.BICUBIC, True)
    aspect_ratio = float(im.size[0]) / im.size[1]
    rotated_aspect_ratio = float(rotated.size[0]) / rotated.size[1]
    alp = math.fabs(angle) * math.pi / 180

    if aspect_ratio < 1:
        total_height = float(im.size[0]) / rotated_aspect_ratio
    else:
        total_height = float(im.size[1])

    h = abs(total_height / (aspect_ratio * abs(math.sin(alp)) + abs(math.cos(alp))))
    w = abs(h * aspect_ratio)

    a = rotated.size[0]*0.5
    b = rotated.size[1]*0.5

    return rotated.crop((a - w*0.5, b - h*0.5, a + w*0.5, b + h*0.5))

def add_rotated_crop_images(image_hash, angle_list):
    print("Adding rotated images...")
    new_image_hash = {}

    for k, v in tqdm(image_hash.items()):
        k_new = k.split(".")[0]
        ext = ".jpg" #feel free to change the extension
        for angle in angle_list:
            new_image_hash[k_new + "_" + str(int(angle)) + ext] = rotate_and_crop(v, angle)

    return new_image_hash

def resize_images(image_hash, dim):
    print("Resizing images...")
    for k, v in tqdm(image_hash.items()):
        image_hash[k] = v.resize(dim)

    return image_hash


def save_images(path, image_hash):
    path = Path(path)                 
    path.mkdir(parents=True, exist_ok=True)

    for k, v in image_hash.items():
        save_path = path / k          
        v.save(save_path)
        print(f"Image {k} saved at {save_path}")

def run_data_augmentation(source_dest_pairs,
    new_size=(400, 400),
    num_angles=17,
    angles=None,
    verbose=True,
    ensure_dirs=True,
):
    if angles is None:
        angle_list = np.linspace(0, 360, num_angles, endpoint=False)
    else:
        angle_list = np.array(angles, dtype=float)
    summary = {}

    for source, dest in source_dest_pairs:
        source = Path(source)
        dest = Path(dest)
        if ensure_dirs:
            dest.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"[BSDS AUG] Reading from: {source}")
            print(f"[BSDS AUG] Saving to  : {dest}")
            print(f"[BSDS AUG] new_size={new_size}, angles={angle_list.tolist()}")

        image_hash = resize_images(
            add_rotated_crop_images(
                add_flipped_images(
                    rotate_if_needed(
                        load_images(str(source))
                    )
                ),
                angle_list
            ),
            new_size
        )
        save_images(str(dest), image_hash)
        
        saved = len(image_hash) if hasattr(image_hash, "__len__") else 0
        summary[str(dest)] = saved

        if verbose:
            print(f"[BSDS AUG] Saved: {saved}\n")

    return summary