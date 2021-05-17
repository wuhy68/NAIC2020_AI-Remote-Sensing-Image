import os
import glob
import numpy as np

from PIL import Image
from tqdm import tqdm
from concurrent import futures
from itertools import repeat
from collections import Counter


def rgb2gray(rgb):
    r, g, b = rgb
    return 256 * 256 * r + 256 * g + b


#                  black        red          green        blue
UNIQUE_COLORS = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (0, 255, 255)]
#                   yellow           cyan
UNIQUE_COLORS = [rgb2gray(x) for x in UNIQUE_COLORS]  # illegal colors {16711935, 16777215}


def get_file_name(path):
    return os.path.basename(path).rpartition('.')[0]


def process_one(mask_path, save_mask_dir):
    rgb_mask = np.array(Image.open(mask_path))
    flag = np.all((rgb_mask == 0) | (rgb_mask == 255), axis=-1)
    rgb_mask[~flag] = [0, 0, 0]
    r, g, b = [rgb_mask[..., i] for i in range(3)]
    mask = rgb2gray((r, g, b))
    colors = np.unique(mask, return_counts=True)
    # print([(a, b) for a, b in zip(*colors)])
    # assert all(x in UNIQUE_COLORS for x in colors)
    new_mask = np.zeros_like(mask) + 255
    for new_cls, old_cls in enumerate(UNIQUE_COLORS):
        flag = (mask == old_cls)
        new_mask[flag] = new_cls

    save_path = os.path.join(save_mask_dir, get_file_name(mask_path) + '.png')
    Image.fromarray(new_mask.astype(np.uint8)).save(save_path)
    return new_mask


def main():
    mask_dir = '/home/wenfeng/datasets/GID/Large-scale Classification_5classes/label_5classes'
    save_mask_dir = mask_dir + '_p'
    os.makedirs(save_mask_dir, exist_ok=True)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.tif')))

    print(UNIQUE_COLORS)

    counter = Counter()
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        for mask in tqdm(pool.map(process_one, mask_paths, repeat(save_mask_dir)),
                      total=len(mask_paths),
                      ncols=80):
            counter += Counter({a: b for a, b in zip(*np.unique(mask, return_counts=True))})
        print(counter)


if __name__ == '__main__':
    main()
