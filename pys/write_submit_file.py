import os
import mmcv
import mutils
import argparse
import numpy as np

from mmseg.datasets import build_dataset
from tqdm import tqdm
from zipfile import ZipFile

parser = argparse.ArgumentParser()
parser.add_argument('cfg')
parser.add_argument('pkl_file')
parser.add_argument('--zip_file', default='data/res.zip', type=str)


def get_file_name(path):
    return os.path.basename(path).rpartition('.')[0]


def out2mask(mask):
    mask = np.array(mask, dtype=np.int32)
    mask = (mask + 1) * 100
    return mask


def main():
    cfg = mmcv.Config.fromfile(args.cfg)
    outs = mmcv.load(args.pkl_file)
    test_set = build_dataset(cfg.data.test)
    with ZipFile(args.zip_file, 'w') as myzip:
        for img_info, binary in zip(tqdm(test_set.img_infos, ncols=80), outs):
            mask = mutils.binary2array(binary, np.int32)
            mask = out2mask(mask)
            image_name = get_file_name(img_info['filename'])
            save_path = os.path.join('results', image_name + '.png')
            out_binary = mutils.array2binary(mask)
            myzip.writestr(save_path, out_binary)


if __name__ == '__main__':
    args = parser.parse_args()
    main()
