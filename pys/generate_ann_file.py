import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('image_dir', type=str)
parser.add_argument('--mask_dir', default=None, type=str)
parser.add_argument('--save_path', default=None, type=str)
parser.add_argument('--image_ext', default='.tif', type=str)


def get_file_name(path):
    return os.path.basename(path).rpartition('.')[0]


def main():
    image_files = sorted([os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)])
    if args.mask_dir is not None:
        mask_files = sorted([os.path.join(args.mask_dir, x) for x in os.listdir(args.mask_dir)])
        assert len(image_files) == len(mask_files)
        assert all(get_file_name(a) == get_file_name(b) for a, b in zip(image_files, mask_files))
    else:
        mask_files = None

    anns = []
    for i, img_path in enumerate(image_files):
        ann = dict(filename=img_path)
        if mask_files:
            ann.update(ann=dict(
                seg_map=mask_files[i],
            ))
        anns.append(ann)

    print('{} image pairs!'.format(len(image_files)))
    if args.save_path:
        with open(args.save_path, 'w') as f:
            json.dump(anns, f, indent=2)
        print('Result saved at {}!'.format(args.save_path))


if __name__ == '__main__':
    args = parser.parse_args()
    main()

