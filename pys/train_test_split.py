import os
import json
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('ann_file', default=str)
parser.add_argument('--seed', default=1996, type=int)
parser.add_argument('--train_save_path', default=None, type=str)
parser.add_argument('--eval_save_path', default=None, type=str)
NUM_EVAL_IMAGES = 5000


def main():
    with open(args.ann_file) as f:
        anns = json.load(f)
    ann_dir = os.path.dirname(args.ann_file)
    random.seed(args.seed)
    random.shuffle(anns)

    train_anns = anns[:-NUM_EVAL_IMAGES]
    eval_anns = anns[-NUM_EVAL_IMAGES:]
    print('{} training and {} eval image pairs!'.format(len(train_anns), len(eval_anns)))
    if args.train_save_path is None:
        train_save_path = os.path.join(ann_dir, 'train_sub.json')
    else:
        train_save_path = args.train_save_path

    if args.eval_save_path is None:
        eval_save_path = os.path.join(ann_dir, 'eval_{}.json'.format(NUM_EVAL_IMAGES))
    else:
        eval_save_path = args.eval_save_path

    with open(train_save_path, 'w') as f:
        json.dump(train_anns, f, indent=2)

    with open(eval_save_path, 'w') as f:
        json.dump(eval_anns, f, indent=2)


if __name__ == '__main__':
    args = parser.parse_args()
    main()

