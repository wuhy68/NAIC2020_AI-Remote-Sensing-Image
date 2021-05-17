#!/usr/bin/env bash

set -e

CONFIG_FILE='configs/pspnet/pspnet_r101-d8_512x512_80k_naic.py'
GPU_NUM=4
export CUDA_VISIBLE_DEVICES='0,1,2,3'
#export CUDA_VISIBLE_DEVICES='4,5,6,7'

if [[ "$1" == "prepare_data" ]]; then

    python pys/generate_ann_file.py ~/datasets/NAIC2020/train/image/ \
        --mask_dir ~/datasets/NAIC2020/train/label \
        --save_path data/train.json


    python pys/generate_ann_file.py ~/datasets/NAIC2020/image_A/ \
        --save_path data/test_A.json

elif [[ "$1" == "train_test_split" ]]; then

    python pys/train_test_split.py data/train.json

elif [[ "$1" == "train" ]]; then

    ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}

elif [[ "$1" == "test" ]]; then
    CKPT_PATH="$2"
    SAVE_PATH="data/$3"
#    python tools/test.py ${CONFIG_FILE} ${CKPT_PATH} --out ${SAVE_PATH}
    ./tools/dist_test.sh ${CONFIG_FILE} ${CKPT_PATH} ${GPU_NUM} --out ${SAVE_PATH} ${@:4}
	python pys/write_submit_file.py ${CONFIG_FILE} ${SAVE_PATH}
else

    echo "No such things as $1"

fi
