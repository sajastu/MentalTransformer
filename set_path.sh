#!/usr/bin/env bash

#user_space_tr=$(python disk.py)
#
#! ln -sf $user_space_tr /home/code-base/user_space/
#
#for f in /home/code-base/user_space/tr*
#do
#  export SAVE_MODEL_DIR="${f}"
#done
SAVE_MODEL_DIR=/trainman-mount/trainman-k8s-storage-349d2c46-5192-4e7b-8567-ada9d1d9b2de/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL=bart
#export M_ID=bart-sentence
export M_ID=bartCurr-xsum-woSmoothing

export DS_BASE_DIR=$SAVE_MODEL_DIR/webis-tldr/splits
export DS_BASE_DIR=reddit_tifu

export MODEL_OUTPUT_DIR=$SAVE_MODEL_DIR/saved_models/$MODEL/$M_ID

echo "SAVE_MODEL_DIR is $SAVE_MODEL_DIR"