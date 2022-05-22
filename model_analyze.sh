#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
export DS_DIR=/disk0/sajad/datasets/news/cnn-dm/paraq-files/

export FILTERED_IDS=("653f8aac8bc1e1aa025ec3aa5d6edd533e6c2fe4")

#    --model_name_or_path /disk1/sajad/sci-trained-models/bart/cnndm-greaseBart/checkpoint-239150/ \
#python3 -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/summarization/run_summarization.py \
CUDA_VISIBLE_DEVICES=1 python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path /disk0/sajad/sci-trained-models/bart/cnndm-greaseBart/checkpoint-239150/ \
    --output_dir /disk0/sajad/sci-trained-models/bart/test/ \
    --per_device_train_batch_size=3 \
    --per_device_eval_batch_size=1  \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --num_train_epochs 5 \
    --text_column article \
    --summary_column highlights \
    --overwrite_output_dir \
    --evaluation_strategy steps --warmup_steps 25000 --logging_steps 100 \
    --predict_with_generate \
    --max_grad_norm 0.1 \
    --eval_steps 47850 --save_steps 47830 \
    --train_file $DS_DIR/test.parquet \
    --validation_file $DS_DIR/test.parquet \
    --test_file $DS_DIR/test.parquet \
    --load_best_model_at_end False \
    --do_predict \
    --use_gnn True \
    --filtered_ids "653f8aac8bc1e1aa025ec3aa5d6edd533e6c2fe4"