#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
export DS_DIR=/disk1/sajad/datasets/medical/mentsum/paraq-files
export HF_DATASETS_CACHE=/disk0/sajad/.cache/huggingface
mkdir -p $HF_DATASETS_CACHE

#export MODEL_NAME=/disk1/sajad/sci-trained-models/grease/mentsum/checkpoint-51110
export MODEL_NAME=facebook/bart-large

#export DS_DIR=/home/sajad/packages/summarization/transformers/sets
#    --model_name_or_path /disk1/sajad/saved_models/bart-finetuned-large-mental/checkpoint-15000/ \
#    --model_name_or_path facebook/bart-large \

#CUDA_VISIBLE_DEVICES=1 python examples/pytorch/summarization/run_summarization.py \
python3 -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir /disk1/sajad/sci-trained-models/grease/mentsum-decoder/ \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=4 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --num_train_epochs 10 \
    --save_total_limit 5 \
    --text_column src \
    --summary_column tldr \
    --overwrite_output_dir \
    --evaluation_strategy steps --warmup_steps 2000 --logging_steps 100 \
    --predict_with_generate \
    --max_grad_norm 0.1 \
    --eval_steps 2700 --save_steps 2700 \
    --train_file $DS_DIR/train.parquet \
    --validation_file $DS_DIR/val.parquet \
    --test_file $DS_DIR/test.parquet \
    --use_gnn True \
    --do_train \
    --do_eval \
    --do_predict \
    --load_best_model_at_end True \
    --greater_is_better True\
    --metric_for_best_model rougeL \
    --report_to wandb \
    --run_name MS-grease-decoder-main \

#    --load_best_model_at_end \

#CUDA_VISIBLE_DEVICES=0 python examples/pytorch/summarization/run_summarization.py  \

#python -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/summarization/run_summarization.py \
#    --task_mode abstractive \
#    --model_name_or_path facebook/bart-large \
#    --do_train \
#    --do_eval \
#    --do_predict \
#    --output_dir /disk1/sajad/bart-outputs/test \
#    --per_device_train_batch_size=1 \
#    --per_device_eval_batch_size=8  \
#    --learning_rate 3e-5 \
#    --weight_decay 0.01 \
#    --adam_beta2 0.98 \
#    --num_train_epochs 5 \
#    --overwrite_output_dir \
#    --evaluation_strategy steps --eval_steps 100 --save_steps 100 --warmup_steps 100 --logging_steps 100 \
#    --text_column document \
#    --summary_column summary \
#    --train_file /home/sajad/transformers/reddit_tifu/train.json \
#    --validation_file /home/sajad/transformers/reddit_tifu/validation.json \
#    --test_file /home/sajad/transformers/reddit_tifu/validation.json \
#    --predict_with_generate