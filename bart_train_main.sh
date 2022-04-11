#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
export DS_DIR=/disk1/sajad/datasets/medical/mental-reddit-reduced/sets
#export DS_DIR=/home/sajad/packages/summarization/transformers/sets
#CUDA_VISIBLE_DEVICES=1 python examples/pytorch/summarization/run_summarization.py \
#    --model_name_or_path /disk1/sajad/saved_models/bart-finetuned-large-mental/checkpoint-15000/ \
python3 -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path facebook/bart-large \
    --do_predict \
    --output_dir /disk1/sajad/saved_models/bart-large \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4  \
    --learning_rate 3e-5 \
    --report_to wandb \
    --run_name bart-large-finetuned-mentalWords-baseline \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --num_train_epochs 10 \
    --text_column src \
    --summary_column tldr \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --evaluation_strategy steps --warmup_steps 1000 --logging_steps 100 \
    --predict_with_generate \
    --max_grad_norm 0.1 \
    --eval_steps 2800 --save_steps 2800 \
    --train_file $DS_DIR/train.json \
    --validation_file $DS_DIR/val.json \
    --test_file $DS_DIR/test.json \
    --do_train \
    --do_eval \
    --greater_is_better True\
    --metric_for_best_model rougeL \
#   --resume_from_checkpoint saved_models/MentBart-mentsum-30kLarge/checkpoint-12000 \

#    --dataset_config "3.0.0" \

#    --dataset_name cnn_dailymail \
#    --dataset_config "3.0.0" \
#    --text_column document \
#    --summary_column summary \
#    --train_file $DS_BASE_DIR/train.json \
#    --validation_file $DS_BASE_DIR/val.json \
#    --test_file $DS_BASE_DIR/test.json \
#    --resume_from_checkpoint /trainman-mount/trainman-k8s-storage-349d2c46-5192-4e7b-8567-ada9d1d9b2de//saved_models/bart-ext/bart-tldr4M-pretrained-superloss/checkpoint-25000/

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