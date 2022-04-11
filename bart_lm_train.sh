#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
#CUDA_VISIBLE_DEVICES=2 python examples/pytorch/summarization/bart_finetune_lm.py \
python3 -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/summarization/bart_finetune_lm.py \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir /disk1/sajad/saved_models/bart-finetuned-large-mental-large-MentalWords \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8  \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --num_train_epochs 10 \
    --text_column masked_src \
    --summary_column original_src \
    --overwrite_output_dir \
    --prediction_loss_only \
    --evaluation_strategy steps --warmup_steps 1000 --logging_steps 100 \
    --max_grad_norm 0.1 \
    --eval_steps 10000 --save_steps 10000 \
    --train_file /disk1/sajad/datasets/medical/mental-reddit-final/sets/train-lm-large-mentalWords.json \
    --validation_file /disk1/sajad/datasets/medical/mental-reddit-final/sets/val-lm-large-mentalWords.json \
    --test_file /disk1/sajad/datasets/medical/mental-reddit-final/sets/test-lm-large-mentalWords.json \
    --report_to wandb \
    --run_name bart-pretrained-MentalWords-120k \
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