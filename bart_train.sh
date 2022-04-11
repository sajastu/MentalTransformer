#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
#CUDA_VISIBLE_DEVICES=1 python examples/pytorch/summarization/run_summarization.py \
python3 -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path allenai/led-base-16384 \
    --do_train \
    --do_eval \
    --do_predict \
    --output_dir /disk1/sajad/saved_models/led-base-govrep-8k \
    --per_device_train_batch_size=3 \
    --per_device_eval_batch_size=4  \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --num_train_epochs 5 \
    --text_column source \
    --summary_column summary \
    --overwrite_output_dir \
    --evaluation_strategy steps --warmup_steps 1000 --logging_steps 100 \
    --prediction_loss_only \
    --max_grad_norm 0.1 \
    --eval_steps 10000 --save_steps 10000 \
    --train_file /disk1/sajad/gov-reports/train.json \
    --validation_file /disk1/sajad/gov-reports/val.json \
    --test_file /disk1/sajad/gov-reports/test.json \
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