
export DS_BASE_DIR_P=/home/code-base/transformers/blink_test_segmented/

export CUDA_VISIBLE_DEVICES=0,1
export MODEL=bart-ext
export DS_DIR=/disk1/sajad/gov-reports/gov-reports/

#python -m torch.distributed.launch --nproc_per_node=8 examples/pytorch/summarization/run_summarization.py \
python3 -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path /disk1/sajad/sci-trained-models/bart/greaseLM-1/checkpoint-87600/ \
    --do_predict \
    --output_dir /disk1/sajad/sci-trained-models/bart/greaseLM-1/no-repeat \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1  \
    --overwrite_output_dir \
    --predict_with_generate \
    --train_file $DS_DIR/train-withIds.json \
    --validation_file $DS_DIR/dev-withIds.json \
    --test_file  $DS_DIR/test-withIds.json \
    --load_best_model_at_end False
#    --dataset_config "3.0.0" \

#python post_stats/integrate_blink_preds.py