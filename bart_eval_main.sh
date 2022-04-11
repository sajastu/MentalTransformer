
export DS_BASE_DIR_P=/home/code-base/transformers/blink_test_segmented/

export CUDA_VISIBLE_DEVICES=0,1
export MODEL=bart-ext
export DS_DIR=/disk1/sajad/datasets/medical/mental-reddit-reduced/sets

#python -m torch.distributed.launch --nproc_per_node=8 examples/pytorch/summarization/run_summarization.py \
CUDA_VISIBLE_DEVICES=0 python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path /disk1/sajad/saved_models/MentBart-mentsum/checkpoint-10000/ \
    --do_predict \
    --output_dir /disk1/sajad/saved_models/MentBart-mentsum/ \
    --per_device_train_batch_size=10 \
    --per_device_eval_batch_size=10  \
    --overwrite_output_dir \
    --predict_with_generate \
    --train_file $DS_DIR/train.json \
    --validation_file $DS_DIR/val.json \
    --test_file  $DS_DIR/test.json \
    --load_best_model_at_end False
#    --dataset_config "3.0.0" \

#python post_stats/integrate_blink_preds.py