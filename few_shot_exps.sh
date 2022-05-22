
export CUDA_VISIBLE_DEVICES=0,1
export DS_DIR=/disk1/sajad/datasets/news/xsum/parq-files/
mkdir -p $HF_DATASETS_CACHE
export MODEL=grease
export USER=shabnam


if [ $SERVER_NAME=="barolo" ]
then
  export USER=sajad
fi

echo "for user $USER..."
export HF_DATASETS_CACHE=/disk0/$USER/.cache/few-shot/


#python src/graph_data_prepartion/preprocess.py --k 10 --seed 111 --only_test True
#python src/graph_data_prepartion/add_graph_to_dataset.py --k 10 --seed 111 --only_test True


for SEED in 123 184 888
do
    for K in 10 100
  do

    if [ $K == 10 ] && [ $SEED == 123 ]
    then
      continue 1
    fi

#    if [ $K -gt 10 ]
#    then
#      if [ $SEED -gt 150 ]
#      then
#        ##### Preprocessing the data given seeds and k numbers
#        python src/graph_data_prepartion/preprocess.py --k $K --seed $SEED
#
#        # Creating Parquet files given the SEED and K numbers (which give the graph path).
#        python src/graph_data_prepartion/add_graph_to_dataset.py --k $K --seed $SEED
#      fi
#    fi

    export WS=$(python -c "print(int((($K * 1))/2))")
    export WStep=$(python -c "print(int(((($K * 1)*30)/2) * .08))")

    echo "Start training with Warmup steps of $WStep, evaluation will be done at each $WS steps"
#        --model_name_or_path  /disk1/sajad/sci-trained-models/grease/checkpoint-239150/ \
#    CUDA_VISIBLE_DEVICES=1 python examples/pytorch/summarization/run_summarization.py \
    python3 -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path /disk1/sajad/sci-trained-models/$MODEL/cnndm-greaseBart/checkpoint-239150/ \
    --output_dir /disk0/$USER/.cache/sci-trained-models/$MODEL/xsum-greaseBart-K$K-SEED$SEED/ \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=16  \
    --save_total_limit 5 \
    --learning_rate 7e-6 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --num_train_epochs 30 \
    --text_column document \
    --summary_column summary \
    --overwrite_output_dir \
    --evaluation_strategy steps \
    --logging_steps 1 \
    --predict_with_generate \
    --max_grad_norm 0.1 \
    --eval_steps $WS --save_steps $WS \
    --train_file $DS_DIR/train.seed$SEED.k$K.parquet \
    --validation_file $DS_DIR/val.seed$SEED.k$K.parquet \
    --test_file $DS_DIR/test.fewShot.parquet \
    --do_predict \
    --do_train \
    --do_eval \
    --load_best_model_at_end True \
    --gradient_accumulation_steps 1 \
    --report_to wandb \
    --run_name $MODEL-fewshot-K$K-SEED$SEED-lr7e-6-lr2e-5 \
    --greater_is_better True\
    --metric_for_best_model rougeL \
    --use_gnn True \
    --warmup_steps $WStep

    rm -rf /disk0/$USER/.cache/sci-trained-models/$MODEL/xsum-K$K-SEED$SEED/checkpoint-*

    python write_to_drive.py \
    --results_path /disk0/$USER/.cache/sci-trained-models/$MODEL/xsum-greaseBart-K$K-SEED$SEED/all_results.json \
    --sheet_key 1sPLEFoMhYuA9ALPfx3xsL04z6tXmn4UisqruwEa_bgA\
    --client_secret /disk1/$USER/client_secret.json \
    --cols AB:AE


  done
done