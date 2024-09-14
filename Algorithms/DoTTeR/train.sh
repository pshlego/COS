#!/bin/bash 
NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=${NUM_GPUS} /home/shpark/OTT_QA_Workspace/Algorithms/DoTTeR/DoTTeR/scripts/train_false_positive_removal.py \
    --train_file /mnt/sde/OTT-QAMountSpace/DoTTeR/preprocessed_data/false_positive_removal/train_intable_bm25_blink_false_positive_removal.pkl \
    --dev_file /mnt/sde/OTT-QAMountSpace/DoTTeR/preprocessed_data/false_positive_removal/dev__blink_false_positive_removal.pkl \
    --seed 42 \
    --effective_batch_size 32 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --model_name_or_path bert-base-cased \
    --do_train_and_eval \
    --logging_steps 10 \
    --output_dir "/mnt/sde/OTT-QAMountSpace/DoTTeR/model/trained_models/false_positive_removal"