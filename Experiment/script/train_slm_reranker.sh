export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node 2 \
-m FlagEmbedding.reranker.run \
--output_dir /mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/slm_reranker_baai_epoch5 \
--model_name_or_path BAAI/bge-reranker-v2-m3 \
--train_data /mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Training_Dataset/edge/reranking_edge_15_negatives.jsonl \
--learning_rate 6e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--dataloader_drop_last True \
--train_group_size 16 \
--max_len 512 \
--weight_decay 0.01 \
--logging_steps 10