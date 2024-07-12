export CUDA_VISIBLE_DEVICES=1
python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path BAAI/bge-reranker-v2-m3 \
--input_file /mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Training_Dataset/edge/reranking_edge_full_negatives.jsonl \
--output_file /mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Training_Dataset/edge/finetune_data_minedHN.jsonl \
--range_for_sampling 2-100 \
--negative_number 20 \
--use_gpu_for_searching