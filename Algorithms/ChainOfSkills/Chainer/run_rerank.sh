#!/bin/bash

# Compute chainer score for OTT-QA hop1 evidences 
CUDA_VISIBLE_DEVICES=0 python /home/shpark/OTT_QA_Workspace/Algorithms/ChainOfSkills/Chainer/rerank_passages.py --retriever_results /mnt/sde/OTT-QAMountSpace/CORE/Dataset/data/retriever_results/OTT-QA/dev_hop1_retrieved_results.json \
--passage_path /mnt/sde/OTT-QAMountSpace/CORE/Dataset/data/knowledge/ott_table_chunks_original.json --output_path /mnt/sde/OTT-QAMountSpace/CORE/Cache_3 --b_size 50 --num_shards 4 --shard_id 0 --do_tables

# Compute chainer score for OTT-QA hop2 evidences
# CUDA_VISIBLE_DEVICES=0 python /home/shpark/OTT_QA_Workspace/Algorithms/ChainOfSkills/Chainer/rerank_passages.py --retriever_results /mnt/sde/OTT-QAMountSpace/CORE/Dataset/data/retriever_results/OTT-QA/dev_hop1_retrieved_results.json \
# --table_pasg_links_path /mnt/sde/OTT-QAMountSpace/CORE/ModelCheckpoints/models/table_chunks_to_passages_shard* \
# --passage_path /mnt/sde/OTT-QAMountSpace/CORE/Dataset/data/knowledge/ott_wiki_passages.json --output_path /mnt/sde/OTT-QAMountSpace/CORE/Cache_2 --b_size 15 --num_shards 4 --shard_id 0

# # Compute chainer score for NQ hop1 evidences 
# CUDA_VISIBLE_DEVICES=0 python rerank_passages.py --retriever_results nq_full_dev.json \
# --output_path your_output_dir_for_score_cache --b_size 50 --num_shards 10 --nq

# # Compute chainer score for NQ hop2 evidences
# CUDA_VISIBLE_DEVICES=0 python rerank_passages.py --retriever_results nq_full_dev.json \
# --output_path your_output_dir_for_score_cache --b_size 50 --num_shards 10 --nq \
# --table_pasg_links_path path_to_nq_linker_results/nq_table_chunks_to_passages_shard* --passage_path psgs_w100.tsv --nq_link


# Run chainer for OTT-QA 
# CUDA_VISIBLE_DEVICES=0 python /home/shpark/OTT_QA_Workspace/Algorithms/ChainOfSkills/Chainer/run_chainer.py --mode ott --retriever_results /mnt/sde/OTT-QAMountSpace/CORE/Dataset/data/retriever_results/OTT-QA/dev_hop1_retrieved_results.json \
# --table_pasg_links_path /mnt/sde/OTT-QAMountSpace/CORE/ModelCheckpoints/models/table_chunks_to_passages_shard* \
# --passage_path /mnt/sde/OTT-QAMountSpace/CORE/Dataset/data/knowledge/ott_wiki_passages.json --table_path /mnt/sde/OTT-QAMountSpace/CORE/Dataset/data/knowledge/ott_table_chunks_original.json \
# --previous_cache /mnt/sde/OTT-QAMountSpace/CORE/Cache_2/final_score_cache.json --output_path /mnt/sde/OTT-QAMountSpace/CORE/Reader --split dev

# # Run chainer for NQ 
# # note that for output_path, it should be different from the input path (retriever_results path)
# python run_chainer.py --mode nq --retriever_results nq_retriever_results_test.json \
# --table_pasg_links_path path_to_nq_linker_results/nq_table_chunks_to_passages_shard* \
# --passage_path psgs_w100.tsv --previous_cache aggregated_NQ_score_cache.json \
# --output_path your_output_dir_for_reader_data 