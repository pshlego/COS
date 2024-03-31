#!/bin/bash

# Command 1
echo "Running generate_dense_embeddings for ott_table..."
CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/UDT-QA/DPR/generate_dense_embeddings.py ctx_src=ott_table out_file=/mnt/sdd/shpark/embeds/ott_table_original target_expert=1 batch_size=2048

# Command 2
echo "Running generate_dense_embeddings for ott_wiki_passages..."
CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/UDT-QA/DPR/generate_dense_embeddings.py ctx_src=ott_wiki_passages out_file=/mnt/sdd/shpark/embeds/ott_wiki_linker target_expert=3 batch_size=2048

# Command 3
echo "Running run_chain_of_skills_ott for span proposal..."
CUDA_VISIBLE_DEVICES=0 python /home/shpark/COS/DPR/run_chain_of_skills_ott.py model_file=/mnt/sdd/shpark/cos/models/cos_nq_ott_hotpot_finetuned_6_experts.ckpt encoder.use_moe=True encoder.moe_type=mod2:attn encoder.num_expert=6 encoder.encoder_model_type=hf_cos encoded_ctx_files=[/mnt/sdd/shpark/cos/embeds/ott_table_original*] qa_dataset=/mnt/sdd/shpark/cos/knowledge/ott_table_chunks_original.json do_span=True ctx_datatsets=[/mnt/sdd/shpark/cos/knowledge/ott_table_chunks_original.json,/mnt/sdd/shpark/cos/knowledge/ott_wiki_passages.json,[/mnt/sdd/shpark/cos/models/table_chunks_to_passages*]] label_question=True hop1_limit=100 hop1_keep=200 batch_size=32

# Command 4
echo "Running run_chain_of_skills_ott for entity linking..."
CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/COS/DPR/run_chain_of_skills_ott.py model_file=/mnt/sdd/shpark/cos/models/cos_nq_ott_hotpot_finetuned_6_experts.ckpt encoder.use_moe=True encoder.moe_type=mod2:attn encoder.num_expert=6 encoder.encoder_model_type=hf_cos encoded_ctx_files=[/mnt/sdd/shpark/cos/embeds/ott_wiki_linker*] qa_dataset=/mnt/sdd/shpark/cos/models/all_table_chunks_span_prediction.json do_link=True ctx_datatsets=[/mnt/sdd/shpark/cos/knowledge/ott_table_chunks_original.json,/mnt/sdd/shpark/cos/knowledge/ott_wiki_passages.json,[/mnt/sdd/shpark/cos/models/table_chunks_to_passages*]] hop1_limit=100 hop1_keep=200

# Command 5
echo "Running run_chain_of_skills_ott for chain of skills..."
CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/UDT-QA/DPR/run_chain_of_skills_ott.py model_file=/mnt/sdd/shpark/cos/models/cos_nq_ott_hotpot_finetuned_6_experts.ckpt encoder.use_moe=True encoder.moe_type=mod2:attn encoder.num_expert=6 encoder.encoder_model_type=hf_cos encoded_ctx_files=[/mnt/sdd/shpark/embeds/ott_table_original*] qa_dataset=/mnt/sdd/shpark/retriever/OTT-QA/ott_dev_q_to_tables_with_bm25neg.json do_cos=True ctx_datatsets=[/mnt/sdd/shpark/knowledge/ott_table_chunks_original.json,/mnt/sdd/shpark/knowledge/ott_wiki_passages.json,[/mnt/sdd/shpark/OTT_table_to_pasg_links/table_chunks_to_passages*]] hop1_limit=100 hop1_keep=200
