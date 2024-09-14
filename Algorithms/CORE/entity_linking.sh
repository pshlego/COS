export CUDA_VISIBLE_DEVICES=0,1,2,3
# python /home/shpark/OTT_QA_Workspace/Algorithms/ChainOfSkills/DPR/dense_retrieve_link.py model_file=/mnt/sde/OTT-QAMountSpace/CORE/ModelCheckpoints/models/core_span_proposal.ckpt \
# qa_dataset=/mnt/sde/OTT-QAMountSpace/CORE/Dataset/data/knowledge/ott_table_chunks_original.json do_span=True label_question=True

python /home/shpark/OTT_QA_Workspace/Algorithms/ChainOfSkills/DPR/dense_retrieve_link.py model_file=/mnt/sde/OTT-QAMountSpace/CORE/ModelCheckpoints/models/core_table_linker.ckpt encoded_ctx_files=[/mnt/sde/OTT-QAMountSpace/CORE/Embeddings/ott_wiki*] \
qa_dataset=/mnt/sde/OTT-QAMountSpace/CORE/ModelCheckpoints/models/all_table_chunks_span_prediction.json do_link=True