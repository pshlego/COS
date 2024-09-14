# python /home/shpark/OTT_QA_Workspace/Algorithms/OTTeR/OTTeR/scripts/eval_ottqa_retrieval.py \
# 	 --raw_data_path /home/shpark/OTT_QA_Workspace/Algorithms/OTT-QA/OTT-QA/released_data/dev.json \
# 	 --eval_only_ans \
# 	 --query_embeddings_path /mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/question_dev.npy \
# 	 --corpus_embeddings_path /mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/indexed_embeddings/table_corpus_blink.npy \
# 	 --id2doc_path /mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/indexed_embeddings/table_corpus_blink/id2doc.json \
#      --output_save_path /mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/indexed_embeddings/dev_output_k100_table_corpus_blink.json \
#      --beam_size 100  2>&1 |tee ./results_retrieval_dev.log

python /home/shpark/OTT_QA_Workspace/Algorithms/DoTTeR/DoTTeR/scripts/eval_ottqa_retrieval.py \
	 --raw_data_path /home/shpark/OTT_QA_Workspace/Algorithms/OTT-QA/OTT-QA/released_data/dev.json \
	 --eval_only_ans \
	 --query_embeddings_path /mnt/sde/OTT-QAMountSpace/DoTTeR/Embeddings/indexed_embeddings/question_dev.npy \
	 --corpus_embeddings_path /mnt/sde/OTT-QAMountSpace/DoTTeR/Embeddings/indexed_embeddings/table_corpus_blink.npy \
	 --id2doc_path /mnt/sde/OTT-QAMountSpace/DoTTeR/Embeddings/indexed_embeddings/table_corpus_blink/id2doc.json \
   --output_save_path /mnt/sde/OTT-QAMountSpace/DoTTeR/Embeddings/indexed_embeddings/dev_output_k100_table_corpus_blink.json \
   --beam_size 100