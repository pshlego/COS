export CUDA_VISIBLE_DEVICES=1
python /home/shpark/OTT_QA_Workspace/Algorithms/OTTeR/OTTeR/scripts/encode_corpus.py \
    --do_predict \
    --encode_table \
    --shared_encoder \
    --predict_batch_size 1024 \
    --model_name roberta-base \
    --predict_file /mnt/sde/OTT-QAMountSpace/OTTeR/Dataset/preprocessed_data/retrieval/table_corpus_blink.pkl \
    --init_checkpoint /mnt/sde/OTT-QAMountSpace/OTTeR/checkpoint-87000/checkpoint_best.pt \
    --embed_save_path /mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/indexed_embeddings/table_corpus_blink \
    --fp16 \
    --max_c_len 512 \
    --num_workers 24  2>&1 |tee ./encode_corpus_table_blink.log

python /home/shpark/OTT_QA_Workspace/Algorithms/OTTeR/OTTeR/scripts/eval_ottqa_retrieval.py \
	 --raw_data_path /home/shpark/OTT_QA_Workspace/Algorithms/OTT-QA/OTT-QA/released_data/dev.json \
	 --eval_only_ans \
	 --query_embeddings_path /mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/question_dev.npy \
	 --corpus_embeddings_path /mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/indexed_embeddings/table_corpus_blink.npy \
	 --id2doc_path /mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/indexed_embeddings/table_corpus_blink/id2doc.json \
     --output_save_path /mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/indexed_embeddings/dev_output_k100_table_corpus_blink.json \
     --beam_size 100  2>&1 |tee ./results_retrieval_dev.log
     