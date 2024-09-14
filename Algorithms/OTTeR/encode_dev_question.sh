export CUDA_VISIBLE_DEVICES=3
python /home/shpark/OTT_QA_Workspace/Algorithms/OTTeR/OTTeR/scripts/encode_corpus.py \
    --do_predict \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --shared_encoder \
    --predict_file /home/shpark/OTT_QA_Workspace/Algorithms/OTT-QA/OTT-QA/released_data/dev.json \
    --init_checkpoint /mnt/sde/OTT-QAMountSpace/OTTeR/checkpoint-87000/checkpoint_best.pt \
    --embed_save_path /mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/question_dev \
    --fp16 \
    --max_c_len 512 \
    --num_workers 8  2>&1 |tee ./encode_corpus_dev.log