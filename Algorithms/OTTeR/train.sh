export CUDA_VISIBLE_DEVICES=0,1,2,3
python /home/shpark/OTT_QA_Workspace/Algorithms/OTTeR/OTTeR/scripts/train_1hop_tb_retrieval.py \
  --do_train \
  --prefix 0 \
  --predict_batch_size 800 \
  --model_name roberta-base \
  --shared_encoder \
  --train_batch_size 64 \
  --fp16 \
  --init_checkpoint /mnt/sde/OTT-QAMountSpace/OTTeR/checkpoint-87000/checkpoint_best.pt \
  --max_c_len 512 \
  --max_q_len 70 \
  --num_train_epochs 20 \
  --warmup_ratio 0.1 \
  --train_file /mnt/sde/OTT-QAMountSpace/OTTeR/Dataset/preprocessed_data/retrieval/train_intable_contra_blink_row.pkl \
  --predict_file /mnt/sde/OTT-QAMountSpace/OTTeR/Dataset/preprocessed_data/retrieval/dev_intable_contra_blink_row.pkl \
  --output_dir /mnt/sde/OTT-QAMountSpace/OTTeR/models/otter \
  2>&1 |tee ./retrieval_training.log

python /home/shpark/OTT_QA_Workspace/Algorithms/OTTeR/OTTeR/scripts/encode_corpus.py \
    --do_predict \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --shared_encoder \
    --predict_file /home/shpark/OTT_QA_Workspace/Algorithms/OTT-QA/OTT-QA/released_data/dev.json \
    --init_checkpoint /mnt/sde/OTT-QAMountSpace/OTTeR/models/otter/checkpoint_best.pt \
    --embed_save_path /mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/question_dev \
    --fp16 \
    --max_c_len 512 \
    --num_workers 8  2>&1 |tee ./encode_corpus_dev.log

python /home/shpark/OTT_QA_Workspace/Algorithms/OTTeR/OTTeR/scripts/encode_corpus.py \
    --do_predict \
    --encode_table \
    --shared_encoder \
    --predict_batch_size 1024 \
    --model_name roberta-base \
    --predict_file /mnt/sde/OTT-QAMountSpace/OTTeR/Dataset/preprocessed_data/retrieval/table_corpus_blink.pkl \
    --init_checkpoint /mnt/sde/OTT-QAMountSpace/OTTeR/models/otter/checkpoint_best.pt \
    --embed_save_path /mnt/sde/OTT-QAMountSpace/OTTeR/Embeddings/indexed_embeddings/table_corpus_blink \
    --fp16 \
    --max_c_len 512 \
    --num_workers 24  2>&1 |tee ./encode_corpus_table_blink.log