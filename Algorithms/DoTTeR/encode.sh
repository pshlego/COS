export CUDA_VISIBLE_DEVICES=0,1,2,3
export BASIC_PATH="."
export RATE_MODEL_PATH=/mnt/sde/OTT-QAMountSpace/DoTTeR/model/trained_models/RATE/best_checkpoint
export RT_MODEL_PATH=/mnt/sde/OTT-QAMountSpace/DoTTeR/model/trained_models/dotter

# python /home/shpark/OTT_QA_Workspace/Algorithms/DoTTeR/DoTTeR/scripts/encode_corpus.py \
#   --do_predict \
#   --predict_batch_size 100 \
#   --model_name roberta-base \
#   --shared_encoder \
#   --predict_file /home/shpark/OTT_QA_Workspace/Algorithms/OTT-QA/OTT-QA/released_data/dev.json \
#   --init_checkpoint /mnt/sde/OTT-QAMountSpace/DoTTeR/model/trained_models/dotter/checkpoint_best.pt \
#   --embed_save_path /mnt/sde/OTT-QAMountSpace/DoTTeR/Embeddings/indexed_embeddings/question_dev \
#   --inject_summary \
#   --injection_scheme "column" \
#   --rate_model_path /mnt/sde/OTT-QAMountSpace/DoTTeR/model/trained_models/RATE/best_checkpoint\
#   --normalize_summary_table \
#   --max_c_len 512 \
#   --num_workers 8

export DATA_PATH=/mnt/sde/OTT-QAMountSpace/DoTTeR/preprocessed_data/retrieval
export TABLE_CORPUS=table_corpus_blink

python /home/shpark/OTT_QA_Workspace/Algorithms/DoTTeR/DoTTeR/scripts/encode_corpus.py \
    --do_predict \
    --encode_table \
    --shared_encoder \
    --predict_batch_size 1200 \
    --model_name roberta-base \
    --predict_file /mnt/sde/OTT-QAMountSpace/DoTTeR/preprocessed_data/retrieval/table_corpus_blink.pkl \
    --init_checkpoint /mnt/sde/OTT-QAMountSpace/DoTTeR/model/trained_models/dotter/checkpoint_best.pt \
    --embed_save_path /mnt/sde/OTT-QAMountSpace/DoTTeR/Embeddings/indexed_embeddings/table_corpus_blink \
    --inject_summary \
    --injection_scheme "column" \
    --rate_model_path /mnt/sde/OTT-QAMountSpace/DoTTeR/model/trained_models/RATE/best_checkpoint\
    --normalize_summary_table \
    --max_c_len 512 \
    --num_workers 24