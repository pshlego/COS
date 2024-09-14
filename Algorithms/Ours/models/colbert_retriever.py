import json
from ColBERT.colbert import Searcher
from ColBERT.colbert.infra import ColBERTConfig
from utils.utils import read_jsonl, disablePrint, enablePrint

class ColBERTRetriever:
    def __init__(self,
                 index_name = "top1_edge_embeddings_v2_trained_1_epoch_bsize_512.nbits2",
                 ids_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/edge/top_1/index_to_chunk_id_edge_topk_1.json", 
                 collection_path = "/mnt/sdf/OTT-QAMountSpace/Dataset/Ours/Embedding_Dataset/edge/top_1/edge_topk_1.tsv", 
                 index_root_path = "/mnt/sdc/shpark/OTT-QAMountSpace/Embeddings", 
                 checkpoint_path = "/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/ColBERT/edge_trained_v2"
                ):

        print("Loading id mappings...")
        self.id_to_key = json.load(open(ids_path))
        print("Loaded id mappings!")

        print("Loading index...")
        disablePrint()
        self.searcher = Searcher(index=index_name, config=ColBERTConfig(), collection=collection_path, index_root=index_root_path, checkpoint=checkpoint_path)
        enablePrint()
        print("Loaded index complete!")

    def search(self, query, k=10000):
        retrieved_info = self.searcher.search(query, k = k)
        retrieved_id_list = retrieved_info[0]
        retrieved_score_list = retrieved_info[2]
        
        retrieved_key_list = [self.id_to_key[str(id)] for id in retrieved_id_list]
        
        return retrieved_key_list, retrieved_score_list