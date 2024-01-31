# Retrieval
## star
#CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/COS/Graph_constructer/subgraph_retriever_colbert.py graph_collection_name_list=[preprocess_table_graph_author_w_score_2_star] doc_ids=/mnt/sdd/shpark/colbert/data/index_to_chunk_id_star.json hierarchical_level=star
CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/COS/Graph_constructer/subgraph_retriever_colbert.py graph_collection_name_list=[preprocess_table_graph_author_w_score_2_star] doc_ids=/mnt/sdd/shpark/colbert/data/index_to_chunk_id_star.json hierarchical_level=star scale=larger
# CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/COS/Graph_constructer/subgraph_retriever_colbert.py graph_collection_name_list=[preprocess_table_graph_author_w_score_2_star] doc_ids=/mnt/sdd/shpark/colbert/data/index_to_chunk_id_star.json hierarchical_level=star full_length_search=True

## edge
#CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/COS/Graph_constructer/subgraph_retriever_colbert.py graph_collection_name_list=[preprocess_table_graph_author_w_score_2_edge] doc_ids=/mnt/sdd/shpark/colbert/data/index_to_chunk_id_edge.json hierarchical_level=edge
CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/COS/Graph_constructer/subgraph_retriever_colbert.py graph_collection_name_list=[preprocess_table_graph_author_w_score_2_edge] doc_ids=/mnt/sdd/shpark/colbert/data/index_to_chunk_id_edge.json hierarchical_level=edge scale=larger
# CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/COS/Graph_constructer/subgraph_retriever_colbert.py graph_collection_name_list=[preprocess_table_graph_author_w_score_2_edge] doc_ids=/mnt/sdd/shpark/colbert/data/index_to_chunk_id_edge.json hierarchical_level=edge full_length_search=True

## both
#CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/COS/Graph_constructer/subgraph_retriever_colbert.py graph_collection_name_list=[preprocess_table_graph_author_w_score_2_star,preprocess_table_graph_author_w_score_2_edge] doc_ids=/mnt/sdd/shpark/colbert/data/index_to_chunk_id_both.json hierarchical_level=both
CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/COS/Graph_constructer/subgraph_retriever_colbert.py graph_collection_name_list=[preprocess_table_graph_author_w_score_2_star,preprocess_table_graph_author_w_score_2_edge] doc_ids=/mnt/sdd/shpark/colbert/data/index_to_chunk_id_both.json hierarchical_level=both scale=larger
# CUDA_VISIBLE_DEVICES=0,1,2 python /home/shpark/COS/Graph_constructer/subgraph_retriever_colbert.py graph_collection_name_list=[preprocess_table_graph_author_w_score_2_star,preprocess_table_graph_author_w_score_2_edge] doc_ids=/mnt/sdd/shpark/colbert/data/index_to_chunk_id_both.json hierarchical_level=both full_length_search=True