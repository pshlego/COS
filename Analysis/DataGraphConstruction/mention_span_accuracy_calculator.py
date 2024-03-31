import json
from tqdm import tqdm
import argparse


ground_truth_mentions_path = "/mnt/sde/shpark/graph_constructer/mention_detector/gold/gt_dev_entities_chunks_w_exception_handling.json"
cos_inferred_mentions_path = "/mnt/sde/shpark/graph_constructer/mention_detector/detected_mentions_from_dev_table_chunks_w_exception_handling.json"

analysis_result_path = "/mnt/sdf/OTT-QAMountSpace/AnalysisResults/COS/DataGraphConstructor/mention_detection_details.json"


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_mentions_path", type = str, default = ground_truth_mentions_path)
    parser.add_argument("--cos_inferred_mentions_path", type = str, default = cos_inferred_mentions_path)
    parser.add_argument("--metric_type", type = str, choices = ["strict_f1", "finegrained_f1"], default = None)
        
    return parser.parse_args()

def main():

    args = parseArguments()

    # 1. Parsing
    chunk_id_to_mentions_gt  = parseMentionFile(ground_truth_mentions_path)
    chunk_id_to_mentions_cos = parseMentionFile(cos_inferred_mentions_path)
    
    
    # 2. Analysis
    strict_result, finegrained_result, chunk_id_to_details = evlaluateF1(args.metric_type, chunk_id_to_mentions_gt, chunk_id_to_mentions_cos)
    
    
    # 3. Print results
    print("[[ Strict F1 Results ]]")
    print("Recall: ", strict_result["recall"])
    print("Precision: ", strict_result["precision"])
    print("F1: ", strict_result["f1"])
    
    print()
    print("[[ Fine-grained F1 Results ]]")
    print("Recall: ", finegrained_result["recall"])
    print("Precision: ", finegrained_result["precision"])
    print("F1: ", finegrained_result["f1"])
    
    with open(analysis_result_path, "w") as file:
        json.dump(chunk_id_to_details, file, indent = 4)




######################################################
# Analysis                                           #
######################################################

def evlaluateF1(metric_type, chunk_id_to_mentions_gt, chunk_id_to_mentions_cos):
    
    # 1. Calculate strict f1 results
    strict_result = {}
    chunk_id_to_details = {}
    tp = 0
    fp = 0
    fn = 0
    for chunk_id in chunk_id_to_mentions_gt:
    
        gt_mentions = chunk_id_to_mentions_gt[chunk_id]
        if chunk_id in chunk_id_to_mentions_cos: cos_mentions = chunk_id_to_mentions_cos.get(chunk_id, [])
        else: cos_mentions = []
        
        details = {}
        true_positives = []
        false_positives = []
        false_negatives = []

        for gt_mention in gt_mentions:
            
            for cos_mention in cos_mentions:
                tp_found = False
                if gt_mention["full_word_start"] == cos_mention["full_word_start"] and gt_mention["full_word_end"] == cos_mention["full_word_end"]:
                    tp += 1
                    tp_found = True
                    true_positives.append(simplifyMention(gt_mention))
                    break
            
            if not tp_found:
                fn += 1
                false_negatives.append(simplifyMention(gt_mention))
        
        for cos_mention in cos_mentions:
            found_in_gt = False
            for gt_mention in gt_mentions:
                if gt_mention["full_word_start"] == cos_mention["full_word_start"] and gt_mention["full_word_end"] == cos_mention["full_word_end"]:
                    found_in_gt = True
                    break

            if not found_in_gt:
                fp += 1
                false_positives.append(simplifyMention(cos_mention))
                
        details["True Positives"] = true_positives
        details["False Negatives"] = false_negatives
        details["False Positives"] = false_positives
        
        chunk_id_to_details[chunk_id] = details
    
    strict_result["recall"]     = tp / (tp + fn)
    strict_result["precision"]  = tp / (tp + fp)
    strict_result["f1"]         = 2 * strict_result["precision"] * strict_result["recall"] / (strict_result["precision"] + strict_result["recall"])
    
    
    # 2. Calculate finegrained f1 result
    finegrained_result = {}
    tp = 0
    fp = 0
    fn = 0
    for chunk_id in chunk_id_to_mentions_gt:
    
        gt_mentions = chunk_id_to_mentions_gt[chunk_id]
        if chunk_id in chunk_id_to_mentions_cos: cos_mentions = chunk_id_to_mentions_cos.get(chunk_id, [])
        else: cos_mentions = []
        
        gt_mention_token_set = set()
        cos_mention_token_set = set()

        for gt_mention in gt_mentions:
            gt_mention_token_set.update(set(range(gt_mention["full_word_start"], gt_mention["full_word_end"] + 1)))
        
        for cos_mention in cos_mentions:
            cos_mention_token_set.update(set(range(cos_mention["full_word_start"], cos_mention["full_word_end"] + 1)))
    
        tp += len(gt_mention_token_set.intersection(cos_mention_token_set))
        fp += len(cos_mention_token_set - gt_mention_token_set)
        fn += len(gt_mention_token_set - cos_mention_token_set)
        
    finegrained_result["recall"]     = tp / (tp + fn)
    finegrained_result["precision"]  = tp / (tp + fp)
    finegrained_result["f1"]         = 2 * finegrained_result["precision"] * finegrained_result["recall"] / (finegrained_result["precision"] + finegrained_result["recall"])
    
    
    return strict_result, finegrained_result, chunk_id_to_details









######################################################
# Parsing                                            #
######################################################

def parseMentionFile(file_path):
    with open(file_path, "r") as file:
        chunk_infos = json.load(file)
        
    chunk_id_to_mentions = {}
        
    for chunk_info in tqdm(chunk_infos):
        chunk_id = chunk_info["chunk_id"]
        mentions = chunk_info["grounding"]
    
        chunk_id_to_mentions[chunk_id] = mentions
    
    return chunk_id_to_mentions



######################################################
# Utils                                              #
######################################################

def simplifyMention(mention):
    simplified_mention = {}
    
    simplified_mention["mention"] = mention["mention"]
    simplified_mention["token_idx_start"] = mention["full_word_start"]
    simplified_mention["token_idx_end"] = mention["full_word_end"]
    simplified_mention["row"] = mention["row_id"]

    return simplified_mention









if __name__ == "__main__":
    main()