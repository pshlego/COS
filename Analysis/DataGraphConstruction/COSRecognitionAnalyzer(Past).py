import json
from tqdm import tqdm


ground_truth_hyperlink_path = "/mnt/sdf/shpark/mnt_sdc/shpark/cos/cos/OTT-QA/ott_dev_linking.json"
cos_recognized_hyperlink_path = "/mnt/sdf/shpark/mnt_sdc/shpark/graph/graph/for_test/all_table_chunks_span_prediction.json"



def main():

    # 1. Parsing
    chunk_to_entities_gt = parseGroundTruthHyperlinks()
    chunk_to_entities_cos = parseAndFilterCOSHyperlinks(chunk_to_entities_gt)
    
    # 2. Analysis
    precision, recall, f1, chunk_id_to_unmatched_entities = evlaluateF1(chunk_to_entities_gt, chunk_to_entities_cos)
    
    # 3. Print results
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    
    with open("Unmatched entities.json", "w") as file:
        json.dump(chunk_id_to_unmatched_entities, file, indent=4)
    with open("/mnt/sdf/shpark/mnt_sdc/shpark/cos_predicted_entities.json", "w") as file:
        json.dump(chunk_to_entities_cos, file, indent=4)




######################################################
# Analysis                                           #
######################################################

def evlaluateF1(chunk_to_entities_gt, chunk_to_entities_cos):
    tp = 0
    fp = 0
    fn = 0
    
    chunk_id_to_unmatched_entities = {}
    
    for chunk_id in chunk_to_entities_gt:
    
        gt_entities = chunk_to_entities_gt[chunk_id]
        try:
            cos_entities = chunk_to_entities_cos.get(chunk_id, [])
        except:
            cos_entities = []
        
        unmatched_entities = {}
        gt_unmatched_entities = []
        cos_unmatched_entities = []
        for gt_entity in gt_entities:
            if gt_entity in cos_entities:
                tp += 1
            else:
                fn += 1
                gt_unmatched_entities.append(gt_entity)
        
        for cos_entity in cos_entities:
            if cos_entity not in gt_entities:
                fp += 1
                cos_unmatched_entities.append(cos_entity)
                
        unmatched_entities["gt"] = gt_unmatched_entities
        unmatched_entities["cos"] = cos_unmatched_entities
        chunk_id_to_unmatched_entities[chunk_id] = unmatched_entities
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1, chunk_id_to_unmatched_entities









######################################################
# Parsing                                            #
######################################################


def getGroundTruthHyperlinks():
    with open(ground_truth_hyperlink_path, 'r') as f:
        data = json.load(f)
    return data


def parseGroundTruthHyperlinks():
    gt_object = getGroundTruthHyperlinks()
    chunk_to_entities = {}
    
    for table_hyperlinks in gt_object:
        chunk_id = table_hyperlinks['chunk_id']
        entities = []
        for positive_ctx in table_hyperlinks['positive_ctxs']:
            entity = (positive_ctx["bert_start"], positive_ctx["bert_end"], positive_ctx["grounding"])
            entities.append(entity)
        
        chunk_to_entities[chunk_id] = entities
    
    return chunk_to_entities





def getCOSHyperlinks():
    with open(cos_recognized_hyperlink_path, 'r') as f:
        data = json.load(f)
    return data

def parseAndFilterCOSHyperlinks(chunk_to_entities_gt):
    cos_object = getCOSHyperlinks()
    chunk_to_entities_cos = {}
    
    for chunk_info in tqdm(cos_object):
        chunk_id = chunk_info['chunk_id']
        if chunk_id not in chunk_to_entities_gt:
            continue
        entities = []
        for grounding in chunk_info['grounding']:
            entity = (grounding["full_word_start"], grounding["full_word_end"], grounding["mention"])
            entities.append(entity)
        
        chunk_to_entities_cos[chunk_id] = entities
    
    return chunk_to_entities_cos










if __name__ == "__main__":
    main()