import json
from Algorithms.ChainOfSkills.FiE_reader.hotpot_evaluate_v1 import f1_score, exact_match_score
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def main():
    path = "/mnt/sdf/OTT-QAMountSpace/ExperimentResults/reading_output/few_shot_5.jsonl" #_oracle.jsonl"
    reading_outputs = read_jsonl(path)
    ems, f1s = [], []
    for reading_output in reading_outputs:
        gold_answers = reading_output["gold"]
        predicted_answer = reading_output["pred"]
        if predicted_answer != "":
            if predicted_answer[-1] == ".":
                predicted_answer = predicted_answer[:-1]

        ems.append(max([exact_match_score(predicted_answer, gold_answer) for gold_answer in gold_answers]))
        f1s.append(max([f1_score(predicted_answer, gold_answer)[0] for gold_answer in gold_answers]))

    avg_f1 = sum(f1s) / len(f1s)
    avg_em = sum(ems) / len(ems)

    print(f"Average F1: {avg_f1}")
    print(f"Average EM: {avg_em}")


if __name__ == "__main__":
    main()