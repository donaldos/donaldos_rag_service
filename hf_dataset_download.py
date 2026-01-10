hf_tokenizer = "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"
import json
from datasets import load_dataset

if __name__ == "__main__":    
    dataset = load_dataset("allganize/RAG-Evaluation-Dataset-KO", token=hf_tokenizer)
    ds = dataset["test"]   # 핵심!

    dataset.save_to_disk("./RAG_KO_dataset")  # split 포함 전체 저장

    with open("rag_eval_ko.jsonl", "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


    with open("rag_eval_ko.jsonl", "r", encoding="utf-8") as f:
        for row in f:
            row = json.loads(row)
            print(row['target_file_name'])