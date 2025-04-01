import json

with open("magistradoDataset2.json", "r", encoding="utf-8") as f:
    datos = json.load(f) 

with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for item in datos:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")