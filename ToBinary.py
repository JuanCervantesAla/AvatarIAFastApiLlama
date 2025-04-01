import json
import numpy as np
from pathlib import Path

def jsonl_to_bin(jsonl_path, bin_path):
    texts = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'].encode('utf-8'))
    
    
    np_array = np.array(texts, dtype=object)
    np.save(bin_path, np_array, allow_pickle=True)

jsonl_to_bin('dataset.jsonl', 'dataset.bin')
print("Conversion Done")