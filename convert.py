from datasets import Dataset
import json

# Your original dataset (list of dictionaries with "text" field)
import json
with open("datasetF_refinado_antialucinaciones.json", "r", encoding="utf-8") as f:
    original_data = json.load(f)

# --- Convert to color dataset format ---
def convert_qa_to_color_format(original_data):
    converted_data = []
    for entry in original_data:
        # Split into question and answer
        parts = entry["text"].split("\n")
        if len(parts) != 2:
            continue  # Skip malformed entries
        
        question = parts[0].replace("### Pregunta: ", "").strip()
        answer = parts[1].replace("### Respuesta: ", "").strip()
        
        converted_data.append({
            "input": question,
            "output": answer
        })
    return converted_data

# Convert the data
converted_data = convert_qa_to_color_format(original_data)

# --- Save to a Hugging Face Dataset ---
hf_dataset = Dataset.from_list(converted_data)

# Print the first 2 entries to verify
print("Converted Dataset Samples:")
for i in range(2):
    print(f"{i+1}. Input: {hf_dataset[i]['input']}")
    print(f"   Output: {hf_dataset[i]['output']}\n")

# --- (Optional) Save to JSON/CSV ---
# Save as JSON
with open("converted_dataset.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

# Save as CSV
import pandas as pd
pd.DataFrame(converted_data).to_csv("converted_dataset.csv", index=False)

print("✅ Conversion complete! Saved to 'converted_dataset.json' and 'converted_dataset.csv'.")