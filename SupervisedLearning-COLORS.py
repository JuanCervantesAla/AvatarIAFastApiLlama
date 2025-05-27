# 1. Install necessary dependencies
!pip install -q transformers datasets accelerate peft bitsandbytes

# 2. Load dataset - using a different approach since the original dataset isn't available
from datasets import Dataset
import pandas as pd

# Sample color data (replace this with your actual data)
color_data = {
    "Description": [
        "A vibrant red like a ripe tomato",
        "Deep blue resembling the ocean",
        "Bright yellow like the sun",
        "Soft green like fresh grass"
    ],
    "Color Name": [
        "Tomato Red",
        "Ocean Blue",
        "Sun Yellow",
        "Grass Green"
    ]
}

# Create a pandas DataFrame
df = pd.DataFrame(color_data)

# Convert to Hugging Face dataset
hf_dataset = Dataset.from_pandas(df)

# 3. Preprocess the data
filtered_data = [
    {"input": entry["Description"], "output": entry["Color Name"]}
    for entry in hf_dataset
    if entry.get("Description") and entry.get("Color Name")
]

hf_dataset = Dataset.from_list(filtered_data)

# 4. Load model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto"
)

# 5. Tokenize the data
def tokenize(example):
    prompt = f"Descripción: {example['input']}\nNombre del color:"
    target = example["output"]
    full_input = prompt + " " + target
    tokens = tokenizer(full_input, truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = hf_dataset.map(tokenize)

# 6. Configure and run training
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./color_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

# 7. Test the trained model
input_text = "Descripción: A captivating blend of medium purple and blue hues.\nNombre del color:"
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

output = model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))