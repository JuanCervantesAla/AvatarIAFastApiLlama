# ========================================
# 1. Instalar dependencias (hazlo antes en tu terminal):
# pip install transformers datasets accelerate peft bitsandbytes
# ========================================

# ========================================
# 2. Cargar tu dataset JSON
# ========================================
from datasets import Dataset
import json

with open('dt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

if isinstance(data, dict):
    data = [data]

dataset = Dataset.from_list(data)
dataset = dataset.select(range(2000))

print("\nPrimeras 2 muestras:")
for i in range(min(2, len(dataset))):
    print(f"Input: {dataset[i]['input']}")
    print(f"Output: {dataset[i]['output']}\n")

# ========================================
# 3. Cargar modelo y tokenizer (con soporte para FP16)
# ========================================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,     # ✅ FP16 para RTX 3090
    device_map="auto"
)

# ========================================
# 4. Preprocesamiento y tokenización
# ========================================
def format_instruction(example):
    return {
        "text": f"### Input: {example['input']}\n### Output: {example['output']}"
    }

dataset = dataset.map(format_instruction)

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# ========================================
# 5. Configurar entrenamiento
# ========================================
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./qwen-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=100,
    learning_rate=5e-5,
    fp16=True,                    # ✅ Usa fp16 para la 3090
    optim="adamw_torch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# ========================================
# 6. ¡Entrenar el modelo!
# ========================================
print("Iniciando entrenamiento...")
trainer.train()

# ========================================
# 7. Probar el modelo entrenado
# ========================================
input_text = "### Input: ¿Cuál es tu película favorita?\n### Output:"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True
)

print("\nRespuesta generada:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
