#COMENTARIOS EN ESPANOL
#AUTOR: JUAN JOSE CERVANTES

#Instalar dependencias en colab
!pip install -q transformers datasets accelerate peft bitsandbytes

#Para cargar el dataset con toda la informacion
from datasets import Dataset
import json

with open('dt.json', 'r', encoding='utf-8') as f:#Abrimos el dataset como archivo
    data = json.load(f)

if isinstance(data, dict):
    data = [data]

dataset = Dataset.from_list(data)
dataset = dataset.select(range(2000))#Seleccionamos un rango, archivo hasta la fecha tiene 10k registros

print("\nPrimeras 2 muestras:")#Muestras de lo que viene en el dataset
for i in range(min(2, len(dataset))):
    print(f"Input: {dataset[i]['input']}")
    print(f"Output: {dataset[i]['output']}\n")


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"#De la libreria transformers de huggingface tomamos el modelo de Deepseek

tokenizer = AutoTokenizer.from_pretrained(#Lo tokenizamos
    model_name,
    trust_remote_code=True,
    padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(#Configuraciones del modelo
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

#Procesamiento de como viene el dataset
#Separar las preguntas(Input) de las respuestas(Output)
def format_instruction(example):
    return {
        "text": f"### Input: {example['input']}\n### Output: {example['output']}"
    }

dataset = dataset.map(format_instruction)#Convertimos el dataset

def tokenize(example):#Tokenizamos con un maximo de 128 por ahora
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)#Tokenizamos en base al dataset


from transformers import Trainer, TrainingArguments

#Configuracion del entrenamiento, como steps, y batch_size que son los mas importantes y la salida del modelo
training_args = TrainingArguments(
    output_dir="./qwen-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=100,
    learning_rate=5e-5,
    bf16=True,                   
    optim="adamw_torch",
    report_to="none"
)

#Al entrenador le pasamos todos los parametros como modelo, argumentos, tokenizador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

#Se inicia el entrenamiento
print("Iniciando entrenamiento...")
trainer.train()

#Prueba sencilla del entrenamiento
input_text = "### Input: ¿Cuál es tu película favorita?\n### Output:"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

#La salida de como deberia generar respuestas como el maximo de tokens(longitud de la respuesta) y temperatura("creatividad")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True
)

print("\nRespuesta generada:")#Respuesta que es generada
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
