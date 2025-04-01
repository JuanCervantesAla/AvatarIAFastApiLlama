
# Importar todas las librerías necesarias
from transformers import TrainerCallback
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
import pandas as pd

# 1. Cargar modelo y tokenizador con configuración adecuada
model_name = "datificate/gpt2-small-spanish"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Configurar token de padding

# Cargar modelo con configuración segura
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Cambiado a float32 para mayor estabilidad
    device_map="auto"
)

# 2. Configurar LoRA específicamente para GPT-2
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],  # Módulos correctos para GPT-2
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    fan_in_fan_out=True  # Necesario para modelos GPT-2
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Preparar dataset de historia con formato correcto
history_data = [
    {"text": "### Pregunta: ¿Quién fue Simón Bolívar?\n### Respuesta: Simón Bolívar fue un líder militar venezolano que liberó a varios países sudamericanos del dominio español."},
    {"text": "### Pregunta: ¿Cuándo comenzó la Segunda Guerra Mundial?\n### Respuesta: La Segunda Guerra Mundial comenzó el 1 de septiembre de 1939 con la invasión alemana de Polonia."},
    {"text": "### Pregunta: ¿Qué fue el Renacimiento?\n### Respuesta: El Renacimiento fue un movimiento cultural que surgió en Italia en el siglo XIV, caracterizado por el redescubrimiento del arte y filosofía clásica."},
    {"text": "### Pregunta: ¿Quién fue Cleopatra?\n### Respuesta: Cleopatra fue la última gobernante del Reino Ptolemaico de Egipto, conocida por sus alianzas con Julio César y Marco Antonio."},
    {"text": "### Pregunta: ¿Qué causó la caída del Imperio Romano?\n### Respuesta: La caída del Imperio Romano de Occidente en 476 d.C. fue causada por invasiones bárbaras, crisis económica y división interna."}
]

# Convertir a Dataset de Hugging Face
dataset = Dataset.from_pandas(pd.DataFrame(history_data))
dataset = dataset.train_test_split(test_size=0.2)

# 4. Configurar el tokenizador y colador de datos correctamente
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["text"]  # Eliminar la columna original después del tokenizado
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # No usar masked language modeling
)

# 5. Configurar los argumentos de entrenamiento (versión estable)
training_args = TrainingArguments(
    output_dir="./gpt2-history-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    
    # Configuración de precisión mixta (desactivada para mayor estabilidad)
    fp16=False,
    bf16=False,
    
    # Configuración esencial
    optim="adamw_torch",
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    
    # Configuración de logging y guardado
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

# 6. Configurar el Trainer (sin callback problemático)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=None,  # Sin evaluación durante el entrenamiento
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 7. Entrenar el modelo con manejo de errores
try:
    print("Iniciando entrenamiento...")
    trainer.train()
    print("¡Entrenamiento completado con éxito!")
except Exception as e:
    print(f"Error durante el entrenamiento: {str(e)}")
    print("Intentando con configuración más conservadora...")
    
    training_args.per_device_train_batch_size = 1
    training_args.gradient_accumulation_steps = 4
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator
    )
    trainer.train()

# 8. Guardar el modelo entrenado
model.save_pretrained("gpt2-history-lora")
tokenizer.save_pretrained("gpt2-history-lora")

# 9. Función para generar respuestas (mejorada)
def generate_history_response(question, max_new_tokens=150):
    # Configurar padding a la izquierda para generación
    tokenizer.padding_side = "left"
    
    prompt = f"### Pregunta: {question}\n### Respuesta:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding="max_length").to(model.device)
    
    # Generar con max_new_tokens en lugar de max_length
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,  # Usar max_new_tokens en lugar de max_length
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Manejo seguro de la respuesta
    if "### Respuesta:" in full_response:
        return full_response.split("### Respuesta:")[1].strip()
    return full_response.strip()

# Probar el modelo con la nueva función
print("\nPrueba del modelo entrenado:")
print("Respuesta 1:", generate_history_response("¿Quién fue Napoleón Bonaparte?"))
print("Respuesta 2:", generate_history_response("¿Qué fue la Revolución Industrial?"))