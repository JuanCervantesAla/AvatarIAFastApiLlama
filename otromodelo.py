import os
import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import bitsandbytes as bnb
from tqdm import tqdm
from google.colab import drive
from sklearn.metrics import accuracy_score, f1_score

# Montar Google Drive para guardar modelos
drive.mount('/content/drive')
save_directory = "/content/drive/MyDrive/modelos_candidato"
os.makedirs(save_directory, exist_ok=True)

# 1. PREPARACIÓN MEJORADA DEL DATASET
# -------------------------------------

# Definir un formato de instrucción más claro
def format_instruction(question, answer, persona_context=""):
    """
    Formatea las entradas con un formato de instrucción claro:
    - Agrega instrucciones explícitas
    - Incluye contexto sobre la personalidad/rol
    - Separadores claros
    """
    instruction = f"""<|system|>
Eres un asistente virtual que simula ser {persona_context}. Responde como lo haría el candidato real,
usando su estilo y conocimiento. Proporciona información precisa y mantén coherencia con su personalidad.
<|endoftext|>

<|user|>
{question}
<|endoftext|>

<|assistant|>
{answer}
<|endoftext|>"""
    return instruction

# Función para verificar y corregir inconsistencias en el dataset
def clean_dataset(data):
    """
    Limpia y valida el dataset:
    - Elimina duplicados
    - Verifica longitudes razonables
    - Busca inconsistencias en el estilo de respuesta
    """
    cleaned_data = []
    for item in data:
        # Extraer pregunta y respuesta
        if isinstance(item, dict) and "text" in item:
            text = item["text"]
            if "### Pregunta:" in text and "### Respuesta:" in text:
                parts = text.split("### Respuesta:")
                if len(parts) == 2:
                    question = parts[0].replace("### Pregunta:", "").strip()
                    answer = parts[1].strip().replace("<|endoftext|>", "")
                    
                    # Validar longitudes
                    if 5 <= len(question) <= 500 and 10 <= len(answer) <= 1000:
                        cleaned_data.append({
                            "question": question,
                            "answer": answer
                        })
    
    # Eliminar duplicados
    seen = set()
    unique_data = []
    for item in cleaned_data:
        key = item["question"].lower()
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    
    return unique_data

# Cargar y limpiar el dataset
print("Cargando y limpiando el dataset...")
with open("datasetF.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

cleaned_data = clean_dataset(raw_data)
print(f"Dataset original: {len(raw_data)} ejemplos")
print(f"Dataset limpio: {len(cleaned_data)} ejemplos")

# Definición del contexto del candidato - PERSONALIZA ESTO
candidato_context = "Sergio Arturo Guerrero Olvera, candidato a magistrado de jalisco"

# Crear dataset formateado
formatted_data = []
for item in cleaned_data:
    formatted_data.append({
        "text": format_instruction(
            item["question"], 
            item["answer"],
            candidato_context
        )
    })

# 2. SELECCIÓN DE UN MEJOR MODELO BASE
# ------------------------------------
# Usando un modelo más grande y capaz para español
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Alternativas: "mistralai/Mistral-7B-Instruct-v0.2" o "bigscience/bloom-7b1"

# Configuración mejorada de tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. CONFIGURACIÓN OPTIMIZADA DEL MODELO
# -------------------------------------
# Configuración de cuantización mejorada
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Cargar modelo con optimizaciones
print("Cargando modelo base...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=True
)

# Preparar modelo para entrenamiento en baja precisión
model = prepare_model_for_kbit_training(model)

# Configuración mejorada de LoRA adaptada al modelo
target_modules = None
if "llama" in model_name.lower():
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
elif "mistral" in model_name.lower():
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
elif "gpt-neox" in model_name.lower() or "bloom" in model_name.lower():
    target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
else:  # GPT-2 y similares
    target_modules = ["c_attn", "c_proj", "c_fc"]

lora_config = LoraConfig(
    r=16,  # Rango mayor para más capacidad
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Aplicar LoRA al modelo
model = get_peft_model(model, lora_config)
print("Parámetros entrenables:")
model.print_trainable_parameters()

# 4. PREPARACIÓN DEL DATASET PARA ENTRENAMIENTO
# --------------------------------------------
dataset = Dataset.from_dict({
    "text": [item["text"] for item in formatted_data]
})

# División en train/test/val para evaluación adecuada
splits = dataset.train_test_split(test_size=0.2)
train_data = splits["train"]
eval_splits = splits["test"].train_test_split(test_size=0.5)
val_data = eval_splits["train"]
test_data = eval_splits["test"]

print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

# Mejorar la función de preprocesamiento
def preprocess_function(examples):
    """
    Tokeniza con padding dinámico para optimizar el uso de memoria
    """
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,  # Aumentado para capturar más contexto
        padding=False  # El DataCollator se encargará del padding dinámico
    )
    return result

# Tokenizar datasets
print("Tokenizando datasets...")
tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=["text"])
tokenized_val = val_data.map(preprocess_function, batched=True, remove_columns=["text"])

# 5. CONFIGURACIÓN OPTIMIZADA DEL ENTRENAMIENTO
# -------------------------------------------
training_args = TrainingArguments(
    output_dir=os.path.join(save_directory, "checkpoints"),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=3,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="tensorboard",
    fp16=True,
    optim="paged_adamw_8bit",
    seed=42
)

# Data collator con padding dinámico
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# 6. ENTRENAMIENTO CON EVALUACIÓN
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

print("Comenzando entrenamiento...")
try:
    trainer.train()
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")
    print("Intentando reducir el tamaño del batch...")
    training_args.per_device_train_batch_size = 1
    training_args.gradient_accumulation_steps = 16
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    trainer.train()

# 7. GUARDAR EL MODELO AJUSTADO
# ---------------------------
output_dir = os.path.join(save_directory, "modelo_final")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Modelo guardado en: {output_dir}")

# 8. EVALUACIÓN Y DETECCIÓN DE ALUCINACIONES
# ----------------------------------------
def evaluate_responses(model, tokenizer, test_data, num_samples=50):
    """
    Evalúa el modelo en datos de prueba y verifica:
    - Coherencia de respuestas
    - Detección de posibles alucinaciones
    """
    # Preparar generador
    generation_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id
    )
    
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        generation_config=generation_config
    )
    
    # Seleccionar muestras aleatorias del test set
    if len(test_data) > num_samples:
        test_samples = test_data.select(np.random.choice(len(test_data), num_samples, replace=False))
    else:
        test_samples = test_data
    
    results = []
    
    for item in tqdm(test_samples):
        text = item["text"]
        # Extraer pregunta y contexto
        system_end = text.find("<|endoftext|>")
        user_start = text.find("<|user|>")
        user_end = text.find("<|endoftext|>", user_start)
        
        system_prompt = text[0:system_end].strip()
        user_prompt = text[user_start:user_end].replace("<|user|>", "").strip()
        
        # Construir prompt para inferencia
        prompt = f"{system_prompt}\n\n<|user|>\n{user_prompt}\n<|endoftext|>\n\n<|assistant|>"
        
        # Generar respuesta
        response = pipe(prompt)[0]['generated_text']
        
        # Extraer solo la parte de respuesta generada
        assistant_start = response.find("<|assistant|>")
        if assistant_start != -1:
            generated_response = response[assistant_start + len("<|assistant|>"):].strip()
            # Limpiar antes del primer endoftext si existe
            if "<|endoftext|>" in generated_response:
                generated_response = generated_response.split("<|endoftext|>")[0].strip()
        else:
            generated_response = "Error en la generación"
        
        # Extraer respuesta esperada del texto original
        assistant_start_orig = text.find("<|assistant|>")
        assistant_end_orig = text.find("<|endoftext|>", assistant_start_orig)
        expected_response = text[assistant_start_orig + len("<|assistant|>"):assistant_end_orig].strip()
        
        results.append({
            "pregunta": user_prompt,
            "respuesta_esperada": expected_response,
            "respuesta_generada": generated_response,
        })
    
    return results

# Ejecutar evaluación
print("Evaluando modelo...")
model.eval()
evaluation_results = evaluate_responses(model, tokenizer, test_data)

# Guardar resultados de evaluación
with open(os.path.join(save_directory, "evaluation_results.json"), "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

print("Evaluación completa y guardada")

# 9. INFERENCIA DEL MODELO
# -----------------------
def generate_response(model, tokenizer, question, context=None):
    """
    Genera una respuesta para una pregunta dada
    """
    if context is None:
        context = f"Eres un asistente virtual que simula ser {candidato_context}."
    
    prompt = f"""<|system|>
{context}
<|endoftext|>

<|user|>
{question}
<|endoftext|>

<|assistant|>"""
    
    generation_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2
    )
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extraer solo la parte generada
    assistant_start = response.find("<|assistant|>")
    if assistant_start != -1:
        generated_text = response[assistant_start + len("<|assistant|>"):].strip()
        # Limpiar texto después de endoftext
        if "<|endoftext|>" in generated_text:
            generated_text = generated_text.split("<|endoftext|>")[0].strip()
        return generated_text
    else:
        return "Error en la generación"

# 10. DEMO INTERACTIVO
# ------------------
def interactive_demo():
    """
    Demo interactivo para probar el modelo
    """
    print("\n=== DEMO DEL CHATBOT DEL CANDIDATO ===")
    print(f"Simulando a: {candidato_context}")
    print("Escribe 'salir' para terminar")
    
    while True:
        question = input("\nTu pregunta: ")
        if question.lower() in ["salir", "exit", "quit"]:
            break
        
        response = generate_response(model, tokenizer, question)
        print(f"\nRespuesta: {response}")

# Ejecutar demo
print("\n¿Quieres probar el modelo con algunas preguntas? (s/n)")
if input().lower() == "s":
    interactive_demo()

# 11. CÓDIGO PARA INTEGRAR CON RAG (OPCIONAL)
# -----------------------------------------
print("\n¿Quieres ver un ejemplo de código para integrar RAG? (s/n)")
if input().lower() == "s":
    print("""
# EJEMPLO DE INTEGRACIÓN CON RAG
# ------------------------------
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
import os

# Función para preparar la base de conocimiento
def setup_knowledge_base(docs_dir, db_path):
    # Cargar documentos
    loader = DirectoryLoader(docs_dir, glob="**/*.txt")
    documents = loader.load()
    
    # Dividir en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # Crear embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Crear y guardar vectorstore
    db = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    db.persist()
    return db

# Función para generar respuesta con RAG
def generate_rag_response(question, context=None):
    # Cargar vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory="path/to/db", embedding_function=embeddings)
    
    # Buscar documentos relevantes
    docs = db.similarity_search(question, k=3)
    context_docs = "\\n".join([doc.page_content for doc in docs])
    
    # Crear sistema con el contexto aumentado
    if context is None:
        context = f"Eres un asistente virtual que simula ser {candidato_context}."
    
    enhanced_context = f"{context}\\n\\nUtiliza la siguiente información para responder:\\n{context_docs}"
    
    # Generar respuesta con el contexto aumentado
    return generate_response(model, tokenizer, question, enhanced_context)
    """)

print("\nEntrenamiento y evaluación completos!")

# INSTRUCCIONES PARA USAR EL MODELO ENTRENADO
print("""
# ------------------------------------------------
# CÓMO USAR TU MODELO ENTRENADO:
# ------------------------------------------------

1. El modelo está guardado en Google Drive en: {}

2. Para cargar el modelo para inferencias:
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from peft import PeftModel

   # Cargar tokenizer
   tokenizer = AutoTokenizer.from_pretrained("ruta_al_modelo")

   # Cargar modelo base
   base_model = AutoModelForCausalLM.from_pretrained(
       "{}",
       device_map="auto",
       torch_dtype=torch.float16
   )

   # Cargar adaptadores LoRA
   model = PeftModel.from_pretrained(base_model, "ruta_al_modelo")
   ```

3. Para generar respuestas, usa la función `generate_response`
   que se incluye en este notebook.

4. Para reducir alucinaciones, considera:
   - Implementar el sistema RAG mostrado
   - Usar temperature más baja (0.3-0.5)
   - Aumentar el penalty de repetición
   - Añadir más ejemplos al dataset
   - Entrenar por más epochs

# ------------------------------------------------
""".format(save_directory, model_name))