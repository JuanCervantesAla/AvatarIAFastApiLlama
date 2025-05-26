!pip install -q sentence-transformers transformers accelerate

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import json

qa_pipeline = pipeline(
    "question-answering",
    model="PlanTL-GOB-ES/roberta-base-bne-sqac",
    tokenizer="PlanTL-GOB-ES/roberta-base-bne-sqac"
)

# 1. Cargar archivo JSON
with open("rag.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [entry["text"] for entry in data]

# 2. Modelo de embeddings multilingüe
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = embedding_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

# 3. Función de búsqueda semántica
def search(query, k=3):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = torch.matmul(embeddings, query_embedding)
    top_k = torch.topk(scores, k=k)
    return [texts[i] for i in top_k.indices]

# 4. Cargar modelo generador (FLAN-T5 Base)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# 5. Función para responder preguntas
def responder(pregunta):
    docs = search(pregunta, k=1)
    contexto = docs[0]
    respuesta = qa_pipeline(question=pregunta, context=contexto)
    return respuesta["answer"]

# 🔍 Ejemplo de uso
respuesta = responder("¿Por qué se desechó el caso SG-AG-0007-2025?")
print("✅ Respuesta:")
print(respuesta)
