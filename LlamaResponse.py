from fuzzywuzzy import fuzz
from llama_cpp import Llama

political_keywords = {"presidente", "elecciones", "gobierno", "partido", "política", "INE", "morena", "PAN", "PRI", "AMLO"}

def is_political_question(prompt: str) -> bool:
    for keyword in political_keywords:
        if fuzz.partial_ratio(prompt.lower(), keyword) > 80:
            return True
        return False

def get_Response(prompt: str):
    if is_political_question(prompt):
        return "Lo siento, no puedo responder preguntas sobre política."

    llm = Llama(model_path="./model/llama-pro-8b.Q4_K_M.gguf", n_threads=3)
    formatted_prompt = (
        "Eres un asistente que solo responde sobre tecnología y programación. "
        "Si te preguntan sobre política, historia o religión, responde con 'No puedo responder eso.'\n\n"
        f"Q: {prompt}\nA:"
    )

    response = llm(formatted_prompt, max_tokens=130, temperature=0.98)
    text = response["choices"][0]["text"]

    clean_response = text.strip().split("\n")[0]
    print(clean_response)
    return clean_response

