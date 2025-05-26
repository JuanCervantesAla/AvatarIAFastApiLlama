from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json

class PersonaAI:
    def __init__(self, base_model_path, peft_model_path, persona_json_path):
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base_model, peft_model_path)
        
        # Load persona knowledge
        with open(persona_json_path, 'r', encoding='utf-8') as f:
            self.persona_data = json.load(f)
        
        # Extract topics from persona data
        self.topics = self._extract_topics()
        
        # Define persona name and characteristics
        self.persona_name = "PersonaName"  # Replace with actual name
        self.persona_traits = ["trait1", "trait2"]  # Replace with actual traits
        
    def _extract_topics(self):
        """Extract main topics from the persona data"""
        topics = set()
        for item in self.persona_data:
            # This is a simple implementation - you might want to use NLP for better topic extraction
            words = item["text"].lower().split()
            for word in words:
                if len(word) > 5 and word not in ("about", "would", "should", "could", "their"):
                    topics.add(word)
        return list(topics)[:100]  # Keep top 100 topics
    
    def _is_relevant_question(self, question):
        """Check if the question is relevant to the persona's knowledge"""
        question_lower = question.lower()
        
        # Check if any topic is mentioned in the question
        for topic in self.topics:
            if topic in question_lower:
                return True
                
        # If no direct topic match, return True anyway but with lower confidence
        return True, 0.5
    
    def generate_response(self, question, max_length=150):
        """Generate a response to the question in the persona's style"""
        # Check if question is relevant
        is_relevant = self._is_relevant_question(question)
        
        if not is_relevant:
            return f"Como {self.persona_name}, no tengo información sobre ese tema específico."
        
        # Create a prompt that makes the model respond as the persona
        prompt = f"""A continuación hay una conversación con {self.persona_name}.
{self.persona_name} es conocido por {', '.join(self.persona_traits)}.

Pregunta: {question}
{self.persona_name}:"""
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs["input_ids"][0]) + max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean up the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        return response

# Part 2: Simple Web Interface - app.py
from flask import Flask, request, render_template, jsonify
from model_inference import PersonaAI

app = Flask(__name__)

# Initialize the AI model
persona_ai = PersonaAI(
    base_model_path="datificate/gpt2-small-spanish",  # Replace with your base model
    peft_model_path="./modelo_ajustado2",  # Your fine-tuned model path
    persona_json_path="./datasetF.json"  # Your persona dataset
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message', '')
    
    if not question:
        return jsonify({'error': 'No message provided'}), 400
    
    response = persona_ai.generate_response(question)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

# Part 3: HTML Template - templates/index.html
"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat con Persona AI</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .chat-container {
            height: 500px;
            border: 1px solid #ccc;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 20px;
        }
        .message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e6f7ff;
            text-align: right;
        }
        .ai-message {
            background-color: #f2f2f2;
        }
        .input-container {
            display: flex;
        }
        #user-input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            margin-left: 10px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>Chat con <span id="persona-name">Persona</span></h1>
    <div class="chat-container" id="chat-container"></div>
    <div class="input-container">
        <input type="text" id="user-input" placeholder="Escribe tu pregunta...">
        <button onclick="sendMessage()">Enviar</button>
    </div>

    <script>
        // Set the persona name
        document.getElementById('persona-name').textContent = "PersonaName"; // Replace with actual name
        
        function addMessage(message, isUser) {
            const chatContainer = document.getElementById('chat-container');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
            messageDiv.textContent = message;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        async function sendMessage() {
            const userInput = document.getElementById('user-input');
            const message = userInput.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, true);
            userInput.value = '';
            
            try {
                // Send message to backend
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message })
                });
                
                const data = await response.json();
                
                // Add AI response to chat
                addMessage(data.response, false);
            } catch (error) {
                console.error('Error:', error);
                addMessage('Lo siento, ha ocurrido un error.', false);
            }
        }
        
        // Allow sending with Enter key
        document.getElementById('user-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

# Part 4: Advanced Features - knowledge_matcher.py
# This is an optional enhancement to better match questions to knowledge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class KnowledgeMatcher:
    def __init__(self, persona_data):
        self.persona_data = persona_data
        self.texts = [item["text"] for item in persona_data]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.text_vectors = self.vectorizer.fit_transform(self.texts)
        
    def find_relevant_knowledge(self, question, top_k=3):
        """Find the most relevant knowledge entries for a given question"""
        question_vector = self.vectorizer.transform([question])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(question_vector, self.text_vectors)[0]
        
        # Get indices of top k most similar texts
        top_indices = np.argsort(similarity_scores)[-top_k:][::-1]
        
        # Return relevant texts and their scores
        results = []
        for idx in top_indices:
            if similarity_scores[idx] > 0.1:  # Threshold for relevance
                results.append({
                    "text": self.texts[idx],
                    "score": float(similarity_scores[idx])
                })
        
        return results