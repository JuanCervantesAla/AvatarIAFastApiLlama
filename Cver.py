# ========================================
# Fine-tuning Script for RTX 3090
# Install: pip install transformers datasets accelerate torch
# ========================================

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ========================================
# 1. Load and prepare dataset
# ========================================
def load_dataset():
    try:
        with open('dt.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single dict and list of dicts
        if isinstance(data, dict):
            data = [data]
        
        # Validate data format
        for i, item in enumerate(data[:3]):
            if 'input' not in item or 'output' not in item:
                print(f"Warning: Item {i} missing 'input' or 'output' key")
                print(f"Keys found: {list(item.keys())}")
        
        dataset = Dataset.from_list(data)
        
        # Limit to 2000 samples for faster training
        if len(dataset) > 2000:
            dataset = dataset.select(range(2000))
        
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Sample data: {dataset[0]}")
        
        return dataset
    
    except FileNotFoundError:
        print("Error: dt.json file not found!")
        print("Please ensure your dataset file exists in the current directory.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# ========================================
# 2. Load model and tokenizer
# ========================================
def load_model_and_tokenizer():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True  # Helps with memory efficiency
    )
    
    print(f"Model loaded on device: {model.device}")
    return model, tokenizer

# ========================================
# 3. Data preprocessing
# ========================================
def format_instruction(example, tokenizer):
    """Format the data for instruction tuning"""
    # Get the actual EOS token string, not the token ID
    eos_token = tokenizer.decode([tokenizer.eos_token_id]) if tokenizer.eos_token_id is not None else ""
    return {
        "text": f"### Input: {example['input']}\n### Output: {example['output']}{eos_token}"
    }

def preprocess_dataset(dataset, tokenizer):
    """Tokenize and prepare dataset for training"""
    
    # Format instructions
    print("Formatting instructions...")
    dataset = dataset.map(lambda x: format_instruction(x, tokenizer))
    
    # Tokenize
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # We'll pad in the data collator
            max_length=256,  # Increased from 128 for better context
            return_tensors=None  # Return lists, not tensors
        )
        
        # Set labels equal to input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    return tokenized_dataset

# ========================================
# 4. Training setup
# ========================================
def setup_training(model, tokenizer, train_dataset):
    """Setup training arguments and trainer"""
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Training arguments optimized for RTX 3090
    training_args = TrainingArguments(
        output_dir="./qwen-finetuned",
        overwrite_output_dir=True,
        
        # Training hyperparameters
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Adjust based on your GPU memory
        gradient_accumulation_steps=2,   # Effective batch size = 4 * 2 = 8
        learning_rate=5e-5,
        
        # Memory optimization
        fp16=True,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        
        # Logging and saving
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        
        # Optimization
        optim="adamw_torch",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        
        # Disable wandb and other reporting
        report_to=[],
        
        # Evaluation (optional)
        # eval_steps=100,
        # evaluation_strategy="steps",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    return trainer

# ========================================
# 5. Testing function
# ========================================
def test_model(model, tokenizer, test_inputs=None):
    """Test the fine-tuned model"""
    
    if test_inputs is None:
        test_inputs = [
            "### Input: ¿Cuál es tu película favorita?\n### Output:",
            "### Input: Explica qué es la inteligencia artificial\n### Output:",
            "### Input: ¿Cómo está el clima hoy?\n### Output:"
        ]
    
    model.eval()
    print("\n" + "="*50)
    print("TESTING THE MODEL")
    print("="*50)
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\nTest {i}:")
        print(f"Input: {input_text.split('### Output:')[0].replace('### Input: ', '')}")
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and print response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = full_response.split("### Output:")[-1].strip()
        print(f"Output: {response_only}")
        print("-" * 30)

# ========================================
# 6. Main execution
# ========================================
def main():
    try:
        # Load dataset
        dataset = load_dataset()
        if dataset is None:
            return
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # Preprocess dataset
        train_dataset = preprocess_dataset(dataset, tokenizer)
        
        # Setup training
        trainer = setup_training(model, tokenizer, train_dataset)
        
        # Train the model
        print("\n" + "="*50)
        print("STARTING TRAINING")
        print("="*50)
        trainer.train()
        
        # Save the model
        print("\nSaving model...")
        trainer.save_model()
        tokenizer.save_pretrained("./qwen-finetuned")
        
        # Test the model
        test_model(model, tokenizer)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("Model saved to: ./qwen-finetuned")
        
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA Out of Memory Error!")
        print("Try reducing batch_size or max_length in the code.")
        print("Current settings: batch_size=4, max_length=256")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()