from ctransformers import AutoModelForCausalLM
import time
import os
import threading
import gc
import hashlib

class StudyBuddyBackend:
    def __init__(self, model_path):
        print("Loading model... This may take a minute or two.")
        
        # Force garbage collection before loading
        gc.collect()
        
        # Performance optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",
            max_new_tokens=512,  # Reduced for speed
            context_length=1024,  # Reduced context length
            gpu_layers=0,  # CPU-only
            threads=os.cpu_count(),  # Use all cores
            batch_size=1  # Safer value
        )
        
        # Initialize lock and cache
        self.lock = threading.Lock()
        self.response_cache = {}
        
        print("Model loaded successfully!")

    def answer_question(self, question, temperature=0.3):
        """Helps solve a study problem step-by-step"""
        # Create a cache key
        cache_key = hashlib.md5(f"solve_{question}_{temperature}".encode()).hexdigest()
        
        # Check if we have a cached response
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Ensure question isn't too long
        if len(question) > 500:
            question = question[:500] + "..."
        
        # Compact prompt
        prompt = f"<s>[INST] You are a helpful study assistant. Solve this step by step: {question} [/INST]</s>"
        
        with self.lock:
            # Force garbage collection
            gc.collect()
            
            start_time = time.time()
            try:
                response = self.model(
                    prompt, 
                    temperature=temperature,
                    top_k=40,  # Limit token choices for speed
                    top_p=0.95  # Limit token diversity for speed
                )
            except Exception as e:
                response = f"Error generating response: {str(e)}"
            end_time = time.time()
        
        # Create and cache result
        result = {
            "answer": response,
            "time_taken": round(end_time - start_time, 2)
        }
        self.response_cache[cache_key] = result
        return result
    
    def generate_practice_questions(self, question, num_questions=3, temperature=0.3):
        """Generates similar practice questions"""
        # Create a cache key
        cache_key = hashlib.md5(f"practice_{question}_{num_questions}_{temperature}".encode()).hexdigest()
        
        # Check cache
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
            
        # Ensure question isn't too long
        if len(question) > 500:
            question = question[:500] + "..."
            
        # Compact prompt
        prompt = f"<s>[INST] Create {num_questions} practice questions similar to: {question} [/INST]</s>"
        
        with self.lock:
            # Force garbage collection
            gc.collect()
            
            try:
                response = self.model(
                    prompt, 
                    temperature=temperature,
                    top_k=40,
                    top_p=0.95
                )
            except Exception as e:
                response = f"Error generating practice questions: {str(e)}"
        
        # Cache the result
        self.response_cache[cache_key] = response
        return response
