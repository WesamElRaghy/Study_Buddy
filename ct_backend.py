from ctransformers import AutoModelForCausalLM
import time
import os

class StudyBuddyBackend:
    def __init__(self, model_path):
        print("Loading model... This may take a minute or two.")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",
            max_new_tokens=1024,
            context_length=4096
        )
        print("Model loaded successfully!")

    def answer_question(self, question, temperature=0.7):
        """Helps solve a study problem step-by-step"""
        prompt = f"""<s>[INST] You are a helpful study assistant. 
        Help me solve this problem step by step: {question} [/INST]</s>"""
        
        start_time = time.time()
        response = self.model(prompt, temperature=temperature)
        end_time = time.time()
        
        return {
            "answer": response,
            "time_taken": round(end_time - start_time, 2)
        }
    
    def generate_practice_questions(self, question, num_questions=3, temperature=0.7):
        """Generates similar practice questions"""
        prompt = f"""<s>[INST] You are a helpful study assistant.
        Create {num_questions} practice questions similar to this one: {question}
        Make sure each question is clearly numbered and tests similar concepts. [/INST]</s>"""
        
        return self.model(prompt, temperature=temperature)
