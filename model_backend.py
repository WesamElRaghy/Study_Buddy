from llama_cpp import Llama
import time

class StudyBuddyBackend:
    def __init__(self, model_path):
        print("Loading model... This may take a minute or two.")
        self.model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=4
        )
        print("Model loaded successfully!")

    def answer_question(self, question, temperature=0.7):
        """Helps solve a study problem step-by-step"""
        prompt = f"""<s>[INST] You are a helpful study assistant. 
        Help me solve this problem step by step: {question} [/INST]</s>"""
        
        start_time = time.time()
        response = self.model(prompt, max_tokens=1024, temperature=temperature, stop=["</s>"])
        end_time = time.time()
        
        return {
            "answer": response["choices"][0]["text"],
            "time_taken": round(end_time - start_time, 2)
        }
    
    def generate_practice_questions(self, question, num_questions=3, temperature=0.7):
        """Generates similar practice questions"""
        prompt = f"""<s>[INST] You are a helpful study assistant.
        Create {num_questions} practice questions similar to this one: {question}
        Make sure each question is clearly numbered and tests similar concepts. [/INST]</s>"""
        
        response = self.model(prompt, max_tokens=1024, temperature=temperature, stop=["</s>"])
        
        return response["choices"][0]["text"]
