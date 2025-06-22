from ctransformers import AutoModelForCausalLM
import time
import os
import threading
import gc
import hashlib
import re
import io
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class StudyBuddyBackend:
    def __init__(self, model_path):
        print("Loading model... This may take a minute or two.")
        gc.collect()
        
        # Initialize Mistral model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",
            max_new_tokens=512,
            context_length=1024,
            gpu_layers=0,
            threads=os.cpu_count(),
            batch_size=1
        )
        
        # Initialize embeddings and text splitter
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n# ", "\n## ", "\n### ", "\n", "\.", "\!", "\?", " "]
        )
        
        # Initialize lock, cache, and FAISS index
        self.lock = threading.Lock()
        self.response_cache = {}
        self.faiss_index = None
        self.corpus_path = "textbook_corpus"
        self.pdf_processed = False  # Track if a PDF has been processed
        
        print("Model and embeddings loaded successfully!")

    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[-_]{2,}', '', text)
        return text.strip()

    def extract_text_from_pdf(self, file_data: bytes, max_pages: int = 5) -> str:
        """Extract text from PDF with OCR fallback."""
        try:
            pdf_file = io.BytesIO(file_data)
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            num_pages = min(len(reader.pages), max_pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text() or ""
                text += self.clean_text(page_text) + " "
            text = self.clean_text(text)

            if text.strip() and len(text) > 50:
                return text

            images = convert_from_bytes(file_data, first_page=1, last_page=max_pages, thread_count=1)
            text = ""
            for page_num, image in enumerate(images, 1):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_data = img_byte_arr.getvalue()
                image = Image.open(io.BytesIO(img_data)).convert('L')
                page_text = pytesseract.image_to_string(image, config='--psm 6')
                text += f"[Page {page_num}] {self.clean_text(page_text)} "
            return self.clean_text(text)
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"

    def process_pdf(self, pdf_data: bytes) -> bool:
        """Process PDF and create FAISS index."""
        if self.pdf_processed:
            print("PDF already processed. Skipping reprocessing.")
            return True

        start_time = time.time()
        pdf_text = []
        try:
            pdf_file = io.BytesIO(pdf_data)
            reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text() or ""
                text = self.clean_text(text)
                pdf_text.append(f"[Page {page_num + 1}] {text}")
        except Exception as e:
            print(f"Error extracting textbook: {e}")
            return False

        texts = []
        metadatas = []
        for i, page_text in enumerate(pdf_text):
            splits = self.text_splitter.split_text(page_text)
            texts.extend(splits)
            metadatas.extend([{"page": i + 1}] * len(splits))

        try:
            self.faiss_index = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            os.makedirs(self.corpus_path, exist_ok=True)
            self.faiss_index.save_local(self.corpus_path)
            self.pdf_processed = True  # Mark PDF as processed
            print(f"FAISS index creation time: {time.time() - start_time:.2f} seconds")
            return True
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            return False

    def reset_pdf(self):
        """Reset the PDF processing state and clear FAISS index."""
        self.faiss_index = None
        self.pdf_processed = False
        if os.path.exists(self.corpus_path):
            import shutil
            shutil.rmtree(self.corpus_path)
        print("PDF processing state reset.")

    def answer_pdf_question(self, question: str, temperature=0.3) -> dict:
        """Answer a question based on the processed PDF."""
        if not self.pdf_processed or self.faiss_index is None:
            return {"answer": "Error: No PDF processed. Please upload a PDF first.", "time_taken": 0}

        cache_key = hashlib.md5(f"pdf_{question}_{temperature}".encode()).hexdigest()
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        if len(question) > 500:
            question = question[:500] + "..."

        try:
            docs = self.faiss_index.similarity_search(question, k=2)
            context = "\n".join([f"[Page {doc.metadata['page']}] {self.clean_text(doc.page_content)}" for doc in docs])
            prompt = f"<s>[INST] You are a study assistant. Answer based on this context:\n{context}\nQuestion: {question} [/INST]</s>"

            with self.lock:
                gc.collect()
                start_time = time.time()
                response = self.model(
                    prompt,
                    temperature=temperature,
                    top_k=40,
                    top_p=0.95
                )
                end_time = time.time()

            result = {
                "answer": response,
                "time_taken": round(end_time - start_time, 2)
            }
            self.response_cache[cache_key] = result
            return result
        except Exception as e:
            return {"answer": f"Error answering question: {str(e)}", "time_taken": 0}

    def answer_question(self, question, temperature=0.3):
        """Helps solve a study problem step-by-step."""
        cache_key = hashlib.md5(f"solve_{question}_{temperature}".encode()).hexdigest()
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        if len(question) > 500:
            question = question[:500] + "..."

        prompt = f"<s>[INST] You are a helpful study assistant. Solve this step by step: {question} [/INST]</s>"
        with self.lock:
            gc.collect()
            start_time = time.time()
            try:
                response = self.model(
                    prompt,
                    temperature=temperature,
                    top_k=40,
                    top_p=0.95
                )
            except Exception as e:
                response = f"Error generating response: {str(e)}"
            end_time = time.time()

        result = {
            "answer": response,
            "time_taken": round(end_time - start_time, 2)
        }
        self.response_cache[cache_key] = result
        return result

    def generate_practice_questions(self, question, num_questions=3, temperature=0.3):
        """Generates similar practice questions."""
        cache_key = hashlib.md5(f"practice_{question}_{num_questions}_{temperature}".encode()).hexdigest()
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        if len(question) > 500:
            question = question[:500] + "..."

        prompt = f"<s>[INST] Create {num_questions} practice questions similar to: {question} [/INST]</s>"
        with self.lock:
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

        self.response_cache[cache_key] = response
        return response
