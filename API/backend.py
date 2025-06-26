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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import Optional, List, Dict
from schemas import StructuredQueryResponse
import logging
from collections import OrderedDict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Memory profiling disabled.")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StudyBuddyBackend:
    """Backend for StudyBuddy RAG API, integrating Mistral-7B and FAISS for educational content processing."""
    
    def __init__(self, model_path: str):
        """Initialize Mistral-7B model and FAISS embeddings."""
        logger.debug("Loading model...")
        gc.collect()
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            logger.debug(f"[{time.time()}] Memory usage before model load: {process.memory_info().rss / 1024**2:.2f} MB")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",
            max_new_tokens=512,
            context_length=1024,
            gpu_layers=0,
            threads=os.cpu_count(),
            batch_size=1
        )
        
        logger.debug("Initializing HuggingFaceEmbeddings with sentence-transformers")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
                cache_folder=os.path.join(os.getcwd(), "embedding_cache")
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFaceEmbeddings: {e}")
            raise
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n# ", "\n## ", "\n### ", "\n", "\.", "\!", "\?", " "]
        )
        
        self.lock = threading.Lock()
        self.response_cache = OrderedDict()
        self.cache_max_size = 50
        self.faiss_index = None
        self.corpus_path = "textbook_corpus"
        self.pdf_processed = False
        self.max_pages = int(os.getenv("MAX_PAGES", "300"))
        
        if PSUTIL_AVAILABLE:
            logger.debug(f"[{time.time()}] Memory usage after model load: {process.memory_info().rss / 1024**2:.2f} MB")
        logger.info("Model and embeddings loaded successfully!")

    def _update_cache(self, key: str, value: any):
        """Update cache with size limit."""
        self.response_cache[key] = value
        if len(self.response_cache) > self.cache_max_size:
            self.response_cache.popitem(last=False)

    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace and non-ASCII characters."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[-_]{2,}', '', text)
        return text.strip()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximate: 1 token ~ 4 chars)."""
        return len(text) // 4 + 1

    def extract_text_from_pdf(self, pdf_data: bytes, max_pages: int = 300) -> str:
        """Extract text from a PDF with OCR fallback."""
        logger.debug("Extracting text from PDF")
        start_time = time.time()
        try:
            pdf_file = io.BytesIO(pdf_data)
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            num_pages = min(len(reader.pages), max_pages)
            logger.debug(f"Extracting text from {num_pages} of {len(reader.pages)} pages")
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text() or ""
                text += self.clean_text(page_text) + " "
            text = self.clean_text(text)

            if text.strip() and len(text) > 50:
                logger.debug(f"PDF text extracted in {time.time() - start_time:.2f} seconds")
                return text

            pdf_file.seek(0)
            images = convert_from_bytes(pdf_data, first_page=1, last_page=num_pages, thread_count=1)
            text = ""
            for page_num, image in enumerate(images, 1):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_data = img_byte_arr.getvalue()
                image = Image.open(io.BytesIO(img_data)).convert('L')
                page_text = pytesseract.image_to_string(image, config='--psm 6')
                text += f"[Page {page_num}] {self.clean_text(page_text)} "
            logger.debug(f"PDF OCR completed in {time.time() - start_time:.2f} seconds")
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def extract_text_from_image(self, image_data: bytes) -> str:
        """Extract text from an image using pytesseract."""
        logger.debug("Extracting text from image")
        start_time = time.time()
        try:
            image = Image.open(io.BytesIO(image_data)).convert('L')
            text = pytesseract.image_to_string(image, config='--psm 6')
            logger.debug(f"Image text extracted in {time.time() - start_time:.2f} seconds")
            return self.clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""

    def process_pdf(self, pdf_data: bytes, page_range: Optional[tuple] = None) -> bool:
        """Process a PDF and create a FAISS index."""
        logger.debug("Processing PDF for FAISS index")
        start_time = time.time()
        if self.pdf_processed:
            logger.info("PDF already processed. Skipping reprocessing.")
            return True

        pdf_text = []
        try:
            pdf_file = io.BytesIO(pdf_data)
            reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(reader.pages)
            start_page = page_range[0] - 1 if page_range else 0
            end_page = min(page_range[1], total_pages) if page_range else min(total_pages, self.max_pages)
            num_pages = end_page - start_page
            logger.debug(f"Processing {num_pages} of {total_pages} pages (range: {start_page + 1}-{end_page})")
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                logger.debug(f"[{time.time()}] Memory usage before processing: {process.memory_info().rss / 1024**2:.2f} MB")
            for page_num in range(start_page, end_page):
                page = reader.pages[page_num]
                text = page.extract_text() or ""
                text = self.clean_text(text)
                pdf_text.append(f"[Page {page_num + 1}] {text}")
        except Exception as e:
            logger.error(f"Error extracting textbook: {e}")
            return False

        documents = []
        for i, page_text in enumerate(pdf_text):
            splits = self.text_splitter.split_text(page_text)
            documents.extend([Document(page_content=split, metadata={"page": start_page + i + 1}) for split in splits])

        try:
            gc.collect()
            logger.debug("Creating FAISS index")
            self.faiss_index = FAISS.from_documents(documents, self.embeddings)
            os.makedirs(self.corpus_path, exist_ok=True)
            self.faiss_index.save_local(self.corpus_path)
            self.pdf_processed = True
            logger.info(f"FAISS index created in {time.time() - start_time:.2f} seconds")
            if PSUTIL_AVAILABLE:
                logger.debug(f"[{time.time()}] Memory usage after indexing: {process.memory_info().rss / 1024**2:.2f} MB")
            return True
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            if "faiss.swigfaiss_avx512" in str(e):
                logger.warning("FAISS AVX512 not available, using AVX2 fallback")
            return False

    def reset_pdf(self):
        """Reset PDF processing state and clear FAISS index."""
        logger.debug("Resetting PDF state")
        self.faiss_index = None
        self.pdf_processed = False
        if os.path.exists(self.corpus_path):
            import shutil
            shutil.rmtree(self.corpus_path)
        logger.info("PDF processing state reset.")

    def answer_pdf_question(self, question: str, temperature: float = 0.3) -> StructuredQueryResponse:
        """Answer a question based on the processed PDF, returning a structured response."""
        logger.debug(f"Answering PDF question: {question[:50]}...")
        start_time = time.time()
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            logger.debug(f"[{time.time()}] Memory usage before query: {process.memory_info().rss / 1024**2:.2f} MB")
        
        if not self.pdf_processed or self.faiss_index is None:
            logger.error("No PDF processed")
            return StructuredQueryResponse(
                question_number=None,
                question_text=None,
                context=[],
                task="Provide a detailed answer based on the textbook context.",
                answer="Error: No PDF processed. Please upload a PDF first.",
                time_taken=0
            )

        cache_key = hashlib.md5(f"pdf_{question}_{temperature:.2f}".encode()).hexdigest()
        if cache_key in self.response_cache:
            logger.debug("Returning cached response")
            return self.response_cache[cache_key]

        if len(question) > 500:
            question = question[:500] + "..."

        try:
            gc.collect()
            search_start = time.time()
            docs = self.faiss_index.similarity_search(question, k=2)
            logger.debug(f"FAISS search completed in {time.time() - search_start:.2f} seconds")
            context = [{"page": str(doc.metadata["page"]), "content": self.clean_text(doc.page_content)} for doc in docs]
            context_text = "\n".join([f"[Page {doc['page']}] {doc['content']}" for doc in context])
            prompt = f"<s>[INST] You are a study assistant. Provide a detailed answer based on this context:\n{context_text}\nQuestion: {question} [/INST]</s>"
            token_count = self.estimate_tokens(prompt)
            logger.debug(f"Estimated prompt tokens: {token_count}")
            if token_count > 1024:
                logger.warning(f"Prompt tokens ({token_count}) exceed context_length (1024). Truncating context.")
                context_text = context_text[:800]
                prompt = f"<s>[INST] You are a study assistant. Provide a detailed answer based on this context:\n{context_text}\nQuestion: {question} [/INST]</s>"

            with self.lock:
                gc.collect()
                model_start = time.time()
                response = self.model(
                    prompt,
                    temperature=temperature,
                    top_k=40,
                    top_p=0.95
                )
                logger.debug(f"Model inference completed in {time.time() - model_start:.2f} seconds")
                response_text = str(response)
                response_tokens = self.estimate_tokens(response_text)
                logger.debug(f"Generated response tokens: {response_tokens}")

            result = StructuredQueryResponse(
                question_number=None,
                question_text=None,
                context=context,
                task="Provide a detailed answer based on the textbook context.",
                answer=response_text,
                time_taken=round(time.time() - start_time, 2)
            )
            self._update_cache(cache_key, result)
            if PSUTIL_AVAILABLE:
                logger.debug(f"[{time.time()}] Memory usage after query: {process.memory_info().rss / 1024**2:.2f} MB")
            logger.info(f"PDF question answered in {result.time_taken} seconds")
            return result
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return StructuredQueryResponse(
                question_number=None,
                question_text=None,
                context=[],
                task="Provide a detailed answer based on the textbook context.",
                answer=f"Error answering question: {str(e)}",
                time_taken=0
            )
        finally:
            gc.collect()

    def process_question_file_with_query(self, pdf_data: bytes, question_number: int, temperature: float = 0.3) -> Optional[StructuredQueryResponse]:
        """Extract a specific question from a PDF and query the textbook."""
        logger.debug(f"Processing question file, question number: {question_number}")
        start_time = time.time()
        if not self.pdf_processed or self.faiss_index is None:
            logger.error("No textbook processed")
            return None

        try:
            pdf_text = self.extract_text_from_pdf(pdf_data)
            if not pdf_text:
                logger.error("No text extracted from question file")
                return None
            cleaned_text = self.clean_text(pdf_text)
            
            questions = re.split(r'\s*(?=\d+\.\s)', cleaned_text)
            questions = [q.strip() for q in questions if q.strip() and re.match(r'^\d+\.\s', q)]
            
            target_question = None
            for q in questions:
                if q.startswith(f"{question_number}."):
                    target_question = q
                    break
            
            if not target_question:
                logger.error(f"Question {question_number} not found")
                return None
            
            gc.collect()
            search_start = time.time()
            docs = self.faiss_index.similarity_search(target_question, k=2)
            logger.debug(f"FAISS search completed in {time.time() - search_start:.2f} seconds")
            context = [{"page": str(doc.metadata["page"]), "content": self.clean_text(doc.page_content)} for doc in docs]
            
            result = self.answer_pdf_question(target_question, temperature)
            if "Error" in result.answer:
                logger.error(f"Query error: {result.answer}")
                return None
            
            response = StructuredQueryResponse(
                question_number=question_number,
                question_text=target_question,
                context=context,
                task="",
                answer=result.answer,
                time_taken=result.time_taken
            )
            logger.info(f"Question file query processed in {time.time() - start_time:.2f} seconds")
            return response
        except Exception as e:
            logger.error(f"Error processing question file: {e}")
            return None
        finally:
            gc.collect()

    def process_image_with_query(self, image_data: bytes, query: str, temperature: float = 0.3) -> Optional[StructuredQueryResponse]:
        """Extract text from an image and query the textbook."""
        logger.debug(f"Processing image query: {query[:50]}...")
        start_time = time.time()
        if not self.pdf_processed or self.faiss_index is None:
            logger.error("No textbook processed")
            return None

        try:
            image_text = self.extract_text_from_image(image_data)
            if not image_text:
                logger.error("No text extracted from image")
                return None
            
            gc.collect()
            search_start = time.time()
            docs = self.faiss_index.similarity_search(query, k=2)
            logger.debug(f"FAISS search completed in {time.time() - search_start:.2f} seconds")
            context = [{"page": str(doc.metadata["page"]), "content": self.clean_text(doc.page_content)} for doc in docs]
            
            result = self.answer_pdf_question(query, temperature)
            if "Error" in result.answer:
                logger.error(f"Query error: {result.answer}")
                return None
            
            response = StructuredQueryResponse(
                question_number=None,
                question_text=image_text,
                context=context,
                task="",
                answer=result.answer,
                time_taken=result.time_taken
            )
            logger.info(f"Image query processed in {time.time() - start_time:.2f} seconds")
            return response
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None
        finally:
            gc.collect()

    def grade_solutions(self, solutions_data: bytes, questions_data: bytes) -> Optional[List[Dict]]:
        """Grade solutions from a PDF against textbook context."""
        logger.debug("Grading solutions")
        start_time = time.time()
        if not self.pdf_processed or self.faiss_index is None:
            logger.error("No textbook processed")
            return None

        try:
            solutions_text = self.extract_text_from_pdf(solutions_data)
            questions_text = self.extract_text_from_pdf(questions_data)
            if not solutions_text or not questions_text:
                logger.error("No text extracted from solutions or questions")
                return None
            
            solutions_cleaned = self.clean_text(solutions_text)
            questions_cleaned = self.clean_text(questions_text)
            
            solutions = re.split(r'\s*(?=\d+\.\s)', solutions_cleaned)
            solutions = [s.strip() for s in solutions if s.strip() and re.match(r'^\d+\.\s', s)]
            questions = re.split(r'\s*(?=\d+\.\s)', questions_cleaned)
            questions = [q.strip() for q in questions if q.strip() and re.match(r'^\d+\.\s', q)]
            
            results = []
            for idx, solution in enumerate(solutions):
                if idx >= len(questions):
                    break
                question = questions[idx]
                question_number = idx + 1
                
                gc.collect()
                search_start = time.time()
                docs = self.faiss_index.similarity_search(question, k=2)
                logger.debug(f"FAISS search completed in {time.time() - search_start:.2f} seconds")
                context = "\n".join([f"[Page {doc.metadata['page']}] {self.clean_text(doc.page_content)}" for doc in docs])
                if len(context) > 800:
                    context = context[:800]
                    logger.warning("Context truncated to 800 characters to fit context_length=1024")
                prompt = f"<s>[INST] Evaluate this solution for correctness based on the textbook context:\nContext: {context}\nQuestion: {question}\nSolution: {solution} [/INST]</s>"
                token_count = self.estimate_tokens(prompt)
                logger.debug(f"Estimated prompt tokens: {token_count}")
                if token_count > 1024:
                    logger.warning(f"Prompt tokens ({token_count}) exceed context_length (1024). Truncating context.")
                    context = context[:600]
                    prompt = f"<s>[INST] Evaluate this solution for correctness based on the textbook context:\nContext: {context}\nQuestion: {question}\nSolution: {solution} [/INST]</s>"

                with self.lock:
                    gc.collect()
                    model_start = time.time()
                    response = self.model(
                        prompt,
                        temperature=0.36,
                        top_k=40,
                        top_p=0.95
                    )
                    logger.debug(f"Model inference completed in {time.time() - model_start:.2f} seconds")
                    response_text = str(response)
                    response_tokens = self.estimate_tokens(response_text)
                    logger.debug(f"Generated response tokens: {response_tokens}")

                results.append({
                    "question_number": str(question_number),
                    "question_text": question,
                    "solution_text": solution,
                    "context": [{"page": str(doc.metadata["page"]), "content": self.clean_text(doc.page_content)} for doc in docs],
                    "task": "Evaluate the solution for correctness based on the textbook context.",
                    "answer": response_text,
                    "time_taken": f"{round(time.time() - start_time, 1)}"
                })
            
            logger.info(f"Solutions graded in {time.time() - start_time:.2f} seconds")
            return results
        except Exception as e:
            logger.error(f"Error grading solutions: {e}")
            return None
        finally:
            gc.collect()

    def answer_question(self, question: str, temperature: float = 0.36) -> Dict:
        """Solve a study problem step-by-step."""
        logger.debug(f"[{time.time()}] Answering question: {question[:50]}...")
        start_time = time.time()
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            logger.debug(f"[{time.time()}] Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

        cache_key = hashlib.md5(f"solve_{question}_{temperature:.2f}".encode()).hexdigest()
        if cache_key in self.response_cache:
            logger.info(f"[{time.time()}] Cached response found")
            return self.response_cache[cache_key]

        if len(question) > 500:
            logger.warning(f"[{time.time()}] Question truncated to 500 chars")
            question = question[:500] + "..."

        prompt = f"<s>[INST] You are a helpful study assistant. Solve this step by step: {question} [/INST]</s>"
        token_count = self.estimate_tokens(prompt)
        logger.debug(f"[{time.time()}] Estimated tokens: {token_count}")

        try:
            with self.lock:
                gc.collect()
                model_start = time.time()
                response = self.model(
                    prompt,
                    temperature=temperature,
                    top_k=40,
                    top_p=0.95
                )
                logger.debug(f"[{time.time()}] Model inference: {time.time() - model_start:.2f}s")
                response_text = str(response)
                elapsed_time = time.time() - start_time

            result = {
                "answer": response_text,
                "elapsed_time": f"{elapsed_time:.2f}"
            }
            self._update_cache(cache_key, result)
            logger.info(f"[{time.time()}] Question answered in {elapsed_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"[{time.time()}] Error: {str(e)}")
            return {"answer": f"Error: {str(e)}", "elapsed_time": "0.00"}
        finally:
            gc.collect()

    def generate_practice_questions(self, question: str, number: int = 3, temperature: float = 0.25) -> str:
        """Generate similar practice questions."""
        logger.debug(f"[{time.time()}] Generating {number} practice questions for: {question[:50]}...")
        start_time = time.time()
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            logger.debug(f"[{time.time()}] Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

        cache_key = hashlib.md5(f"practice_{question}_{number}_{temperature:.2f}".encode()).hexdigest()
        if cache_key in self.response_cache:
            logger.info(f"[{time.time()}] Cached response found")
            return self.response_cache[cache_key]

        if len(question) > 25:
            logger.warning(f"[{time.time()}] Question truncated to 25 chars")
            question = question[:25] + "..."

        prompt = f"<s>[INST] Create {number} practice questions similar to: {question} [/INST]</s>"
        try:
            with self.lock:
                gc.collect()
                model_start = time.time()
                response = self.model(
                    prompt,
                    temperature=temperature,
                    top_k=40,
                    top_p=0.95
                )
                logger.debug(f"[{time.time()}] Model inference: {time.time() - model_start:.2f}s")
                response_text = str(response)
                elapsed_time = time.time() - start_time
                
            self._update_cache(cache_key, response_text)
            logger.info(f"[{time.time()}] {number} practice questions generated in {elapsed_time:.2f}s")
            return response_text

        except Exception as e:
            logger.error(f"[{time.time()}] Error: {str(e)}")
            return f"Error: {str(e)}"
        finally:
            gc.collect()