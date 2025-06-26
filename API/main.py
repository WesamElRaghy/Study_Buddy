from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
import os
import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from backend import StudyBuddyBackend
from schemas import (
    TextbookQuery, PracticeQuestionsRequest, StructuredQueryResponse,
    UserCreate, UserResponse, Token, LessonCreate, LessonResponse, EnrollmentCreate
)
from database import init_db, SessionLocal, User, UserRole, Lesson, Enrollment
from auth import (
    get_db, get_password_hash, verify_password, create_access_token,
    get_current_user, get_current_student, get_current_teacher,
    get_current_admin
)
import uuid
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is not set in environment variables")

app = FastAPI(title="StudyBuddy RAG API", description="Unified API for study assistant and user management.")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.getcwd(), "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
try:
    logger.debug(f"Checking model path: {MODEL_PATH}")
    backend = StudyBuddyBackend(MODEL_PATH)
except Exception as e:
    logger.error(f"Failed to initialize backend: {e}")
    raise RuntimeError(f"Failed to initialize backend: {str(e)}")

@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/auth/signup", response_model=UserResponse, tags=["Authentication"])
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if not user.password and not user.google_id:
        raise HTTPException(status_code=400, detail="Password or Google ID required")
    hashed_password = get_password_hash(user.password) if user.password else None
    new_user = User(
        uid=str(uuid.uuid4()),
        email=user.email,
        password_hash=hashed_password,
        google_id=user.google_id,
        full_name=user.full_name,
        role=UserRole(user.role)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/auth/token", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token(data={"sub": user.uid})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# @app.post("/auth/google/mobile", response_model=Token, tags=["Authentication"])
# async def google_mobile_login(id_token: str = Form(...), db: Session = Depends(get_db)):
#     return await handle_google_mobile_callback(id_token, db)

# @app.get("/auth/google/login", tags=["Authentication"])
# async def google_login(request: Request):
#     redirect_uri = "http://localhost:8000/auth/google/callback"
#     return await oauth.google.authorize_redirect(request, redirect_uri)

@app.put("/users/{uid}", response_model=UserResponse, tags=["User"])
async def update_user(uid: str, full_name: Optional[str] = Form(None), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.uid != uid:
        raise HTTPException(status_code=403, detail="Not authorized")
    user = db.query(User).filter(User.uid == uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if full_name:
        user.full_name = full_name
    db.commit()
    db.refresh(user)
    return user

@app.delete("/users/{uid}", tags=["User"])
async def delete_user(uid: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.uid != uid:
        raise HTTPException(status_code=403, detail="Not authorized")
    user = db.query(User).filter(User.uid == uid).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"status": "success"}

@app.post("/lessons", response_model=LessonResponse, tags=["Lesson"])
async def create_lesson(lesson: LessonCreate, current_teacher: User = Depends(get_current_teacher), db: Session = Depends(get_db)):
    new_lesson = Lesson(
        title=lesson.title,
        description=lesson.description,
        teacher_id=current_teacher.id
    )
    db.add(new_lesson)
    db.commit()
    db.refresh(new_lesson)
    return new_lesson

@app.get("/lessons", response_model=List[LessonResponse], tags=["Lesson"])
async def list_lessons(db: Session = Depends(get_db)):
    return db.query(Lesson).all()

@app.get("/lessons/teacher/{teacher_id}", response_model=List[LessonResponse], tags=["Lesson"])
async def get_teacher_lessons(teacher_id: int, db: Session = Depends(get_db)):
    return db.query(Lesson).filter(Lesson.teacher_id == teacher_id).all()

@app.post("/enrollments", tags=["Enrollment"])
async def enroll(enrollment: EnrollmentCreate, current_student: User = Depends(get_current_student), db: Session = Depends(get_db)):
    if db.query(Lesson).filter(Lesson.id == enrollment.lesson_id).first() is None:
        raise HTTPException(status_code=404, detail="Lesson not found")
    if db.query(Enrollment).filter_by(student_id=current_student.id, lesson_id=enrollment.lesson_id).first():
        raise HTTPException(status_code=400, detail="Already enrolled")
    db.add(Enrollment(student_id=current_student.id, lesson_id=enrollment.lesson_id))
    db.commit()
    return {"status": "success"}

@app.get("/enrollments/{student_id}", response_model=List[int], tags=["Enrollment"])
async def list_enrollments(student_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.id != student_id and current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    return [e.lesson_id for e in db.query(Enrollment).filter_by(student_id=student_id).all()]

@app.post("/process-textbook/", tags=["RAG"])
async def process_textbook(file: UploadFile = File(...), start_page: Optional[int] = Form(None), end_page: Optional[int] = Form(None), current_user: User = Depends(get_current_user)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    try:
        pdf_data = await file.read()
        success = backend.process_pdf(pdf_data, (start_page, end_page) if start_page and end_page else None)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process textbook")
        return {"status": "success", "index_path": backend.corpus_path}
    except Exception as e:
        logger.error(f"Textbook processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/textbook-query/", response_model=StructuredQueryResponse, tags=["RAG"])
async def textbook_query(query: TextbookQuery, current_user: User = Depends(get_current_user)):
    try:
        return backend.answer_pdf_question(query.query, query.temperature)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve-problem/", tags=["RAG"])
async def solve_problem(query: TextbookQuery, current_user: User = Depends(get_current_user)):
    try:
        return backend.answer_question(query.query, query.temperature)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-practice-questions/", tags=["RAG"])
async def generate_questions(req: PracticeQuestionsRequest, current_user: User = Depends(get_current_user)):
    try:
        return {"questions": backend.generate_practice_questions(req.question, req.num_questions, req.temperature)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset-pdf/", tags=["RAG"])
async def reset_pdf(current_user: User = Depends(get_current_user)):
    try:
        backend.reset_pdf()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)