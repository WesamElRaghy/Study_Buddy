from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from starlette.config import Config
from authlib.integrations.starlette_client import OAuth
from database import SessionLocal, User, UserRole
import uuid
import os
from datetime import datetime, timedelta

# Environment and secrets
SECRET_KEY = os.getenv("SECRET_KEY", "insecure-dev-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth setup
# config = Config(environ={
#     "GOOGLE_CLIENT_ID": os.getenv("GOOGLE_CLIENT_ID", ""),
#     "GOOGLE_CLIENT_SECRET": os.getenv("GOOGLE_CLIENT_SECRET", ""),
# })
# oauth = OAuth(config)
# oauth.register(
#     name='google',
#     server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
#     client_kwargs={'scope': 'openid email profile'},
# )

# OAuth2 password bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Hashing
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Token creation
def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Token verification
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid = payload.get("sub")
        if uid is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.uid == uid).first()
    if user is None:
        raise credentials_exception
    return user

# Role-based dependencies
def get_current_teacher(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != UserRole.teacher:
        raise HTTPException(status_code=403, detail="Only teachers can access this endpoint")
    return current_user

def get_current_student(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != UserRole.student:
        raise HTTPException(status_code=403, detail="Only students can access this endpoint")
    return current_user

def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Only admins can access this endpoint")
    return current_user

# Google OAuth: Web callback
# async def handle_google_web_callback(request: Request, db: Session):
#     token = await oauth.google.authorize_access_token(request)
#     user_info = await oauth.google.parse_id_token(request, token)
#     email = user_info.get("email")
#     google_id = user_info.get("sub")
#     full_name = user_info.get("name")

#     if not email or not google_id:
#         raise HTTPException(status_code=400, detail="Invalid Google response")

#     user = db.query(User).filter(User.email == email).first()
#     if not user:
#         user = User(
#             uid=str(uuid.uuid4()),
#             email=email,
#             google_id=google_id,
#             full_name=full_name,
#             role=UserRole.student
#         )
#         db.add(user)
#         db.commit()
#         db.refresh(user)

#     access_token = create_access_token(data={"sub": user.uid})
#     return {"access_token": access_token, "token_type": "bearer"}

# Google OAuth: Mobile client
# async def handle_google_mobile_callback(request: Request, db: Session):
#     data = await request.json()
#     token_id = data.get("token_id")
#     if not token_id:
#         raise HTTPException(status_code=400, detail="Missing token_id from mobile client")

#     # Validate token manually (you can use a Google library or your own logic)
#     from google.oauth2 import id_token
#     from google.auth.transport import requests as google_requests

#     try:
#         idinfo = id_token.verify_oauth2_token(token_id, google_requests.Request(), os.getenv("GOOGLE_CLIENT_ID", ""))
#     except Exception:
#         raise HTTPException(status_code=401, detail="Invalid Google token")

#     email = idinfo.get("email")
#     google_id = idinfo.get("sub")
#     full_name = idinfo.get("name")

#     if not email or not google_id:
#         raise HTTPException(status_code=400, detail="Google ID token missing data")

#     user = db.query(User).filter(User.email == email).first()
#     if not user:
#         user = User(
#             uid=str(uuid.uuid4()),
#             email=email,
#             google_id=google_id,
#             full_name=full_name,
#             role=UserRole.student
#         )
#         db.add(user)
#         db.commit()
#         db.refresh(user)

#     access_token = create_access_token(data={"sub": user.uid})
#     return {"access_token": access_token, "token_type": "bearer"}
