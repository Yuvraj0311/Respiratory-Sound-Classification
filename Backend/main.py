"""
main.py - FastAPI Backend for A.I.R.A. (AI Respiratory Assistant)
FIXED VERSION FOR DEPLOYMENT
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, EmailStr
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, Any, List
import os
import logging
from datetime import datetime, timedelta
import uuid
import shutil
from contextlib import asynccontextmanager
from jose import jwt, JWTError

from llm_agent import LungScopeChatbot
from database import DataRetrievalAgent
from audio_classifier import RespiratoryAudioClassifier
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_backend_path(relative_path: str) -> str:
    """Resolve paths from either the backend directory or the repo root."""
    if os.path.exists(relative_path):
        return relative_path
    return os.path.join(BASE_DIR, relative_path)


def safe_upload_filename(filename: Optional[str]) -> str:
    """Keep uploaded filenames local and filesystem-friendly."""
    base_name = os.path.basename(filename or "upload")
    safe_name = "".join(
        char if char.isalnum() or char in {".", "_", "-"} else "_"
        for char in base_name
    )
    return safe_name or "upload"


AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac'}
XRAY_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# ---------------- JWT CONFIG ---------------- #
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION_12345")
if SECRET_KEY == "CHANGE_THIS_IN_PRODUCTION_12345":
    logging.warning("WARNING: Using default JWT secret key! Set JWT_SECRET_KEY environment variable in production!")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# ---------------- LOGGING ---------------- #
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lungscope_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- GLOBAL VARIABLES ---------------- #
chatbot = None
audio_classifier = None
xray_classifier = None

# ---------------- LIFESPAN HANDLER ---------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all agents and resources on startup; clean up on shutdown."""
    global chatbot, audio_classifier, xray_classifier

    logger.info("A.I.R.A Backend starting up...")
    os.makedirs("uploads/audio", exist_ok=True)
    os.makedirs("uploads/xray", exist_ok=True)
    os.makedirs("uploads/avatars", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger.info("Directories created")

    try:
        chatbot = LungScopeChatbot(model_name="gpt-4.1-nano")
        logger.info("LLM Chatbot initialized")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        chatbot = None

    try:
        audio_classifier = RespiratoryAudioClassifier(
            model_path=resolve_backend_path("models/trained_model.keras"),
            sample_rate=22050,
            duration=20.0,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            target_length=432
        )
        logger.info("Audio Classifier initialized")
    except Exception as e:
        logger.error(f"Failed to initialize audio classifier: {e}")
        audio_classifier = None

    try:
        from xray_classifier import LungXRayClassifier

        xray_classifier = LungXRayClassifier(
            model_path=resolve_backend_path("models/resnet18_finetuned.pth")
        )
        logger.info("X-ray Classifier initialized")
    except Exception as e:
        logger.error(f"Failed to initialize X-ray classifier: {e}")
        xray_classifier = None

    yield

    logger.info("A.I.R.A Backend shutting down...")
    if chatbot:
        try:
            chatbot.cleanup()
        except:
            pass
    logger.info("Cleanup complete")


# ---------------- FASTAPI APP ---------------- #
app = FastAPI(
    title="A.I.R.A API",
    description="AI Healthcare Assistant for Respiratory Analysis and Smart Diagnostics",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ---------------- CORS MIDDLEWARE (FIXED FOR DEPLOYMENT) ---------------- #
# Get allowed origins from environment variable or use defaults
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Configure this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PYDANTIC MODELS ---------------- #
class UserRegister(BaseModel):
    email: EmailStr
    username: str
    full_name: str
    password: str
    age_range: Optional[str] = None
    gender: Optional[str] = None
    smoking_status: Optional[str] = None
    has_hypertension: Optional[bool] = False
    has_diabetes: Optional[bool] = False
    has_asthma_history: Optional[bool] = False
    previous_respiratory_infections: Optional[int] = 0
    current_medications: Optional[str] = ""
    allergies: Optional[str] = ""

class UserLogin(BaseModel):
    username_or_email: str
    password: str

class GoogleAuthData(BaseModel):
    email: EmailStr
    name: str
    picture: str
    google_id: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_info: Dict[str, Any]

class ChatRequest(BaseModel):
    patient_id: str = Field(..., example="PATIENT_12345")
    conversation_id: Optional[str] = Field(None, example="3f766aa8-4a7f-44cc-a056-58b667b621c7")
    query: str = Field(..., example="What should I do about my breathing difficulty?")
    recent_messages: Optional[List[str]] = Field(None, example=[
        "I feel short of breath today.",
        "It gets worse when I walk and I hear wheezing."
    ])
    audio_result: Optional[Dict[str, Any]] = Field(None, example={
        "disease": "COPD",
        "confidence": 0.9759,
        "severity": "moderate"
    })
    xray_result: Optional[Dict[str, Any]] = Field(None, example={
        "finding": "Viral Pneumonia",
        "confidence": 0.91,
        "severity": "high"
    })

class ChatResponse(BaseModel):
    response: str
    confidence: Optional[float] = None
    disease_classification: Optional[str] = None
    xray_classification: Optional[str] = None
    follow_up: Optional[str] = None
    turn_count: Optional[int] = None  
    should_request_audio: Optional[bool] = None  
    should_request_xray: Optional[bool] = None
    recommended_tool: Optional[str] = None
    tool_selection_reason: Optional[str] = None
    diagnostic_tool_decision: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class AudioAnalysisResponse(BaseModel):
    disease: str
    confidence: float
    severity: str
    timestamp: str

class XRayAnalysisResponse(BaseModel):
    finding: str
    confidence: float
    severity: str
    timestamp: str
    modality: Optional[str] = None
    knowledge_disease: Optional[str] = None
    clinical_note: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    top_predictions: Optional[List[Dict[str, Any]]] = None

class PatientDataResponse(BaseModel):
    patient_id: str
    email: str
    username: str
    full_name: str
    age_range: Optional[str]
    gender: Optional[str]
    smoking_status: Optional[str]
    comorbidities: List[str]
    current_medications: Optional[str]
    allergies: Optional[str]
    avatar: Optional[str] = None
    has_hypertension: Optional[bool] = False
    has_diabetes: Optional[bool] = False
    has_asthma_history: Optional[bool] = False
    previous_respiratory_infections: Optional[int] = 0
    last_consultation_date: Optional[str] = None

# ---------------- ENDPOINTS ---------------- #
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "A.I.R.A API",
        "version": "2.0.0",
        "status": "operational",
        "environment": "production" if os.getenv("RENDER") else "development",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "register": "/register",
            "login": "/login",
            "google_login": "/google-login",
            "chat": "/api/chat",
            "audio_analysis": "/api/analyze-audio",
            "xray_analysis": "/api/analyze-xray",
            "full_analysis": "/api/full-analysis",
            "full_xray_analysis": "/api/full-xray-analysis",
            "patient_data": "/api/patient/{patient_id}"
        }
    }

# ---------------- HEALTH CHECK ---------------- #
@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    global chatbot, audio_classifier, xray_classifier
    
    return { 
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "services": {
            "chatbot": "available" if chatbot else "unavailable",
            "audio_classifier": "available" if audio_classifier else "unavailable",
            "xray_classifier": "available" if xray_classifier else "unavailable",
            "database": "configured" if os.getenv("POSTGRES_DB") else "not configured"
        }
    }

# ---------------- AUTHENTICATION ---------------- #
@app.post("/register", tags=["Authentication"])
async def register_patient(
    email: str = Form(...),
    username: str = Form(...),
    full_name: str = Form(...),
    password: str = Form(...),
    age_range: str = Form(None),
    gender: str = Form(None),
    smoking_status: str = Form(None),
    has_hypertension: bool = Form(False),
    has_diabetes: bool = Form(False),
    has_asthma_history: bool = Form(False),
    previous_respiratory_infections: int = Form(0),
    current_medications: str = Form(""),
    allergies: str = Form(""),
    avatar: UploadFile = File(None)
):
    """Register a new patient with email, username, and password"""
    
    patient_id = f"PT_{uuid.uuid4().hex[:12].upper()}"
    
    avatar_path = None
    if avatar:
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        file_ext = os.path.splitext(avatar.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format. Allowed formats: {', '.join(allowed_extensions)}"
            )
        
        os.makedirs("uploads/avatars", exist_ok=True)
        avatar_filename = f"{username}_{uuid.uuid4()}{file_ext}"
        avatar_path = os.path.join("uploads/avatars", avatar_filename)
        
        with open(avatar_path, "wb") as buffer:
            shutil.copyfileobj(avatar.file, buffer)
        
        avatar_path = f"/uploads/avatars/{avatar_filename}"
    
    db_data = {
        'patient_id': patient_id,
        'email': email.lower(),
        'username': username,
        'full_name': full_name,
        'password': password,
        'age_range': age_range,
        'gender': gender,
        'smoking_status': smoking_status,
        'has_hypertension': has_hypertension,
        'has_diabetes': has_diabetes,
        'has_asthma_history': has_asthma_history,
        'previous_respiratory_infections': previous_respiratory_infections,
        'current_medications': current_medications,
        'allergies': allergies,
        'avatar': avatar_path,
        'auth_provider': 'local'
    }
    
    agent = DataRetrievalAgent()
    try:
        success = agent.db_manager.create_patient(db_data)
        if not success:
            if avatar_path and os.path.exists(avatar_path.lstrip('/')):
                os.remove(avatar_path.lstrip('/'))
            raise HTTPException(
                status_code=409,
                detail="Username or email already exists. Please choose different credentials."
            )
        
        logger.info(f"New patient registered: {username}")
        return JSONResponse(
            content={
                "message": "Registration successful! You can now login.",
                "patient_id": patient_id,
                "username": username,
                "email": email,
                "avatar": avatar_path
            },
            status_code=201
        )
    except HTTPException:
        raise
    except Exception as e:
        if avatar_path and os.path.exists(avatar_path.lstrip('/')):
            os.remove(avatar_path.lstrip('/'))
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Registration failed: {str(e)}"
        )
    finally:
        agent.cleanup()

@app.post("/login", response_model=Token, tags=["Authentication"])
async def login_for_access_token(
    username_or_email: str = Form(...),
    password: str = Form(...)
):
    """Login with username/email and password"""
    agent = DataRetrievalAgent()
    try:
        user = agent.db_manager.get_patient_for_auth(username_or_email.lower())
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials. Please check your username/email and password."
            )
        
        if user.get('auth_provider') == 'google':
            raise HTTPException(
                status_code=401,
                detail="This account uses Google Sign-In. Please use 'Sign in with Google'."
            )
        
        if not agent.db_manager.verify_password(password, user['password_hash']):
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials. Please check your username/email and password."
            )
        
        patient_profile = agent.db_manager.get_patient_data(user['patient_id'])
        if not patient_profile:
            raise HTTPException(
                status_code=500,
                detail="Failed to load patient profile."
            )
        
        access_token = create_access_token(data={"sub": user["patient_id"]})
        
        avatar_url = patient_profile.get("avatar") or "https://ui-avatars.com/api/?name=Patient&size=150&background=4A90E2&color=ffffff"
        
        logger.info(f"Patient logged in: {user['username']}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_info={
                "patient_id": patient_profile.get("patient_id"),
                "username": patient_profile.get("username"),
                "name": patient_profile.get("full_name"),
                "email": patient_profile.get("email"),
                "avatar": avatar_url,
                "age_range": patient_profile.get("age_range"),
                "gender": patient_profile.get("gender"),
                "auth_provider": user.get("auth_provider", "local")
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Login failed: {str(e)}"
        )
    finally:
        agent.cleanup()

@app.post("/google-login", response_model=Token, tags=["Authentication"])
async def google_login(google_data: GoogleAuthData):
    """Login or register with Google OAuth"""
    agent = DataRetrievalAgent()
    try:
        user = agent.db_manager.get_patient_for_auth(google_data.email.lower())
        
        if user:
            patient_profile = agent.db_manager.get_patient_data(user['patient_id'])
        else:
            patient_id = f"PT_{uuid.uuid4().hex[:12].upper()}"
            username = google_data.email.split('@')[0]
            
            counter = 1
            original_username = username
            while agent.db_manager.get_patient_for_auth(username):
                username = f"{original_username}{counter}"
                counter += 1
            
            db_data = {
                'patient_id': patient_id,
                'email': google_data.email.lower(),
                'username': username,
                'full_name': google_data.name,
                'password': None,
                'avatar': google_data.picture,
                'auth_provider': 'google'
            }
            
            success = agent.db_manager.create_patient(db_data)
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create account with Google"
                )
            
            patient_profile = agent.db_manager.get_patient_data(patient_id)
            logger.info(f"New Google user registered: {google_data.email}")
        
        access_token = create_access_token(data={"sub": patient_profile["patient_id"]})
        
        logger.info(f"Google login successful: {google_data.email}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_info={
                "patient_id": patient_profile.get("patient_id"),
                "username": patient_profile.get("username"),
                "name": patient_profile.get("full_name"),
                "email": patient_profile.get("email"),
                "avatar": patient_profile.get("avatar") or google_data.picture,
                "age_range": patient_profile.get("age_range"),
                "gender": patient_profile.get("gender"),
                "auth_provider": "google"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google login error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Google login failed: {str(e)}"
        )
    finally:
        agent.cleanup()

# ---------------- API ENDPOINTS ---------------- #
@app.post("/api/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat_with_ai(request: ChatRequest):
    """Handles chat interactions with the AI assistant."""
    global chatbot
    
    if not chatbot:
        raise HTTPException(
            status_code=503, 
            detail="Chatbot service not available. Please check backend logs."
        )
    
    try:
        logger.info(f"Chat request from patient: {request.patient_id[:8]}...")
        
        response = chatbot.query(
            user_query=request.query,
            patient_id=request.patient_id,
            conversation_id=request.conversation_id,
            recent_messages=request.recent_messages,
            audio_result=request.audio_result,
            xray_result=request.xray_result
        )
        
        return ChatResponse(**response)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Chat failed: {str(e)}"
        )

# ---------------- AUDIO ANALYSIS ---------------- #
@app.post("/api/analyze-audio", tags=["Audio Analysis"])
async def analyze_respiratory_audio(
    patient_id: str = Form(...), 
    user_query: str = Form(""),
    file: UploadFile = File(...)
):
    """Analyzes a respiratory audio file for disease classification."""
    global audio_classifier
    
    if not audio_classifier:
        raise HTTPException(
            status_code=503, 
            detail="Audio classifier not available. Please check if the model file exists."
        )
    
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    
    if file_ext not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid audio file format. Allowed formats: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    file_id = str(uuid.uuid4())
    file_path = os.path.join("uploads/audio", f"{file_id}_{safe_upload_filename(file.filename)}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Analyzing audio file for patient: {patient_id[:8]}...")
        
        result = audio_classifier.predict(file_path)
        
        logger.info(f"Prediction: {result['disease']} ({result['confidence']:.4f})")
        
        return AudioAnalysisResponse(
            disease=result["disease"],
            confidence=result["confidence"],
            severity=result["severity"],
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in audio analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

# ---------------- X-RAY ANALYSIS ---------------- #
@app.post("/api/analyze-xray", response_model=XRayAnalysisResponse, tags=["X-ray Analysis"])
async def analyze_chest_xray(
    patient_id: str = Form(...),
    user_query: str = Form(""),
    file: UploadFile = File(...)
):
    """Analyzes a chest X-ray image for screening classification."""
    global xray_classifier

    if not xray_classifier:
        raise HTTPException(
            status_code=503,
            detail="X-ray classifier not available. Please check if the model file exists and PyTorch is installed."
        )

    file_ext = os.path.splitext(file.filename or "")[1].lower()
    if file_ext not in XRAY_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid X-ray image format. Allowed formats: {', '.join(sorted(XRAY_IMAGE_EXTENSIONS))}"
        )

    file_id = str(uuid.uuid4())
    file_path = os.path.join("uploads/xray", f"{file_id}_{safe_upload_filename(file.filename)}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Analyzing X-ray image for patient: {patient_id[:8]}...")
        result = xray_classifier.predict(file_path)
        logger.info(f"X-ray prediction: {result['finding']} ({result['confidence']:.4f})")

        return XRayAnalysisResponse(
            finding=result["finding"],
            confidence=result["confidence"],
            severity=result["severity"],
            timestamp=datetime.now().isoformat(),
            modality=result.get("modality"),
            knowledge_disease=result.get("knowledge_disease"),
            clinical_note=result.get("clinical_note"),
            probabilities=result.get("probabilities"),
            top_predictions=result.get("top_predictions")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in X-ray analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

# ---------------- COMBINED WORKFLOW ---------------- #
@app.post("/api/full-analysis", tags=["Complete Analysis"])
async def complete_respiratory_analysis(
    file: UploadFile = File(..., description="Respiratory audio file"),
    patient_id: str = Form(..., description="Patient identifier"),
    query: str = Form(..., description="Patient's medical question")
):
    """Complete end-to-end analysis workflow"""
    file_path = None
    try:
        if not audio_classifier or not chatbot:
            raise HTTPException(status_code=503, detail="Services not fully initialized")
        
        logger.info(f"Full analysis request from patient: {patient_id[:8]}***")
        
        file_ext = os.path.splitext(file.filename or "")[1].lower()
        if file_ext not in AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Invalid audio format. Supported: .wav, .mp3, .flac")
        
        file_id = str(uuid.uuid4())
        file_path = os.path.join("uploads/audio", f"{file_id}_{safe_upload_filename(file.filename)}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info("Classifying audio...")
        audio_result = audio_classifier.predict(file_path)
        logger.info(f"Audio classified as: {audio_result['disease']} (confidence: {audio_result['confidence']:.2f})")
        
        logger.info("Generating AI response with RAG pipeline...")
        chat_response = chatbot.query(
            user_query=query,
            patient_id=patient_id,
            audio_result=audio_result
        )
        logger.info("Chat response generated successfully")
        
        return {
            "audio_analysis": {
                "disease": audio_result["disease"],
                "confidence": audio_result["confidence"],
                "severity": audio_result.get("severity", "N/A"),
                "probabilities": audio_result.get("probabilities", {})
            },
            "ai_response": chat_response,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Full analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Audio analysis failed: {str(e)}"
        )
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up audio file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete audio file: {e}")

@app.post("/api/full-xray-analysis", tags=["Complete Analysis"])
async def complete_xray_analysis(
    file: UploadFile = File(..., description="Chest X-ray image file"),
    patient_id: str = Form(..., description="Patient identifier"),
    query: str = Form(..., description="Patient's medical question")
):
    """Complete X-ray analysis workflow with RAG response."""
    file_path = None
    try:
        if not xray_classifier or not chatbot:
            raise HTTPException(status_code=503, detail="Services not fully initialized")

        logger.info(f"Full X-ray analysis request from patient: {patient_id[:8]}***")

        file_ext = os.path.splitext(file.filename or "")[1].lower()
        if file_ext not in XRAY_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid X-ray image format. Supported: {', '.join(sorted(XRAY_IMAGE_EXTENSIONS))}"
            )

        file_id = str(uuid.uuid4())
        file_path = os.path.join("uploads/xray", f"{file_id}_{safe_upload_filename(file.filename)}")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info("Classifying X-ray image...")
        xray_result = xray_classifier.predict(file_path)
        logger.info(
            f"X-ray classified as: {xray_result['finding']} "
            f"(confidence: {xray_result['confidence']:.2f})"
        )

        logger.info("Generating AI response with RAG pipeline...")
        chat_response = chatbot.query(
            user_query=query,
            patient_id=patient_id,
            xray_result=xray_result
        )
        logger.info("Chat response generated successfully")

        return {
            "xray_analysis": {
                "finding": xray_result["finding"],
                "confidence": xray_result["confidence"],
                "severity": xray_result.get("severity", "N/A"),
                "knowledge_disease": xray_result.get("knowledge_disease"),
                "probabilities": xray_result.get("probabilities", {})
            },
            "ai_response": chat_response,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Full X-ray analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"X-ray analysis failed: {str(e)}"
        )
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up X-ray file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete X-ray file: {e}")

# ---------------- PATIENT DATA MANAGEMENT ---------------- #
@app.get("/api/patient/{patient_id}", response_model=PatientDataResponse, tags=["Patient Data"])
async def get_patient_data(patient_id: str):
    """Retrieves medical information for a specific patient."""
    agent = DataRetrievalAgent()
    try:
        patient_data = agent.db_manager.get_patient_data(patient_id)
        if not patient_data:
            raise HTTPException(
                status_code=404, 
                detail="Patient not found"
            )
        
        comorbidities = []
        if patient_data.get('has_hypertension'): 
            comorbidities.append('Hypertension')
        if patient_data.get('has_diabetes'): 
            comorbidities.append('Diabetes')
        if patient_data.get('has_asthma_history'): 
            comorbidities.append('Asthma History')
        
        return PatientDataResponse(
            patient_id=patient_data['patient_id'],
            email=patient_data.get('email'),
            username=patient_data.get('username'),
            full_name=patient_data.get('full_name'),
            age_range=patient_data.get('age_range'),
            gender=patient_data.get('gender'),
            smoking_status=patient_data.get('smoking_status'),
            comorbidities=comorbidities,
            current_medications=patient_data.get('current_medications'),
            allergies=patient_data.get('allergies'),
            avatar=patient_data.get('avatar'),
            has_hypertension=patient_data.get('has_hypertension'),
            has_diabetes=patient_data.get('has_diabetes'),
            has_asthma_history=patient_data.get('has_asthma_history'),
            previous_respiratory_infections=patient_data.get('previous_respiratory_infections'),
            last_consultation_date=patient_data.get('last_consultation_date')
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get patient data error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve patient data: {str(e)}"
        )
    finally:
        agent.cleanup()

# ---------------- PATIENT HISTORY ---------------- #
@app.get("/api/patient/{patient_id}/history", tags=["Patient Data"])
async def get_patient_history(patient_id: str, limit: int = 5):
    """Retrieves recent medical history for a patient."""
    agent = DataRetrievalAgent()
    try:
        history = agent.db_manager.get_patient_history(patient_id, limit)
        return {
            "patient_id": patient_id, 
            "history_count": len(history), 
            "history": history
        }
    except Exception as e:
        logger.error(f"Get patient history error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve history: {str(e)}"
        )
    finally:
        agent.cleanup()

# ---------------- MODEL INFO ---------------- #
@app.get("/api/model/info", tags=["Model Information"])
async def get_model_info():
    """Gets information about the loaded audio classification model."""
    global audio_classifier
    
    if not audio_classifier:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded"
        )
    
    try:
        return JSONResponse(content=audio_classifier.get_model_info())
    except Exception as e:
        logger.error(f"Get model info error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get model info: {str(e)}"
        )

@app.get("/api/model/xray-info", tags=["Model Information"])
async def get_xray_model_info():
    """Gets information about the loaded X-ray classification model."""
    global xray_classifier

    if not xray_classifier:
        raise HTTPException(
            status_code=503,
            detail="X-ray model not loaded"
        )

    try:
        return JSONResponse(content=xray_classifier.get_model_info())
    except Exception as e:
        logger.error(f"Get X-ray model info error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get X-ray model info: {str(e)}"
        )

# ---------------- UPDATE PATIENT PROFILE ---------------- #
@app.put("/api/patient/{username}/update", tags=["Patient Data"])
async def update_patient_profile(
    username: str,
    email: Optional[str] = Form(None),
    full_name: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    age_range: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    smoking_status: Optional[str] = Form(None),
    has_hypertension: Optional[bool] = Form(None),
    has_diabetes: Optional[bool] = Form(None),
    has_asthma_history: Optional[bool] = Form(None),
    previous_respiratory_infections: Optional[int] = Form(None),
    current_medications: Optional[str] = Form(None),
    allergies: Optional[str] = Form(None),
    last_consultation_date: Optional[str] = Form(None),
    avatar: Optional[UploadFile] = File(None)
):
    """Update patient profile information including avatar and password"""
    agent = DataRetrievalAgent()
    try:
        user = agent.db_manager.get_patient_for_auth(username)
        if not user:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        patient_id = user['patient_id']
        patient = agent.db_manager.get_patient_data(patient_id)
        
        update_data = {}
        
        form_data = {
            "email": email, "full_name": full_name, "age_range": age_range, 
            "gender": gender, "smoking_status": smoking_status,
            "has_hypertension": has_hypertension, "has_diabetes": has_diabetes,
            "has_asthma_history": has_asthma_history, 
            "previous_respiratory_infections": previous_respiratory_infections,
            "current_medications": current_medications, "allergies": allergies,
            "last_consultation_date": last_consultation_date, "password": password
        }

        for key, value in form_data.items():
            if value is not None:
                update_data[key] = value

        if avatar:
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            file_ext = os.path.splitext(avatar.filename)[1].lower()
            
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image format. Allowed: {', '.join(allowed_extensions)}"
                )
            
            os.makedirs("uploads/avatars", exist_ok=True)
            
            old_avatar = patient.get('avatar')
            if old_avatar and not old_avatar.startswith('http') and os.path.exists(old_avatar.lstrip('/')):
                try:
                    os.remove(old_avatar.lstrip('/'))
                except Exception as e:
                    logger.warning(f"Could not delete old avatar: {e}")
            
            avatar_filename = f"{username}_{uuid.uuid4()}{file_ext}"
            avatar_path = os.path.join("uploads/avatars", avatar_filename)
            
            with open(avatar_path, "wb") as buffer:
                shutil.copyfileobj(avatar.file, buffer)
            
            update_data["avatar"] = f"/uploads/avatars/{avatar_filename}"

        if not update_data:
             return JSONResponse(content={"message": "No fields to update"}, status_code=200)

        success = agent.db_manager.update_patient(patient_id, update_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update patient profile")

        return JSONResponse(content={
            "message": "Profile updated successfully",
            "updated_fields": list(update_data.keys())
        }, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        agent.cleanup()

# ---------------- STATIC FILES ---------------- #
if os.path.exists("uploads"):
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
else:
    os.makedirs("uploads/audio", exist_ok=True)
    os.makedirs("uploads/xray", exist_ok=True)
    os.makedirs("uploads/avatars", exist_ok=True)
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
    
# ---------------- ERROR HANDLERS ---------------- #
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "path": str(request.url),
            "available_endpoints": "/docs"
        }
    )

# ---------------- MAIN EXECUTION BLOCK ---------------- #
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting A.I.R.A Backend Server...")
    logger.info("API Documentation available at: http://localhost:8000/docs")
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False if os.getenv("RENDER") else True,
        log_level="info"
    )
