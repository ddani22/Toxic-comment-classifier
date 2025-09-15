from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import secrets
import time
from collections import defaultdict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Toxic Comment Classifier API",
    description="API para clasificar comentarios como tóxicos o no tóxicos",
    version="1.0.0"
)

# Configurar CORS para permitir peticiones desde diferentes dominios
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Autenticación básica
security = HTTPBasic()

# Credenciales simples (en producción usar base de datos)
VALID_CREDENTIALS = {
    "admin": "secret123",
    "user": "password456"
}

# Rate limiting simple (en producción usar Redis)
request_counts = defaultdict(list)
RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # seconds

# Cargar modelo y vectorizador
try:
    clf = joblib.load("artifacts/model.pkl")
    vectorizer = joblib.load("artifacts/vectorizer.pkl")
    logger.info("Modelo y vectorizador cargados exitosamente")
except Exception as e:
    logger.error(f"Error cargando artefactos: {e}")
    clf = None
    vectorizer = None

# Modelos Pydantic para validación de datos
class CommentRequest(BaseModel):
    text: str
    include_probability: Optional[bool] = False

class CommentResponse(BaseModel):
    text: str
    is_toxic: bool
    label: str
    probability: Optional[float] = None

class BatchRequest(BaseModel):
    comments: List[str]
    include_probability: Optional[bool] = False

class BatchResponse(BaseModel):
    results: List[CommentResponse]
    total_processed: int

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verificar credenciales de usuario"""
    username = credentials.username
    password = credentials.password
    
    if username not in VALID_CREDENTIALS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario no válido",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    if not secrets.compare_digest(password, VALID_CREDENTIALS[username]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Contraseña incorrecta",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return username

def check_rate_limit(username: str):
    """Verificar rate limiting por usuario"""
    current_time = time.time()
    user_requests = request_counts[username]
    
    # Limpiar requests antiguos
    user_requests[:] = [req_time for req_time in user_requests 
                        if current_time - req_time < RATE_WINDOW]
    
    # Verificar límite
    if len(user_requests) >= RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit excedido. Máximo {RATE_LIMIT} requests por minuto."
        )
    
    # Registrar nueva request
    user_requests.append(current_time)

def get_current_user(username: str = Depends(verify_credentials)):
    """Dependency que combina autenticación y rate limiting"""
    check_rate_limit(username)
    return username

@app.get("/")
async def root():
    """Endpoint de salud del servicio"""
    return {
        "message": "Toxic Comment Classifier API",
        "status": "active",
        "model_loaded": clf is not None and vectorizer is not None
    }

@app.get("/health")
async def health_check():
    """Endpoint detallado de salud"""
    return {
        "status": "healthy" if clf is not None and vectorizer is not None else "unhealthy",
        "model_available": clf is not None,
        "vectorizer_available": vectorizer is not None,
        "timestamp": time.time()
    }

@app.post("/classify", response_model=CommentResponse)
async def classify_comment(
    request: CommentRequest,
    username: str = Depends(get_current_user)
):
    """Clasificar un solo comentario"""
    if clf is None or vectorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    try:
        # Vectorizar texto
        X_new = vectorizer.transform([request.text])
        
        # Predecir
        prediction = clf.predict(X_new)[0]
        is_toxic = bool(prediction)
        label = "Tóxico" if is_toxic else "No tóxico"
        
        # Calcular probabilidad si se solicita
        probability = None
        if request.include_probability:
            prob_scores = clf.predict_proba(X_new)[0]
            probability = float(prob_scores[1])  # Probabilidad de ser tóxico
        
        logger.info(f"Usuario {username} clasificó: '{request.text[:50]}...' -> {label}")
        
        return CommentResponse(
            text=request.text,
            is_toxic=is_toxic,
            label=label,
            probability=probability
        )
        
    except Exception as e:
        logger.error(f"Error en clasificación: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )

@app.post("/classify/batch", response_model=BatchResponse)
async def classify_batch(
    request: BatchRequest,
    username: str = Depends(get_current_user)
):
    """Clasificar múltiples comentarios en lote"""
    if clf is None or vectorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    if len(request.comments) > 100:  # Límite de batch
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Máximo 100 comentarios por batch"
        )
    
    try:
        results = []
        
        for comment in request.comments:
            # Vectorizar
            X_new = vectorizer.transform([comment])
            
            # Predecir
            prediction = clf.predict(X_new)[0]
            is_toxic = bool(prediction)
            label = "Tóxico" if is_toxic else "No tóxico"
            
            # Probabilidad opcional
            probability = None
            if request.include_probability:
                prob_scores = clf.predict_proba(X_new)[0]
                probability = float(prob_scores[1])
            
            results.append(CommentResponse(
                text=comment,
                is_toxic=is_toxic,
                label=label,
                probability=probability
            ))
        
        logger.info(f"Usuario {username} procesó batch de {len(request.comments)} comentarios")
        
        return BatchResponse(
            results=results,
            total_processed=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error en batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor"
        )

@app.get("/stats")
async def get_stats(username: str = Depends(get_current_user)):
    """Obtener estadísticas de uso (simple)"""
    total_requests = sum(len(requests) for requests in request_counts.values())
    
    return {
        "total_requests": total_requests,
        "active_users": len(request_counts),
        "your_requests_last_minute": len(request_counts[username])
    }