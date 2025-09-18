from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import joblib
import secrets
import time
import logging
import os
import jwt
from collections import defaultdict
from datetime import datetime, timedelta

# Configurar logging, configura el sistema de logs para registrar eventos importantes de la API.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crea la aplicación FastAPI
app = FastAPI(
    title="Toxic Comment Classifier API",
    description="API para clasificar comentarios como tóxicos o no tóxicos",
    version="1.0.0"
)

# Configurar CORS, permite que navegadores web desde cualquier dominio puedan hacer peticiones a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite peticiones desde cualquier dominio (⚠️ inseguro en producción). En producción, especifica dominios específicos
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
clf = None
vectorizer = None
detailed_clf = None
detailed_vectorizer = None
toxicity_categories = None

try:
    # Artifacts básicos (obligatorios)
    clf = joblib.load('artifacts/model.pkl')
    vectorizer = joblib.load('artifacts/vectorizer.pkl')
    logger.info("Modelo básico cargado correctamente")
except Exception as e:
    logger.error(f"Error cargando modelo básico: {e}")

try:
    # Artifacts detallados (opcionales)
    detailed_clf = joblib.load('artifacts/detailed_model.pkl')
    detailed_vectorizer = joblib.load('artifacts/vectorizer_detailed.pkl')
    toxicity_categories = joblib.load('artifacts/toxicity_categories.pkl')
    logger.info("Modelos de clasificación detallada cargados correctamente")
except Exception as e:
    logger.warning(f"Clasificación detallada no disponible: {e}")

# Definir categorías de toxicidad
TOXICITY_CATEGORIES = [
    'obscene', 'identity_attack', 'insult', 'threat', 'asian', 'atheist', 
    'bisexual', 'black', 'buddhist', 'christian', 'female', 'heterosexual', 
    'hindu', 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability', 
    'jewish', 'latino', 'male', 'muslim', 'other_disability', 'other_gender', 
    'other_race_or_ethnicity', 'other_religion', 'other_sexual_orientation', 
    'physical_disability', 'psychiatric_or_mental_illness', 'transgender', 
    'white', 'sexual_explicit'
]

# Modelos Pydantic para validación de datos
class CommentRequest(BaseModel):
    text: str
    include_probability: bool = False
    include_detailed_classification: bool = False
    
class DetailedClassification(BaseModel):
    category: str
    probability: float
    is_present: bool

class CommentResponse(BaseModel):
    text: str
    is_toxic: bool
    label: str
    probability: Optional[float] = None
    detailed_classification: Optional[List[DetailedClassification]] = None
    timestamp: str

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
    """Endpoint raíz"""
    return {
        "message": "Toxic Comment Classifier API",
        "status": "active",
        "basic_model_loaded": clf is not None and vectorizer is not None,
        "detailed_model_loaded": detailed_clf is not None and detailed_vectorizer is not None
    }

@app.get("/health")
async def health_check():
    """Health check detallado"""
    return {
        "status": "healthy" if clf is not None and vectorizer is not None else "unhealthy",
        "basic_model_available": clf is not None,
        "basic_vectorizer_available": vectorizer is not None,
        "detailed_model_available": detailed_clf is not None,
        "detailed_vectorizer_available": detailed_vectorizer is not None,
        "toxicity_categories_available": toxicity_categories is not None,
        "timestamp": datetime.now().isoformat()
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
            detail="Modelo básico no disponible"
        )
    
    try:
        # Clasificación básica (siempre disponible)
        X_new = vectorizer.transform([request.text])
        prediction = clf.predict(X_new)[0]
        is_toxic = bool(prediction)
        label = "Tóxico" if is_toxic else "No tóxico"
        
        # Probabilidad básica
        probability = None
        if request.include_probability:
            prob_scores = clf.predict_proba(X_new)[0]
            probability = float(prob_scores[1]) if len(prob_scores) > 1 else 0.0
        
        # Clasificación detallada (solo si está disponible y es tóxico)
        detailed_classification = None
        if (request.include_detailed_classification and 
            is_toxic and 
            detailed_clf is not None and 
            detailed_vectorizer is not None and
            toxicity_categories is not None):
            
            try:
                logger.info("Iniciando clasificación detallada...")
                X_detailed = detailed_vectorizer.transform([request.text])
                detailed_predictions = detailed_clf.predict(X_detailed)[0]
                detailed_probabilities = detailed_clf.predict_proba(X_detailed)
                
                logger.info(f"Predicciones detalladas: {detailed_predictions.shape}")
                logger.info(f"Categorías disponibles: {len(toxicity_categories)}")
                
                detailed_classification = []
                for i, category in enumerate(toxicity_categories):
                    if i < len(detailed_predictions):
                        # Obtener probabilidad de clase positiva (corregido)
                        prob = 0.0
                        if i < len(detailed_probabilities):
                            # detailed_probabilities[i] es array con [prob_negative, prob_positive]
                            proba_array = detailed_probabilities[i][0]
                            if len(proba_array) > 1:
                                prob = float(proba_array[1])  # Probabilidad de clase positiva
                        
                        is_present = bool(detailed_predictions[i])
                        
                        detailed_classification.append(DetailedClassification(
                            category=category,
                            probability=prob,
                            is_present=is_present
                        ))
                        
                        if is_present:
                            logger.info(f"Categoría detectada: {category} (prob: {prob:.3f})")
                
                logger.info(f"Total categorías en respuesta: {len(detailed_classification)}")
                        
            except Exception as detail_error:
                logger.error(f"Error en clasificación detallada: {detail_error}")
                # Continuar sin clasificación detallada
        
        logger.info(f"Usuario {username} clasificó: '{request.text[:50]}...' -> {label}")
        
        return CommentResponse(
            text=request.text,
            is_toxic=is_toxic,
            label=label,
            probability=probability,
            detailed_classification=detailed_classification,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error al clasificar comentario: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor: {str(e)}"
        )

@app.post("/classify-batch")
async def classify_batch_comments(
    comments: List[str],
    include_detailed: bool = False,
    username: str = Depends(get_current_user)
):
    """Clasificar múltiples comentarios"""
    if clf is None or vectorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo básico no disponible"
        )
    
    try:
        results = []
        
        for comment in comments:
            request = CommentRequest(
                text=comment,
                include_probability=True,
                include_detailed_classification=include_detailed
            )
            
            # Reutilizar la lógica del endpoint individual
            result = await classify_comment(request, username)
            results.append(result)
        
        return {
            "results": results,
            "total_processed": len(comments),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en clasificación por lotes: {e}")
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