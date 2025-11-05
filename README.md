<div align="center">

# 🔎 Toxic Comment Classifier

### *Clasificador Inteligente de Comentarios Tóxicos con Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.48.1-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Aplicación completa de Machine Learning para detectar y clasificar toxicidad en comentarios de texto.**

[Demo](#-demo-rápida) • [Características](#-características-principales) • [Instalación](#-instalación) • [API](#-api-rest) • [Documentación](#-estructura-del-proyecto)

</div>

---

## 📋 Descripción

**Toxic Comment Classifier** es una solución end-to-end de **procesamiento de lenguaje natural (NLP)** que identifica automáticamente contenido tóxico, ofensivo o inapropiado en comentarios escritos en inglés. El sistema combina dos niveles de análisis:

🎯 **Clasificación Binaria**: Determina si un comentario es tóxico o no tóxico  
🔬 **Análisis Multi-etiqueta**: Identifica hasta **29 categorías específicas** de toxicidad

### 💡 Casos de Uso

- **Moderación de Contenido**: Filtrado automático en redes sociales, foros y blogs
- **Análisis de Sentimiento**: Evaluación de feedback y comentarios de usuarios
- **Investigación**: Estudio de patrones de lenguaje ofensivo y comportamiento online
- **Sistemas de Alerta**: Detección temprana de amenazas o acoso

---

## ✨ Características Principales

### 🤖 Clasificación Inteligente

- **Modelo Básico**: Clasificación binaria rápida (Tóxico/No Tóxico)
- **Modelo Detallado**: Análisis multi-etiqueta con 29 categorías:
  - `obscene`, `insult`, `threat`, `identity_attack`
  - Identidades específicas: `racial`, `religious`, `gender`, `sexual_orientation`
  - `sexual_explicit` y muchas más

### 🖥️ Interfaz Web Interactiva (Streamlit)

- **UI Amigable**: Interfaz limpia y responsive
- **Análisis en Tiempo Real**: Resultados instantáneos con métricas de confianza
- **Visualización Avanzada**: 
  - Probabilidades de clasificación
  - Categorías detectadas con niveles de confianza
  - Indicadores visuales de alerta

### 🚀 API REST (FastAPI)

- **Endpoints RESTful**: Integración fácil con otros sistemas
- **Autenticación**: HTTP Basic Auth con rate limiting
- **Documentación Automática**: Swagger UI en `/docs`
- **Batch Processing**: Clasificación de múltiples comentarios

### 📊 Pipeline ML Completo

- **Notebooks Jupyter**: Exploración de datos, entrenamiento y evaluación
- **Modelos Persistidos**: Artifacts listos para producción
- **Reproducibilidad**: Scripts de smoke test y validación

---

## 🚀 Demo Rápida

### Opción 1: Streamlit App

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/toxic-comment-classifier.git
cd toxic-comment-classifier

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app/app.py
```

Abre `http://localhost:8501` en tu navegador y ¡empieza a clasificar!

### Opción 2: Docker

```bash
# Construir imagen
docker build -t toxic-classifier .

# Ejecutar contenedor
docker run -p 8501:8501 toxic-classifier
```

### Opción 3: API REST

```bash
# Ejecutar API
python run_api.py

# Acceder a documentación
# http://localhost:8000/docs
```

---

## 💻 Instalación

### Requisitos Previos

- Python 3.11+
- pip
- (Opcional) Docker

### Paso a Paso

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/toxic-comment-classifier.git
cd toxic-comment-classifier
```

2. **Crear entorno virtual**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Verificar instalación**
```bash
python scripts/smoke_test.py
```

---

## 🎮 Uso

### Interfaz Web (Streamlit)

```bash
streamlit run app/app.py
```

**Funcionalidades:**
- ✍️ Escribe o pega cualquier comentario
- 🔍 Click en "Clasificar Comentario"
- 📊 Visualiza resultados con probabilidades
- 🔬 Activa "Análisis Detallado" para comentarios tóxicos

### API REST (FastAPI)

**Iniciar servidor:**
```bash
python run_api.py
# API disponible en http://localhost:8000
```

**Ejemplo de uso:**

```python
import requests
from requests.auth import HTTPBasicAuth

url = "http://localhost:8000/classify"
auth = HTTPBasicAuth("admin", "secret123")

response = requests.post(
    url,
    json={
        "text": "You are an idiot!",
        "include_probability": True,
        "include_detailed_classification": True
    },
    auth=auth
)

print(response.json())
```

**Endpoints disponibles:**
- `POST /classify` - Clasificar comentario individual
- `POST /classify-batch` - Clasificación por lotes
- `GET /health` - Estado del sistema
- `GET /stats` - Estadísticas de uso
- `GET /docs` - Documentación interactiva (Swagger)

### Uso Programático

```python
import joblib

# Cargar modelos
vectorizer = joblib.load('artifacts/vectorizer.pkl')
clf = joblib.load('artifacts/model.pkl')

# Clasificar
comment = "This is a great article!"
X = vectorizer.transform([comment])
prediction = clf.predict(X)[0]
probability = clf.predict_proba(X)[0]

print(f"Toxic: {bool(prediction)}")
print(f"Confidence: {probability[1]:.2%}")
```

---

## 🏗️ Estructura del Proyecto

```
toxic-comment-classifier/
│
├── 📁 app/                          # Aplicación Streamlit
│   └── app.py                       # Interfaz web principal
│
├── 📁 api/                          # API REST
│   ├── __init__.py
│   └── main.py                      # Endpoints FastAPI
│
├── 📁 artifacts/                    # Modelos entrenados
│   ├── model.pkl                    # Modelo básico
│   ├── vectorizer.pkl               # TF-IDF vectorizador
│   ├── detailed_model.pkl           # Modelo multi-etiqueta
│   ├── vectorizer_detailed.pkl      # Vectorizador detallado
│   └── toxicity_categories.pkl      # Lista de categorías
│
├── 📁 notebooks/                    # Análisis y entrenamiento
│   ├── 01_exploracion_dataset.ipynb     # EDA
│   ├── 02_entrenamiento_modelo.ipynb    # Modelo básico
│   ├── 03_guardar_modelo.ipynb          # Persistencia
│   └── 04_modelo_clasificacion_detallada.ipynb  # Modelo multi-etiqueta
│
├── 📁 data/                         # Datasets (local)
│   └── data.csv
│
├── 📁 scripts/                      # Utilidades
│   └── smoke_test.py                # Tests de validación
│
├── 📄 requirements.txt              # Dependencias Python
├── 📄 Dockerfile                    # Configuración Docker
├── 📄 run_api.py                    # Script para ejecutar API
└── 📄 README.md                     # Este archivo
```

---

## 🛠️ Stack Tecnológico

### Machine Learning & Data Science
- **scikit-learn** `1.7.1` - Modelos de clasificación (Logistic Regression)
- **pandas** `2.3.2` - Manipulación de datos
- **numpy** `2.3.2` - Operaciones numéricas
- **joblib** `1.5.1` - Serialización de modelos

### Frameworks Web
- **Streamlit** `1.48.1` - Interfaz web interactiva
- **FastAPI** `0.116.1` - API REST de alto rendimiento
- **Uvicorn** `0.35.0` - Servidor ASGI

### Visualización
- **matplotlib** `3.10.5` - Gráficos estáticos
- **Altair** `5.5.0` - Visualizaciones interactivas

### Otros
- **PyJWT** `2.8.0` - Autenticación JWT
- **cryptography** `41.0.7` - Seguridad
- **Docker** - Containerización

---

## 📊 Rendimiento del Modelo

### Modelo Básico (Binario)
- **Algoritmo**: Logistic Regression
- **Vectorización**: TF-IDF (10,000 features)
- **Accuracy**: ~92%
- **Vocabulario**: 10,000 palabras

### Modelo Detallado (Multi-etiqueta)
- **Algoritmo**: MultiOutputClassifier + Logistic Regression
- **Vectorización**: TF-IDF extendido
- **Categorías**: 29 tipos de toxicidad
- **Vocabulario**: 10,000+ palabras

---

## 🔐 Seguridad

### API
- Autenticación HTTP Basic (usuario/contraseña)
- Rate Limiting: 100 requests/minuto por usuario
- Validación de entrada con Pydantic

### Producción
> ⚠️ **Nota**: Este proyecto es educativo/demostrativo. Para producción:
> - Implementar autenticación robusta (OAuth2, JWT)
> - Usar base de datos para credenciales
> - Configurar CORS específico
> - Implementar rate limiting con Redis
> - Agregar logging y monitoreo
> - Usar HTTPS

---

## 🧪 Testing

### Smoke Test
```bash
python scripts/smoke_test.py
```

Verifica:
- ✅ Carga correcta de todos los artifacts
- ✅ Funcionalidad de predicción básica
- ✅ Funcionalidad de predicción detallada
- ✅ Integridad del vocabulario

---

## 📚 Notebooks

### 1. Exploración del Dataset
`notebooks/01_exploracion_dataset.ipynb`
- Análisis exploratorio de datos (EDA)
- Distribución de clases
- Análisis de palabras frecuentes

### 2. Entrenamiento Modelo Básico
`notebooks/02_entrenamiento_modelo.ipynb`
- Preprocesamiento de texto
- Vectorización TF-IDF
- Entrenamiento Logistic Regression
- Evaluación de métricas

### 3. Guardar Modelos
`notebooks/03_guardar_modelo.ipynb`
- Serialización con joblib
- Validación de artifacts

### 4. Clasificación Detallada
`notebooks/04_modelo_clasificacion_detallada.ipynb`
- Modelo multi-etiqueta
- 29 categorías de toxicidad
- Análisis granular

---

## 🚢 Despliegue

### Local
```bash
streamlit run app/app.py
```

### Docker
```bash
docker build -t toxic-classifier .
docker run -p 8501:8501 toxic-classifier
```

### Streamlit Cloud
1. Sube el repo a GitHub
2. Conecta con [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy automático

### Heroku / Railway / Render
Compatible con cualquier plataforma que soporte Docker

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas! 

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 👨‍💻 Autor

**Daniel Moreno**

- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- LinkedIn: [Tu Perfil](https://linkedin.com/in/tu-perfil)

---

## 🙏 Agradecimientos

- Dataset basado en [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- Inspirado en investigaciones de NLP y moderación de contenido
- Comunidad de scikit-learn y Streamlit

---
