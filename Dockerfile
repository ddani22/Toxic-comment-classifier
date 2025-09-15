FROM python:3.11-slim

# Do not write .pyc files and ensure stdout/stderr are not buffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements and install if present
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
    && if [ -s requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy project source (includes artifacts/ with model files)
COPY . .

## Create a non-root user and set ownership of the app directory
# Use UID/GID 1000 to better match common host users (harmless if already taken)
RUN groupadd -g 1000 appuser || true \
    && useradd -m -u 1000 -g appuser -s /bin/bash appuser || true \
    && chown -R appuser:appuser /app

ENV HOME=/home/appuser

# Expose Streamlit default port
EXPOSE 8501

# Switch to the non-root user for running the app
USER appuser

# Run the Streamlit app (bind to 0.0.0.0 so it is accessible from outside)
CMD ["streamlit", "run", "app/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
