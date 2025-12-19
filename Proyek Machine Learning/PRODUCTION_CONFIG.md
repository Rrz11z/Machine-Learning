# Production Configuration Guide

## 🌍 Environment Configuration

### Local Development (.env)
```env
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True
DEBUG=True
```

### Production Environment
```env
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=False
DEBUG=False
WORKERS=4
```

---

## 🔐 Security Configuration

### CORS Settings (main.py)
```python
CORSMiddleware(
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**For Production:**
```python
CORSMiddleware(
    allow_origins=["https://yourdomain.com", "https://app.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Recommended Headers
```python
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

---

## 📦 Deployment Configurations

### Gunicorn (Production WSGI Server)
```bash
# Install
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:8000 main:app

# With additional settings
gunicorn -w 4 -b 0.0.0.0:8000 \
    --timeout 60 \
    --access-logfile - \
    --error-logfile - \
    main:app
```

### Nginx Configuration (Reverse Proxy)
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /path/to/cert.crt;
    ssl_certificate_key /path/to/key.key;
    
    client_max_body_size 20M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Systemd Service (Linux/Ubuntu)
```ini
[Unit]
Description=Drug Effectiveness Predictor
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/drug-predictor
Environment="PATH=/var/www/drug-predictor/.venv/bin"
ExecStart=/var/www/drug-predictor/.venv/bin/gunicorn -w 4 -b 127.0.0.1:8000 main:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Supervisor Configuration
```ini
[program:drug-predictor]
command=/var/www/drug-predictor/.venv/bin/gunicorn -w 4 -b 127.0.0.1:8000 main:app
directory=/var/www/drug-predictor
user=www-data
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/drug-predictor.log
```

---

## 🐳 Docker Advanced Configuration

### Multi-stage Dockerfile (Optimized)
```dockerfile
# Build stage
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s CMD python -c "import requests; requests.get('http://localhost:8000/health')"
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "main:app"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: drug-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: drug-predictor
  template:
    metadata:
      labels:
        app: drug-predictor
    spec:
      containers:
      - name: drug-predictor
        image: your-registry/drug-predictor:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_PORT
          value: "8000"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: drug-predictor-service
spec:
  selector:
    app: drug-predictor
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 📊 Monitoring & Logging

### Logging Configuration
```python
import logging
from pythonjsonlogger import jsonlogger

# JSON logging for production
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)
```

### Application Monitoring (Prometheus)
```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
request_count = Counter('requests_total', 'Total requests')
prediction_latency = Histogram('prediction_latency', 'Prediction latency')

@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type='text/plain')
```

### Error Tracking (Sentry)
```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FastApiIntegration()]
)
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions (.github/workflows/deploy.yml)
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: docker build -t drug-predictor:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push drug-predictor:${{ github.sha }}
    
    - name: Deploy to server
      run: |
        ssh user@server "cd /app && docker-compose pull && docker-compose up -d"
```

---

## 📈 Performance Tuning

### Database Connection Pooling
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:password@localhost/db',
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40
)
```

### Caching Strategy
```python
from fastapi_cache2 import FastAPICache2
from fastapi_cache2.backends.redis import RedisBackend
from aioredis import create_redis_pool

@app.on_event("startup")
async def startup():
    redis = await create_redis_pool('redis://localhost')
    FastAPICache2.init(RedisBackend(redis), prefix="fastapi-cache")
```

### Response Compression
```python
from fastapi.middleware.gzip import GZIPMiddleware

app.add_middleware(GZIPMiddleware, minimum_size=1000)
```

---

## 🛡️ Backup & Recovery

### Database Backup
```bash
# Daily backup script (backup.sh)
#!/bin/bash
BACKUP_DIR="/backups/drug-predictor"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup models
cp -r models/ $BACKUP_DIR/models_$DATE/

# Backup database
pg_dump database_name > $BACKUP_DIR/db_$DATE.sql

# Compress
tar -czf $BACKUP_DIR/backup_$DATE.tar.gz $BACKUP_DIR/

# Upload to cloud (S3, GCS, etc.)
aws s3 cp $BACKUP_DIR/backup_$DATE.tar.gz s3://your-bucket/backups/
```

---

**Last Updated**: December 2025  
**Status**: Production Ready
