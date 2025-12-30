#!/bin/bash

# SMS Spam Classifier ML Service Startup Script

echo "Starting SMS Spam Classifier ML Service..."

# Python 가상환경 확인
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# 가상환경 활성화
source venv/bin/activate

# 의존성 설치
echo "Installing dependencies..."
pip install -r ml_requirements.txt

# 서버 시작
echo "Starting FastAPI server on port 8000..."
python ml_service.py
