

@echo off
cd /d "C:\Users\User\OneDrive\Desktop\dr-web-feedback\backend"
start python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
timeout /t 25 /nobreak >nul
start http://127.0.0.1:8000/
pause










