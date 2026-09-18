@echo off
echo ==========================================
echo AI/ML Wine Classification Project
echo ==========================================

python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

python src/train.py
if errorlevel 1 exit /b 1

python src/evaluation.py
if errorlevel 1 exit /b 1

echo.
echo Project execution completed.
echo To make a new prediction, run:
echo python src/predict.py
pause
