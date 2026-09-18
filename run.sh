#!/bin/bash
set -e

echo "=========================================="
echo "AI/ML Wine Classification Project"
echo "=========================================="

python3 -m pip install -r requirements.txt
python3 src/train.py
python3 src/evaluation.py

echo ""
echo "Project execution completed."
echo "To make a new prediction, run:"
echo "python3 src/predict.py"
