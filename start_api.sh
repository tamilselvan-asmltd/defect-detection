#!/bin/bash
echo "Starting Flask API in background..."
nohup python api.py > api.log 2>&1 &
echo "Flask API started. Check api.log for output."