#!/bin/bash
cd "$(dirname "$0")"
source /root/piper/piper_env/bin/activate 2>/dev/null || true
python -m uvicorn app:app --host 0.0.0.0 --port 8000
