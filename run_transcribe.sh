#!/bin/bash
cd "/Users/kian/Developer/transcribe"
source .venv/bin/activate
# Load API key from a .env file if it exists
if [ -f .env ]; then
    export $(cat .env | xargs)
fi
python3 transcribe.py
