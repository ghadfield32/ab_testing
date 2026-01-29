#!/usr/bin/env python3
"""
Quick dataset download script that auto-loads credentials from .env
"""
import os
from pathlib import Path

# Load environment variables from .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    print(f"Loading credentials from: {env_file}")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    print(f"✅ KAGGLE_API_TOKEN loaded")
else:
    print(f"⚠️  No .env file found at: {env_file}")
    print("Please set KAGGLE_API_TOKEN environment variable manually")

# Run the setup script
import subprocess
result = subprocess.run(
    ["python", "setup_datasets.py", "--download"],
    cwd=Path(__file__).parent
)
exit(result.returncode)
