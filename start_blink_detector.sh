#!/bin/bash
echo "🚀 Starting AI Blink Detector Setup..."

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install required packages
echo "📦 Installing/Updating dependencies..."
pip install --upgrade pip
pip install flask opencv-python mediapipe numpy scipy

echo "✨ System ready. Launching application..."
# Run the demo app
python blink_app.py
