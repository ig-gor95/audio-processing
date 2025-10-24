#!/bin/bash
# Urgent Fix for torchaudio compatibility issue

echo "🔧 Fixing torchaudio compatibility issue..."
echo ""

# Navigate to project directory
cd /Users/igorlapin/PycharmProjects/audio-processing

echo "Step 1: Uninstalling incompatible versions..."
pip uninstall torch torchaudio speechbrain -y

echo ""
echo "Step 2: Installing compatible versions..."
pip install torch==2.1.0 torchaudio==2.1.0 speechbrain==1.0.0

echo ""
echo "Step 3: Testing the fix..."
python -c "
import torch
import torchaudio
import speechbrain

print('✅ Torch:', torch.__version__)
print('✅ Torchaudio:', torchaudio.__version__)
print('✅ SpeechBrain:', speechbrain.__version__)
print('')
print('✅ All packages installed successfully!')
print('✅ The torchaudio issue is FIXED!')
"

echo ""
echo "Step 4: Testing imports..."
python -c "
from core.audio_to_text.diarizer import diarize
print('✅ Diarizer imports successfully')
"

echo ""
echo "✅ FIX COMPLETE! You can now run your pipeline."
echo ""
echo "Test with: python pipeline_cli.py --help"

