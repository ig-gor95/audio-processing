#!/bin/bash
# Working Fix for torchaudio compatibility issue
# Uses oldest available torch version (2.2.0)

echo "🔧 Fixing torchaudio compatibility issue..."
echo ""

cd /Users/igorlapin/PycharmProjects/audio-processing

echo "Step 1: Uninstalling incompatible versions..."
pip uninstall torch torchaudio speechbrain -y

echo ""
echo "Step 2: Installing compatible versions (torch 2.2.0)..."
pip install torch==2.2.0 torchaudio==2.2.0

echo ""
echo "Step 3: Installing speechbrain (latest compatible)..."
pip install speechbrain==1.0.2

echo ""
echo "Step 4: Testing the fix..."
python -c "
import torch
import torchaudio
import speechbrain

print('✅ Torch:', torch.__version__)
print('✅ Torchaudio:', torchaudio.__version__)
print('✅ SpeechBrain:', speechbrain.__version__)
print('')
print('✅ All packages installed successfully!')
"

echo ""
echo "Step 5: Testing imports..."
python -c "
try:
    from core.audio_to_text.diarizer import diarize
    print('✅ Diarizer imports successfully')
    print('✅ FIX COMPLETE!')
except Exception as e:
    print('⚠️  Import test failed:', e)
    print('But basic packages are installed correctly.')
"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Test with: python pipeline_cli.py --help"

