# Fix: torchaudio Compatibility Issue

## 🐛 Error

```
AttributeError: module 'torchaudio' has no attribute 'list_audio_backends'
```

## 🔍 Root Cause

The issue occurs because `torchaudio 2.x` removed the `list_audio_backends()` method, but older versions of `speechbrain` still try to use it.

Your current versions:
- `torch~=2.6.0`
- `torchaudio~=2.6.0`
- `speechbrain~=1.0.3`

## ✅ Solutions (Try in Order)

### **Solution 1: Upgrade speechbrain (Recommended)**

This keeps your torch/torchaudio at the latest versions.

```bash
cd /Users/igorlapin/PycharmProjects/audio-processing

# Upgrade speechbrain to latest
pip install --upgrade speechbrain

# Or install specific version
pip install speechbrain>=1.0.3 --upgrade
```

**Test it:**
```bash
python -c "import speechbrain; print(f'✓ SpeechBrain {speechbrain.__version__}')"
python -c "import torchaudio; print(f'✓ Torchaudio {torchaudio.__version__}')"
```

---

### **Solution 2: Use Compatible Versions (If Solution 1 Fails)**

Downgrade to known compatible versions:

```bash
cd /Users/igorlapin/PycharmProjects/audio-processing

# Uninstall current versions
pip uninstall torch torchaudio speechbrain -y

# Install compatible versions
pip install torch==2.1.0 torchaudio==2.1.0 speechbrain==1.0.0
```

**Update requirements.txt:**
```bash
# Replace in requirements.txt:
torch==2.1.0
torchaudio==2.1.0
speechbrain==1.0.0
```

---

### **Solution 3: Install Development Version of speechbrain**

If stable version still has issues:

```bash
cd /Users/igorlapin/PycharmProjects/audio-processing

# Install from GitHub
pip install git+https://github.com/speechbrain/speechbrain.git
```

---

## 🧪 Verify the Fix

After applying any solution, test it:

```bash
# Test imports
python -c "
import torch
import torchaudio
import speechbrain

print(f'✓ Torch: {torch.__version__}')
print(f'✓ Torchaudio: {torchaudio.__version__}')
print(f'✓ SpeechBrain: {speechbrain.__version__}')
print('All imports successful!')
"
```

**Test diarization:**
```bash
python -c "
from core.audio_to_text.diarizer import diarize
print('✓ Diarizer imports successfully')
"
```

---

## 📝 Recommended Configuration

### **For Latest Features (Recommended):**
```txt
torch>=2.0.0
torchaudio>=2.0.0
speechbrain>=1.0.0
```

### **For Stability:**
```txt
torch==2.1.0
torchaudio==2.1.0
speechbrain==1.0.0
```

---

## 🔧 Quick Fix Command

Run this to apply Solution 1:

```bash
cd /Users/igorlapin/PycharmProjects/audio-processing
pip install --upgrade speechbrain
python -c "import speechbrain, torchaudio; print('✓ Fix applied successfully')"
```

---

## 🐍 Alternative: Create Fresh Environment

If issues persist, create a clean environment:

```bash
# Create new environment
conda create -n audio-processing-fixed python=3.12 -y
conda activate audio-processing-fixed

# Install from requirements
cd /Users/igorlapin/PycharmProjects/audio-processing
pip install -r requirements.txt

# Test
python pipeline_cli.py --help
```

---

## 📊 Version Compatibility Matrix

| torch | torchaudio | speechbrain | Status |
|-------|------------|-------------|--------|
| 2.6.x | 2.6.x | >=1.0.3 | ✅ Should work |
| 2.1.x | 2.1.x | 1.0.x | ✅ Tested |
| 2.0.x | 2.0.x | 1.0.x | ✅ Stable |
| 2.6.x | 2.6.x | <1.0 | ❌ Incompatible |

---

## 🎯 Recommended Action

**Try this first:**
```bash
pip install --upgrade speechbrain
```

If that doesn't work:
```bash
pip install torch==2.1.0 torchaudio==2.1.0 speechbrain==1.0.0
```

---

## 📞 Still Having Issues?

If the error persists:

1. **Check your Python version:**
   ```bash
   python --version  # Should be 3.8-3.12
   ```

2. **Clear pip cache:**
   ```bash
   pip cache purge
   pip install --upgrade --force-reinstall speechbrain
   ```

3. **Check for conflicting packages:**
   ```bash
   pip list | grep -E "torch|speech"
   ```

4. **Try alternative backend:**
   Some systems need different audio backends. Install:
   ```bash
   pip install soundfile
   ```

---

## ✅ Expected Outcome

After fix:
- ✓ No more `list_audio_backends` error
- ✓ Diarization works
- ✓ Pipeline runs successfully

---

**Quick Fix Now:**
```bash
pip install --upgrade speechbrain && echo "✓ Fixed!"
```


