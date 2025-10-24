# Quick Reference Card

## 🚀 Common Commands

### CLI Commands
```bash
# Full pipeline
python pipeline_cli.py run-all --input-folder ~/audio_files

# Individual stages
python pipeline_cli.py transcribe --input-folder ~/audio_files
python pipeline_cli.py detect-criteria
python pipeline_cli.py llm-process

# Status & validation
python pipeline_cli.py status
python pipeline_cli.py validate-config

# Custom config
python pipeline_cli.py run-all --config my_config.yaml --input-folder ~/audio
```

### Python API
```python
from core.pipeline import AudioProcessingPipeline

# Quick run
pipeline = AudioProcessingPipeline()
result = pipeline.run_full_pipeline(audio_folder="~/audio")

# Check results
print(f"Duration: {result.get_duration():.2f}s")
print(f"Success: {len(result.transcription_stats['successful'])}")
```

## 📁 Where Is Everything?

### Current Structure (Quick Lookup)
```
configs/pipeline_config.yaml          # Main configuration
core/service/                         # Service layer (NEW)
core/pipeline/                        # Pipeline orchestrator (NEW)
core/audio_to_text/diarization_utils.py  # Diarization helpers (NEW)
pipeline_cli.py                       # CLI interface (NEW)
```

### Recommended Structure (Future)
```
config/default.yaml                   # Configuration
src/domain/models/                    # Business entities
src/domain/services/                  # Business logic
src/infrastructure/database/          # Data access
src/infrastructure/ml/                # ML code
src/application/pipeline/             # Orchestration
src/interface/cli/                    # CLI
```

## 🔧 Configuration Quick Edits

```yaml
# config/pipeline_config.yaml

# Change parallel workers
pipeline:
  max_workers: 5  # <- Edit this

# Change input folder
pipeline:
  default_input_folder: "~/your/path"  # <- Edit this

# Skip problematic dialogs
llm_processing:
  skip_indices: [380, 1313, 1275]  # <- Add indices

# Adjust retry behavior
llm_processing:
  max_retries: 3  # <- Edit this
  retry_delay_seconds: 5  # <- Edit this
```

## 🐛 Common Issues & Quick Fixes

### Issue: No input folder specified
```yaml
# Add to config/pipeline_config.yaml
pipeline:
  default_input_folder: "~/Documents/Аудио Бринекс/2/"
```

### Issue: Out of memory
```yaml
# Reduce workers in config
pipeline:
  max_workers: 2  # Reduce from 5
```

### Issue: LLM fails on specific dialogs
```yaml
# Add to skip list in config
llm_processing:
  skip_indices: [380, 1313, 1275, YOUR_ID]
```

### Issue: Files not reprocessing
```yaml
# In config
transcription:
  skip_processed: false  # Force reprocess
```

## 📊 Status Checks

### Check pipeline status
```bash
python pipeline_cli.py status
```

### Check from Python
```python
from core.repository.audio_dialog_repository import AudioDialogRepository

repo = AudioDialogRepository()
dialogs = repo.find_all()
print(f"Total: {len(dialogs)}")
print(f"Processed: {sum(1 for d in dialogs if d.status.value == 'PROCESSED')}")
```

## 📝 File Naming Convention

### Services (Business Logic)
```
*_service.py          # TranscriptionService, CriteriaDetectionService
```

### Repositories (Data Access)
```
*_repository.py       # AudioDialogRepository, DialogRowRepository
```

### Models (Domain Entities)
```
audio_dialog.py       # AudioDialog model
dialog_row.py         # DialogRow model
```

### Detectors (Pattern Matching)
```
*_detector.py         # SwearDetector, SalesDetector
```

### Utilities
```
*_utils.py            # diarization_utils, text_utils
```

## 🎯 Quick Migration Steps

### 1. Preview Migration (Safe)
```bash
python migrate_structure.py --dry-run
```

### 2. Execute Migration
```bash
python migrate_structure.py
```

### 3. Update One Import Example
```python
# Old
from core.repository.entity.audio_dialog import AudioDialog

# New
from src.domain.models.audio_dialog import AudioDialog
```

## 🧪 Testing Quick Guide

### Run All Tests (Future)
```bash
pytest tests/
```

### Run Specific Tests
```bash
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests
pytest tests/e2e/                     # End-to-end tests
```

### Run with Coverage
```bash
pytest --cov=src tests/
```

## 📚 Documentation Quick Links

| Document | Purpose |
|----------|---------|
| `README_REFACTORING.md` | Overview of refactoring |
| `QUICKSTART.md` | Get started in 5 minutes |
| `REFACTORING_GUIDE.md` | Detailed architecture |
| `REFACTORING_SUMMARY.md` | Executive summary |
| `RECOMMENDED_STRUCTURE.md` | Ideal file structure |
| `STRUCTURE_COMPARISON.md` | Before/after comparison |
| `QUICK_REFERENCE.md` | This file |
| `example_usage.py` | Code examples |

## 💡 Pro Tips

### Tip 1: Use Config Overrides
```bash
# Development
python pipeline_cli.py run-all --config config/development.yaml

# Production
python pipeline_cli.py run-all --config config/production.yaml
```

### Tip 2: Monitor Progress
```bash
# Terminal 1: Run pipeline
python pipeline_cli.py run-all --input-folder ~/audio

# Terminal 2: Watch status
watch -n 30 "python pipeline_cli.py status"
```

### Tip 3: Batch Processing
```python
from pathlib import Path
from core.service.transcription_service import TranscriptionService

service = TranscriptionService()
files = list(Path("~/audio").glob("*.mp3"))

# Process in batches
batch_size = 10
for i in range(0, len(files), batch_size):
    batch = files[i:i+batch_size]
    service.process_multiple_files(batch)
```

### Tip 4: Error Log Analysis
```python
import json

# Read error log
with open('logs/errors.json', 'r') as f:
    errors = json.load(f)

# Find common error patterns
error_types = {}
for error in errors:
    stage = error['stage']
    error_types[stage] = error_types.get(stage, 0) + 1

print(error_types)  # See which stage fails most
```

## 🔍 Where to Find Specific Code

| What | Where (Current) | Where (Recommended) |
|------|-----------------|---------------------|
| Audio models | `core/repository/entity/` | `src/domain/models/` |
| Transcription | `core/audio_to_text/` | `src/infrastructure/ml/audio_to_text/` |
| Detectors | `core/post_processors/text_processing/detector/` | `src/infrastructure/ml/nlp/detectors/` |
| Repositories | `core/repository/` | `src/infrastructure/database/repositories/` |
| Business logic | `core/service/` | `src/domain/services/` |
| Pipeline | `core/pipeline/` | `src/application/pipeline/` |
| CLI | `pipeline_cli.py` | `src/interface/cli/` |
| Config | `configs/` | `config/` |
| Patterns | `core/post_processors/config/` | `config/criteria/patterns/` |

## 🎨 Code Style Quick Reference

### Imports Order
```python
# 1. Standard library
import os
import sys
from pathlib import Path

# 2. Third-party
import pandas as pd
import numpy as np

# 3. Local application
from src.domain.models import AudioDialog
from src.infrastructure.database.repositories import AudioDialogRepository
```

### Type Hints
```python
from typing import List, Optional, Dict
from pathlib import Path
import uuid

def process_files(
    files: List[Path],
    config: Optional[Dict] = None
) -> uuid.UUID:
    """Process audio files."""
    pass
```

### Docstrings
```python
def transcribe_audio(audio_path: Path) -> TranscriptionResult:
    """
    Transcribe audio file to text.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        TranscriptionResult with text and metadata
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        TranscriptionError: If transcription fails
    """
    pass
```

## 🚦 Service Status Indicators

### Check Service Health
```python
from core.service.transcription_service import TranscriptionService
from core.service.llm_processing_service import LLMProcessingService

# Transcription
trans_service = TranscriptionService()
# Check if initialized

# LLM
llm_service = LLMProcessingService()
stats = llm_service.get_processing_stats()
print(f"LLM ready: {stats['total_dialogs']} dialogs")
```

## 📞 Getting Help

### Check Logs
```bash
tail -f logs/pipeline.log              # Main log
tail -f logs/transcription.log         # Transcription log
tail -f logs/llm_processing.log        # LLM log
tail -f logs/errors.log                # Errors only
```

### Validate Setup
```bash
python pipeline_cli.py validate-config
```

### Debug Mode
```yaml
# config/pipeline_config.yaml
logging:
  level: DEBUG  # Change from INFO to DEBUG
```

## ⚡ Performance Tuning

### For Speed
```yaml
pipeline:
  max_workers: 10  # More parallel processing
  
transcription:
  omp_num_threads: 2
  mkl_num_threads: 2
```

### For Stability
```yaml
pipeline:
  max_workers: 2  # Less parallel load
  
llm_processing:
  max_retries: 5  # More attempts
  retry_delay_seconds: 10  # Longer delay

error_handling:
  continue_on_error: true  # Don't stop on failures
```

### For Memory
```yaml
pipeline:
  max_workers: 2  # Reduce concurrent files
  max_files_per_run: 100  # Process in smaller batches
```

## 🎯 Decision Tree

```
Need to process audio files?
├─ Yes → python pipeline_cli.py transcribe --input-folder ~/audio
└─ No
    └─ Need to analyze existing transcripts?
        ├─ Yes → python pipeline_cli.py detect-criteria
        └─ No
            └─ Need LLM analysis?
                ├─ Yes → python pipeline_cli.py llm-process
                └─ No → python pipeline_cli.py status
```

## 📋 Checklist: Before Running

- [ ] Config file edited (`config/pipeline_config.yaml`)
- [ ] Input folder exists and has audio files
- [ ] Database is running
- [ ] Ollama is running (for LLM stage)
- [ ] Enough disk space (for models and output)
- [ ] Validated config (`python pipeline_cli.py validate-config`)

## 🎓 Learning Path

1. **Day 1:** Read `QUICKSTART.md`, run status command
2. **Day 2:** Process small batch, review `REFACTORING_GUIDE.md`
3. **Day 3:** Understand services, read code in `core/service/`
4. **Week 2:** Review `RECOMMENDED_STRUCTURE.md`, plan migration
5. **Week 3-4:** Execute migration, write tests

---

**💡 Remember:** Keep this file handy for quick reference during development!

