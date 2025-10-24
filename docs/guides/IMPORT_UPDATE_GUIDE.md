# Import Update Guide

## ⚠️ IMPORTANT: Imports Need Updating

The project structure has been migrated! **Files have been copied** (not moved) to the new structure, so the old files still exist and work. However, to use the new structure, you need to update imports.

## Strategy: Two Approaches

### Approach 1: Use Old Structure (Works Now)
Keep using the old files - nothing needs to change! The old structure in `core/` still works.

### Approach 2: Use New Structure (Recommended)
Update imports to use the new `src/` structure. This guide shows you how.

## Import Mapping Table

### Domain Models

| Old Import | New Import |
|------------|------------|
| `from core.repository.entity.audio_dialog import AudioDialog` | `from src.domain.models.audio_dialog import AudioDialog` |
| `from core.repository.entity.dialog_rows import DialogRow` | `from src.domain.models.dialog_row import DialogRow` |
| `from core.repository.entity.dialog_criteria import DialogCriteria` | `from src.domain.models.dialog_criteria import DialogCriteria` |

### Domain Services

| Old Import | New Import |
|------------|------------|
| `from core.service.transcription_service import TranscriptionService` | `from src.domain.services.transcription_service import TranscriptionService` |
| `from core.service.criteria_detection_service import CriteriaDetectionService` | `from src.domain.services.criteria_detection_service import CriteriaDetectionService` |
| `from core.service.llm_processing_service import LLMProcessingService` | `from src.domain.services.llm_processing_service import LLMProcessingService` |

### Repositories

| Old Import | New Import |
|------------|------------|
| `from core.repository.audio_dialog_repository import AudioDialogRepository` | `from src.infrastructure.database.repositories.audio_dialog_repository import AudioDialogRepository` |
| `from core.repository.dialog_criteria_repository import DialogCriteriaRepository` | `from src.infrastructure.database.repositories.dialog_criteria_repository import DialogCriteriaRepository` |
| `from core.repository.dialog_rows_repository import DialogRowRepository` | `from src.infrastructure.database.repositories.dialog_row_repository import DialogRowRepository` |

### ML / Audio to Text

| Old Import | New Import |
|------------|------------|
| `from core.audio_to_text.audio_to_text_processor import audio_to_text_processor` | `from src.infrastructure.ml.audio_to_text.processor import audio_to_text_processor` |
| `from core.audio_to_text.transcriber import transcribe` | `from src.infrastructure.ml.audio_to_text.transcriber import transcribe` |
| `from core.audio_to_text.diarizer import diarize` | `from src.infrastructure.ml.audio_to_text.diarizer import diarize` |
| `from core.audio_to_text.text_to_speaker_resolver import unite_results` | `from src.infrastructure.ml.audio_to_text.speaker_resolver import unite_results` |
| `from core.audio_to_text.diarization_utils import *` | `from src.infrastructure.ml.audio_to_text.diarization_utils import *` |

### ML / NLP

| Old Import | New Import |
|------------|------------|
| `from core.post_processors.text_processing.DialogueAnalyzerPandas import DialogueAnalyzerPandas` | `from src.infrastructure.ml.nlp.dialogue_analyzer import DialogueAnalyzerPandas` |
| `from core.post_processors.text_processing.criteria_utils import *` | `from src.infrastructure.ml.nlp.criteria_utils import *` |
| `from core.post_processors.llm_processing.objections_resolver import resolve_llm_data` | `from src.infrastructure.ml.nlp.llm_analyzer import resolve_llm_data` |

### Detectors

| Old Import | New Import |
|------------|------------|
| `from core.post_processors.text_processing.detector.sales_detector import SalesDetector` | `from src.infrastructure.ml.nlp.detectors.sales_detector import SalesDetector` |
| `from core.post_processors.text_processing.detector.swear_detector import SwearDetector` | `from src.infrastructure.ml.nlp.detectors.swear_detector import SwearDetector` |
| `from core.post_processors.text_processing.detector.*_detector import *` | `from src.infrastructure.ml.nlp.detectors.*_detector import *` |

### Audio Infrastructure

| Old Import | New Import |
|------------|----------||
| `from core.audio_loader import *` | `from src.infrastructure.audio.loader import *` |
| `from core.post_processors.audio_processing.loudness_analyzer import *` | `from src.infrastructure.audio.loudness_analyzer import *` |

### External Services

| Old Import | New Import |
|------------|------------|
| `from lib.saiga import SaigaClient` | `from src.infrastructure.external.ollama_client import SaigaClient` |

### Pipeline & Application

| Old Import | New Import |
|------------|------------|
| `from core.pipeline.audio_processing_pipeline import AudioProcessingPipeline` | `from src.application.pipeline.executor import AudioProcessingPipeline` |
| `from core.pipeline import *` | `from src.application.pipeline import *` |

### DTOs

| Old Import | New Import |
|------------|------------|
| `from core.dto.audio_to_text_result import ProcessingResults` | `from src.application.dto.transcription_result import ProcessingResults` |
| `from core.dto.criteria import CriteriaConfig` | `from src.application.dto.criteria import CriteriaConfig` |
| `from core.dto.diarisation_result import DiarisationResult` | `from src.application.dto.diarization_result import DiarisationResult` |

### Shared Utilities

| Old Import | New Import |
|------------|------------|
| `from lib.yaml_reader import ConfigLoader` | `from src.shared.config_loader import ConfigLoader` |
| `from lib.log_utils import setup_logger` | `from src.shared.logger import setup_logger` |
| `from lib.json_util import *` | `from src.shared.utils.json_utils import *` |
| `from yaml_reader import ConfigLoader` | `from src.shared.config_loader import ConfigLoader` |
| `from log_utils import setup_logger` | `from src.shared.logger import setup_logger` |

### Configuration Paths

| Old Path | New Path |
|----------|----------|
| `"configs/pipeline_config.yaml"` | `"config/default.yaml"` |
| `"configs/config.yaml"` | `"config/base.yaml"` |
| `"configs/criteria_detector_config.yaml"` | `"config/criteria/detector.yaml"` |
| `"../configs/pipeline_config.yaml"` | `"../config/default.yaml"` |
| `"post_processors/config/*_patterns.yaml"` | `"config/criteria/patterns/*_patterns.yaml"` |

## Automated Update Script

Create this script to help update imports:

```python
#!/usr/bin/env python3
"""
Script to update imports in a Python file.
Usage: python update_imports.py <file_path>
"""

import sys
import re
from pathlib import Path

REPLACEMENTS = {
    # Models
    'from core.repository.entity.audio_dialog': 'from src.domain.models.audio_dialog',
    'from core.repository.entity.dialog_rows': 'from src.domain.models.dialog_row',
    'from core.repository.entity.dialog_criteria': 'from src.domain.models.dialog_criteria',
    
    # Services
    'from core.service.transcription_service': 'from src.domain.services.transcription_service',
    'from core.service.criteria_detection_service': 'from src.domain.services.criteria_detection_service',
    'from core.service.llm_processing_service': 'from src.domain.services.llm_processing_service',
    
    # Repositories
    'from core.repository.audio_dialog_repository': 'from src.infrastructure.database.repositories.audio_dialog_repository',
    'from core.repository.dialog_criteria_repository': 'from src.infrastructure.database.repositories.dialog_criteria_repository',
    'from core.repository.dialog_rows_repository': 'from src.infrastructure.database.repositories.dialog_row_repository',
    
    # ML - Audio to Text
    'from core.audio_to_text.audio_to_text_processor': 'from src.infrastructure.ml.audio_to_text.processor',
    'from core.audio_to_text.transcriber': 'from src.infrastructure.ml.audio_to_text.transcriber',
    'from core.audio_to_text.diarizer': 'from src.infrastructure.ml.audio_to_text.diarizer',
    'from core.audio_to_text.text_to_speaker_resolver': 'from src.infrastructure.ml.audio_to_text.speaker_resolver',
    'from core.audio_to_text.diarization_utils': 'from src.infrastructure.ml.audio_to_text.diarization_utils',
    
    # ML - NLP
    'from core.post_processors.text_processing.DialogueAnalyzerPandas': 'from src.infrastructure.ml.nlp.dialogue_analyzer',
    'from core.post_processors.text_processing.criteria_utils': 'from src.infrastructure.ml.nlp.criteria_utils',
    'from core.post_processors.llm_processing.objections_resolver': 'from src.infrastructure.ml.nlp.llm_analyzer',
    'from core.post_processors.text_processing.detector.': 'from src.infrastructure.ml.nlp.detectors.',
    
    # Audio
    'from core.audio_loader': 'from src.infrastructure.audio.loader',
    'from core.post_processors.audio_processing.loudness_analyzer': 'from src.infrastructure.audio.loudness_analyzer',
    
    # External
    'from lib.saiga': 'from src.infrastructure.external.ollama_client',
    
    # Pipeline
    'from core.pipeline.audio_processing_pipeline': 'from src.application.pipeline.executor',
    'from core.pipeline': 'from src.application.pipeline',
    
    # DTOs
    'from core.dto.audio_to_text_result': 'from src.application.dto.transcription_result',
    'from core.dto.criteria': 'from src.application.dto.criteria',
    'from core.dto.diarisation_result': 'from src.application.dto.diarization_result',
    
    # Shared
    'from lib.yaml_reader': 'from src.shared.config_loader',
    'from lib.log_utils': 'from src.shared.logger',
    'from lib.json_util': 'from src.shared.utils.json_utils',
    'from yaml_reader': 'from src.shared.config_loader',
    'from log_utils': 'from src.shared.logger',
    
    # Config paths
    '"configs/pipeline_config.yaml"': '"config/default.yaml"',
    '"configs/config.yaml"': '"config/base.yaml"',
    '"configs/criteria_detector_config.yaml"': '"config/criteria/detector.yaml"',
    '"../configs/pipeline_config.yaml"': '"../config/default.yaml"',
    '"post_processors/config/': '"config/criteria/patterns/',
}

def update_file(file_path):
    """Update imports in a file."""
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return
    
    content = path.read_text()
    original = content
    
    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)
    
    if content != original:
        path.write_text(content)
        print(f"✓ Updated: {file_path}")
    else:
        print(f"  No changes: {file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_imports.py <file_path>")
        sys.exit(1)
    
    update_file(sys.argv[1])
```

## Manual Update Steps

### Step 1: Files in `src/` already have correct imports
The migrated files in `src/` directory need their imports updated to reference other files in `src/`.

### Step 2: Update one file at a time
1. Open a file in `src/`
2. Find all imports from `core.`, `lib.`, etc.
3. Replace using the table above
4. Test the file

### Step 3: Update config paths
Any references to `configs/` should become `config/`

## Testing Your Changes

After updating imports:

```bash
# Test import
python -c "from src.domain.models.audio_dialog import AudioDialog; print('✓ Import works')"

# Test CLI (if updated)
python src/interface/cli/main.py --help

# Test services
python -c "from src.domain.services.transcription_service import TranscriptionService; print('✓ Service import works')"
```

## Priority Update List

Update these files first (in order):

1. **✅ DONE:** Files in `src/` (already copied with correct structure)
2. **Shared utilities** in `src/shared/`
3. **Infrastructure** in `src/infrastructure/`
4. **Domain services** in `src/domain/services/`
5. **Application layer** in `src/application/`
6. **Interface layer** in `src/interface/`

## Which Files Need Updating?

### Files in `src/` Directory
ALL files in `src/` need their imports updated to reference other `src/` files.

**Example:**

File: `src/domain/services/transcription_service.py`

```python
# OLD (won't work)
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.audio_to_text.audio_to_text_processor import audio_to_text_processor
from log_utils import setup_logger

# NEW (correct)
from src.infrastructure.database.repositories.audio_dialog_repository import AudioDialogRepository
from src.infrastructure.ml.audio_to_text.processor import audio_to_text_processor
from src.shared.logger import setup_logger
```

## Quick Reference: Most Common Updates

```python
# Models
from src.domain.models.audio_dialog import AudioDialog
from src.domain.models.dialog_row import DialogRow

# Services
from src.domain.services.transcription_service import TranscriptionService

# Repositories
from src.infrastructure.database.repositories.audio_dialog_repository import AudioDialogRepository

# ML
from src.infrastructure.ml.audio_to_text.processor import audio_to_text_processor
from src.infrastructure.ml.nlp.detectors.sales_detector import SalesDetector

# Shared
from src.shared.logger import setup_logger
from src.shared.config_loader import ConfigLoader

# Application
from src.application.pipeline.executor import AudioProcessingPipeline
```

## Gradual Migration Strategy

You can migrate gradually:

1. **Week 1:** Keep using old structure (`core/`, `lib/`, `configs/`)
2. **Week 2:** Update imports in `src/` to work with new structure
3. **Week 3:** Test new structure thoroughly
4. **Week 4:** Switch to using new structure completely
5. **Week 5:** Remove old `core/` directory (after confirming everything works)

## Notes

- **Old files still work!** The migration copied files, it didn't move them
- **No rush:** You can migrate imports gradually
- **Test thoroughly:** After updating imports, test each component
- **Keep backups:** The old structure is your backup

## Status Tracking

Track which files you've updated:

- [ ] src/domain/models/*.py
- [ ] src/domain/services/*.py
- [ ] src/infrastructure/database/repositories/*.py
- [ ] src/infrastructure/ml/audio_to_text/*.py
- [ ] src/infrastructure/ml/nlp/*.py
- [ ] src/infrastructure/ml/nlp/detectors/*.py
- [ ] src/application/pipeline/*.py
- [ ] src/application/dto/*.py
- [ ] src/interface/cli/*.py
- [ ] src/shared/*.py

## Getting Help

If imports aren't working:
1. Check the path is correct
2. Check `__init__.py` files exist in all directories
3. Check Python can find `src/` (add to PYTHONPATH if needed)
4. Use the import update script above

---

**Remember:** The old structure still works! Take your time with the migration.

