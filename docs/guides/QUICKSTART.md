# Quick Start Guide

This guide will help you get started with the refactored audio processing pipeline.

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy and customize the configuration file:

```bash
cp configs/pipeline_config.yaml configs/my_config.yaml
```

2. Edit `configs/my_config.yaml`:

```yaml
pipeline:
  default_input_folder: "/path/to/your/audio/files"
  max_workers: 5
  max_files_per_run: 5000
```

## Usage

### Method 1: Command Line Interface (Recommended)

#### Run the Complete Pipeline

```bash
python pipeline_cli.py run-all --input-folder ~/audio_files
```

#### Run Individual Stages

```bash
# Step 1: Transcription
python pipeline_cli.py transcribe --input-folder ~/audio_files

# Step 2: Criteria Detection
python pipeline_cli.py detect-criteria

# Step 3: LLM Processing
python pipeline_cli.py llm-process
```

#### Check Status

```bash
python pipeline_cli.py status
```

Output:
```
============================================================
PIPELINE STATUS
============================================================

Audio Dialogs:
  - Total: 1500
  - Processed: 1450
  - Pending: 50

Criteria Detection:
  - Unprocessed rows: 125

LLM Processing:
  - Processed: 1380
  - Pending: 120
============================================================
```

### Method 2: Python API

Create a script `process_my_audio.py`:

```python
from core.pipeline import AudioProcessingPipeline

# Initialize pipeline
pipeline = AudioProcessingPipeline()

# Run full pipeline
result = pipeline.run_full_pipeline(
    audio_folder="~/Documents/Аудио Бринекс/2/"
)

# Print results
print(f"Duration: {result.get_duration():.2f} seconds")
print(f"Completed stages: {result.stages_completed}")

if result.transcription_stats:
    print(f"Successful: {len(result.transcription_stats['successful'])}")
    print(f"Failed: {len(result.transcription_stats['failed'])}")
```

Run it:
```bash
python process_my_audio.py
```

### Method 3: Service Layer (Advanced)

For more control, use services directly:

```python
from pathlib import Path
from core.service.transcription_service import TranscriptionService
from core.service.criteria_detection_service import CriteriaDetectionService
from core.service.llm_processing_service import LLMProcessingService

# Transcription
transcription_service = TranscriptionService()
audio_file = Path("path/to/audio.mp3")
file_uuid = transcription_service.process_audio_file(audio_file)

# Criteria Detection
criteria_service = CriteriaDetectionService()
stats = criteria_service.process_dialogs()

# LLM Processing
llm_service = LLMProcessingService()
success = llm_service.process_dialog(file_uuid)
```

## Common Workflows

### Workflow 1: Process New Audio Files

```bash
# 1. Add new audio files to your input folder
cp new_audio/*.mp3 ~/audio_files/

# 2. Run transcription
python pipeline_cli.py transcribe --input-folder ~/audio_files

# 3. Run criteria detection
python pipeline_cli.py detect-criteria

# 4. Run LLM processing
python pipeline_cli.py llm-process
```

### Workflow 2: Reprocess Criteria for All Dialogs

```bash
# Just run criteria detection - it processes unprocessed rows
python pipeline_cli.py detect-criteria
```

### Workflow 3: Process with Custom Configuration

```bash
python pipeline_cli.py run-all \
  --input-folder ~/audio_files \
  --config configs/my_custom_config.yaml
```

### Workflow 4: Monitor Progress

Open two terminals:

**Terminal 1:** Run processing
```bash
python pipeline_cli.py run-all --input-folder ~/audio_files
```

**Terminal 2:** Monitor status
```bash
# Check status every 30 seconds
watch -n 30 "python pipeline_cli.py status"
```

## Configuration Tips

### Optimize for Performance

```yaml
pipeline:
  max_workers: 10  # Increase for more parallelism
  max_files_per_run: 10000

transcription:
  omp_num_threads: 2
  mkl_num_threads: 2
```

### Optimize for Stability

```yaml
pipeline:
  max_workers: 2  # Reduce for stability

error_handling:
  continue_on_error: true  # Don't stop on single file failure
  save_error_log: true

llm_processing:
  max_retries: 5  # More retry attempts
  retry_delay_seconds: 10
```

### Skip Problematic Files

If certain dialogs cause issues:

```yaml
llm_processing:
  skip_indices: [380, 1313, 1275, 1500, 1501]  # Add problematic indices
```

## Troubleshooting

### Issue: "Module not found" errors

```bash
# Ensure you're in the project root
cd /path/to/audio-processing

# Install dependencies
pip install -r requirements.txt
```

### Issue: "No input folder specified"

Set default in config:
```yaml
pipeline:
  default_input_folder: "~/Documents/Аудио Бринекс/2/"
```

Or pass via CLI:
```bash
python pipeline_cli.py transcribe --input-folder ~/audio_files
```

### Issue: Database connection errors

Check database configuration in `configs/config.yaml`:
```yaml
db:
  dbname: neiro-insight
  user: postgres
  password: your_password
  host: localhost
  port: 5432
```

### Issue: LLM service not responding

1. Check if Ollama is running:
```bash
curl http://localhost:11434/v1/models
```

2. Check model configuration in `configs/config.yaml`:
```yaml
saiga:
  url: http://localhost:11434/v1
  model-name: gemma3:12b
```

### Issue: Out of memory

Reduce parallel workers:
```yaml
pipeline:
  max_workers: 2  # Reduce from 5
```

## Next Steps

1. **Review Configuration:** Customize `configs/pipeline_config.yaml` for your needs
2. **Read Documentation:** See `REFACTORING_GUIDE.md` for detailed architecture
3. **Test with Sample:** Run pipeline on a small set of files first
4. **Monitor Logs:** Check `logs/` directory for detailed execution logs
5. **Backup Database:** Before large runs, backup your database

## Getting Help

- Check logs in `logs/` directory
- Review error log: `logs/errors.json`
- See `REFACTORING_GUIDE.md` for detailed documentation
- Run validation: `python pipeline_cli.py validate-config`

## Migrating from Old Scripts

If you were using:
- `transcribe_files_runner.py` → Use `pipeline_cli.py transcribe`
- `pd_criteria_detector_runner.py` → Use `pipeline_cli.py detect-criteria`
- `llm_detector_runner.py` → Use `pipeline_cli.py llm-process`

The old scripts still work but the new pipeline offers:
- Better error handling
- Progress tracking
- Configuration management
- Unified interface

