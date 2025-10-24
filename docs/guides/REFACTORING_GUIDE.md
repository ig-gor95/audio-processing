# Audio Processing Pipeline - Refactoring Guide

This document explains the refactoring changes made to improve code organization, readability, and maintainability.

## Overview of Changes

The project has been refactored from three separate runner scripts into a unified, modular pipeline architecture:

**Before:**
- `transcribe_files_runner.py` - Manual transcription script
- `pd_criteria_detector_runner.py` - Manual criteria detection script  
- `llm_detector_runner.py` - Manual LLM processing script

**After:**
- Unified `AudioProcessingPipeline` orchestrator
- Service layer for each stage
- Configuration-driven execution
- Command-line interface (CLI)

## New Architecture

### 1. Configuration Layer (`configs/pipeline_config.yaml`)

All hardcoded values are now centralized in configuration:

```yaml
pipeline:
  max_workers: 5
  max_files_per_run: 5000
  default_input_folder: "~/Documents/Аудио Бринекс/2/"

diarization:
  window_size: 0.9
  hop_size: 0.25
  switch_penalty: 0.18

llm_processing:
  skip_indices: [380, 1313, 1275]
  max_retries: 3
```

**Benefits:**
- Easy to modify settings without code changes
- Environment-specific configurations
- Better documentation of parameters

### 2. Utility Layer (`core/audio_to_text/diarization_utils.py`)

Extracted diarization helper functions with full documentation:

- `make_overlapping_windows()` - Generate overlapping audio windows
- `merge_labeled_windows()` - Merge consecutive speaker segments
- `viterbi_labels_by_centroids()` - Refine speaker labels

**Benefits:**
- Reusable across different contexts
- Well-documented with docstrings and examples
- Easier to test and maintain

### 3. Service Layer

#### `TranscriptionService` (`core/service/transcription_service.py`)

Handles audio transcription and diarization:

```python
service = TranscriptionService()
file_uuid = service.process_audio_file(audio_path)
results = service.process_multiple_files(audio_files)
```

**Features:**
- Automatic file status tracking
- Configurable reprocessing logic
- Batch processing support

#### `CriteriaDetectionService` (`core/service/criteria_detection_service.py`)

Handles linguistic criteria detection:

```python
service = CriteriaDetectionService()
stats = service.process_dialogs()
```

**Features:**
- Batch processing
- Configurable criteria selection
- Status tracking

#### `LLMProcessingService` (`core/service/llm_processing_service.py`)

Handles LLM-based dialog analysis:

```python
service = LLMProcessingService()
stats = service.process_all_dialogs()
success = service.process_dialog(dialog_id)
```

**Features:**
- Automatic retry logic with exponential backoff
- Robust JSON parsing with multiple fallback strategies
- Skip list support for problematic dialogs
- Progress tracking

### 4. Pipeline Orchestration (`core/pipeline/audio_processing_pipeline.py`)

Unified pipeline that orchestrates all stages:

```python
from core.pipeline import AudioProcessingPipeline, PipelineStage

pipeline = AudioProcessingPipeline()

# Run full pipeline
result = pipeline.run_full_pipeline(audio_folder="path/to/audio")

# Run single stage
result = pipeline.run_single_stage(
    PipelineStage.TRANSCRIPTION,
    audio_folder="path/to/audio"
)
```

**Features:**
- Parallel processing with configurable workers
- Comprehensive error handling
- Detailed execution statistics
- Error logging to file
- Continue-on-error support

### 5. Command-Line Interface (`pipeline_cli.py`)

Easy-to-use CLI for pipeline execution:

```bash
# Run full pipeline
python pipeline_cli.py run-all --input-folder ~/audio_files

# Run individual stages
python pipeline_cli.py transcribe --input-folder ~/audio_files
python pipeline_cli.py detect-criteria
python pipeline_cli.py llm-process

# Check status
python pipeline_cli.py status

# Validate configuration
python pipeline_cli.py validate-config
```

## Migration Guide

### Old Way

```python
# Step 1: Run transcription (manually edit folder path)
if __name__ == "__main__":
    folder_path = f"{Path.home()}/Documents/Аудио Бринекс/2/"
    audio_files = list(Path(folder_path).glob("*"))
    process_files_parallel(audio_files, max_files=5000)

# Step 2: Run criteria detection (separate script)
if __name__ == "__main__":
    analyzer = DialogueAnalyzerPandas()
    analyzer.analyze_dialogue()

# Step 3: Run LLM processing (separate script)
if __name__ == "__main__":
    dialogs = audio_dialog_repository.find_all()
    for dialog in dialogs:
        # ... manual iteration
```

### New Way (Option 1: CLI)

```bash
# Run everything in one command
python pipeline_cli.py run-all --input-folder ~/Documents/Аудио\ Бринекс/2/

# Or run stages individually
python pipeline_cli.py transcribe --input-folder ~/Documents/Аудио\ Бринекс/2/
python pipeline_cli.py detect-criteria
python pipeline_cli.py llm-process
```

### New Way (Option 2: Python API)

```python
from core.pipeline import AudioProcessingPipeline

# Configure once
pipeline = AudioProcessingPipeline()

# Run everything
result = pipeline.run_full_pipeline(
    audio_folder="~/Documents/Аудио Бринекс/2/"
)

# Check results
print(f"Duration: {result.get_duration():.2f}s")
print(f"Completed: {result.stages_completed}")
print(f"Failed: {result.stages_failed}")
print(result)  # Detailed JSON output
```

## Key Improvements

### 1. **Separation of Concerns**

- **Old:** Business logic mixed with orchestration in runner files
- **New:** Clear separation between services, pipeline, and CLI

### 2. **Configuration Management**

- **Old:** Hardcoded values scattered throughout code
- **New:** Centralized configuration file

### 3. **Error Handling**

- **Old:** Basic try-catch with limited recovery
- **New:** Comprehensive error handling with:
  - Retry logic with configurable attempts
  - Continue-on-error mode
  - Detailed error logging
  - Error aggregation and reporting

### 4. **Code Reusability**

- **Old:** Helper functions embedded in runner files
- **New:** Reusable utilities in dedicated modules

### 5. **Testability**

- **Old:** Difficult to test individual components
- **New:** Each service can be tested independently

### 6. **Documentation**

- **Old:** Minimal inline comments
- **New:** 
  - Comprehensive docstrings
  - Type hints
  - Configuration documentation
  - Usage examples

### 7. **Monitoring & Debugging**

- **Old:** Scattered print statements
- **New:**
  - Structured logging
  - Execution statistics
  - Status command for pipeline state
  - Detailed error reports

## Configuration Options

### Pipeline Settings

```yaml
pipeline:
  max_workers: 5              # Parallel processing threads
  max_files_per_run: 5000     # Max files to process per run
  default_input_folder: "..."  # Default audio folder
  temp_folder: "data/input/temp/"
```

### Transcription Settings

```yaml
transcription:
  omp_num_threads: 1          # OpenMP threads
  mkl_num_threads: 1          # MKL threads
  reprocess_before_date: "2025-09-06"  # Reprocess old files
  skip_processed: true        # Skip already processed files
```

### Diarization Settings

```yaml
diarization:
  window_size: 0.9            # Window size in seconds
  hop_size: 0.25              # Hop size in seconds
  min_segment_length: 0.8     # Minimum segment length
  max_gap: 0.15               # Max gap to merge segments
  switch_penalty: 0.18        # Speaker switch penalty
```

### LLM Processing Settings

```yaml
llm_processing:
  skip_existing: true         # Skip dialogs with existing data
  skip_indices: [380, 1313, 1275]  # Problematic dialog indices
  min_dialog_length: 5        # Minimum chars to process
  max_retries: 3              # Retry attempts
  retry_delay_seconds: 5      # Delay between retries
```

### Error Handling

```yaml
error_handling:
  continue_on_error: true     # Continue if one file fails
  save_error_log: true        # Save errors to file
  error_log_path: "logs/errors.json"
```

## Usage Examples

### 1. Process New Audio Files

```bash
python pipeline_cli.py transcribe --input-folder ~/new_audio_files/
```

### 2. Reprocess Criteria Detection

```bash
python pipeline_cli.py detect-criteria
```

### 3. Run LLM on Pending Dialogs

```bash
python pipeline_cli.py llm-process
```

### 4. Check Pipeline Status

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

### 5. Custom Configuration

```bash
python pipeline_cli.py run-all \
  --input-folder ~/audio_files \
  --config my_custom_config.yaml
```

### 6. Programmatic Usage

```python
from core.pipeline import AudioProcessingPipeline, PipelineStage
from pathlib import Path

# Initialize pipeline
pipeline = AudioProcessingPipeline()

# Process specific files
audio_files = [
    Path("file1.mp3"),
    Path("file2.mp3"),
    Path("file3.mp3")
]

result = pipeline.run_full_pipeline(audio_files=audio_files)

# Access results
print(f"Successful: {len(result.transcription_stats['successful'])}")
print(f"Failed: {len(result.transcription_stats['failed'])}")

# Save results
with open('results.json', 'w') as f:
    f.write(str(result))
```

## Backward Compatibility

The old runner files still exist and can be used:

```python
# Old way still works
python core/transcribe_files_runner.py
python core/pd_criteria_detector_runner.py
python core/llm_detector_runner.py
```

However, using the new pipeline is **strongly recommended** for:
- Better error handling
- Progress tracking
- Configuration management
- Easier maintenance

## Next Steps

1. **Update `transcribe_files_runner.py`** to use `TranscriptionService`:
   ```python
   from core.service.transcription_service import TranscriptionService
   # Update implementation
   ```

2. **Update `pd_criteria_detector_runner.py`** to use `CriteriaDetectionService`

3. **Update `llm_detector_runner.py`** to use `LLMProcessingService`

4. **Add unit tests** for services and utilities

5. **Consider adding**:
   - Database migrations for new fields
   - Monitoring/metrics integration
   - Web dashboard for pipeline status
   - Scheduled execution (cron/systemd)

## Troubleshooting

### Issue: "No input folder specified"

**Solution:** Either:
- Pass `--input-folder` to CLI
- Set `default_input_folder` in config
- Use programmatic API with explicit path

### Issue: LLM processing fails with JSON errors

**Solution:** The service includes multiple fallback strategies:
- Strips markdown code fences
- Extracts first JSON blob
- Handles both objects and arrays

If issues persist, add the problematic dialog index to `skip_indices` in config.

### Issue: Files not being reprocessed

**Solution:** Check `skip_processed` and `reprocess_before_date` in config:
```yaml
transcription:
  skip_processed: false  # Set to false to reprocess all
```

### Issue: Out of memory with parallel processing

**Solution:** Reduce `max_workers`:
```yaml
pipeline:
  max_workers: 2  # Reduce from 5 to 2
```

## Summary

This refactoring transforms the audio processing system from a collection of scripts into a well-structured, maintainable pipeline with:

✅ Unified orchestration  
✅ Configuration-driven execution  
✅ Comprehensive error handling  
✅ CLI interface  
✅ Reusable services  
✅ Better documentation  
✅ Improved testability  
✅ Progress tracking  
✅ Backward compatibility  

The new architecture makes it easier to:
- Add new processing stages
- Test individual components
- Monitor pipeline execution
- Handle errors gracefully
- Scale processing capacity
- Maintain and extend the codebase

