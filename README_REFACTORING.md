# Audio Processing Pipeline - Refactored Architecture

This project has been refactored to provide a clean, maintainable, and scalable architecture for audio processing with transcription, diarization, criteria detection, and LLM-based analysis.

## 🎯 What Changed

**Before:** Three separate manual scripts  
**After:** Unified, modular pipeline with CLI and Python API

## 📁 New Structure

```
audio-processing/
├── configs/
│   ├── pipeline_config.yaml          # Centralized configuration
│   └── criteria_detector_config.yaml
├── core/
│   ├── audio_to_text/
│   │   ├── diarization_utils.py      # ✨ NEW: Extracted utilities
│   │   └── ...
│   ├── service/                       # ✨ NEW: Service layer
│   │   ├── transcription_service.py
│   │   ├── criteria_detection_service.py
│   │   └── llm_processing_service.py
│   ├── pipeline/                      # ✨ NEW: Pipeline orchestration
│   │   ├── __init__.py
│   │   └── audio_processing_pipeline.py
│   ├── transcribe_files_runner.py    # Old (preserved)
│   ├── pd_criteria_detector_runner.py # Old (preserved)
│   ├── llm_detector_runner.py        # Old (preserved)
│   ├── transcribe_files_runner_new.py # ✨ NEW: Refactored version
│   ├── pd_criteria_detector_runner_new.py
│   └── llm_detector_runner_new.py
├── pipeline_cli.py                    # ✨ NEW: Command-line interface
├── example_usage.py                   # ✨ NEW: Usage examples
├── QUICKSTART.md                      # ✨ NEW: Quick start guide
├── REFACTORING_GUIDE.md               # ✨ NEW: Detailed guide
└── README_REFACTORING.md             # This file
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run the complete pipeline
python pipeline_cli.py run-all --input-folder ~/audio_files

# Or run stages individually
python pipeline_cli.py transcribe --input-folder ~/audio_files
python pipeline_cli.py detect-criteria
python pipeline_cli.py llm-process

# Check status
python pipeline_cli.py status
```

### Python API

```python
from core.pipeline import AudioProcessingPipeline

pipeline = AudioProcessingPipeline()
result = pipeline.run_full_pipeline(audio_folder="~/audio_files")

print(f"Duration: {result.get_duration():.2f}s")
print(f"Completed: {result.stages_completed}")
```

## ✨ Key Features

### 1. **Unified Pipeline Orchestration**
- Single entry point for all processing stages
- Automatic dependency management
- Progress tracking and reporting

### 2. **Configuration Management**
- All settings in one place (`configs/pipeline_config.yaml`)
- Environment-specific configurations
- No hardcoded values

### 3. **Comprehensive Error Handling**
- Automatic retry logic
- Continue-on-error mode
- Detailed error logging
- Graceful failure recovery

### 4. **Service Layer Architecture**
- Clean separation of concerns
- Reusable components
- Easy to test and maintain

### 5. **CLI Interface**
- User-friendly command-line tools
- Status checking
- Configuration validation

### 6. **Better Documentation**
- Comprehensive docstrings
- Type hints throughout
- Usage examples
- Architecture guides

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)** - Detailed architecture and migration guide
- **[example_usage.py](example_usage.py)** - Code examples

## 🔄 Migration Path

### Old Way (Still Works)

```python
# Step 1
python core/transcribe_files_runner.py

# Step 2
python core/pd_criteria_detector_runner.py

# Step 3
python core/llm_detector_runner.py
```

### New Way (Recommended)

```bash
# One command for all stages
python pipeline_cli.py run-all --input-folder ~/audio_files
```

**Benefits of New Way:**
- ✅ Better error handling
- ✅ Progress tracking
- ✅ Centralized configuration
- ✅ Automatic retry logic
- ✅ Detailed execution reports

## 📊 Example Output

```
============================================================
PIPELINE EXECUTION SUMMARY
============================================================
Duration: 3547.23 seconds
Completed stages: transcription, criteria_detection, llm_processing

Transcription:
  - Successful: 1450
  - Skipped: 45
  - Failed: 5

Criteria Detection:
  - Success: True
  - Rows processed: 12,543

LLM Processing:
  - Total dialogs: 1500
  - Processed: 1480
  - Skipped: 15
  - Failed: 5

============================================================
```

## 🛠️ Configuration Example

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
  max_retries: 3
  retry_delay_seconds: 5
  skip_indices: [380, 1313, 1275]

error_handling:
  continue_on_error: true
  save_error_log: true
```

## 🎓 Usage Examples

### Example 1: Process New Audio Files

```bash
python pipeline_cli.py transcribe --input-folder ~/new_audio/
```

### Example 2: Reprocess Criteria

```bash
python pipeline_cli.py detect-criteria
```

### Example 3: Check Status

```bash
python pipeline_cli.py status
```

Output:
```
Audio Dialogs:
  - Total: 1500
  - Processed: 1450
  - Pending: 50

LLM Processing:
  - Processed: 1380
  - Pending: 120
```

### Example 4: Python API

```python
from core.pipeline import AudioProcessingPipeline, PipelineStage

pipeline = AudioProcessingPipeline()

# Run only transcription
result = pipeline.run_single_stage(
    PipelineStage.TRANSCRIPTION,
    audio_folder="~/audio_files"
)

# Access detailed results
for file in result.transcription_stats['successful']:
    print(f"✓ {file}")

for file in result.transcription_stats['failed']:
    print(f"✗ {file}")
```

## 🔍 Architecture Highlights

### Service Layer

Each processing stage has a dedicated service:

- **TranscriptionService**: Audio → Text with speaker diarization
- **CriteriaDetectionService**: Linguistic analysis
- **LLMProcessingService**: Advanced LLM-based analysis

### Pipeline Orchestrator

The `AudioProcessingPipeline` class:
- Manages service lifecycle
- Handles parallel execution
- Tracks progress and errors
- Generates detailed reports

### Configuration System

Centralized YAML configuration:
- Pipeline settings
- Stage-specific parameters
- Error handling options
- Logging configuration

## 🐛 Troubleshooting

### Common Issues

**No input folder specified:**
```yaml
# Set in config
pipeline:
  default_input_folder: "~/audio_files"
```

**Out of memory:**
```yaml
# Reduce workers
pipeline:
  max_workers: 2
```

**LLM parsing errors:**
```yaml
# Add to skip list
llm_processing:
  skip_indices: [380, 1313, 1275]
```

For more troubleshooting tips, see [QUICKSTART.md](QUICKSTART.md).

## 📈 Performance Tips

### Optimize for Speed
- Increase `max_workers` (5-10)
- Use SSD storage
- Enable parallel processing

### Optimize for Stability
- Decrease `max_workers` (2-3)
- Enable `continue_on_error`
- Increase `max_retries`

## 🤝 Contributing

The refactored architecture makes it easier to:
- Add new processing stages
- Implement custom analyzers
- Extend the pipeline
- Write unit tests

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

This refactoring improves:
- Code maintainability
- Error resilience
- User experience
- System scalability

---

**Need Help?**
- Read [QUICKSTART.md](QUICKSTART.md) for basic usage
- See [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md) for detailed docs
- Check [example_usage.py](example_usage.py) for code samples
- Run `python pipeline_cli.py --help` for CLI options

