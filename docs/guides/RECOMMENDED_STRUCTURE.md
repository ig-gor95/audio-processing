# Recommended Project Structure

## Current Issues

1. ❌ Everything mixed in `core/` directory
2. ❌ No clear separation between domain logic and infrastructure
3. ❌ Runner files mixed with business logic
4. ❌ No tests directory structure
5. ❌ Configuration scattered
6. ❌ Utilities not well organized

## Recommended Structure

```
audio-processing/
│
├── README.md                          # Main project readme
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Development dependencies
├── setup.py                          # Package setup
├── .env.example                      # Environment variables template
├── .gitignore
│
├── config/                           # 📁 All configuration files
│   ├── __init__.py
│   ├── default.yaml                 # Default configuration
│   ├── development.yaml             # Development overrides
│   ├── production.yaml              # Production overrides
│   ├── criteria/                    # Criteria-specific configs
│   │   ├── detector_config.yaml
│   │   ├── patterns/
│   │   │   ├── greeting_patterns.yaml
│   │   │   ├── swear_patterns.yaml
│   │   │   └── ...
│   └── models/                      # Model configurations
│       └── diarization_config.yaml
│
├── src/                              # 📁 Main source code
│   ├── __init__.py
│   │
│   ├── domain/                       # 📁 Business logic & domain models
│   │   ├── __init__.py
│   │   ├── models/                  # Domain models/entities
│   │   │   ├── __init__.py
│   │   │   ├── audio_dialog.py
│   │   │   ├── dialog_row.py
│   │   │   ├── dialog_criteria.py
│   │   │   └── processing_result.py
│   │   │
│   │   └── services/                # Domain services (business logic)
│   │       ├── __init__.py
│   │       ├── audio_processing_service.py
│   │       ├── transcription_service.py
│   │       ├── criteria_detection_service.py
│   │       └── llm_analysis_service.py
│   │
│   ├── infrastructure/               # 📁 Infrastructure concerns
│   │   ├── __init__.py
│   │   │
│   │   ├── database/                # Database layer
│   │   │   ├── __init__.py
│   │   │   ├── connection.py
│   │   │   ├── repositories/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_repository.py
│   │   │   │   ├── audio_dialog_repository.py
│   │   │   │   ├── dialog_row_repository.py
│   │   │   │   └── criteria_repository.py
│   │   │   └── migrations/          # Database migrations
│   │   │       └── ...
│   │   │
│   │   ├── ml/                      # ML/AI infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── audio_to_text/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── transcriber.py
│   │   │   │   ├── diarizer.py
│   │   │   │   └── speaker_resolver.py
│   │   │   ├── nlp/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── text_analyzer.py
│   │   │   │   └── llm_client.py
│   │   │   └── models/              # Pre-trained models
│   │   │       └── ...
│   │   │
│   │   ├── audio/                   # Audio processing utilities
│   │   │   ├── __init__.py
│   │   │   ├── loader.py
│   │   │   ├── preprocessor.py
│   │   │   └── loudness_analyzer.py
│   │   │
│   │   └── external/                # External service integrations
│   │       ├── __init__.py
│   │       ├── ollama_client.py
│   │       └── storage_client.py
│   │
│   ├── application/                  # 📁 Application layer
│   │   ├── __init__.py
│   │   │
│   │   ├── pipeline/                # Pipeline orchestration
│   │   │   ├── __init__.py
│   │   │   ├── pipeline_executor.py
│   │   │   ├── stage_manager.py
│   │   │   └── result_aggregator.py
│   │   │
│   │   ├── use_cases/               # Use cases (application logic)
│   │   │   ├── __init__.py
│   │   │   ├── transcribe_audio.py
│   │   │   ├── detect_criteria.py
│   │   │   ├── analyze_with_llm.py
│   │   │   └── generate_report.py
│   │   │
│   │   └── dto/                     # Data Transfer Objects
│   │       ├── __init__.py
│   │       ├── transcription_result.py
│   │       ├── criteria_result.py
│   │       └── llm_result.py
│   │
│   ├── interface/                    # 📁 Interface adapters
│   │   ├── __init__.py
│   │   │
│   │   ├── cli/                     # Command-line interface
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── commands/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── transcribe_cmd.py
│   │   │   │   ├── detect_cmd.py
│   │   │   │   ├── analyze_cmd.py
│   │   │   │   └── status_cmd.py
│   │   │   └── formatters/
│   │   │       ├── __init__.py
│   │   │       └── result_formatter.py
│   │   │
│   │   ├── api/                     # REST API (if needed)
│   │   │   ├── __init__.py
│   │   │   ├── app.py
│   │   │   └── routes/
│   │   │       └── ...
│   │   │
│   │   └── web/                     # Web dashboard (if needed)
│   │       └── ...
│   │
│   └── shared/                       # 📁 Shared utilities
│       ├── __init__.py
│       ├── config_loader.py
│       ├── logger.py
│       ├── exceptions.py
│       ├── constants.py
│       └── utils/
│           ├── __init__.py
│           ├── file_utils.py
│           ├── text_utils.py
│           └── validation_utils.py
│
├── tests/                            # 📁 Test suite
│   ├── __init__.py
│   ├── conftest.py                  # Pytest configuration
│   │
│   ├── unit/                        # Unit tests
│   │   ├── __init__.py
│   │   ├── domain/
│   │   │   ├── test_audio_dialog.py
│   │   │   └── test_services.py
│   │   ├── infrastructure/
│   │   │   └── test_repositories.py
│   │   └── application/
│   │       └── test_use_cases.py
│   │
│   ├── integration/                 # Integration tests
│   │   ├── __init__.py
│   │   ├── test_transcription_flow.py
│   │   ├── test_criteria_detection.py
│   │   └── test_pipeline.py
│   │
│   ├── e2e/                         # End-to-end tests
│   │   ├── __init__.py
│   │   └── test_full_pipeline.py
│   │
│   └── fixtures/                    # Test fixtures
│       ├── audio/
│       │   └── sample.mp3
│       └── data/
│           └── sample_dialog.json
│
├── scripts/                          # 📁 Utility scripts
│   ├── setup_database.py
│   ├── migrate_old_data.py
│   ├── benchmark_performance.py
│   └── export_reports.py
│
├── docs/                             # 📁 Documentation
│   ├── index.md
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── domain_model.md
│   │   └── pipeline_design.md
│   ├── guides/
│   │   ├── quickstart.md
│   │   ├── configuration.md
│   │   └── deployment.md
│   ├── api/
│   │   └── api_reference.md
│   └── images/
│       └── architecture_diagram.png
│
├── data/                             # 📁 Data files (gitignored)
│   ├── input/                       # Input audio files
│   ├── output/                      # Output results
│   ├── temp/                        # Temporary processing files
│   └── models/                      # Downloaded ML models
│
├── logs/                             # 📁 Log files (gitignored)
│   ├── app.log
│   ├── errors.log
│   └── pipeline.log
│
├── notebooks/                        # 📁 Jupyter notebooks
│   ├── exploratory/
│   │   └── audio_analysis.ipynb
│   └── experiments/
│       └── model_tuning.ipynb
│
└── deployments/                      # 📁 Deployment configurations
    ├── docker/
    │   ├── Dockerfile
    │   ├── docker-compose.yml
    │   └── .dockerignore
    ├── kubernetes/
    │   └── ...
    └── systemd/
        └── audio-processing.service
```

## Key Principles Applied

### 1. **Clean Architecture / Hexagonal Architecture**

```
┌─────────────────────────────────────────────────────┐
│                    Interface                        │
│              (CLI, API, Web)                        │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│                 Application                         │
│         (Use Cases, Pipeline)                       │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│                   Domain                            │
│         (Business Logic, Models)                    │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              Infrastructure                         │
│    (Database, ML, External Services)                │
└─────────────────────────────────────────────────────┘
```

**Benefits:**
- Business logic independent of frameworks
- Easy to test
- Flexible to change infrastructure
- Clear dependencies flow

### 2. **Separation of Concerns**

Each directory has a single responsibility:
- `domain/` - Business rules and entities
- `application/` - Application-specific logic
- `infrastructure/` - Technical implementations
- `interface/` - User/system interfaces

### 3. **Testability**

- Unit tests for domain logic
- Integration tests for repositories
- E2E tests for full workflows
- Test fixtures organized

### 4. **Configuration Management**

- Environment-based configs
- Override mechanism
- Type-safe config loading
- Separate concerns (DB, ML, Pipeline)

## Migration Strategy

### Phase 1: Prepare New Structure (Week 1)

```bash
# Create new directory structure
mkdir -p src/{domain,infrastructure,application,interface,shared}
mkdir -p tests/{unit,integration,e2e,fixtures}
mkdir -p config/{criteria,models}
mkdir -p scripts docs
```

### Phase 2: Move Domain Logic (Week 2)

1. **Move models:**
   ```
   core/repository/entity/ → src/domain/models/
   ```

2. **Create domain services:**
   ```python
   # src/domain/services/transcription_service.py
   class TranscriptionService:
       """Pure business logic, no infrastructure"""
       def should_process(self, dialog: AudioDialog) -> bool:
           # Business rules
   ```

### Phase 3: Move Infrastructure (Week 2-3)

1. **Move repositories:**
   ```
   core/repository/ → src/infrastructure/database/repositories/
   ```

2. **Move ML code:**
   ```
   core/audio_to_text/ → src/infrastructure/ml/audio_to_text/
   core/post_processors/ → src/infrastructure/ml/nlp/
   ```

### Phase 4: Move Application Layer (Week 3)

1. **Move pipeline:**
   ```
   core/pipeline/ → src/application/pipeline/
   ```

2. **Create use cases:**
   ```python
   # src/application/use_cases/transcribe_audio.py
   class TranscribeAudioUseCase:
       def __init__(
           self,
           transcription_service: TranscriptionService,
           repository: AudioDialogRepository
       ):
           ...
   ```

### Phase 5: Move Interfaces (Week 4)

1. **Move CLI:**
   ```
   pipeline_cli.py → src/interface/cli/main.py
   ```

2. **Organize commands:**
   ```
   Create command pattern for each CLI command
   ```

### Phase 6: Configuration & Documentation (Week 4)

1. **Reorganize configs:**
   ```
   configs/ → config/
   Split into environment-based configs
   ```

2. **Update documentation:**
   ```
   Update all docs to reflect new structure
   ```

## Comparison: Before & After

### Current Structure (Issues)
```
core/
├── audio_loader.py                    ❌ Mixed with other files
├── transcribe_files_runner.py        ❌ Runner mixed with logic
├── audio_to_text/                    ❌ Infrastructure + domain mixed
├── repository/                       ❌ In core instead of infrastructure
│   └── entity/                       ❌ Should be domain models
└── service/                          ❌ Mixed responsibilities
```

### Recommended Structure (Clean)
```
src/
├── domain/                           ✅ Pure business logic
│   ├── models/                       ✅ Domain entities
│   └── services/                     ✅ Domain services
├── infrastructure/                   ✅ Technical implementations
│   ├── database/repositories/        ✅ Data access
│   └── ml/                          ✅ ML infrastructure
├── application/                      ✅ Application logic
│   ├── pipeline/                     ✅ Orchestration
│   └── use_cases/                    ✅ Application flows
└── interface/                        ✅ User interfaces
    └── cli/                          ✅ CLI commands
```

## Detailed Example: Transcription Flow

### Current (Mixed Responsibilities)
```python
# core/transcribe_files_runner.py
# Everything in one file: orchestration, business logic, infrastructure

def run_pipeline(audio_file):
    # DB logic
    existing = audio_dialog_repository.find_by_filename(...)
    
    # Business logic
    if existing.status == AudioDialogStatus.PROCESSED:
        return
    
    # Infrastructure
    result = audio_to_text_processor(audio_file)
    
    # More DB logic
    dialog_row_repository.save_bulk(...)
```

### Recommended (Separated)
```python
# src/domain/services/transcription_service.py
class TranscriptionService:
    """Pure business logic"""
    def should_process(self, dialog: AudioDialog) -> bool:
        if dialog.status == AudioDialogStatus.PROCESSED:
            if dialog.updated_at < self.reprocess_date:
                return True
            return False
        return True

# src/infrastructure/ml/audio_to_text/transcriber.py
class AudioTranscriber:
    """Infrastructure - ML implementation"""
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        # ML logic here

# src/application/use_cases/transcribe_audio.py
class TranscribeAudioUseCase:
    """Application logic - orchestration"""
    def __init__(
        self,
        transcription_service: TranscriptionService,
        transcriber: AudioTranscriber,
        repository: AudioDialogRepository
    ):
        self.service = transcription_service
        self.transcriber = transcriber
        self.repository = repository
    
    def execute(self, audio_path: Path) -> UUID:
        # Find or create dialog
        dialog = self.repository.find_or_create(audio_path.name)
        
        # Check if should process (business rule)
        if not self.service.should_process(dialog):
            return None
        
        # Process (infrastructure)
        result = self.transcriber.transcribe(audio_path)
        
        # Save (infrastructure)
        dialog.update_from_result(result)
        self.repository.save(dialog)
        
        return dialog.id

# src/interface/cli/commands/transcribe_cmd.py
class TranscribeCommand:
    """CLI interface"""
    def __init__(self, use_case: TranscribeAudioUseCase):
        self.use_case = use_case
    
    def execute(self, input_folder: str):
        files = Path(input_folder).glob("*")
        for file in files:
            self.use_case.execute(file)
```

## Benefits of New Structure

### 1. **Testability**
```python
# Easy to test business logic without infrastructure
def test_should_process():
    service = TranscriptionService(reprocess_date=...)
    dialog = AudioDialog(status=ProcessingStatus.PROCESSED, ...)
    assert service.should_process(dialog) == False
```

### 2. **Flexibility**
```python
# Easy to swap implementations
# Old: PostgreSQL → New: MongoDB
old_repo = PostgresAudioDialogRepository()
new_repo = MongoAudioDialogRepository()
# Interface stays the same!
```

### 3. **Clarity**
- Domain logic in one place
- Infrastructure in another
- Clear dependency directions
- Easy to onboard new developers

### 4. **Scalability**
- Add new use cases easily
- Extend with new interfaces (Web API, gRPC)
- Plug in new ML models
- Add features without breaking existing code

## Implementation Guide

### Step 1: Create Package Structure
```python
# src/__init__.py
__version__ = "2.0.0"

# src/domain/__init__.py
from .models import AudioDialog, DialogRow
from .services import TranscriptionService

# src/application/__init__.py
from .use_cases import TranscribeAudioUseCase

# src/interface/__init__.py
# Interface adapters
```

### Step 2: Update Imports
```python
# Old
from core.repository.audio_dialog_repository import AudioDialogRepository

# New
from src.infrastructure.database.repositories import AudioDialogRepository
```

### Step 3: Use Dependency Injection
```python
# src/application/container.py (Dependency Injection Container)
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # Repositories
    audio_dialog_repo = providers.Singleton(AudioDialogRepository)
    
    # Services
    transcription_service = providers.Factory(
        TranscriptionService,
        config=config.transcription
    )
    
    # Use Cases
    transcribe_use_case = providers.Factory(
        TranscribeAudioUseCase,
        service=transcription_service,
        repository=audio_dialog_repo
    )
```

## Tools & Best Practices

### Project Management
```bash
# Use poetry for dependency management
poetry init
poetry add <package>

# Or pipenv
pipenv install
```

### Code Quality
```bash
# Black for formatting
black src/

# isort for import sorting
isort src/

# mypy for type checking
mypy src/

# pylint for linting
pylint src/

# pytest for testing
pytest tests/
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
```

## Summary

### Current Issues → Solutions

| Issue | Solution |
|-------|----------|
| Mixed responsibilities | Layered architecture |
| Hard to test | Pure business logic in domain |
| Tightly coupled | Dependency injection |
| No clear structure | Clean architecture |
| Configuration scattered | Centralized config/ |
| No tests | Comprehensive test suite |
| Hard to extend | Open/closed principle |

### Next Steps

1. ✅ **Review this document**
2. ✅ **Create new directory structure**
3. ✅ **Start with domain layer** (models + services)
4. ✅ **Move infrastructure** (repos + ML)
5. ✅ **Create application layer** (use cases)
6. ✅ **Update interfaces** (CLI)
7. ✅ **Write tests** as you go
8. ✅ **Update documentation**

This structure will make your codebase:
- **Professional** - Industry-standard architecture
- **Maintainable** - Easy to understand and modify
- **Testable** - Each layer can be tested independently
- **Scalable** - Easy to add new features
- **Flexible** - Easy to swap implementations

