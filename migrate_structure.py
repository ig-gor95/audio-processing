#!/usr/bin/env python3
"""
Migration Script for New Project Structure

This script helps migrate from the current structure to the recommended
Clean Architecture structure.

Usage:
    python migrate_structure.py --dry-run    # Preview changes
    python migrate_structure.py              # Execute migration
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple


class StructureMigrator:
    """Handles migration to new project structure."""
    
    def __init__(self, project_root: Path, dry_run: bool = False):
        self.root = project_root
        self.dry_run = dry_run
        self.moves: List[Tuple[Path, Path]] = []
    
    def create_directory_structure(self):
        """Create the new directory structure."""
        directories = [
            # Source code
            "src",
            "src/domain",
            "src/domain/models",
            "src/domain/services",
            "src/infrastructure",
            "src/infrastructure/database",
            "src/infrastructure/database/repositories",
            "src/infrastructure/database/migrations",
            "src/infrastructure/ml",
            "src/infrastructure/ml/audio_to_text",
            "src/infrastructure/ml/nlp",
            "src/infrastructure/ml/nlp/detectors",
            "src/infrastructure/ml/models",
            "src/infrastructure/audio",
            "src/infrastructure/external",
            "src/application",
            "src/application/pipeline",
            "src/application/use_cases",
            "src/application/dto",
            "src/interface",
            "src/interface/cli",
            "src/interface/cli/commands",
            "src/interface/cli/formatters",
            "src/shared",
            "src/shared/utils",
            
            # Configuration
            "config",
            "config/criteria",
            "config/criteria/patterns",
            "config/models",
            
            # Tests
            "tests",
            "tests/unit",
            "tests/unit/domain",
            "tests/unit/infrastructure",
            "tests/unit/application",
            "tests/integration",
            "tests/e2e",
            "tests/fixtures",
            "tests/fixtures/audio",
            "tests/fixtures/data",
            
            # Documentation
            "docs",
            "docs/architecture",
            "docs/guides",
            "docs/api",
            
            # Scripts
            "scripts",
            
            # Data (these should be in .gitignore)
            "data",
            "data/input",
            "data/output",
            "data/temp",
            "data/models",
            
            # Logs (should be in .gitignore)
            "logs",
        ]
        
        print("Creating new directory structure...")
        for directory in directories:
            path = self.root / directory
            if self.dry_run:
                print(f"  [DRY RUN] Would create: {path}")
            else:
                path.mkdir(parents=True, exist_ok=True)
                (path / "__init__.py").touch(exist_ok=True)
                print(f"  ✓ Created: {path}")
    
    def plan_migrations(self):
        """Plan file migrations."""
        
        # Domain models
        self.add_move(
            "core/repository/entity/audio_dialog.py",
            "src/domain/models/audio_dialog.py"
        )
        self.add_move(
            "core/repository/entity/dialog_rows.py",
            "src/domain/models/dialog_row.py"
        )
        self.add_move(
            "core/repository/entity/dialog_criteria.py",
            "src/domain/models/dialog_criteria.py"
        )
        
        # Domain services (will be created from existing service code)
        self.add_move(
            "core/service/transcription_service.py",
            "src/domain/services/transcription_service.py"
        )
        self.add_move(
            "core/service/criteria_detection_service.py",
            "src/domain/services/criteria_detection_service.py"
        )
        self.add_move(
            "core/service/llm_processing_service.py",
            "src/domain/services/llm_processing_service.py"
        )
        
        # Infrastructure - Database
        self.add_move(
            "core/repository/audio_dialog_repository.py",
            "src/infrastructure/database/repositories/audio_dialog_repository.py"
        )
        self.add_move(
            "core/repository/dialog_criteria_repository.py",
            "src/infrastructure/database/repositories/dialog_criteria_repository.py"
        )
        self.add_move(
            "core/repository/dialog_rows_repository.py",
            "src/infrastructure/database/repositories/dialog_row_repository.py"
        )
        
        # Infrastructure - ML / Audio to Text
        self.add_move(
            "core/audio_to_text/audio_to_text_processor.py",
            "src/infrastructure/ml/audio_to_text/processor.py"
        )
        self.add_move(
            "core/audio_to_text/transcriber.py",
            "src/infrastructure/ml/audio_to_text/transcriber.py"
        )
        self.add_move(
            "core/audio_to_text/diarizer.py",
            "src/infrastructure/ml/audio_to_text/diarizer.py"
        )
        self.add_move(
            "core/audio_to_text/pyannote_diarizer.py",
            "src/infrastructure/ml/audio_to_text/pyannote_diarizer.py"
        )
        self.add_move(
            "core/audio_to_text/text_to_speaker_resolver.py",
            "src/infrastructure/ml/audio_to_text/speaker_resolver.py"
        )
        self.add_move(
            "core/audio_to_text/diarization_utils.py",
            "src/infrastructure/ml/audio_to_text/diarization_utils.py"
        )
        
        # Infrastructure - ML / NLP
        self.add_move(
            "core/post_processors/text_processing/DialogueAnalyzerPandas.py",
            "src/infrastructure/ml/nlp/dialogue_analyzer.py"
        )
        self.add_move(
            "core/post_processors/text_processing/criteria_utils.py",
            "src/infrastructure/ml/nlp/criteria_utils.py"
        )
        self.add_move(
            "core/post_processors/llm_processing/objections_resolver.py",
            "src/infrastructure/ml/nlp/llm_analyzer.py"
        )
        
        # Infrastructure - ML / Detectors (copy entire directory)
        # Note: These should be moved individually in practice
        
        # Infrastructure - Audio
        self.add_move(
            "core/audio_loader.py",
            "src/infrastructure/audio/loader.py"
        )
        self.add_move(
            "core/post_processors/audio_processing/loudness_analyzer.py",
            "src/infrastructure/audio/loudness_analyzer.py"
        )
        
        # Infrastructure - External
        self.add_move(
            "lib/saiga.py",
            "src/infrastructure/external/ollama_client.py"
        )
        
        # Application - Pipeline
        self.add_move(
            "core/pipeline/audio_processing_pipeline.py",
            "src/application/pipeline/executor.py"
        )
        
        # Application - DTOs
        self.add_move(
            "core/dto/audio_to_text_result.py",
            "src/application/dto/transcription_result.py"
        )
        self.add_move(
            "core/dto/criteria.py",
            "src/application/dto/criteria.py"
        )
        self.add_move(
            "core/dto/diarisation_result.py",
            "src/application/dto/diarization_result.py"
        )
        
        # Interface - CLI
        self.add_move(
            "pipeline_cli.py",
            "src/interface/cli/main.py"
        )
        
        # Shared utilities
        self.add_move(
            "lib/yaml_reader.py",
            "src/shared/config_loader.py"
        )
        self.add_move(
            "lib/log_utils.py",
            "src/shared/logger.py"
        )
        self.add_move(
            "lib/json_util.py",
            "src/shared/utils/json_utils.py"
        )
        
        # Configuration files
        self.add_move(
            "configs/pipeline_config.yaml",
            "config/default.yaml"
        )
        self.add_move(
            "configs/config.yaml",
            "config/base.yaml"
        )
        self.add_move(
            "configs/criteria_detector_config.yaml",
            "config/criteria/detector.yaml"
        )
        
        # Move pattern configs
        patterns = [
            "await_request_exit_patterns.yaml",
            "await_request_patterns.yaml",
            "axis_attention_pattern.yaml",
            "name_patterns.yaml",
            "non_professional_patterns.yaml",
            "ongoing_sale_patterns.yaml",
            "order_pattern.yaml",
            "parasites_patterns.yaml",
            "phrase_patterns.yaml",
            "sales_patterns.yaml",
            "stopwords_patterns.yaml",
            "swear_patterns.yaml",
            "working_hours_patterns.yaml",
        ]
        for pattern in patterns:
            self.add_move(
                f"core/post_processors/config/{pattern}",
                f"config/criteria/patterns/{pattern}"
            )
    
    def add_move(self, source: str, destination: str):
        """Add a file move operation."""
        src_path = self.root / source
        dst_path = self.root / destination
        
        if src_path.exists():
            self.moves.append((src_path, dst_path))
        else:
            print(f"  ⚠️  Source not found: {source}")
    
    def execute_migrations(self):
        """Execute the planned migrations."""
        print(f"\n{'=' * 60}")
        print("Executing file migrations...")
        print(f"{'=' * 60}\n")
        
        for src, dst in self.moves:
            if self.dry_run:
                print(f"[DRY RUN] Would move:")
                print(f"  FROM: {src.relative_to(self.root)}")
                print(f"  TO:   {dst.relative_to(self.root)}")
            else:
                try:
                    # Ensure destination directory exists
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file (don't move yet, to be safe)
                    shutil.copy2(src, dst)
                    print(f"  ✓ Copied: {src.relative_to(self.root)} → {dst.relative_to(self.root)}")
                except Exception as e:
                    print(f"  ✗ Error moving {src.name}: {e}")
    
    def create_init_files(self):
        """Create __init__.py files with proper imports."""
        print("\nCreating __init__.py files...")
        
        init_files = {
            "src/__init__.py": '''"""Audio Processing Pipeline."""
__version__ = "2.0.0"
''',
            "src/domain/__init__.py": '''"""Domain layer - Business logic and models."""
from src.domain.models import AudioDialog, DialogRow, DialogCriteria
from src.domain.services import (
    TranscriptionService,
    CriteriaDetectionService,
    LLMProcessingService
)

__all__ = [
    "AudioDialog",
    "DialogRow",
    "DialogCriteria",
    "TranscriptionService",
    "CriteriaDetectionService",
    "LLMProcessingService",
]
''',
            "src/domain/models/__init__.py": '''"""Domain models."""
from .audio_dialog import AudioDialog
from .dialog_row import DialogRow
from .dialog_criteria import DialogCriteria

__all__ = ["AudioDialog", "DialogRow", "DialogCriteria"]
''',
            "src/infrastructure/__init__.py": '''"""Infrastructure layer."""
''',
            "src/application/__init__.py": '''"""Application layer."""
from src.application.pipeline import AudioProcessingPipeline

__all__ = ["AudioProcessingPipeline"]
''',
        }
        
        for file_path, content in init_files.items():
            full_path = self.root / file_path
            if self.dry_run:
                print(f"  [DRY RUN] Would create: {file_path}")
            else:
                with open(full_path, 'w') as f:
                    f.write(content)
                print(f"  ✓ Created: {file_path}")
    
    def create_gitignore(self):
        """Create or update .gitignore."""
        gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
data/
logs/
*.log
.DS_Store
.env
config/local.yaml
config/production.yaml

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Models
*.ckpt
*.pth
*.pt
*.joblib

# Temp files
temp/
tmp/
*.tmp
'''
        
        gitignore_path = self.root / ".gitignore"
        if self.dry_run:
            print(f"[DRY RUN] Would create/update: .gitignore")
        else:
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content)
            print(f"✓ Created/updated: .gitignore")
    
    def generate_report(self):
        """Generate migration report."""
        print(f"\n{'=' * 60}")
        print("MIGRATION REPORT")
        print(f"{'=' * 60}\n")
        print(f"Total files to migrate: {len(self.moves)}")
        print(f"Mode: {'DRY RUN (no changes made)' if self.dry_run else 'EXECUTED'}")
        print("\nNext steps:")
        print("1. Review the new structure")
        print("2. Update import statements in files")
        print("3. Run tests to ensure everything works")
        print("4. Update documentation")
        print("5. Commit changes")
        print(f"\n{'=' * 60}\n")
    
    def migrate(self):
        """Run the full migration."""
        print("Starting migration to new structure...")
        print(f"Project root: {self.root}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'EXECUTE'}\n")
        
        self.create_directory_structure()
        self.plan_migrations()
        self.execute_migrations()
        self.create_init_files()
        self.create_gitignore()
        self.generate_report()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate project to new Clean Architecture structure"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing them"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    migrator = StructureMigrator(args.project_root, args.dry_run)
    migrator.migrate()
    
    if args.dry_run:
        print("\n💡 This was a dry run. Run without --dry-run to execute migration.")
    else:
        print("\n✅ Migration complete!")
        print("⚠️  Important: Review changes, update imports, and test thoroughly.")


if __name__ == "__main__":
    main()

