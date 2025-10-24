"""
Transcription Runner (Refactored)

This is a refactored version of the original transcribe_files_runner.py
that uses the new service architecture.

For backward compatibility, the old file is preserved. Use this version
for new code.

Usage:
    python core/transcribe_files_runner_new.py

Or use the CLI:
    python pipeline_cli.py transcribe --input-folder <path>
"""

import concurrent.futures
import time
from pathlib import Path
from typing import List

from core.service.transcription_service import TranscriptionService
from log_utils import setup_logger
from yaml_reader import ConfigLoader

logger = setup_logger(__name__)


def process_files_parallel(
    audio_files: List[Path],
    max_workers: int = 5,
    max_files: int = 2000,
    config_path: str = None
) -> dict:
    """
    Process audio files in parallel using TranscriptionService.
    
    Args:
        audio_files: List of audio file paths
        max_workers: Number of parallel workers
        max_files: Maximum files to process
        config_path: Optional path to config file
        
    Returns:
        Dictionary with processing statistics
    """
    start_time = time.time()
    
    # Initialize service
    service = TranscriptionService()
    
    # Limit files
    files_to_process = audio_files[:max_files]
    logger.info(f"Processing {len(files_to_process)} files with {max_workers} workers")
    
    # Statistics
    stats = {
        'successful': [],
        'skipped': [],
        'failed': []
    }
    
    # Process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(service.process_audio_file, file): file
            for file in files_to_process
        }
        
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_file), 1):
            file = future_to_file[future]
            try:
                file_uuid = future.result()
                if file_uuid is None:
                    stats['skipped'].append(file.name)
                else:
                    stats['successful'].append(file.name)
                
                logger.info(f"Completed {idx}/{len(files_to_process)}: {file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {file.name}: {str(e)}")
                stats['failed'].append(file.name)
    
    # Summary
    end_time = time.time()
    logger.info(f"Total processed: {len(stats['successful'])}")
    logger.info(f"Total skipped: {len(stats['skipped'])}")
    logger.info(f"Total failed: {len(stats['failed'])}")
    logger.info(f"Execution time: {end_time - start_time:.2f} seconds")
    
    return stats


if __name__ == "__main__":
    # Load configuration
    config = ConfigLoader("../configs/pipeline_config.yaml").get_all()
    
    # Get folder path from config or use default
    folder_path = config.get('pipeline', {}).get(
        'default_input_folder',
        f"{Path.home()}/Documents/Аудио Бринекс/2/"
    )
    
    # Get audio files
    audio_files = list(Path(folder_path).expanduser().glob("*"))
    logger.info(f"Found {len(audio_files)} files in {folder_path}")
    
    # Get settings from config
    max_workers = config.get('pipeline', {}).get('max_workers', 5)
    max_files = config.get('pipeline', {}).get('max_files_per_run', 5000)
    
    # Process
    process_files_parallel(audio_files, max_workers=max_workers, max_files=max_files)

