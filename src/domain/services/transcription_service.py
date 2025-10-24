"""
Transcription Service - Handles audio transcription and diarization pipeline.

This service orchestrates the process of converting audio files to text
with speaker diarization, managing database operations and error handling.
"""

import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import soundfile as sf

from core.audio_to_text.audio_to_text_processor import audio_to_text_processor
from core.post_processors.text_processing.detector.sales_detector import SalesDetector
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.dialog_rows_repository import DialogRowRepository
from core.repository.entity.audio_dialog import AudioDialogStatus, AudioDialog
from core.repository.entity.dialog_rows import DialogRow
from log_utils import setup_logger
from yaml_reader import ConfigLoader

logger = setup_logger(__name__)


class TranscriptionService:
    """Service for handling audio transcription and diarization."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the transcription service.
        
        Args:
            config: Optional configuration dictionary. If None, loads from pipeline_config.yaml
        """
        if config is None:
            config_loader = ConfigLoader("../configs/pipeline_config.yaml")
            self.config = config_loader.get_all()
        else:
            self.config = config
        
        # Initialize repositories
        self.audio_dialog_repo = AudioDialogRepository()
        self.dialog_row_repo = DialogRowRepository()
        self.dialog_criteria_repo = DialogCriteriaRepository()
        
        # Initialize detectors
        self.sales_detector = SalesDetector()
        
        # Extract configuration
        self.transcription_config = self.config.get('transcription', {})
        self.reprocess_before_date = datetime.strptime(
            self.transcription_config.get('reprocess_before_date', '2025-09-06'),
            '%Y-%m-%d'
        )
        self.skip_processed = self.transcription_config.get('skip_processed', True)
    
    def get_audio_duration(self, audio_file: Path) -> float:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Duration in seconds, or 0.0 on error
        """
        try:
            with sf.SoundFile(str(audio_file)) as f:
                duration = len(f) / f.samplerate
            logger.debug(f"Audio duration for {audio_file.name}: {duration:.2f}s")
            return duration
        except Exception as e:
            logger.error(f"Error getting duration for {audio_file.name}: {str(e)}")
            return 0.0
    
    def should_process_file(self, audio_file: Path) -> tuple[bool, Optional[uuid.UUID]]:
        """
        Determine if an audio file should be processed.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Tuple of (should_process, file_uuid)
            - should_process: True if file should be processed
            - file_uuid: UUID for the dialog (existing or new)
        """
        existing_dialog = self.audio_dialog_repo.find_by_filename(audio_file.name)
        
        # New file - create entry and process
        if existing_dialog is None:
            file_uuid = uuid.uuid4()
            self.audio_dialog_repo.save(
                AudioDialog(
                    id=file_uuid,
                    file_name=audio_file.name,
                    duration=self.get_audio_duration(audio_file),
                    status=AudioDialogStatus.NOT_PROCESSED
                )
            )
            logger.info(f"New file registered: {audio_file.name}")
            return True, file_uuid
        
        # Already processed - check if reprocessing needed
        if existing_dialog.status == AudioDialogStatus.PROCESSED:
            if not self.skip_processed and existing_dialog.updated_at < self.reprocess_before_date:
                logger.info(f"Reprocessing old dialog: {audio_file.name}")
                self._clean_existing_dialog_data(existing_dialog.id)
                return True, existing_dialog.id
            else:
                logger.debug(f"Skipping already processed file: {audio_file.name}")
                return False, existing_dialog.id
        
        # Partial or failed processing - continue
        logger.info(f"Continuing processing for: {audio_file.name}")
        return True, existing_dialog.id
    
    def _clean_existing_dialog_data(self, dialog_id: uuid.UUID):
        """
        Clean existing dialog data before reprocessing.
        
        Args:
            dialog_id: UUID of the dialog to clean
        """
        rows = self.dialog_row_repo.find_by_dialog_id(dialog_id)
        for row in rows:
            self.dialog_criteria_repo.delete_by_dialog_row_fk_id(row.id)
        self.dialog_row_repo.delete_all_by_dialog_id(dialog_id)
        logger.debug(f"Cleaned data for dialog {dialog_id}")
    
    def _set_threading_environment(self):
        """Set threading environment variables for optimal performance."""
        os.environ["OMP_NUM_THREADS"] = str(self.transcription_config.get('omp_num_threads', 1))
        os.environ["MKL_NUM_THREADS"] = str(self.transcription_config.get('mkl_num_threads', 1))
    
    def process_audio_file(self, audio_file: Path) -> Optional[uuid.UUID]:
        """
        Process a single audio file through transcription and diarization.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            UUID of the processed dialog, or None if skipped
            
        Raises:
            Exception: If processing fails
        """
        # Set threading environment
        self._set_threading_environment()
        
        # Check if file should be processed
        should_process, file_uuid = self.should_process_file(audio_file)
        if not should_process:
            return None
        
        logger.info(f"Starting transcription for: {audio_file.name}")
        start_time = time.time()
        
        try:
            # Run audio-to-text processing
            result = audio_to_text_processor(audio_file)
            
            # Convert results to dialog rows
            dialog_rows = [
                DialogRow(
                    audio_dialog_fk_id=file_uuid,
                    row_num=r.phrase_id,
                    row_text=r.text.replace("Продолжение следует...", '')
                                   .replace('Субтитры сделал DimaTorzok', ''),
                    speaker_id=r.speaker_id,
                    start=r.start_time,
                    end=r.end_time
                )
                for r in result.items
            ]
            
            # Run sales detection
            self.sales_detector(dialog_rows)
            
            # Save to database
            self.dialog_row_repo.save_bulk(dialog_rows)
            
            # Update dialog status
            execution_time = time.time() - start_time
            self.audio_dialog_repo.update_status(
                file_uuid,
                AudioDialogStatus.PROCESSED,
                execution_time
            )
            
            logger.info(f"Completed transcription for: {audio_file.name} in {execution_time:.2f}s")
            
            # Log dialog rows (optional, can be disabled for performance)
            if logger.level <= 10:  # DEBUG level
                for row in sorted(dialog_rows, key=lambda x: x.row_num):
                    row.print()
            
            return file_uuid
            
        except Exception as e:
            logger.error(f"Error processing {audio_file.name}: {str(e)}", exc_info=True)
            # Update status to failed
            self.audio_dialog_repo.update_status(
                file_uuid,
                AudioDialogStatus.NOT_PROCESSED,
                time.time() - start_time
            )
            raise
    
    def process_multiple_files(self, audio_files: list[Path]) -> dict[str, list]:
        """
        Process multiple audio files and return results summary.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            Dictionary with 'successful', 'skipped', and 'failed' file lists
        """
        results = {
            'successful': [],
            'skipped': [],
            'failed': []
        }
        
        for audio_file in audio_files:
            try:
                file_uuid = self.process_audio_file(audio_file)
                if file_uuid is None:
                    results['skipped'].append(audio_file.name)
                else:
                    results['successful'].append(audio_file.name)
            except Exception as e:
                logger.error(f"Failed to process {audio_file.name}: {str(e)}")
                results['failed'].append(audio_file.name)
        
        return results

