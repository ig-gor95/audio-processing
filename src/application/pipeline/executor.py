"""
Audio Processing Pipeline Orchestrator.

This module provides a unified interface for running the complete audio processing
pipeline including transcription, criteria detection, and LLM processing.
"""

import concurrent.futures
import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

from core.service.transcription_service import TranscriptionService
from core.service.criteria_detection_service import CriteriaDetectionService
from core.service.llm_processing_service import LLMProcessingService
from log_utils import setup_logger
from yaml_reader import ConfigLoader

logger = setup_logger(__name__)


class PipelineStage(Enum):
    """Enum for pipeline stages."""
    TRANSCRIPTION = "transcription"
    CRITERIA_DETECTION = "criteria_detection"
    LLM_PROCESSING = "llm_processing"


class PipelineResult:
    """Container for pipeline execution results."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.end_time = None
        self.stages_completed = []
        self.stages_failed = []
        self.transcription_stats = {}
        self.criteria_stats = {}
        self.llm_stats = {}
        self.errors = []
    
    def mark_stage_complete(self, stage: PipelineStage, stats: dict = None):
        """Mark a stage as complete."""
        self.stages_completed.append(stage.value)
        if stats:
            if stage == PipelineStage.TRANSCRIPTION:
                self.transcription_stats = stats
            elif stage == PipelineStage.CRITERIA_DETECTION:
                self.criteria_stats = stats
            elif stage == PipelineStage.LLM_PROCESSING:
                self.llm_stats = stats
    
    def mark_stage_failed(self, stage: PipelineStage, error: str):
        """Mark a stage as failed."""
        self.stages_failed.append(stage.value)
        self.errors.append({
            'stage': stage.value,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    
    def finalize(self):
        """Finalize the result."""
        self.end_time = datetime.now()
    
    def get_duration(self) -> float:
        """Get total duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.get_duration(),
            'stages_completed': self.stages_completed,
            'stages_failed': self.stages_failed,
            'transcription_stats': self.transcription_stats,
            'criteria_stats': self.criteria_stats,
            'llm_stats': self.llm_stats,
            'errors': self.errors
        }
    
    def __str__(self) -> str:
        """String representation."""
        return json.dumps(self.to_dict(), indent=2)


class AudioProcessingPipeline:
    """
    Orchestrates the complete audio processing pipeline.
    
    The pipeline consists of three stages:
    1. Transcription: Audio to text with speaker diarization
    2. Criteria Detection: Linguistic analysis of dialog text
    3. LLM Processing: Advanced analysis using language models
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        # Load configuration
        if config_path:
            config_loader = ConfigLoader(config_path)
        else:
            config_loader = ConfigLoader("../configs/pipeline_config.yaml")
        
        self.config = config_loader.get_all()
        self.pipeline_config = self.config.get('pipeline', {})
        self.error_config = self.config.get('error_handling', {})
        
        # Initialize services
        logger.info("Initializing pipeline services...")
        self.transcription_service = TranscriptionService(self.config)
        self.criteria_service = CriteriaDetectionService(self.config)
        self.llm_service = LLMProcessingService(self.config)
        logger.info("Pipeline services initialized successfully")
        
        # Pipeline settings
        self.max_workers = self.pipeline_config.get('max_workers', 5)
        self.continue_on_error = self.error_config.get('continue_on_error', True)
        self.save_error_log = self.error_config.get('save_error_log', True)
        self.error_log_path = self.error_config.get('error_log_path', 'logs/errors.json')
    
    def _save_error_log(self, result: PipelineResult):
        """Save error log to file."""
        if not self.save_error_log or not result.errors:
            return
        
        try:
            error_log_path = Path(self.error_log_path)
            error_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing errors if file exists
            existing_errors = []
            if error_log_path.exists():
                try:
                    with open(error_log_path, 'r') as f:
                        existing_errors = json.load(f)
                except Exception:
                    pass
            
            # Append new errors
            existing_errors.extend(result.errors)
            
            # Save
            with open(error_log_path, 'w') as f:
                json.dump(existing_errors, f, indent=2)
            
            logger.info(f"Error log saved to {error_log_path}")
            
        except Exception as e:
            logger.error(f"Failed to save error log: {str(e)}")
    
    def run_transcription_stage(
        self,
        audio_files: List[Path],
        result: PipelineResult
    ) -> bool:
        """
        Run the transcription stage.
        
        Args:
            audio_files: List of audio files to process
            result: Pipeline result object to update
            
        Returns:
            True if stage completed successfully
        """
        logger.info(f"=== Stage 1: Transcription ({len(audio_files)} files) ===")
        
        try:
            max_files = self.pipeline_config.get('max_files_per_run', 5000)
            files_to_process = audio_files[:max_files]
            
            logger.info(f"Processing {len(files_to_process)} files with {self.max_workers} workers")
            
            # Process files in parallel
            stats = {'successful': [], 'skipped': [], 'failed': []}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.transcription_service.process_audio_file, file): file
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
                        
                        logger.info(f"Progress: {idx}/{len(files_to_process)} files")
                        
                    except Exception as e:
                        error_msg = f"Error processing {file.name}: {str(e)}"
                        logger.error(error_msg)
                        stats['failed'].append(file.name)
                        
                        if not self.continue_on_error:
                            raise
            
            result.mark_stage_complete(PipelineStage.TRANSCRIPTION, stats)
            logger.info(f"Transcription complete: {len(stats['successful'])} successful, "
                       f"{len(stats['skipped'])} skipped, {len(stats['failed'])} failed")
            
            return len(stats['successful']) > 0 or len(stats['skipped']) > 0
            
        except Exception as e:
            error_msg = f"Transcription stage failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.mark_stage_failed(PipelineStage.TRANSCRIPTION, error_msg)
            return False
    
    def run_criteria_detection_stage(self, result: PipelineResult) -> bool:
        """
        Run the criteria detection stage.
        
        Args:
            result: Pipeline result object to update
            
        Returns:
            True if stage completed successfully
        """
        logger.info("=== Stage 2: Criteria Detection ===")
        
        try:
            stats = self.criteria_service.process_dialogs()
            result.mark_stage_complete(PipelineStage.CRITERIA_DETECTION, stats)
            logger.info("Criteria detection complete")
            return stats.get('success', False)
            
        except Exception as e:
            error_msg = f"Criteria detection stage failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.mark_stage_failed(PipelineStage.CRITERIA_DETECTION, error_msg)
            return False
    
    def run_llm_processing_stage(self, result: PipelineResult) -> bool:
        """
        Run the LLM processing stage.
        
        Args:
            result: Pipeline result object to update
            
        Returns:
            True if stage completed successfully
        """
        logger.info("=== Stage 3: LLM Processing ===")
        
        try:
            stats = self.llm_service.process_all_dialogs()
            result.mark_stage_complete(PipelineStage.LLM_PROCESSING, stats)
            logger.info(f"LLM processing complete: {stats['processed']} processed, "
                       f"{stats['skipped']} skipped, {stats['failed']} failed")
            return stats['processed'] > 0 or stats['skipped'] > 0
            
        except Exception as e:
            error_msg = f"LLM processing stage failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.mark_stage_failed(PipelineStage.LLM_PROCESSING, error_msg)
            return False
    
    def run_full_pipeline(
        self,
        audio_folder: Optional[str] = None,
        audio_files: Optional[List[Path]] = None,
        stages: Optional[List[PipelineStage]] = None
    ) -> PipelineResult:
        """
        Run the complete pipeline or selected stages.
        
        Args:
            audio_folder: Path to folder containing audio files (for transcription)
            audio_files: List of specific audio files to process (overrides audio_folder)
            stages: List of stages to run. If None, runs all stages.
            
        Returns:
            PipelineResult with execution details
        """
        result = PipelineResult()
        
        # Default to all stages
        if stages is None:
            stages = [
                PipelineStage.TRANSCRIPTION,
                PipelineStage.CRITERIA_DETECTION,
                PipelineStage.LLM_PROCESSING
            ]
        
        logger.info(f"Starting pipeline with stages: {[s.value for s in stages]}")
        
        try:
            # Stage 1: Transcription
            if PipelineStage.TRANSCRIPTION in stages:
                # Get audio files
                if audio_files is None:
                    if audio_folder is None:
                        audio_folder = self.pipeline_config.get('default_input_folder')
                    
                    if audio_folder:
                        folder_path = Path(audio_folder).expanduser()
                        audio_files = list(folder_path.glob("*"))
                        logger.info(f"Found {len(audio_files)} files in {folder_path}")
                    else:
                        logger.warning("No audio folder or files specified, skipping transcription")
                        audio_files = []
                
                if audio_files:
                    success = self.run_transcription_stage(audio_files, result)
                    if not success and not self.continue_on_error:
                        result.finalize()
                        return result
            
            # Stage 2: Criteria Detection
            if PipelineStage.CRITERIA_DETECTION in stages:
                success = self.run_criteria_detection_stage(result)
                if not success and not self.continue_on_error:
                    result.finalize()
                    return result
            
            # Stage 3: LLM Processing
            if PipelineStage.LLM_PROCESSING in stages:
                success = self.run_llm_processing_stage(result)
                if not success and not self.continue_on_error:
                    result.finalize()
                    return result
            
            result.finalize()
            logger.info(f"Pipeline completed in {result.get_duration():.2f} seconds")
            
            # Save error log if needed
            self._save_error_log(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            result.finalize()
            self._save_error_log(result)
            raise
    
    def run_single_stage(
        self,
        stage: PipelineStage,
        audio_folder: Optional[str] = None,
        audio_files: Optional[List[Path]] = None
    ) -> PipelineResult:
        """
        Run a single pipeline stage.
        
        Args:
            stage: The stage to run
            audio_folder: Path to audio folder (for transcription stage)
            audio_files: List of audio files (for transcription stage)
            
        Returns:
            PipelineResult with execution details
        """
        return self.run_full_pipeline(
            audio_folder=audio_folder,
            audio_files=audio_files,
            stages=[stage]
        )

