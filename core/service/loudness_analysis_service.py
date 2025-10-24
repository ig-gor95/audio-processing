"""
Loudness Analysis Service - Handles audio loudness analysis for dialogs.

This service orchestrates the process of analyzing loudness characteristics
for dialog rows, including mean, max, min loudness, dynamic range, and percentiles.
"""

import multiprocessing as mp
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict

from core.post_processors.audio_processing.loudness_analyzer import LoudnessAnalyzer
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.repository.dialog_rows_repository import DialogRowRepository
from log_utils import setup_logger
from yaml_reader import ConfigLoader

logger = setup_logger(__name__)


def _process_single_file_worker(args):
    """
    Worker function for parallel processing.
    
    This function is defined at module level to be picklable for multiprocessing.
    
    Args:
        args: Tuple of (audio_path, phrases, analyzer_config)
        
    Returns:
        Tuple of (audio_path, results, error_message)
    """
    audio_path, phrases, analyzer_config = args
    analyzer = LoudnessAnalyzer(**analyzer_config)
    
    try:
        results = analyzer.process_file(audio_path, phrases)
        return audio_path, results, None
    except Exception as e:
        return audio_path, [], str(e)


class LoudnessAnalysisService:
    """Service for handling audio loudness analysis."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the loudness analysis service.
        
        Args:
            config: Optional configuration dictionary. If None, loads from pipeline_config.yaml
        """
        if config is None:
            config_loader = ConfigLoader("../configs/pipeline_config.yaml")
            self.config = config_loader.get_all()
        else:
            self.config = config
        
        self.audio_dialog_repo = AudioDialogRepository()
        self.dialog_row_repo = DialogRowRepository()
        
        loudness_config = self.config.get('loudness_analysis', {})
        self.sample_rate = loudness_config.get('sample_rate', 16000)
        self.hop_length = loudness_config.get('hop_length', 256)
        self.max_workers = loudness_config.get('max_workers', mp.cpu_count())
        self.batch_size = loudness_config.get('batch_size', 1000)
        self.skip_existing = loudness_config.get('skip_existing', True)
        
        logger.info("LoudnessAnalysisService initialized")
    
    def _get_phrases_for_dialog(self, dialog_id: uuid.UUID, audio_path: Path) -> List[Dict]:
        """
        Get phrases (dialog rows) that need loudness analysis.
        
        Args:
            dialog_id: UUID of the dialog
            audio_path: Path to the audio file
            
        Returns:
            List of phrase dictionaries with start_time, end_time, and row_id
        """
        rows = self.dialog_row_repo.find_by_dialog_id(dialog_id)
        phrases = []
        
        for row in rows:
            if self.skip_existing and row.mean_loudness is not None:
                logger.debug(f"Skipping row {row.id} - already has loudness data")
                continue
            
            phrases.append({
                'start_time': row.start,
                'end_time': row.end,
                'row_id': row.id
            })
        
        logger.debug(f"Found {len(phrases)} phrases to analyze for dialog {dialog_id}")
        return phrases
    
    def process_dialog(self, dialog_id: uuid.UUID, audio_path: Path) -> bool:
        """
        Process a single dialog for loudness analysis.
        
        Args:
            dialog_id: UUID of the dialog
            audio_path: Path to the audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            phrases = self._get_phrases_for_dialog(dialog_id, audio_path)
            
            if not phrases:
                logger.info(f"No phrases to analyze for dialog {dialog_id}")
                return True
            
            logger.info(f"Analyzing loudness for dialog {dialog_id}: {len(phrases)} phrases")
            start_time = time.time()
            
            analyzer = LoudnessAnalyzer(
                sample_rate=self.sample_rate,
                hop_length=self.hop_length
            )
            
            results = analyzer.process_file(str(audio_path), phrases)
            
            elapsed = time.time() - start_time
            logger.info(f"Completed loudness analysis for dialog {dialog_id} in {elapsed:.2f}s")
            
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"Error analyzing loudness for dialog {dialog_id}: {str(e)}", exc_info=True)
            return False
    
    def process_dialogs(self, dialog_ids: Optional[List[uuid.UUID]] = None) -> dict:
        """
        Process multiple dialogs for loudness analysis using multiprocessing.
        
        Args:
            dialog_ids: Optional list of dialog UUIDs to process.
                       If None, processes all dialogs with unanalyzed rows.
        
        Returns:
            Dictionary with processing statistics:
            - total: Total dialogs to process
            - processed: Number successfully processed
            - skipped: Number skipped (no phrases to analyze)
            - failed: Number that failed
        """
        logger.info("Starting loudness analysis for dialogs")
        start_time = time.time()
        
        file_phrase_mapping = {}
        
        if dialog_ids:
            dialogs = [self.audio_dialog_repo.find_by_id(did) for did in dialog_ids]
            dialogs = [d for d in dialogs if d is not None]
        else:
            dialogs = self.audio_dialog_repo.find_all()
        
        logger.info(f"Preparing {len(dialogs)} dialogs for loudness analysis")
        
        for dialog in dialogs:
            phrases = self._get_phrases_for_dialog(dialog.id, Path(dialog.file_name))
            
            if phrases:
                audio_path = self._find_audio_file(dialog.file_name)
                if audio_path and audio_path.exists():
                    file_phrase_mapping[str(audio_path)] = phrases
                else:
                    logger.warning(f"Audio file not found for dialog {dialog.id}: {dialog.file_name}")
        
        if not file_phrase_mapping:
            logger.info("No dialogs need loudness analysis")
            return {
                'total': len(dialogs),
                'processed': 0,
                'skipped': len(dialogs),
                'failed': 0
            }
        
        logger.info(f"Processing {len(file_phrase_mapping)} audio files in parallel")
        
        analyzer_config = {
            'sample_rate': self.sample_rate,
            'hop_length': self.hop_length
        }
        
        stats = {
            'total': len(file_phrase_mapping),
            'processed': 0,
            'skipped': 0,
            'failed': 0
        }
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for audio_path, phrases in file_phrase_mapping.items():
                future = executor.submit(
                    _process_single_file_worker,
                    (audio_path, phrases, analyzer_config)
                )
                futures[future] = audio_path
            
            for future in as_completed(futures):
                audio_path = futures[future]
                try:
                    audio_path_result, results, error = future.result()
                    
                    if error:
                        logger.error(f"Failed to process {audio_path}: {error}")
                        stats['failed'] += 1
                    elif results:
                        stats['processed'] += 1
                        if stats['processed'] % 10 == 0:
                            logger.info(f"Progress: {stats['processed']}/{stats['total']} files analyzed")
                    else:
                        stats['skipped'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {audio_path}: {str(e)}")
                    stats['failed'] += 1
        
        elapsed = time.time() - start_time
        logger.info(f"Loudness analysis completed in {elapsed:.2f}s")
        logger.info(f"Results: {stats}")
        
        return stats
    
    def _find_audio_file(self, filename: str) -> Optional[Path]:
        """
        Find audio file path from filename.
        
        This is a simplified version - you may need to customize based on your file structure.
        
        Args:
            filename: Name of the audio file
            
        Returns:
            Path to audio file or None if not found
        """
        common_locations = [
            Path.home() / "Documents" / "Аудио Бринекс" / "2",
            Path.home() / "Documents" / "Аудио Бринекс" / "4",
            Path("./data/input"),
        ]
        
        for location in common_locations:
            if location.exists():
                audio_path = location / filename
                if audio_path.exists():
                    return audio_path
        
        logger.debug(f"Could not find audio file: {filename}")
        return None
    
    def get_analysis_stats(self) -> dict:
        """
        Get statistics about loudness analysis coverage.
        
        Returns:
            Dictionary with analysis statistics
        """
        all_rows = self.dialog_row_repo.find_all()
        
        total_rows = len(all_rows)
        analyzed_rows = sum(1 for row in all_rows if row.mean_loudness is not None)
        pending_rows = total_rows - analyzed_rows
        
        return {
            'total_rows': total_rows,
            'analyzed': analyzed_rows,
            'pending': pending_rows,
            'coverage_percent': (analyzed_rows / total_rows * 100) if total_rows > 0 else 0
        }

