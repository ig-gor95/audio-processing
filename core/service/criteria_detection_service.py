"""
Criteria Detection Service - Handles linguistic analysis of dialog text.

This service orchestrates the detection of various linguistic criteria
including greetings, names, sales patterns, and other text features.
"""

from typing import Optional

from core.post_processors.text_processing.DialogueAnalyzerPandas import DialogueAnalyzerPandas
from log_utils import setup_logger
from yaml_reader import ConfigLoader

logger = setup_logger(__name__)


class CriteriaDetectionService:
    """Service for detecting linguistic criteria in dialog text."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the criteria detection service.
        
        Args:
            config: Optional configuration dictionary. If None, loads from pipeline_config.yaml
        """
        if config is None:
            config_loader = ConfigLoader("../configs/pipeline_config.yaml")
            self.config = config_loader.get_all()
        else:
            self.config = config
        
        # Extract configuration
        self.criteria_config = self.config.get('criteria_detection', {})
        self.enabled = self.criteria_config.get('enabled', True)
        self.batch_size = self.criteria_config.get('batch_size', 1000)
        
        # Initialize analyzer
        logger.info("Initializing DialogueAnalyzerPandas...")
        self.analyzer = DialogueAnalyzerPandas()
        logger.info("DialogueAnalyzerPandas initialized successfully")
    
    def process_dialogs(self, dialog_ids: Optional[list] = None) -> dict:
        """
        Process dialogs for criteria detection.
        
        Args:
            dialog_ids: Optional list of audio_dialog UUIDs to process.
                       If None, processes all unprocessed dialogs.
        
        Returns:
            Dictionary with processing statistics:
            - rows_processed: Number of dialog rows processed
            - success: Whether processing completed successfully
            - dialog_count: Number of dialogs processed
            
        Raises:
            Exception: If processing fails
        """
        if not self.enabled:
            logger.info("Criteria detection is disabled in configuration")
            return {
                'rows_processed': 0,
                'success': False,
                'message': 'Disabled',
                'dialog_count': 0
            }
        
        if dialog_ids:
            logger.info(f"Starting criteria detection for {len(dialog_ids)} specific dialogs")
        else:
            logger.info("Starting criteria detection for all unprocessed dialogs")
        
        try:
            # Run the analyzer
            self.analyzer.analyze_dialogue(dialog_ids)
            
            logger.info("Criteria detection completed successfully")
            
            return {
                'rows_processed': None,  # Could track this if needed
                'success': True,
                'message': 'Completed',
                'dialog_count': len(dialog_ids) if dialog_ids else None
            }
            
        except Exception as e:
            logger.error(f"Error during criteria detection: {str(e)}", exc_info=True)
            raise
    
    def get_analysis_stats(self) -> dict:
        """
        Get statistics about the current analysis state.
        
        Returns:
            Dictionary with analysis statistics
        """
        # This could query the database for statistics
        # For now, return basic info
        return {
            'analyzer_initialized': self.analyzer is not None,
            'enabled': self.enabled,
            'batch_size': self.batch_size
        }

