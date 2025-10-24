"""
LLM Processing Service - Handles LLM-based analysis of dialog transcripts.

This service orchestrates the processing of dialog text through LLM
(Large Language Model) for advanced analysis and extraction.
"""

import json
import re
import time
from typing import Optional
import uuid

from core.post_processors.llm_processing.objections_resolver import resolve_llm_data
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.service.dialog_row_util_service import print_dialog_to_text
from log_utils import setup_logger
from yaml_reader import ConfigLoader

logger = setup_logger(__name__)


class LLMProcessingService:
    """Service for LLM-based dialog analysis."""
    
    # Regex patterns for JSON extraction
    _CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the LLM processing service.
        
        Args:
            config: Optional configuration dictionary. If None, loads from pipeline_config.yaml
        """
        if config is None:
            config_loader = ConfigLoader("../configs/pipeline_config.yaml")
            self.config = config_loader.get_all()
        else:
            self.config = config
        
        # Initialize repository
        self.audio_dialog_repo = AudioDialogRepository()
        
        # Extract configuration
        self.llm_config = self.config.get('llm_processing', {})
        self.skip_existing = self.llm_config.get('skip_existing', True)
        self.skip_indices = set(self.llm_config.get('skip_indices', []))
        self.min_dialog_length = self.llm_config.get('min_dialog_length', 5)
        self.max_retries = self.llm_config.get('max_retries', 3)
        self.retry_delay = self.llm_config.get('retry_delay_seconds', 5)
        self.strip_code_fences = self.llm_config.get('strip_code_fences', True)
        self.extract_first_json = self.llm_config.get('extract_first_json_blob', True)
    
    def _strip_code_fences(self, text: str) -> str:
        """
        Remove markdown code fences from text.
        
        Args:
            text: Text potentially containing code fences
            
        Returns:
            Text with code fences removed
        """
        return self._CODE_FENCE_RE.sub("", text).strip()
    
    def _extract_first_json_blob(self, text: str) -> Optional[str]:
        """
        Extract the first complete JSON object or array from text.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            First complete JSON blob, or None if not found
        """
        # Find first opening brace or bracket
        start = None
        for i, ch in enumerate(text):
            if ch in "{[":
                start = i
                break
        
        if start is None:
            return None
        
        # Track nested braces/brackets
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        
        for j in range(start, len(text)):
            c = text[j]
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return text[start:j+1]
        
        return None
    
    def _parse_llm_response(self, raw_text: str) -> dict:
        """
        Parse LLM response text into a dictionary.
        
        Attempts multiple strategies:
        1. Parse as-is
        2. Strip code fences and parse
        3. Extract first JSON blob and parse
        
        Args:
            raw_text: Raw text response from LLM
            
        Returns:
            Parsed dictionary, or empty dict if parsing fails
        """
        if raw_text is None:
            logger.warning("LLM returned None")
            return {}
        
        # Try different parsing strategies
        candidates = [raw_text]
        
        if self.strip_code_fences:
            candidates.append(self._strip_code_fences(raw_text))
        
        if self.extract_first_json:
            blob = self._extract_first_json_blob(candidates[-1])
            if blob:
                candidates.append(blob)
        
        # Try parsing each candidate
        for i, candidate in enumerate(candidates):
            try:
                val = json.loads(candidate)
                
                # Return dict as-is, wrap arrays
                if isinstance(val, dict):
                    logger.debug(f"Successfully parsed JSON (strategy {i+1})")
                    return val
                if isinstance(val, list):
                    logger.debug(f"Successfully parsed JSON array, wrapping (strategy {i+1})")
                    return {"data": val}
                    
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse failed for candidate {i+1}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error parsing candidate {i+1}: {e}")
                continue
        
        logger.warning("All JSON parsing strategies failed, returning empty dict")
        return {}
    
    def _process_single_dialog_with_retry(
        self,
        dialog_id: uuid.UUID,
        dialog_text: str
    ) -> Optional[dict]:
        """
        Process a single dialog with retry logic.
        
        Args:
            dialog_id: UUID of the dialog
            dialog_text: Text content of the dialog
            
        Returns:
            Parsed LLM response dict, or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM processing attempt {attempt + 1}/{self.max_retries} for dialog {dialog_id}")
                
                # Call LLM
                llm_response = resolve_llm_data(dialog_text)
                
                # Parse response
                llm_data = self._parse_llm_response(llm_response)
                
                return llm_data
                
            except Exception as e:
                logger.warning(f"LLM processing attempt {attempt + 1} failed: {str(e)}")
                
                # Wait before retry (except on last attempt)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All retry attempts exhausted for dialog {dialog_id}")
                    return None
        
        return None
    
    def process_dialog(self, dialog_id: uuid.UUID) -> bool:
        """
        Process a single dialog through LLM.
        
        Args:
            dialog_id: UUID of the dialog to process
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get dialog text
            dialog_text = print_dialog_to_text(dialog_id)
            
            # Skip if too short
            if len(dialog_text) < self.min_dialog_length:
                logger.debug(f"Skipping dialog {dialog_id} - too short ({len(dialog_text)} chars)")
                return False
            
            # Process with retry
            llm_data = self._process_single_dialog_with_retry(dialog_id, dialog_text)
            
            if llm_data is None:
                return False
            
            # Save to database
            self.audio_dialog_repo.update_llm_data(dialog_id, llm_data)
            logger.info(f"Successfully processed dialog {dialog_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing dialog {dialog_id}: {str(e)}", exc_info=True)
            return False
    
    def process_all_dialogs(self) -> dict:
        """
        Process all dialogs that need LLM processing.
        
        Returns:
            Dictionary with processing statistics:
            - total: Total dialogs found
            - processed: Number successfully processed
            - skipped: Number skipped (already processed or in skip list)
            - failed: Number that failed processing
        """
        # Get all dialogs
        dialogs = self.audio_dialog_repo.find_all()
        total = len(dialogs)
        
        logger.info(f"Found {total} dialogs to process")
        
        stats = {
            'total': total,
            'processed': 0,
            'skipped': 0,
            'failed': 0
        }
        
        for idx, dialog in enumerate(dialogs, start=1):
            # Skip if already has LLM data
            if self.skip_existing and dialog.llm_data_short is not None:
                logger.debug(f"[{idx}/{total}] Skipping dialog {dialog.id} - already processed")
                stats['skipped'] += 1
                continue
            
            # Skip if in skip list
            if idx in self.skip_indices:
                logger.info(f"[{idx}/{total}] Skipping dialog {dialog.id} - in skip list")
                stats['skipped'] += 1
                continue
            
            logger.info(f"[{idx}/{total}] Processing dialog {dialog.id}")
            
            # Process dialog
            success = self.process_dialog(dialog.id)
            
            if success:
                stats['processed'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"LLM processing complete: {stats}")
        return stats
    
    def get_processing_stats(self) -> dict:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary with processing state statistics
        """
        all_dialogs = self.audio_dialog_repo.find_all()
        total = len(all_dialogs)
        
        processed = sum(1 for d in all_dialogs if d.llm_data_short is not None)
        pending = total - processed
        
        return {
            'total_dialogs': total,
            'processed': processed,
            'pending': pending,
            'skip_existing': self.skip_existing,
            'skip_indices_count': len(self.skip_indices)
        }

