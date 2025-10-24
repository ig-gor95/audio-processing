"""
LLM Detection Runner (Refactored)

This is a refactored version of the original llm_detector_runner.py
that uses the new service architecture.

For backward compatibility, the old file is preserved. Use this version
for new code.

Usage:
    python core/llm_detector_runner_new.py

Or use the CLI:
    python pipeline_cli.py llm-process
"""

from core.service.llm_processing_service import LLMProcessingService
from log_utils import setup_logger

logger = setup_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting LLM processing")
    
    # Initialize service
    service = LLMProcessingService()
    
    # Process all dialogs
    stats = service.process_all_dialogs()
    
    # Log results
    logger.info(f"LLM processing completed:")
    logger.info(f"  - Total: {stats['total']}")
    logger.info(f"  - Processed: {stats['processed']}")
    logger.info(f"  - Skipped: {stats['skipped']}")
    logger.info(f"  - Failed: {stats['failed']}")

