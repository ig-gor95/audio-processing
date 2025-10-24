"""
Criteria Detection Runner (Refactored)

This is a refactored version of the original pd_criteria_detector_runner.py
that uses the new service architecture.

For backward compatibility, the old file is preserved. Use this version
for new code.

Usage:
    python core/pd_criteria_detector_runner_new.py

Or use the CLI:
    python pipeline_cli.py detect-criteria
"""

from core.service.criteria_detection_service import CriteriaDetectionService
from log_utils import setup_logger

logger = setup_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting criteria detection")
    
    # Initialize service
    service = CriteriaDetectionService()
    
    # Process dialogs
    stats = service.process_dialogs()
    
    # Log results
    logger.info(f"Criteria detection completed: {stats}")

