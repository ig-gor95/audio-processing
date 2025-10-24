#!/usr/bin/env python3
"""
Audio Processing Pipeline CLI

Command-line interface for running the audio processing pipeline.
Supports running individual stages or the full pipeline.

Examples:
    # Run full pipeline
    python pipeline_cli.py run-all --input-folder ~/audio_files

    # Run only transcription
    python pipeline_cli.py transcribe --input-folder ~/audio_files

    # Run only criteria detection
    python pipeline_cli.py detect-criteria

    # Run only LLM processing
    python pipeline_cli.py llm-process

    # Check pipeline status
    python pipeline_cli.py status
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from core.pipeline import AudioProcessingPipeline, PipelineStage, PipelineResult
from log_utils import setup_logger

logger = setup_logger(__name__)


def format_result_summary(result: PipelineResult) -> str:
    """Format pipeline result as a readable summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("PIPELINE EXECUTION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Duration: {result.get_duration():.2f} seconds")
    lines.append(f"Completed stages: {', '.join(result.stages_completed) or 'None'}")
    lines.append(f"Failed stages: {', '.join(result.stages_failed) or 'None'}")
    lines.append("")
    
    # Transcription stats
    if result.transcription_stats:
        stats = result.transcription_stats
        lines.append("Transcription:")
        lines.append(f"  - Successful: {len(stats.get('successful', []))}")
        lines.append(f"  - Skipped: {len(stats.get('skipped', []))}")
        lines.append(f"  - Failed: {len(stats.get('failed', []))}")
        lines.append("")
    
    # Criteria detection stats
    if result.criteria_stats:
        stats = result.criteria_stats
        lines.append("Criteria Detection:")
        lines.append(f"  - Success: {stats.get('success', False)}")
        if stats.get('rows_processed'):
            lines.append(f"  - Rows processed: {stats['rows_processed']}")
        lines.append("")
    
    # LLM processing stats
    if result.llm_stats:
        stats = result.llm_stats
        lines.append("LLM Processing:")
        lines.append(f"  - Total dialogs: {stats.get('total', 0)}")
        lines.append(f"  - Processed: {stats.get('processed', 0)}")
        lines.append(f"  - Skipped: {stats.get('skipped', 0)}")
        lines.append(f"  - Failed: {stats.get('failed', 0)}")
        lines.append("")
    
    # Errors
    if result.errors:
        lines.append("ERRORS:")
        for error in result.errors:
            lines.append(f"  - [{error['stage']}] {error['error']}")
        lines.append("")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def run_full_pipeline(args) -> int:
    """Run the complete pipeline."""
    try:
        logger.info("Starting full pipeline execution")
        
        pipeline = AudioProcessingPipeline(args.config)
        
        # Determine input
        audio_folder = args.input_folder or pipeline.pipeline_config.get('default_input_folder')
        
        if not audio_folder:
            logger.error("No input folder specified. Use --input-folder or set default_input_folder in config.")
            return 1
        
        # Run pipeline
        result = pipeline.run_full_pipeline(audio_folder=audio_folder)
        
        # Print summary
        print("\n" + format_result_summary(result))
        
        # Return 0 if no errors, 1 otherwise
        return 0 if not result.errors else 1
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        return 1


def run_transcription(args) -> int:
    """Run only the transcription stage."""
    try:
        logger.info("Starting transcription stage")
        
        pipeline = AudioProcessingPipeline(args.config)
        
        # Determine input
        audio_folder = args.input_folder or pipeline.pipeline_config.get('default_input_folder')
        
        if not audio_folder:
            logger.error("No input folder specified. Use --input-folder or set default_input_folder in config.")
            return 1
        
        # Run stage
        result = pipeline.run_single_stage(
            PipelineStage.TRANSCRIPTION,
            audio_folder=audio_folder
        )
        
        # Print summary
        print("\n" + format_result_summary(result))
        
        return 0 if not result.errors else 1
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        return 1


def run_criteria_detection(args) -> int:
    """Run only the criteria detection stage."""
    try:
        logger.info("Starting criteria detection stage")
        
        pipeline = AudioProcessingPipeline(args.config)
        
        # Run stage
        result = pipeline.run_single_stage(PipelineStage.CRITERIA_DETECTION)
        
        # Print summary
        print("\n" + format_result_summary(result))
        
        return 0 if not result.errors else 1
        
    except Exception as e:
        logger.error(f"Criteria detection failed: {str(e)}", exc_info=True)
        return 1


def run_llm_processing(args) -> int:
    """Run only the LLM processing stage."""
    try:
        logger.info("Starting LLM processing stage")
        
        pipeline = AudioProcessingPipeline(args.config)
        
        # Run stage
        result = pipeline.run_single_stage(PipelineStage.LLM_PROCESSING)
        
        # Print summary
        print("\n" + format_result_summary(result))
        
        return 0 if not result.errors else 1
        
    except Exception as e:
        logger.error(f"LLM processing failed: {str(e)}", exc_info=True)
        return 1


def show_status(args) -> int:
    """Show current pipeline status."""
    try:
        from core.service.transcription_service import TranscriptionService
        from core.service.llm_processing_service import LLMProcessingService
        from core.repository.audio_dialog_repository import AudioDialogRepository
        from core.repository.dialog_criteria_repository import DialogCriteriaRepository
        
        print("=" * 60)
        print("PIPELINE STATUS")
        print("=" * 60)
        
        # Audio dialogs
        audio_repo = AudioDialogRepository()
        all_dialogs = audio_repo.find_all()
        processed = sum(1 for d in all_dialogs if d.status.value == 'PROCESSED')
        
        print(f"\nAudio Dialogs:")
        print(f"  - Total: {len(all_dialogs)}")
        print(f"  - Processed: {processed}")
        print(f"  - Pending: {len(all_dialogs) - processed}")
        
        # Criteria detection
        criteria_repo = DialogCriteriaRepository()
        try:
            unprocessed = criteria_repo.pd_get_all_unprocessed_rows()
            print(f"\nCriteria Detection:")
            print(f"  - Unprocessed rows: {len(unprocessed)}")
        except Exception as e:
            print(f"\nCriteria Detection:")
            print(f"  - Status check failed: {str(e)}")
        
        # LLM processing
        llm_processed = sum(1 for d in all_dialogs if d.llm_data_short is not None)
        print(f"\nLLM Processing:")
        print(f"  - Processed: {llm_processed}")
        print(f"  - Pending: {len(all_dialogs) - llm_processed}")
        
        print("\n" + "=" * 60)
        return 0
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}", exc_info=True)
        return 1


def validate_config(args) -> int:
    """Validate the configuration file."""
    try:
        from yaml_reader import ConfigLoader
        
        config_path = args.config or "../configs/pipeline_config.yaml"
        print(f"Validating configuration: {config_path}")
        
        config_loader = ConfigLoader(config_path)
        config = config_loader.get_all()
        
        print("\n✓ Configuration file is valid")
        print("\nConfiguration summary:")
        print(f"  - Max workers: {config.get('pipeline', {}).get('max_workers', 'N/A')}")
        print(f"  - Max files per run: {config.get('pipeline', {}).get('max_files_per_run', 'N/A')}")
        print(f"  - Default input folder: {config.get('pipeline', {}).get('default_input_folder', 'N/A')}")
        print(f"  - Continue on error: {config.get('error_handling', {}).get('continue_on_error', 'N/A')}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Configuration validation failed: {str(e)}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Audio Processing Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run-all --input-folder ~/audio_files
  %(prog)s transcribe --input-folder ~/audio_files --config custom_config.yaml
  %(prog)s detect-criteria
  %(prog)s llm-process
  %(prog)s status
  %(prog)s validate-config
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: configs/pipeline_config.yaml)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # run-all command
    run_all_parser = subparsers.add_parser(
        'run-all',
        help='Run the complete pipeline (all stages)'
    )
    run_all_parser.add_argument(
        '--input-folder',
        type=str,
        help='Path to folder containing audio files'
    )
    
    # transcribe command
    transcribe_parser = subparsers.add_parser(
        'transcribe',
        help='Run only the transcription stage'
    )
    transcribe_parser.add_argument(
        '--input-folder',
        type=str,
        help='Path to folder containing audio files'
    )
    
    # detect-criteria command
    subparsers.add_parser(
        'detect-criteria',
        help='Run only the criteria detection stage'
    )
    
    # llm-process command
    subparsers.add_parser(
        'llm-process',
        help='Run only the LLM processing stage'
    )
    
    # status command
    subparsers.add_parser(
        'status',
        help='Show current pipeline status'
    )
    
    # validate-config command
    subparsers.add_parser(
        'validate-config',
        help='Validate the configuration file'
    )
    
    args = parser.parse_args()
    
    # Show help if no command
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    commands = {
        'run-all': run_full_pipeline,
        'transcribe': run_transcription,
        'detect-criteria': run_criteria_detection,
        'llm-process': run_llm_processing,
        'status': show_status,
        'validate-config': validate_config
    }
    
    command_func = commands.get(args.command)
    if command_func:
        return command_func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

