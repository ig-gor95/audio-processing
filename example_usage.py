#!/usr/bin/env python3
"""
Example Usage of the Audio Processing Pipeline

This script demonstrates different ways to use the refactored pipeline.
"""

from pathlib import Path
from core.pipeline import AudioProcessingPipeline, PipelineStage


def example_1_run_full_pipeline():
    """Example 1: Run the complete pipeline."""
    print("=== Example 1: Run Full Pipeline ===\n")
    
    pipeline = AudioProcessingPipeline()
    
    # Run all stages
    result = pipeline.run_full_pipeline(
        audio_folder="~/Documents/Аудио Бринекс/2/"
    )
    
    print(f"Duration: {result.get_duration():.2f} seconds")
    print(f"Completed: {result.stages_completed}")
    print(f"Failed: {result.stages_failed}")
    print()


def example_2_run_individual_stages():
    """Example 2: Run stages individually."""
    print("=== Example 2: Run Individual Stages ===\n")
    
    pipeline = AudioProcessingPipeline()
    
    # Stage 1: Transcription
    print("Running transcription...")
    result1 = pipeline.run_single_stage(
        PipelineStage.TRANSCRIPTION,
        audio_folder="~/Documents/Аудио Бринекс/2/"
    )
    print(f"Transcription complete: {len(result1.transcription_stats.get('successful', []))} files")
    
    # Stage 2: Criteria Detection
    print("Running criteria detection...")
    result2 = pipeline.run_single_stage(PipelineStage.CRITERIA_DETECTION)
    print(f"Criteria detection complete")
    
    # Stage 3: LLM Processing
    print("Running LLM processing...")
    result3 = pipeline.run_single_stage(PipelineStage.LLM_PROCESSING)
    print(f"LLM processing complete: {result3.llm_stats.get('processed', 0)} dialogs")
    print()


def example_3_process_specific_files():
    """Example 3: Process specific audio files."""
    print("=== Example 3: Process Specific Files ===\n")
    
    pipeline = AudioProcessingPipeline()
    
    # Specify exact files
    audio_files = [
        Path("~/Documents/Аудио Бринекс/2/file1.mp3").expanduser(),
        Path("~/Documents/Аудио Бринекс/2/file2.mp3").expanduser(),
        Path("~/Documents/Аудио Бринекс/2/file3.mp3").expanduser(),
    ]
    
    result = pipeline.run_full_pipeline(audio_files=audio_files)
    
    print(f"Processed {len(audio_files)} files")
    print(f"Successful: {len(result.transcription_stats.get('successful', []))}")
    print()


def example_4_use_services_directly():
    """Example 4: Use services directly for fine-grained control."""
    print("=== Example 4: Use Services Directly ===\n")
    
    from core.service.transcription_service import TranscriptionService
    from core.service.llm_processing_service import LLMProcessingService
    
    # Transcription service
    transcription_service = TranscriptionService()
    audio_file = Path("~/Documents/Аудио Бринекс/2/test.mp3").expanduser()
    
    if audio_file.exists():
        file_uuid = transcription_service.process_audio_file(audio_file)
        print(f"Transcribed file UUID: {file_uuid}")
        
        # LLM service
        if file_uuid:
            llm_service = LLMProcessingService()
            success = llm_service.process_dialog(file_uuid)
            print(f"LLM processing success: {success}")
    else:
        print(f"File not found: {audio_file}")
    print()


def example_5_custom_configuration():
    """Example 5: Use custom configuration."""
    print("=== Example 5: Custom Configuration ===\n")
    
    # Use custom config file
    pipeline = AudioProcessingPipeline(config_path="configs/my_custom_config.yaml")
    
    result = pipeline.run_full_pipeline(
        audio_folder="~/Documents/Аудио Бринекс/2/"
    )
    
    print(f"Pipeline completed with custom config")
    print(f"Duration: {result.get_duration():.2f} seconds")
    print()


def example_6_check_status():
    """Example 6: Check pipeline status."""
    print("=== Example 6: Check Status ===\n")
    
    from core.repository.audio_dialog_repository import AudioDialogRepository
    
    repo = AudioDialogRepository()
    all_dialogs = repo.find_all()
    
    processed = sum(1 for d in all_dialogs if d.status.value == 'PROCESSED')
    with_llm = sum(1 for d in all_dialogs if d.llm_data_short is not None)
    
    print(f"Total dialogs: {len(all_dialogs)}")
    print(f"Processed: {processed}")
    print(f"With LLM data: {with_llm}")
    print(f"Pending transcription: {len(all_dialogs) - processed}")
    print(f"Pending LLM: {len(all_dialogs) - with_llm}")
    print()


def example_7_batch_processing():
    """Example 7: Batch processing with progress tracking."""
    print("=== Example 7: Batch Processing ===\n")
    
    from core.service.transcription_service import TranscriptionService
    
    service = TranscriptionService()
    
    # Get all audio files
    folder = Path("~/Documents/Аудио Бринекс/2/").expanduser()
    audio_files = list(folder.glob("*.mp3"))
    
    print(f"Found {len(audio_files)} files")
    
    # Process in batches
    batch_size = 10
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch)} files")
        
        results = service.process_multiple_files(batch)
        print(f"  Successful: {len(results['successful'])}")
        print(f"  Failed: {len(results['failed'])}")
    print()


def example_8_single_file_complete_pipeline():
    """Example 8: Process ONE file through ALL stages."""
    print("=== Example 8: Complete Pipeline for Single File ===\n")
    
    from core.service.transcription_service import TranscriptionService
    from core.service.criteria_detection_service import CriteriaDetectionService
    from core.service.llm_processing_service import LLMProcessingService
    
    # Specify a single audio file
    audio_file = Path("~/Documents/Аудио Бринекс/4/").expanduser()
    audio_files = list(audio_file.glob("*.mp3"))
    
    if not audio_files:
        print("❌ No audio files found in the folder")
        return
    
    # Take the first file
    single_file = audio_files[0]
    print(f"📁 Processing: {single_file.name}\n")
    
    # Stage 1: Transcription
    print("Stage 1/3: Transcription & Diarization")
    print("-" * 40)
    transcription_service = TranscriptionService()
    
    file_uuid = transcription_service.process_audio_file(single_file)
    
    if file_uuid:
        print(f"✅ Transcribed successfully")
        print(f"   Dialog UUID: {file_uuid}\n")
    else:
        print("⚠️  File was skipped (already processed)\n")
        # Try to get UUID from database
        from core.repository.audio_dialog_repository import AudioDialogRepository
        repo = AudioDialogRepository()
        dialog = repo.find_by_filename(single_file.name)
        if dialog:
            file_uuid = dialog.id
            print(f"   Found existing UUID: {file_uuid}\n")
    
    if not file_uuid:
        print("❌ Could not process file")
        return
    
    # Stage 2: Criteria Detection
    print("Stage 2/3: Criteria Detection")
    print("-" * 40)
    criteria_service = CriteriaDetectionService()
    
    try:
        criteria_stats = criteria_service.process_dialogs()
        print(f"✅ Criteria detection completed")
        print(f"   Success: {criteria_stats.get('success', False)}\n")
    except Exception as e:
        print(f"⚠️  Criteria detection: {str(e)}\n")
    
    # Stage 3: LLM Processing
    print("Stage 3/3: LLM Analysis")
    print("-" * 40)
    llm_service = LLMProcessingService()
    
    try:
        success = llm_service.process_dialog(file_uuid)
        if success:
            print(f"✅ LLM analysis completed")
            print(f"   Dialog UUID: {file_uuid}\n")
        else:
            print(f"⚠️  LLM processing skipped or failed\n")
    except Exception as e:
        print(f"⚠️  LLM processing error: {str(e)}\n")
    
    # Summary
    print("=" * 40)
    print("✅ COMPLETE PIPELINE FINISHED")
    print("=" * 40)
    print(f"File: {single_file.name}")
    print(f"UUID: {file_uuid}")
    print("\nAll stages completed for this file!")
    print("\nYou can now:")
    print("  - View results in database")
    print("  - Generate reports")
    print("  - Process more files")
    print()


def main():
    """Run all examples."""
    print("Audio Processing Pipeline - Usage Examples")
    print("=" * 60)
    print()
    
    # Uncomment the examples you want to run:
    
    # example_1_run_full_pipeline()
    # example_2_run_individual_stages()
    # example_3_process_specific_files()
    # example_4_use_services_directly()
    # example_5_custom_configuration()
    # example_6_check_status()
    # example_7_batch_processing()
    example_8_single_file_complete_pipeline()  # NEW: Process one file through all stages
    
    print("=" * 60)
    print("Examples complete!")


if __name__ == "__main__":
    main()

