#!/usr/bin/env python3
"""
Example Usage of the Audio Processing Pipeline

This script demonstrates different ways to use the refactored pipeline.
"""

import os
import logging
import warnings

# Suppress verbose logging before any imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger('speechbrain').setLevel(logging.ERROR)
logging.getLogger('pymorphy2').setLevel(logging.ERROR)
logging.getLogger('pymorphy3').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)

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
    """Example 8: Process first 10 files through ALL stages."""
    print("=== Example 8: Complete Pipeline for First 10 Files ===\n")
    
    from core.service.transcription_service import TranscriptionService
    from core.service.criteria_detection_service import CriteriaDetectionService
    from core.service.llm_processing_service import LLMProcessingService
    from core.repository.audio_dialog_repository import AudioDialogRepository
    import time
    
    # Specify folder with audio files
    audio_folder = Path("~/Documents/Аудио Бринекс/4/").expanduser()
    audio_files = list(audio_folder.glob("*.mp3"))
    
    if not audio_files:
        print("❌ No audio files found in the folder")
        return
    
    # Take first 10 files
    files_to_process = audio_files[:10]
    print(f"📁 Found {len(audio_files)} files, processing first {len(files_to_process)}\n")
    
    # Initialize services once
    transcription_service = TranscriptionService()
    criteria_service = CriteriaDetectionService()
    llm_service = LLMProcessingService()
    repo = AudioDialogRepository()
    
    # Track statistics
    stats = {
        'transcribed': 0,
        'skipped': 0,
        'criteria_success': 0,
        'llm_success': 0,
        'errors': 0
    }
    
    processed_uuids = []
    start_time = time.time()
    
    # ============================================================
    # STAGE 1: Transcription & Diarization (per file)
    # ============================================================
    print(f"\n{'='*60}")
    print("🎤 STAGE 1/3: TRANSCRIPTION & DIARIZATION")
    print(f"{'='*60}\n")
    
    for idx, audio_file in enumerate(files_to_process, 1):
        print(f"⏳ [{idx}/{len(files_to_process)}] Processing: {audio_file.name}")
        
        try:
            file_uuid = transcription_service.process_audio_file(audio_file)
            
            if file_uuid:
                print(f"   ✅ Transcribed | UUID: {file_uuid}\n")
                stats['transcribed'] += 1
                processed_uuids.append(file_uuid)
            else:
                # Try to get existing UUID
                dialog = repo.find_by_filename(audio_file.name)
                if dialog:
                    file_uuid = dialog.id
                    print(f"   ⚠️  Already processed | UUID: {file_uuid}\n")
                    stats['skipped'] += 1
                    processed_uuids.append(file_uuid)
                else:
                    print(f"   ❌ Could not process file\n")
                    stats['errors'] += 1
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}\n")
            stats['errors'] += 1
    
    # ============================================================
    # STAGE 2: Criteria Detection (BATCH - runs on entire DB once)
    # ============================================================
    print(f"\n{'='*60}")
    print("📊 STAGE 2/3: CRITERIA DETECTION (BATCH)")
    print(f"{'='*60}")
    print("⏳ Running criteria detection on entire database...")
    print("   (This is MUCH faster than per-file processing!)\n")
    
    criteria_start = time.time()
    try:
        criteria_service.process_dialogs()
        criteria_elapsed = time.time() - criteria_start
        print(f"✅ Criteria detection completed in {criteria_elapsed:.2f}s")
        print(f"   Average per file: {criteria_elapsed/len(files_to_process):.2f}s\n")
        stats['criteria_success'] = len(processed_uuids)
    except Exception as e:
        print(f"❌ Criteria detection failed: {str(e)}\n")
    
    # ============================================================
    # STAGE 3: LLM Processing (per file)
    # ============================================================
    print(f"\n{'='*60}")
    print("🤖 STAGE 3/3: LLM ANALYSIS")
    print(f"{'='*60}\n")
    
    for idx, file_uuid in enumerate(processed_uuids, 1):
        print(f"⏳ [{idx}/{len(processed_uuids)}] LLM processing UUID: {file_uuid}")
        
        try:
            success = llm_service.process_dialog(file_uuid)
            if success:
                print(f"   ✅ LLM analysis completed\n")
                stats['llm_success'] += 1
            else:
                print(f"   ⚠️  LLM processing skipped\n")
        except Exception as e:
            print(f"   ⚠️  Warning: {str(e)}\n")
    
    # Final Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("✅ BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"\n📊 Statistics:")
    print(f"   Total files:            {len(files_to_process)}")
    print(f"   Newly transcribed:      {stats['transcribed']}")
    print(f"   Already processed:      {stats['skipped']}")
    print(f"   Criteria detected:      {stats['criteria_success']} (batch)")
    print(f"   LLM analysis done:      {stats['llm_success']}")
    print(f"   Errors:                 {stats['errors']}")
    print(f"\n⏱️  Total time: {elapsed:.2f} seconds")
    print(f"   Average per file: {elapsed/len(files_to_process):.2f} seconds")
    print(f"\n💡 Note: Criteria detection runs on entire DB (much faster than per-file!)")
    print(f"\n✅ All {len(processed_uuids)} files processed through all stages!")
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

