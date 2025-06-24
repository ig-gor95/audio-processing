import concurrent.futures
import threading
import time
import uuid
from pathlib import Path
from typing import List
import soundfile as sf

import audio_dialog_repository as audio_dialog_repository
import dialog_rows_repository as dialog_rows_repository
from core.audio_to_text.audio_to_text_processor import audio_to_text_processor
import os

from core.post_processors.client_detector import detect_manager
from log_utils import setup_logger

logger = setup_logger(__name__)

folder_path = f"{Path.home()}/Documents/Аудио Бринекс/Brinex_in_2025_04/"
audio_files = list(Path(folder_path).glob("*"))

audio_files = [f for f in Path(folder_path).iterdir() if f.is_file()]

print_lock = threading.Lock()
duration_lock = threading.Lock()


def get_duration(audio_file: Path) -> float:
    with print_lock:
        print(f"Processing {audio_file.name}")  # Synchronized print

    try:
        with duration_lock:  # Synchronize file access
            with sf.SoundFile(str(audio_file)) as f:
                duration = len(f) / f.samplerate

        with print_lock:
            print(f"Completed {audio_file.name}: {duration:.2f}s")
        return duration

    except Exception as e:
        with print_lock:
            print(f"Error in {audio_file.name}: {str(e)}")
        return 0.0


def run_pipeline(audio_file: Path):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    start_time = time.time()
    file_uuid = uuid.uuid4()
    existing_dialog = audio_dialog_repository.find_by_filename(audio_file.name)
    if existing_dialog is None:
        audio_dialog_repository.save(file_uuid, audio_file.name, get_duration(audio_file))
    elif existing_dialog["status"] == "PROCESSED":
        logger.info(f"Skipping {audio_file.name}")
        return
    else:
        dialog_rows_repository.delete_all_by_dialog_id(existing_dialog["id"])

    result = audio_to_text_processor(audio_file)
    detect_manager(result)

    # resolve_objections(result)
    end_time = time.time()
    execution_time = end_time - start_time

    for r in result.items:
        dialog_rows_repository.save(
            row_id=uuid.uuid4(),
            audio_dialog_fk_id=file_uuid,
            row_num=r.phrase_id,
            row_text=r.text,
            speaker_id=r.speaker_id,
            start=r.start_time,
            end=r.end_time,
            has_greeting=False,
            has_swear_word=False,
        )

    audio_dialog_repository.update_status(file_uuid, "PROCESSED", execution_time)

def process_files_parallel(audio_files: List[Path], max_workers: int = 4, max_files: int = 100):
    """Process files in parallel with progress tracking"""
    start_time = time.time()
    processed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(run_pipeline, file): file
            for file in audio_files[:max_files]
        }

        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                future.result()  # Get result or raise exception
                processed_count += 1
                logger.info(f"Completed {processed_count}/{min(len(audio_files), max_files)}")
            except Exception as e:
                logger.error(f"Error processing {file.name}: {str(e)}")

    end_time = time.time()
    logger.info(f"Total processed: {processed_count}")
    logger.info(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    first_100_files = audio_files[:200]
    process_files_parallel(first_100_files)