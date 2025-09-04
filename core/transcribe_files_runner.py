import concurrent.futures
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List
import soundfile as sf

from core.audio_to_text.audio_to_text_processor import audio_to_text_processor
from core.post_processors.text_processing.detector.sales_detector import SalesDetector

from core.repository.audio_dialog_repository import AudioDialogRepository
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.dialog_rows_repository import DialogRowRepository
from core.repository.entity.audio_dialog import AudioDialogStatus, AudioDialog
from core.repository.entity.dialog_rows import DialogRow
from core.service.dialog_row_util_service import print_dialog
from log_utils import setup_logger
import os

logger = setup_logger(__name__)

audio_dialog_repository = AudioDialogRepository()
dialog_row_repository = DialogRowRepository()

print_lock = threading.Lock()
duration_lock = threading.Lock()
detector = SalesDetector()
dialog_criteria_repository = DialogCriteriaRepository()

def get_duration(audio_file: Path) -> float:
    with print_lock:
        print(f"Processing {audio_file.name}")

    try:
        with duration_lock:
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
        audio_dialog_repository.save(
            AudioDialog(
                id=file_uuid,
                file_name=audio_file.name,
                duration=get_duration(audio_file),
                status=AudioDialogStatus.NOT_PROCESSED
            )
        )
    elif existing_dialog.status == AudioDialogStatus.PROCESSED and existing_dialog.updated_at < datetime(2025, 8, 2):
        rows = dialog_row_repository.find_by_dialog_id(existing_dialog.id)
        for row in rows:
            dialog_criteria_repository.delete_by_dialog_row_fk_id(row.id)
        dialog_row_repository.delete_all_by_dialog_id(existing_dialog.id)
        logger.info(f"Cleaning dialog {audio_file.name}")
        file_uuid = existing_dialog.id

    # elif existing_dialog.status == AudioDialogStatus.PROCESSED:
    #     return
    else:
        rows = dialog_row_repository.find_by_dialog_id(existing_dialog.id)
        for row in rows:
            dialog_criteria_repository.delete_by_dialog_row_fk_id(row.id)
        dialog_row_repository.delete_all_by_dialog_id(existing_dialog.id)
        file_uuid = existing_dialog.id

    result = audio_to_text_processor(audio_file)

    end_time = time.time()
    execution_time = end_time - start_time
    dialog_rows = [
        DialogRow(
            audio_dialog_fk_id=file_uuid,
            row_num=r.phrase_id,
            row_text=r.text.replace("Продолжение следует...", '').replace('Субтитры сделал DimaTorzok', ''),
            speaker_id=r.speaker_id,
            start=r.start_time,
            end=r.end_time
        )
        for r in result.items
    ]
    detector(dialog_rows)
    rows = sorted(dialog_rows, key=lambda x: x.row_num)
    dialog_row_repository.save_bulk(dialog_rows)
    for row in rows:
        row.print()
    audio_dialog_repository.update_status(file_uuid, AudioDialogStatus.PROCESSED, execution_time)


def process_files_parallel(audio_files: List[Path], max_workers: int = 3, max_files: int = 200):
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
                future.result()
                processed_count += 1
                logger.info(f"Completed {processed_count}/{min(len(audio_files), max_files)}")
            except Exception as e:
                logger.error(f"Error processing {file.name}: {str(e)}")

    end_time = time.time()
    logger.info(f"Total processed: {processed_count}")
    logger.info(f"Execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # print_dialog(uuid.UUID("247699f2-5337-40b3-b6a8-4c3b14449fa8"))
    # folder_path = f"{Path.home()}/Documents/Аудио Бринекс/Brinex_in_2025_04/in-07010101-89384864072-20250402-113058-1743582658.20978046.mp3"
    # audio_file = Path(folder_path)
    # process_files_parallel([audio_file], max_files=5000)

    print_dialog(uuid.UUID('009fc88f-6252-434b-88dd-42b39b1eb4b4'))
    # audio_files = list(Path(folder_path).glob("*"))
    # print(f' Total: {len(audio_files)}')
    # process_files_parallel(audio_files, max_files=5000)
