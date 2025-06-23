import concurrent.futures
import time
import uuid
from pathlib import Path
from typing import List

from core.audio_to_text.audio_to_text_processor import audio_to_text_processor
import os
import pandas as pd
from sqlalchemy import create_engine, update, text
from datetime import datetime
import soundfile as sf

from core.post_processors.client_detector import detect_manager
from log_utils import setup_logger

logger = setup_logger(__name__)

folder_path = f"{Path.home()}/Documents/Аудио Бринекс/Brinex_in_2025_04/"
audio_files = list(Path(folder_path).glob("*"))

audio_files = [f for f in Path(folder_path).iterdir() if f.is_file()]

DB_CONFIG = {
    'dbname': 'neiro-insight',
    'user': 'postgres',
    'password': '1510261105',
    'host': 'localhost',
    'port': '5432'
}
engine = create_engine(
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)


def get_sf_duration(audio_file: Path) -> float:
    info = sf.info(audio_file)
    return info.duration

def update_status(file_id, new_status, processed_time):
    stmt = text("""
        UPDATE audio_dialog 
        SET status = :status, 
            processing_time = :processed_time
        WHERE id = :file_id
    """)
    with engine.connect() as conn:
        conn.execute(stmt, {'file_id': file_id, 'status': new_status, 'processed_time': processed_time})
        conn.commit()

def run_pipeline(audio_file: Path):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    start_time = time.time()
    data = []
    row_data = []
    file_uuid = uuid.uuid4()
    data.append({
        'id': file_uuid,
        'file_name': audio_file.name,
        'status': "NOT_PROCESSED",
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'duration': get_sf_duration(audio_file)
    })

    pd.DataFrame(data).to_sql(
        name='audio_dialog',
        con=engine,
        if_exists='append',
        index=False
    )

    result = audio_to_text_processor(audio_file)
    detect_manager(result)

    # resolve_objections(result)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Время выполнения: {execution_time:.4f} секунд")

    for r in result.items:
        row_data.append({
            'id': uuid.uuid4(),
            'audio_dialog_fk_id': file_uuid,
            'row_num': r.phrase_id,
            'row_text': r.text,
            'speaker_id': r.speaker_id,
            'start': r.start_time,
            'end': r.end_time,
            'has_swear_word': False,
            'has_greeting': False
        })
        if r.is_copy_line:
            continue
        print(f"{r.phrase_id} [{r.start_time:.2f} - {r.end_time:.2f}]: {r.speaker_id} : {r.text}")
    pd.DataFrame(row_data).to_sql(
        name='dialog_rows',
        con=engine,
        if_exists='append',
        index=False
    )
    update_status(file_uuid, "PROCESSED", execution_time)
    for prop in result.objection_prop:
        print(f"Строка с возражением: {result.items[prop.objection_str - 1].text}")
        print(f"Строка с Обработкой возражения: {result.items[prop.manager_offer_str - 1].text}")
        print(f"Решилась ли проблема: {prop.was_resolved}")

def process_files_parallel(audio_files: List[Path], max_workers: int = 3, max_files: int = 100):
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
    process_files_parallel([audio_files[0], audio_files[1]])
    # count = 0
    # start_time = time.time()
    # for audio_file in audio_files:
    #     logger.info(f"Processing {count}: {audio_file.name}")
    #     run_pipeline(audio_file)
    #     if count == 100:
    #         break
    #     count += 1
    #     run_pipeline(audio_file)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Время выполнения: {execution_time:.4f} секунд")