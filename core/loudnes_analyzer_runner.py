import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

from typing import Dict, List

from core.post_processors.audio_processing.loudness_analyzer import LoudnessAnalyzer
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.repository.dialog_rows_repository import DialogRowRepository

dialog_row_repository = DialogRowRepository()


def chunk_list(rows: List, chunk_size: int = 1000) -> List[List]:
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]


def process_single_file(args):
    """Process a single file - designed for parallel execution"""
    audio_path, phrases, analyzer_config = args
    analyzer = LoudnessAnalyzer(**analyzer_config)

    try:
        results = analyzer.process_file(audio_path, phrases)
        return audio_path, results, None
    except Exception as e:
        return audio_path, [], str(e)


def process_files_and_wait(file_phrase_mapping: dict, max_workers: int = None):
    """Submit files and wait for completion, but ignore results"""
    if max_workers is None:
        max_workers = mp.cpu_count()

    analyzer_config = {'sample_rate': 16000, 'hop_length': 256}
    total_files = len(file_phrase_mapping)

    print(f"Starting processing of {total_files} files...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for audio_path, phrases in file_phrase_mapping.items():
            future = executor.submit(process_single_file, (audio_path, phrases, analyzer_config))
            futures.append(future)

        processed_count = 0
        for future in futures:
            future.result()
            processed_count += 1

            if processed_count % 100 == 0:
                print(f"Processed {processed_count}/{total_files} files")

    print(f"All {processed_count} files processed!")


# Example usage for multiple files
if __name__ == "__main__":
    file_phrase_mapping = {}

    audio_dialog_repository = AudioDialogRepository()
    audio_files = glob.glob(f"{Path.home()}/Documents/Аудио Бринекс/2/*")

    chunks = chunk_list(audio_files, 1000)
    for chunk in chunks:
        file_phrase_mapping = {}
        for audio_file in chunk:
            dialog = audio_dialog_repository.find_by_filename(audio_file.split("/")[-1])
            if dialog is None:
                continue
            rows = dialog_row_repository.find_by_dialog_id(dialog.id)
            phrases = []
            for row in rows:
                if row.mean_loudness is not None:
                    continue
                phrases.append({
                    'start_time': row.start,
                    'end_time': row.end,
                    'row_id': row.id
                })
            if len(phrases) > 0:
                file_phrase_mapping[audio_file] = phrases
        process_files_and_wait(
            file_phrase_mapping=file_phrase_mapping,
            max_workers=8,
        )
