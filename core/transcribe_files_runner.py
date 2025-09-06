import concurrent.futures
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
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

def make_overlapping_windows(segments, win=0.9, hop=0.25, min_len=0.8):
    """Генерим перекрывающиеся окна фиксированной длины. Окон < min_len не создаём."""
    out = []
    win = float(win); hop = float(hop); min_len = float(min_len)
    for seg in segments:
        s, e = float(seg['start']), float(seg['end'])
        dur = e - s
        if dur < min_len - 1e-6:
            continue
        if dur <= win + 1e-6:
            out.append({'start': s, 'end': e})
            continue
        t = s
        while t + win <= e + 1e-9:
            out.append({'start': t, 'end': min(t + win, e)})
            t += hop
    return out

def merge_labeled_windows(windows, labels, max_gap=0.15, min_len=0.8):
    """Склеиваем подряд идущие окна одного спикера, фильтруем коротыши."""
    if not windows or len(windows) != len(labels):
        return [], np.array([], dtype=int)
    items = sorted(
        [{'start': float(w['start']), 'end': float(w['end']), 'lab': int(l)}
         for w, l in zip(windows, labels)],
        key=lambda z: (z['start'], z['end'])
    )
    merged = [{'start': items[0]['start'], 'end': items[0]['end'], 'lab': items[0]['lab']}]
    for it in items[1:]:
        cur = merged[-1]
        if it['lab'] == cur['lab'] and (it['start'] - cur['end']) <= max_gap + 1e-9:
            cur['end'] = max(cur['end'], it['end'])
        else:
            merged.append({'start': it['start'], 'end': it['end'], 'lab': it['lab']})
    keep = [m for m in merged if (m['end'] - m['start']) >= (min_len - 1e-6)]
    if not keep:
        return [], np.array([], dtype=int)
    segs = [{'start': k['start'], 'end': k['end']} for k in keep]
    labs = np.array([k['lab'] for k in keep], dtype=int)
    return segs, labs

def viterbi_labels_by_centroids(X: np.ndarray, labels_init: np.ndarray, n_clusters: int, switch_penalty: float = 0.18) -> np.ndarray:
    """
    Уточняет последовательность меток по косинусной близости к центроидам с умеренным штрафом за переключение.
    X: [N, d] L2-нормализованные эмбеддинги окон (как у тебя после l2norm_rows)
    labels_init: начальные метки окон (после кластеризации)
    n_clusters: число кластеров (2)
    switch_penalty: штраф за смену спикера между соседними окнами (0.15–0.25 обычно ок)
    """
    # 1) центроиды по начальным меткам
    C = []
    for k in range(n_clusters):
        idx = np.where(labels_init == k)[0]
        if len(idx) == 0:
            # если кластер пуст — возьмём самый "дальний" от другого (редко, но бывает)
            idx = np.array([np.argmax(np.linalg.norm(X - X.mean(0, keepdims=True), axis=1))])
        c = X[idx].mean(0)
        c /= (np.linalg.norm(c) + 1e-12)
        C.append(c)
    C = np.stack(C, axis=0)  # [K, d]

    # 2) эмиссионные стоимости = -cosine_similarity (чем меньше — тем лучше)
    sims = X @ C.T                           # [N, K], т.к. X и C уже L2-нормированы
    cost = -sims                             # минимизируем

    N, K = cost.shape
    dp = np.zeros((N, K), dtype=np.float32)
    bk = np.zeros((N, K), dtype=np.int32)
    dp[0] = cost[0]

    # 3) динамика с штрафом за переключения
    for i in range(1, N):
        prev = dp[i-1][:, None] + switch_penalty * (1.0 - np.eye(K, dtype=np.float32))
        bk[i] = prev.argmin(axis=0)
        dp[i] = cost[i] + prev.min(axis=0)

    # 4) обратный проход
    y = np.zeros(N, dtype=np.int32)
    y[-1] = dp[-1].argmin()
    for i in range(N - 2, -1, -1):
        y[i] = bk[i + 1, y[i + 1]]
    return y


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
    folder_path = f"{Path.home()}/Documents/Аудио Бринекс/asd/"
    # audio_file = Path(folder_path)
    # process_files_parallel([audio_file], max_files=5000)

    # print_dialog(uuid.UUID('009fc88f-6252-434b-88dd-42b39b1eb4b4'))
    audio_files = list(Path(folder_path).glob("*"))[3:7]
    print(f' Total: {len(audio_files)}')
    process_files_parallel(audio_files, max_files=5000)
