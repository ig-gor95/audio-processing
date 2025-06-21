import time
from pathlib import Path
from log_utils import setup_logger
from core.audio_to_text.audio_to_text_processor import audio_to_text_processor
from core.post_processors.client_detector import detect_manager
from core.post_processors.llm_processing.objections_resolver import resolve_objections

logger = setup_logger(__name__)

if __name__ == "__main__":
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    audio_file = "../data/input/temp/processed_10da0ae8-85d6-49e1-8a75-2741ea4a55b0.wav"

    start_time = time.time()

    result = audio_to_text_processor(Path(audio_file))
    detect_manager(result)

    resolve_objections(result)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Время выполнения: {execution_time:.4f} секунд")


    for r in result.items:
        if r.is_copy_line:
            continue
        print(f"{r.phrase_id} [{r.start_time:.2f} - {r.end_time:.2f}]: {r.speaker_id} : {r.text}")

    for prop in result.objection_prop:
        print(f"Строка с возражением: {result.items[prop.objection_str - 1].text}")
        print(f"Строка с Обработкой возражения: {result.items[prop.manager_offer_str - 1].text}")
        print(f"Решилась ли проблема: {prop.was_resolved}")
