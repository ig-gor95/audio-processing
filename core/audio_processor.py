from pathlib import Path

from core.audio_to_text.audio_to_text_processor import audio_to_text_processor
from core.post_processors.client_detector import detect_manager
from core.post_processors.llm_processing.objections_resolver import resolve_objections

if __name__ == "__main__":
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    audio_file = "../data/input/temp/processed_10da0ae8-85d6-49e1-8a75-2741ea4a55b0.wav"

    result = audio_to_text_processor(Path(audio_file))

    detect_manager(result)

    resolve_objections(result)

    for r in result.items:
        if r.is_copy_line:
            continue
        print(f"[{r.start_time:.2f} - {r.end_time:.2f}]: {r.speaker_id} : {r.text}")
