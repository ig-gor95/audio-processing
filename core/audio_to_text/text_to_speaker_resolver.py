from typing import List, Type

from core.entity.audio_to_text_result import ProcessingResult, ProcessingResults
from core.entity.diarisation_result import DiarizedResult


def unite_results(transcribed_result, diarized_result: DiarizedResult) -> ProcessingResults:
    diarization_result = []
    for i, segment in enumerate(diarized_result.valid_segments):
        speaker = f"Speaker_{diarized_result.labels[i] + 1}"
        diarization_result.append({
            "start": segment['start'],
            "end": segment['end'],
            "speaker": speaker
        })
    processing_results = ProcessingResults()

    for segment in transcribed_result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        max_overlap = 0
        best_speaker = None

        for diarization_segment in diarization_result:
            overlap_start = max(start, diarization_segment["start"])
            overlap_end = min(end, diarization_segment["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diarization_segment["speaker"]

        if best_speaker is None:
            for diarization_segment in diarization_result:
                if diarization_segment["end"] >= start or diarization_segment["start"] <= end:
                    best_speaker = diarization_segment["speaker"]
                    break

        speaker = best_speaker if best_speaker else "Unknown"
        prev_element = processing_results.items[-1] if processing_results.items else None

        is_copy_anomaly = prev_element is not None and prev_element.text == text and prev_element.speaker_id == speaker

        # Если следущая строка от того же спикера, то объединяем строку
        if prev_element is not None and prev_element.speaker_id == speaker and prev_element.end_time == start:
            prev_element.text = f"{prev_element.text} {text}"
            prev_element.end_time = start
            continue
        processing_results.items.append(ProcessingResult(speaker, text, start, end, is_copy_anomaly))
    return processing_results
