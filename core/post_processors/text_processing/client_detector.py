from typing import List

from fuzzywuzzy import fuzz

from core.repository.entity.dialog_rows import DialogRow
from yaml_reader import ConfigLoader


class SalesDetector:
    def __init__(self, config_path: str = "post_processors/config/sales_patterns.yaml"):
        self._config = ConfigLoader(config_path).get("patterns")
        self._threshold = 80

    def __call__(self, dialog_rows: List[DialogRow]):
        speaker_scores = {}

        for line in dialog_rows:
            phrase = line.row_text
            speaker = line.speaker_id
            phrase_lower = phrase.lower()
            score = 0

            for manager_phrase in self._config:
                similarity = fuzz.partial_ratio(manager_phrase.lower(), phrase_lower)
                if similarity >= self._threshold:
                    score += 1
            if speaker not in speaker_scores:
                speaker_scores[speaker] = score
            else:
                speaker_scores[speaker] += score

        manager = max(speaker_scores.items(), key=lambda x: x[1])[0]
        for row in dialog_rows:
            if row.speaker_id == manager:
                row.speaker_id = "SALES"
            else:
                row.speaker_id = "CLIENT"