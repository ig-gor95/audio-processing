from collections import defaultdict
from typing import List

from fuzzywuzzy import fuzz

from core.repository.entity.dialog_rows import DialogRow
from yaml_reader import ConfigLoader


class SalesDetector:
    def __init__(self, config_path: str = "post_processors/config/sales_patterns.yaml"):
        self._config = ConfigLoader(config_path).get("patterns")
        self._threshold = 90
        self._patterns = [pattern.lower() for pattern in self._config]

    def __call__(self, dialog_rows: List[DialogRow]):
        speaker_scores = defaultdict(int)

        for row in dialog_rows:
            if self.detect_by_text(row.row_text.lower()):
                speaker_scores[row.speaker_id] += 1

        if speaker_scores:
            manager = max(speaker_scores.items(), key=lambda x: x[1])[0]
            self._assign_roles(dialog_rows, manager)

    def _assign_roles(self, dialog_rows: List[DialogRow], manager_id: str):
        for row in dialog_rows:
            row.speaker_id = "SALES" if row.speaker_id == manager_id else "CLIENT"

    def detect_by_text(self, phrase: str) -> bool:
        return any(
            fuzz.partial_ratio(pattern, phrase) >= self._threshold
            for pattern in self._patterns
        )
