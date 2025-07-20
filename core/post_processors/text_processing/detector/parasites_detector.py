import re
from typing import Optional

import pymorphy2

from core.post_processors.text_processing.criteria_utils import normalize_text
from yaml_reader import ConfigLoader


class ParasitesDetector:
    def __init__(self, config_path: str = "post_processors/config/parasites_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy2.MorphAnalyzer()
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[str]:
        parasite_patterns = self._config.get('speech_patterns')['parasites']
        return parasite_patterns

    def __call__(self, text: str) -> Optional[str]:
        text = normalize_text(text)
        parasites = list

        for word in self._patterns:
            if re.search(rf'\b{re.escape(word)}\b', text):
                parasites.append(word)

        return ', '.join(sorted(set(parasites)))