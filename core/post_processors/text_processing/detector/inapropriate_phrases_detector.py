import re
from collections import defaultdict
from typing import Optional

import pymorphy2

from core.post_processors.text_processing.criteria_utils import normalize_text
from yaml_reader import ConfigLoader


class InappropriatePhrasesDetector:
    def __init__(self, config_path: str = "post_processors/config/parasites_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> str:
        return self._config.get('speech_patterns')['inappropriate_phrases']

    def __call__(self, text: str) -> Optional[str]:
        text = normalize_text(text)
        inappropriate_phrases = list()

        for phrase in self._patterns:
            if phrase in text:
                inappropriate_phrases.append(phrase)

        return ', '.join(sorted(set(inappropriate_phrases)))