import re
from typing import Optional

import pandas as pd
import pymorphy2

from core.post_processors.text_processing.criteria_utils import normalize_text
from yaml_reader import ConfigLoader

exclude = ['-го']
class SlangDetector:
    def __init__(self, config_path: str = "post_processors/config/parasites_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy2.MorphAnalyzer()
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[str]:
        return self._config.get('speech_patterns')['slang']

    def __call__(self, texts: pd.DataFrame):
        def find_match(text):
            slang_words = list()

            for word in self._patterns:
                if re.search(rf'\b{re.escape(word)}\b', text):
                    if word not in exclude:
                        slang_words.append(word)

            return ', '.join(sorted(set(slang_words)))

        return texts.apply(find_match)
