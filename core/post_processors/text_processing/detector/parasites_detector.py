import re
from functools import lru_cache
from typing import Optional

import pandas as pd
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

    def __call__(self, df: pd.DataFrame, text_column='row_text'):
        # Pre-normalize all texts once (vectorized operation)
        texts = df[text_column].apply(normalize_text)

        patterns = [(word, re.compile(rf'\b{re.escape(word)}\b'))
                    for word in self._patterns]

        @lru_cache(maxsize=1000)
        def cached_find_match(text):
            found = set()
            for word, pattern in patterns:
                if pattern.search(text):
                    found.add(word)
            return ', '.join(sorted(found)) if found else ''

        return texts.apply(cached_find_match)