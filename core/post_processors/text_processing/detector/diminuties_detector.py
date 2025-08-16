import re
from collections import defaultdict
from functools import lru_cache
from typing import Optional

import pandas as pd
import pymorphy2

from core.post_processors.text_processing.criteria_utils import normalize_text
from yaml_reader import ConfigLoader


class DiminutivesDetector:
    def __init__(self, config_path: str = "post_processors/config/parasites_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy2.MorphAnalyzer()
        self.diminutive_suffixes = [
            ('ик', 5),
            ('чик', 6),
            ('к', 4),
            ('очк', 6),
            ('ечк', 6),
            ('оньк', 6),
            ('еньк', 6)
        ]

    def __call__(self, df: pd.DataFrame, text_column='row_text'):
        # Pre-compile regex and pre-process suffixes
        word_pattern = re.compile(r'\b[а-яё]+\b')
        suffixes = [(suf.lower(), min_len) for suf, min_len in self.diminutive_suffixes]

        # Vectorized text normalization
        texts = df[text_column].apply(normalize_text)

        # Cache morph analysis results
        @lru_cache(maxsize=10000)
        def analyze_word(word):
            parsed = self._morph.parse(word)[0]
            return (
                parsed.normal_form,
                any(tag in parsed.tag for tag in ['Name', 'Geox', 'Surn'])
            )

        def find_match(text):
            words = word_pattern.findall(text.lower())
            diminutives = set()

            for word in words:
                if len(word) < 5:
                    continue

                # Fast suffix check first
                if not any(word.endswith(suf) and len(word) >= min_len
                           for suf, min_len in suffixes):
                    continue

                # Then do morph analysis
                normal_form, is_name = analyze_word(word)

                if (normal_form != word) and not is_name:
                    diminutives.add(word)

            return ', '.join(sorted(diminutives)) if diminutives else None

        return texts.apply(find_match)