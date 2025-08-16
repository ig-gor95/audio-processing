import re
from collections import defaultdict
from typing import Optional

import pandas as pd
import pymorphy2

from core.post_processors.text_processing.criteria_utils import normalize_text
from yaml_reader import ConfigLoader


class AbbreviationsDetector:
    def __init__(self, config_path: str = "post_processors/config/parasites_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy2.MorphAnalyzer()
        self._patterns = self._compile_patterns()
        self._threshold = self._compile_patterns()

    def _compile_patterns(self) -> re.Pattern:
        parasite_patterns = self._config.get('speech_patterns')
        parasite_patterns['abbreviations'] = re.compile(parasite_patterns['abbreviations_pattern'])
        return parasite_patterns

    def __call__(self, df: pd.DataFrame, text_column='row_text') -> pd.Series:
        texts = df[text_column].apply(normalize_text)

        abbrev_pattern = self._patterns['abbreviations']

        if isinstance(abbrev_pattern, re.Pattern):
            all_matches = texts.str.findall(abbrev_pattern)

            result = all_matches.apply(lambda x: ', '.join(sorted(set(x))) if x else '')
            return result

        def find_match(text):
            abbrevs = abbrev_pattern.findall(text)
            return ', '.join(sorted(set(abbrevs))) if abbrevs else ''

        return texts.apply(find_match)