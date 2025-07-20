import re
from collections import defaultdict
from typing import Optional

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

    def __call__(self, text: str) ->str:
        text = normalize_text(text)
        abbrevs = self._patterns['abbreviations'].findall(text)

        return ', '.join(sorted(set(abbrevs)))