import re

import pandas as pd
import pymorphy3
from typing import List

from yaml_reader import ConfigLoader


class NamePatternsDetector:
    def __init__(self, config_path: str = "post_processors/config/name_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy3.MorphAnalyzer()
        self.name_patterns = self._compile_patterns()
        self._combined_pattern = self._create_combined_word_pattern()

    def _compile_patterns(self) -> List[str]:
        return self._config.get('patterns')

    def _create_combined_word_pattern(self) -> str:
        if not self.name_patterns:
            return r'(?!)'

        escaped_patterns = []
        for pattern in self.name_patterns:
            escaped_pattern = re.escape(pattern)
            escaped_patterns.append(r'\b' + escaped_pattern + r'\b')

        return '|'.join(escaped_patterns)

    def __call__(self, texts: pd.DataFrame) -> pd.Series:
        matches = texts.str.contains(self._combined_pattern, case=False, na=False, regex=True)

        return matches.astype(int)
