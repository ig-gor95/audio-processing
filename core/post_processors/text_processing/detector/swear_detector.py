from collections import defaultdict

import pandas as pd

from yaml_reader import ConfigLoader
from typing import Optional
import re


class SwearDetector:
    def __init__(self, config_path: str = "post_processors/config/swear_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[re.Pattern]:
        base_words = self._config.get('patterns.swear-words', [])
        char_map = self._config.get('patterns.char_replacements', {})

        compiled_patterns = []
        for word in base_words:
            pattern = ''.join(char_map.get(c, c) for c in word.lower())
            compiled_patterns.append(
                re.compile(fr'\b{pattern}[а-яa-z]*\b', re.IGNORECASE)
            )
        return compiled_patterns

    def __call__(self, df: pd.DataFrame, text_column='row_text'):
        texts = df[text_column].str.lower()  # Vectorized lowercase conversion

        pattern_cache = defaultdict(set)
        for pattern in self._patterns:
            for found in pattern.findall('dummy'):  # Extract capture groups
                pattern_cache[found].add(pattern)

        combined_pattern = re.compile(
            '|'.join(f'(?:{pattern.pattern})' for pattern in self._patterns),
            flags=re.IGNORECASE
        )

        def find_match(text):
            matches = combined_pattern.findall(text)
            unique_matches = set(matches)
            return ', '.join(sorted(unique_matches)) if unique_matches else None

        return texts.apply(find_match)
