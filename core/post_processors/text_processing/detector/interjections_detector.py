import re
from typing import Optional

import pandas as pd

from core.post_processors.text_processing.criteria_utils import normalize_text
from yaml_reader import ConfigLoader


class InterjectionsDetector:
    def __init__(self, config_path: str = "post_processors/config/parasites_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self.exclude = ['эту', 'эти', 'это', 'эта', 'еще', 'ещё', 'а', 'его', 'эго']
        self.interjections_patterns = self._compile_patterns()

    def _compile_patterns(self) -> re.Pattern:
        parasite_patterns = self._config.get('speech_patterns')
        return re.compile(parasite_patterns['interjections'])

    def __call__(self, df: pd.DataFrame, text_column='row_text') -> pd.Series:
        texts = df[text_column].apply(normalize_text)

        interjection_pattern = self.interjections_patterns
        exclude_set = set(self.exclude)

        if isinstance(interjection_pattern, re.Pattern):
            all_matches = texts.str.findall(interjection_pattern)

            return all_matches.apply(
                lambda matches: ', '.join(
                    sorted(set(m for m in matches if m not in exclude_set))
                ) if matches else None
            )

        def find_match(text):
            matches = interjection_pattern.findall(text)
            filtered = [m for m in matches if m not in exclude_set]
            return ', '.join(sorted(set(filtered))) if filtered else None

        return texts.apply(find_match)