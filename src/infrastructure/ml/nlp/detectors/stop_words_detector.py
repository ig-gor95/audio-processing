import re
from rapidfuzz import fuzz, process

import pandas as pd
import pymorphy3
from collections import defaultdict
from yaml_reader import ConfigLoader

class StopWordsDetector:
    def __init__(self, config_path: str = "post_processors/config/stopwords_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy3.MorphAnalyzer()
        self._patterns = self._compile_patterns()
        self._threshold = 98

    def _compile_patterns(self) -> list[re.Pattern]:
        stopwords = self._config.get('patterns')
        patterns = defaultdict(list)
        for base_phrase, variants in stopwords.items():
            patterns[base_phrase].extend(variants)
        return patterns

    def __call__(self, texts: pd.Series):
        texts = texts.str.lower()
        all_variants = [v for variants in self._patterns.values() for v in variants]

        def find_match(text):
            # Проверка на точное совпадение целого слова
            found_exact = {v for v in all_variants if re.search(rf'\b{re.escape(v)}\b', text)}
            if found_exact:
                return ', '.join(sorted(found_exact))
            else:
                return None



        return texts.apply(find_match)