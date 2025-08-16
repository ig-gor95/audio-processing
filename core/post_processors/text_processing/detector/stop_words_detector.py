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
        self._threshold = 95

    def _compile_patterns(self) -> list[re.Pattern]:
        stopwords = self._config.get('patterns')
        patterns = defaultdict(list)
        for base_phrase, variants in stopwords.items():
            patterns[base_phrase].extend(variants)
        return patterns

    def __call__(self, df: pd.DataFrame, text_column='row_text'):
        texts = df[text_column].str.lower()
        all_variants = [v for variants in self._patterns.values() for v in variants]

        def find_match(text):
            # First check for exact matches (much faster)
            found_exact = {v for v in all_variants if v in text}
            if found_exact:
                return ', '.join(sorted(found_exact))

            found_stopwords = set()
            for base_phrase, variants in self._patterns.items():
                best_match = process.extractOne(text, variants, scorer=fuzz.partial_ratio, score_cutoff=self._threshold)
                if best_match:
                    found_stopwords.add(best_match[0])

            return ', '.join(sorted(found_stopwords)) if found_stopwords else None

        return texts.apply(find_match)