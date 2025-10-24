import re
import pandas as pd
import pymorphy3
from rapidfuzz import process, fuzz
from yaml_reader import ConfigLoader

class NonProfessionalPatternsDetector:
    def __init__(self, config_path: str = "post_processors/config/non_professional_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy3.MorphAnalyzer()
        self._patterns = self._compile_patterns()
        self._threshold = 95

    def _compile_patterns(self) -> list[str]:
        return self._config.get('patterns')

    def __call__(self, texts: pd.DataFrame):
        texts = texts.str.lower()

        patterns = list(self._patterns)
        exact_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, patterns)) + r')\b')
        threshold = self._threshold

        def find_match(text):
            exact_matches = set(exact_pattern.findall(text))
            if exact_matches:
                return ', '.join(sorted(exact_matches))
            else:
                return None

        return texts.apply(find_match)