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

    def __call__(self, df: pd.DataFrame, text_column='row_text'):
        texts = df[text_column].str.lower()

        patterns = list(self._patterns)
        exact_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, patterns)) + r')\b')
        threshold = self._threshold

        def find_match(text):
            exact_matches = set(exact_pattern.findall(text))
            if exact_matches:
                return ', '.join(sorted(exact_matches))

            fuzzy_matches = process.extract(
                text,
                patterns,
                scorer=fuzz.partial_ratio,
                score_cutoff=threshold,
                limit=len(patterns)
            )
            found = {match[0] for match in fuzzy_matches}
            return ', '.join(sorted(found)) if found else None

        return texts.apply(find_match)