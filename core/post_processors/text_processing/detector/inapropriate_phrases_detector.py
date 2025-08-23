from rapidfuzz import fuzz, process
import pandas as pd
from core.post_processors.text_processing.criteria_utils import normalize_text
from yaml_reader import ConfigLoader


class InappropriatePhrasesDetector:
    def __init__(self, config_path: str = "post_processors/config/parasites_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._patterns = self._compile_patterns()
        self._threshold = 85

    def _compile_patterns(self) -> str:
        return self._config.get('speech_patterns')['inappropriate_phrases']

    def __call__(self, texts: pd.DataFrame):
        # Pre-compute all patterns for faster access
        patterns = list(self._patterns)

        def find_match(text):
            found_phrases = [p for p in patterns if p in text]
            if found_phrases:
                return ', '.join(sorted(set(found_phrases)))

            matches = process.extract(
                text,
                patterns,
                scorer=fuzz.partial_ratio,
                score_cutoff=self._threshold,
                limit=None
            )

            found_phrases = [match[0] for match in matches]
            return ', '.join(sorted(set(found_phrases))) if found_phrases else None

        return texts.apply(find_match)