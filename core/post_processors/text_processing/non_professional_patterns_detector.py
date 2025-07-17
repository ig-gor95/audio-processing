import pymorphy3
from typing import Optional
from fuzzywuzzy import fuzz
from yaml_reader import ConfigLoader

class NonProfessionalPatternsDetector:
    def __init__(self, config_path: str = "post_processors/config/non_professional_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy3.MorphAnalyzer()
        self._patterns = self._compile_patterns()
        self._threshold = 95

    def _compile_patterns(self) -> list[str]:
        return self._config.get('patterns')

    def __call__(self, text: str) -> Optional[str]:
        text = text.lower()
        found_words = set()

        for variant in self._patterns:
            if (variant in text) or (fuzz.partial_ratio(variant, text) >= self._threshold):
                found_words.add(variant)
                break

        return ', '.join(sorted(found_words)) if found_words else None