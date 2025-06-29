import re
import pymorphy3
from collections import defaultdict
from typing import Optional
from fuzzywuzzy import fuzz
from core.post_processors.criteria_utils import normalize_text
from yaml_reader import ConfigLoader

class StopWordsDetector:
    def __init__(self, config_path: str = "post_processors/config/stopwords_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy3.MorphAnalyzer()
        self._patterns = self._compile_patterns()
        self._threshold = 85

    def _compile_patterns(self) -> list[re.Pattern]:
        stopwords = self._config.get('patterns')
        patterns = defaultdict(list)
        for base_phrase, variants in stopwords.items():
            patterns[base_phrase].extend(variants)

            words = base_phrase.split()
            for word in words:
                parsed = self._morph.parse(word.lower())[0]
                for form in parsed.lexeme:
                    if form.word != word:
                        new_phrase = base_phrase.replace(word, form.word)
                        patterns[base_phrase].append(new_phrase)

        return patterns

    def __call__(self, text: str) -> Optional[str]:
        text = normalize_text(text)
        found_stopwords = set()

        for base_phrase, variants in self._patterns.items():
            for variant in variants:
                if (variant in text) or (fuzz.partial_ratio(variant, text) >= self._threshold):
                    found_stopwords.add(variant)
                    break

        return ', '.join(sorted(found_stopwords)) if found_stopwords else None