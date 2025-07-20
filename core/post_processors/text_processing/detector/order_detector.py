import pymorphy3
from typing import Dict
from fuzzywuzzy import fuzz
from yaml_reader import ConfigLoader


class OrderPatternsDetector:
    def __init__(self, config_path: str = "post_processors/config/order_pattern.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy3.MorphAnalyzer()
        self.offer_patterns = self._compile_offer_patterns()
        self.processing_patterns = self._compile_processing_patterns()
        self.resume_patterns = self._compile_resume_patterns()
        self._threshold = 95

    def _compile_offer_patterns(self) -> list[str]:
        return self._config.get('patterns')['offer']

    def _compile_processing_patterns(self) -> list[str]:
        return self._config.get('patterns')['processing']

    def _compile_resume_patterns(self) -> list[str]:
        return self._config.get('patterns')['resume']

    def __call__(self, text: str) -> Dict:
        text = text.lower()
        result = {'offer': '', 'processing': '', 'resume': ''}

        for variant in self.offer_patterns:
            if (variant in text) or (fuzz.partial_ratio(variant, text) >= self._threshold):
                result['offer'] = variant
                break
        for variant in self.processing_patterns:
            if (variant in text) or (fuzz.partial_ratio(variant, text) >= self._threshold):
                result['processing'] = variant
                break
        for variant in self.resume_patterns:
            if (variant in text) or (fuzz.partial_ratio(variant, text) >= self._threshold):
                result['resume'] = variant
                break

        return result
