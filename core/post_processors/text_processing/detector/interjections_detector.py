import re
from typing import Optional

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

    def __call__(self, text: str) -> Optional[str]:
        text = normalize_text(text)
        interjections = self.interjections_patterns.findall(text)
        interjections = [item for item in interjections if item not in self.exclude]

        return ', '.join(sorted(set(interjections)))