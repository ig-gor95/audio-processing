import re
from collections import defaultdict
from typing import Optional

import pymorphy2

from core.post_processors.text_processing.criteria_utils import normalize_text
from yaml_reader import ConfigLoader


class DiminutivesDetector:
    def __init__(self, config_path: str = "post_processors/config/parasites_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy2.MorphAnalyzer()
        self.diminutive_suffixes = [
            ('ик', 5),
            ('чик', 6),
            ('к', 4),
            ('очк', 6),
            ('ечк', 6),
            ('оньк', 6),
            ('еньк', 6)
        ]

    def __call__(self, text: str) -> Optional[str]:
        text = normalize_text(text)
        diminutives = list()
        words = re.findall(r'\b[а-яё]+\b', text.lower())

        for word in words:
            if len(word) < 5:
                continue

            parsed = self._morph.parse(word)[0]

            has_suffix = any(word.endswith(suf) and len(word) >= min_len
                             for suf, min_len in self.diminutive_suffixes)

            if has_suffix:
                normal_form = parsed.normal_form
                is_name = any(tag in parsed.tag for tag in ['Name', 'Geox', 'Surn'])

                if (normal_form != word) and not is_name:
                    diminutives.append(word)

        return ', '.join(sorted(set(diminutives)))