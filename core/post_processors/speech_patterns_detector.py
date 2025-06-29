import re
from collections import defaultdict
from typing import Optional

import pymorphy2

from core.post_processors.criteria_utils import normalize_text
from yaml_reader import ConfigLoader


class SpeechPatternsDetector:
    def __init__(self, config_path: str = "post_processors/config/parasites_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._patterns = self._compile_patterns()
        self._threshold = self._compile_patterns()
        self._morph = pymorphy2.MorphAnalyzer()

    def _compile_patterns(self) -> list[re.Pattern]:
        parasite_patterns = self._config.get('patterns')
        parasite_patterns['interjections'] = re.compile(parasite_patterns['interjections'])
        parasite_patterns['abbreviations'] = re.compile(parasite_patterns['abbreviations'])

        diminutives = set()
        sample_words = ['день', 'вопрос', 'документ', 'минута', 'секунда', 'деньги', 'зайка', 'солнышко']
        for word in sample_words:
            parsed = self._morph.parse(word)[0]
            for form in parsed.lexeme:
                if 'Dmns' in form.tag:
                    diminutives.add(form.word)

        parasite_patterns['diminutives'] = list(diminutives)

        return parasite_patterns

    def __call__(self, text: str) -> Optional[str]:
        text = normalize_text(text)
        issues = defaultdict(list)

        interjs = self._patterns['interjections'].findall(text)
        if interjs:
            issues['interjections'].extend(interjs)

        for word in self._patterns['parasites']:
            if re.search(rf'\b{re.escape(word)}\b', text):
                issues['parasites'].append(word)

        abbrevs = self._patterns['abbreviations'].findall(text)
        if abbrevs:
            issues['abbreviations'].extend(abbrevs)

        for word in self._patterns['slang']:
            if re.search(rf'\b{re.escape(word)}\b', text):
                issues['slang'].append(word)

        for phrase in self._patterns['inappropriate_phrases']:
            if phrase in text:
                issues['inappropriate_phrases'].append(phrase)

        words = re.findall(r'\b[а-яё]+\b', text.lower())

        safe_words = {
            'мама', 'папа', 'бабушка', 'дедушка',
            'дочка', 'сынок', 'котик', 'зайка',
            'морсакала', 'римского'
        }

        diminutive_suffixes = [
            ('ик', 5),
            ('чик', 6),
            ('к', 4),
            ('очк', 6),
            ('ечк', 6),
            ('оньк', 6),
            ('еньк', 6)
        ]

        for word in words:
            if len(word) < 5:
                continue

            if word in safe_words:
                continue

            parsed = self._morph.parse(word)[0]

            has_suffix = any(word.endswith(suf) and len(word) >= min_len
                             for suf, min_len in diminutive_suffixes)

            if has_suffix:
                normal_form = parsed.normal_form
                is_name = any(tag in parsed.tag for tag in ['Name', 'Geox', 'Surn'])

                if (normal_form != word) and not is_name:
                    issues['diminutives'].append(word)

        # Форматируем результаты
        formatted_issues = {}
        for category, items in issues.items():
            if items:
                formatted_issues[category] = ', '.join(sorted(set(items)))

        return formatted_issues if formatted_issues else None