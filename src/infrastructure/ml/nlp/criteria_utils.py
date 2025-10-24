import re
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Union, Sequence, Optional

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
import pymorphy2
from natasha import MorphVocab, NamesExtractor


def normalize_text(text):
    """Нормализация текста для обработки с учетом возможных ошибок"""
    text = text.lower()
    replacements = {
        'здрасте': 'здравствуйте',
        'здрасьте': 'здравствуйте',
        'здрась': 'здравствуйте',
        'драсьте': 'здравствуйте',
        'доброго дня': 'добрый день',
        'добраго вечера': 'добрый вечер',
        'всего хорошего': 'всего доброго',
        'в общем-то': 'в общем',
        'короче говоря': 'короче',
        'типа того': 'типа',
        'как-бы': 'как бы',
        'т.е.': 'то есть',
        'т.д.': 'так далее',
        'т.п.': 'тому подобное'
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text


def find_phrase(text, phrases, threshold=80):
    text = normalize_text(text)
    for phrase in phrases:
        if fuzz.partial_ratio(phrase, text) >= threshold:
            return phrase
    return None


def find_phrases_in_df(normalized_texts, phrases, threshold=80):
    return normalized_texts.apply(
        lambda text: (
            process.extractOne(
                text,
                phrases,
                scorer=fuzz.partial_ratio,
                score_cutoff=threshold
            )[0] if process.extractOne(
                text,
                phrases,
                scorer=fuzz.partial_ratio,
                score_cutoff=threshold
            ) else None
        )
    )

def find_phrases_in_df_full_word(normalized_texts: pd.Series, phrases) -> pd.Series:
    texts = normalized_texts.astype(str)
    phrases_list = [p for p in phrases if isinstance(p, str) and p.strip()]
    if not phrases_list:
        return pd.Series([None] * len(texts), index=texts.index)

    def _phrase_to_pattern(p: str) -> str:
        tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", p)
        if not tokens:
            return ""
        inner = r"(?:\W+)+".join(map(re.escape, tokens))
        return rf"(?<!\w)(?:{inner})(?!\w)"

    parts = [_phrase_to_pattern(p) for p in phrases_list]
    parts = [p for p in parts if p]
    if not parts:
        return pd.Series([None] * len(texts), index=texts.index)

    pattern = re.compile(rf"({'|'.join(parts)})", flags=re.IGNORECASE | re.UNICODE)

    matches = texts.str.extract(pattern, expand=False)
    return matches.where(matches.notna(), None)

def extract_valid_names_optimized(
    texts: Union[pd.Series, Sequence[Optional[str]]],
    min_name_length: int = 3
) -> Union[pd.Series, list]:
    """Ultra-optimized version with aggressive pre-screening.
    Accepts a pandas Series or any sequence of strings. Returns a Series (if input is Series)
    or a list (if input is a plain sequence)."""

    # Init heavy objects once per call
    morph_vocab = MorphVocab()
    extractor = NamesExtractor(morph_vocab)
    morph_analyzer = pymorphy2.MorphAnalyzer()
    name_cache = defaultdict(bool)

    # Pre-compile patterns for faster filtering
    uppercase_pattern = re.compile(r'[A-ZА-ЯЁ]')
    name_like_pattern = re.compile(
        r'\b[A-ZА-ЯЁ][a-zа-яё]{%d,}\b' % max(min_name_length - 1, 0)
    )

    def is_valid_name(name: str) -> bool:
        if not name:
            return False
        if name in name_cache:
            return name_cache[name]

        if len(name) < min_name_length or not name[0].isupper():
            name_cache[name] = False
            return False

        parsed = morph_analyzer.parse(name)
        # True if any parse has grammemes for first/last name
        ok = any(('Name' in p.tag) or ('Surn' in p.tag) for p in parsed)
        name_cache[name] = bool(ok)
        return name_cache[name]

    # Support both Series and plain sequences
    is_series = isinstance(texts, pd.Series)
    iterable = texts.values if is_series else texts

    results = []
    for text in iterable:
        if not isinstance(text, str) or not text.strip():
            results.append(None)
            continue

        # Ultra-fast pre-screening: skip if no uppercase letters at all
        if not uppercase_pattern.search(text):
            results.append(None)
            continue

        # Additional screening: check for name-like patterns
        if not name_like_pattern.search(text):
            results.append(None)
            continue

        matches = extractor(text)
        valid_names = []

        for match in matches:
            first = getattr(match.fact, 'first', None)
            if first and is_valid_name(first):
                valid_names.append(first)

        results.append(', '.join(sorted(set(valid_names))) if valid_names else None)

    return pd.Series(results, index=texts.index) if is_series else results
