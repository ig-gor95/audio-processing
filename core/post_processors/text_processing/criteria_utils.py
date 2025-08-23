from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count

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

def _process_chunk(texts_chunk, min_name_length):
    chunk_results = []
    morph_vocab = MorphVocab()
    extractor = NamesExtractor(morph_vocab)
    morph_analyzer = pymorphy2.MorphAnalyzer()

    # Local cache for this chunk
    name_cache = defaultdict(bool)

    def is_valid_name(name):
        if name in name_cache:
            return name_cache[name]

        if len(name) < min_name_length or not name[0].isupper():
            name_cache[name] = False
            return False

        parsed = morph_analyzer.parse(name)
        name_cache[name] = any('Name' in p.tag or 'Surn' in p.tag for p in parsed)
        return name_cache[name]

    for text in texts_chunk:
        matches = extractor(text)
        valid_names = []

        for match in matches:
            if match.fact.first is None:
                continue

            name = match.fact.first
            if is_valid_name(name):
                valid_names.append(name)

        chunk_results.append(', '.join(valid_names) if valid_names else None)

    return chunk_results


def extract_valid_names(texts, min_name_length=3, n_workers=None):
    """Parallel name extraction with proper serialization."""
    n_workers = n_workers or cpu_count()

    text_values = texts.values
    text_chunks = np.array_split(text_values, n_workers * 4)

    worker = partial(_process_chunk, min_name_length=min_name_length)

    with Pool(n_workers) as pool:
        results = pool.map(worker, text_chunks)

    return pd.Series(
        [item for sublist in results for item in sublist],
        index=texts.index
    )
