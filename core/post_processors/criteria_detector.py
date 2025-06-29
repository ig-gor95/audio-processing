import re
import uuid

from natasha import MorphVocab, NamesExtractor, Doc, Segmenter, NewsEmbedding, NewsMorphTagger, NewsNERTagger

import pymorphy2

from core.post_processors.criteria_utils import find_phrase
from core.post_processors.swear_detector import SwearDetector
from core.post_processors.speech_patterns_detector import SpeechPatternsDetector
from core.post_processors.stop_words_detector import StopWordsDetector


def extract_valid_names(text, extractor, morph_vocab):
    """Извлекаем имена с учетом контекста представления"""
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    ner_tagger = NewsNERTagger(emb)

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tag_ner(ner_tagger)

    name_phrases = ['зовут', 'меня', 'это', 'мое имя', 'представьтесь', 'меня зовут']
    names = []

    for span in doc.spans:
        if span.type == 'PER':
            start_pos = span.start
            prev_words = ' '.join(token.text for token in doc.tokens[:start_pos]).lower()

            if any(phrase in prev_words for phrase in name_phrases):
                names.append(span.text)

    return names[0] if names else None

def analyze_dialogue_enhanced(dialogue_text, row_id):
    # Инициализация инструментов
    morph_vocab = MorphVocab()
    extractor = NamesExtractor(morph_vocab)
    morph = pymorphy2.MorphAnalyzer()

    # Фразы для проверки (с вариантами)
    greeting_phrases = [
        'здравствуйте', 'добрый день', 'добрый вечер',
        'доброе утро', 'приветствую вас', 'доброго времени суток', 'приветствую'
    ]

    farewell_phrases = [
        'всего доброго', 'хорошего дня', 'хорошего вечера',
        'до свидания', 'благодарю за обращение', 'приятного дня',
        'были рады помочь', 'хорошего настроения', 'был рад помочь', 'была рада помочь'
    ]

    # Загрузка словарей
    swear_detector = SwearDetector()
    stop_words_detector = StopWordsDetector()
    speech_patterns_detector = SpeechPatternsDetector()

    # Белый список допустимых уменьшительных форм
    safe_diminutives = {'мама', 'папа', 'бабушка', 'дедушка', 'дочка', 'сынок', 'котик'}

    # Обработка текста
    text = dialogue_text

    # Основные проверки
    greeting_phrase = find_phrase(text, greeting_phrases)
    farewell_phrase = find_phrase(text, farewell_phrases)
    found_name = extract_valid_names(text, extractor, morph_vocab)

    # Проверка качества речи
    speech_issues = speech_patterns_detector(text)
    stopwords_found = stop_words_detector(text)
    mat_words_found = swear_detector(text)

    # Дополнительная проверка всех уменьшительных форм
    words = re.findall(r'\b[а-яё]+\b', text.lower())
    diminutives = set()
    for word in words:
        parsed = morph.parse(word)[0]
        if 'Dmns' in parsed.tag and word not in safe_diminutives:
            diminutives.add(word)

    # Добавляем найденные уменьшительные формы к существующим проблемам
    if diminutives:
        if speech_issues is None:
            speech_issues = {}
        if 'diminutives' in speech_issues:
            existing = set(speech_issues['diminutives'].split(', '))
            diminutives.update(existing)
        speech_issues['diminutives'] = ', '.join(sorted(diminutives))

    # Формирование флагов
    greeting = 1 if greeting_phrase else 0
    farewell = 1 if farewell_phrase else 0
    name = 1 if found_name else 0
    has_stopwords = 1 if stopwords_found else 0
    has_mat = 1 if mat_words_found else 0
    has_diminutives = 1 if diminutives else 0

    return {
        "dialog_criteria_id": uuid.uuid4(),
        "dialog_row_fk_id": row_id,
        "has_greeting": greeting == 1,
        "greeting_phrase": greeting_phrase,
        "has_name": name == 1,
        "found_name": found_name,
        "has_farewell": farewell == 1,
        "farewell_phrase": farewell_phrase,
        "has_stopwords": has_stopwords == 1,
        "has_swear_words": has_mat == 1,
        "has_diminutives": has_diminutives == 1,
        "interjections": speech_issues.get('interjections', '') if speech_issues else '',
        "parasite_words": speech_issues.get('parasites', '') if speech_issues else '',
        "abbreviations": speech_issues.get('abbreviations', '') if speech_issues else '',
        "slang": speech_issues.get('slang', '') if speech_issues else '',
        "inappropriate_phrases": speech_issues.get('inappropriate_phrases', '') if speech_issues else '',
        "diminutives": speech_issues.get('diminutives', '') if speech_issues else '',
        "stop_words": stopwords_found or '',
        "swear_words": mat_words_found or ''
    }


def process_row_wrapper(args):
    row_text, row_id = args
    return analyze_dialogue_enhanced(row_text, row_id)


from multiprocessing import Pool
from tqdm import tqdm


def process_row_wrapper(args):
    row_text, row_id = args
    return analyze_dialogue_enhanced(row_text, row_id)


def process_rows_parallel(rows, processes=4):
    data = []
    count = 0

    args = [(row['row_text'], row['id']) for row in rows]

    with Pool(processes=processes) as pool:
        for result in tqdm(pool.imap(process_row_wrapper, args), total=len(rows)):
            data.append(result)
            count += 1

    return data