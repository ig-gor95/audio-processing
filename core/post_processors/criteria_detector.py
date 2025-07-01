import re
import uuid
from typing import List

from natasha import MorphVocab, NamesExtractor, Doc, Segmenter, NewsEmbedding, NewsMorphTagger, NewsNERTagger

import pymorphy2

from core.post_processors.criteria_utils import find_phrase
from core.post_processors.swear_detector import SwearDetector
from core.post_processors.speech_patterns_detector import SpeechPatternsDetector
from core.post_processors.stop_words_detector import StopWordsDetector
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.entity.dialog_criteria import DialogCriteria
from core.repository.entity.dialog_rows import DialogRow

dialogCriteriaRepository = DialogCriteriaRepository()
def extract_valid_names(text):
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

def analyze_dialogue_enhanced(dialogue_text, row_id, speaker_id):
    dialogCriteriaRepository.delete_by_dialog_row_fk_id(row_id)
    if speaker_id == 'CLIENT':
        return None
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
    found_name = extract_valid_names(text)

    # Проверка качества речи
    speech_issues = speech_patterns_detector(text)
    stopwords_found = stop_words_detector(text)
    swear_words_found = swear_detector(text)

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

    return DialogCriteria(
        dialog_criteria_id= uuid.uuid4(),
        dialog_row_fk_id=row_id,
        greeting_phrase=greeting_phrase,
        found_name=found_name,
        farewell_phrase=farewell_phrase,
        interjections=speech_issues.get('interjections', '') if speech_issues else '',
        parasite_words=speech_issues.get('parasites', '') if speech_issues else '',
        abbreviations=speech_issues.get('abbreviations', '') if speech_issues else '',
        slang=speech_issues.get('slang', '') if speech_issues else '',
        inappropriate_phrases=speech_issues.get('inappropriate_phrases', '') if speech_issues else '',
        diminutives=speech_issues.get('diminutives', '') if speech_issues else '',
        stop_words=stopwords_found or '',
        swear_words= swear_words_found or '',
    )


def process_row_wrapper(args):
    return analyze_dialogue_enhanced(args['row_text'], args['row_id'], args['speaker_id'])


from multiprocessing import Pool
from tqdm import tqdm



def process_rows_parallel(rows: List[DialogRow], processes=4):
    data = []
    count = 0

    args = [{"row_text": row.row_text, "row_id": row.id, "speaker_id": row.speaker_id} for row in rows]

    with Pool(processes=processes) as pool:
        for result in tqdm(pool.imap(process_row_wrapper, args), total=len(rows)):
            if result is not None:
                data.append(result)
            count += 1

    return data