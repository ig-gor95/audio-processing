import re
import uuid

import pandas as pd
from natasha import MorphVocab, NamesExtractor, Doc, Segmenter, NewsEmbedding, NewsMorphTagger, NewsNERTagger

from fuzzywuzzy import fuzz
from collections import defaultdict
import pymorphy2

from core.post_processors.criteria_utils import normalize_text, find_phrase
from yaml_reader import ConfigLoader


def create_stopword_patterns(morph):
    config = ConfigLoader("config/stopwords_patterns.yaml").get('pattern')
    """Создает паттерны для поиска стоп-слов с учетом склонений"""
    stopwords = {
        'не знаю': ['не знаю', 'незнаю', 'не знаем', 'я не знаю', 'знать не знаю', 'не зная', 'неизвестно'],
        'не владею': ['не владею', 'не владеем', 'не владеет', 'не владеют', 'не владения'],
        'нет информации': ['у меня нет этой информации', 'информации нет', 'нет сведений', 'нет данных', 'инфы нет'],
        'у нас нет': ['у нас нет', 'нету у нас', 'не имеется', 'отсутствует', 'не в наличии'],
        'не могу': ['я не могу', 'не могу', 'не можем', 'не смогу', 'не смог', 'не смогла', 'не смогли'],
        'запрещено': ['запрещено', 'нельзя', 'воспрещается', 'запрещается', 'не разрешается'],
        'вы должны': ['вы должны', 'вам следует', 'вам надо', 'вам нужно', 'вам необходимо'],
        'я же говорил': ['я же вам сказал', 'я же говорил', 'я уже сказал', 'повторяю', 'еще раз повторяю'],
        'невозможно': ['невозможно', 'не представляется возможным', 'нет возможности', 'никак нельзя', 'нереально'],
        'не умею': ['не умею', 'не умеем', 'не умете', 'не умение', 'не обучен'],
        'не решаю': ['не решаю', 'не решаем', 'не решает', 'не решение', 'не в моих полномочиях'],
        'не могу помочь': ['не могу помочь', 'не помогу', 'не помогаю', 'помочь не могу', 'ничем не могу помочь'],
        'не знаю что делать': ['не знаю что делать', 'не знаю как быть', 'без понятия', 'не представляю',
                               'в затруднении'],
        'никаких вариантов': ['никаких вариантов', 'нет вариантов', 'вариантов нет', 'нет решений', 'без вариантов'],
        'не могу сказать': ['не могу сказать', 'не скажу', 'не вправе сказать', 'не могу сообщить', 'не разглашается'],
        'не моя компетенция': ['не моя компетенция', 'не в моей компетенции', 'не по моей части', 'не моего уровня',
                               'не в моих силах'],
        'не могу прокомментировать': ['не могу прокомментировать', 'без комментариев', 'не комментирую',
                                      'воздержусь от комментариев', 'не дам комментарий'],
        'не могу подтвердить': ['не могу подтвердить', 'не подтверждаю', 'подтвердить не могу', 'нет подтверждения',
                                'не подтверждено'],
        'не могу выполнить': ['не могу выполнить', 'не выполню', 'не исполню', 'не выполняется', 'не исполнимо'],
        'не могу решить': ['не могу решить', 'не решу', 'решить не могу', 'не решается', 'не разрешимо'],
        'не могу предоставить': ['не могу предоставить', 'не предоставлю', 'предоставить не могу', 'не предоставляется',
                                 'нет возможности предоставить']
    }

    patterns = defaultdict(list)

    for base_phrase, variants in stopwords.items():
        # Добавляем явные варианты
        patterns[base_phrase].extend(variants)

        # Генерируем формы слов с помощью pymorphy2
        words = base_phrase.split()
        for word in words:
            parsed = morph.parse(word)[0]
            for form in parsed.lexeme:
                if form.word != word:
                    new_phrase = base_phrase.replace(word, form.word)
                    patterns[base_phrase].append(new_phrase)

    return patterns


def create_speech_patterns(morph):
    """Создает паттерны для поиска всех нежелательных элементов речи"""
    patterns = {
        # Междометия и звуки
        'interjections': re.compile(
            r'\b(а+[^а\s]{0,2}|э+[^э\s]{0,2}|мм+[^м\s]{0,2}|уу+[^у\s]{0,2}|ё+[^ё\s]{0,2}|е+[^е\s]{0,2})\b'),

        # Слова-паразиты и раздражители
        'parasites': [
            'ну', 'это', 'вот', 'значит', 'так сказать', 'собственно', 'вообще',
            'короче', 'понимаете', 'в общем', 'то есть', 'в принципе', 'слышите',
            'блин', 'чё', 'чё-то', 'да', 'ага', 'угу', 'ладно', 'окей', 'ясно',
            'типа', 'как бы', 'получается', 'фактически', 'собственно говоря',
            'вот именно', 'в некотором роде', 'как сказать', 'вроде как', 'вроде бы'
        ],

        # Сокращения
        'abbreviations': re.compile(r'\b(спс|пжл|плз|нзч|оч|сч|всм|хз|лол|кек|рофл|имхо|збс|пнх|очк|втф)\b'),

        # Сленг и неформальные выражения
        'slang': [
            'хай', 'хелло', 'го', 'по кайфу', 'норм', 'ок', 'окся', 'офигенно',
            'прикольно', 'круто', 'отстой', 'фигня', 'хрень', 'бред', 'зашибись',
            'ништяк', 'кайф', 'тащусь', 'ржач', 'прусь', 'агонь', 'жжет', 'пеши',
            'рофлишь', 'ауф', 'кринж', 'краш', 'чилить', 'париться', 'залипать'
        ],

        # Неуместные речевые обороты
        'inappropriate_phrases': [
            'ты должен', 'вы должны', 'обязан', 'срочно', 'быстро', 'немедленно',
            'не вопрос', 'без проблем', 'не парься', 'забей', 'фиг с ним', 'по ходу',
            'как бы не так', 'да ладно', 'не гони', 'не вешай нос', 'держи хвост пистолетом',
            'в натуре', 'реально', 'конкретно', 'по факту', 'по-любому', 'жесть'
        ]
    }

    # Создаем морфологический анализатор для уменьшительно-ласкательных форм
    diminutives = set()
    sample_words = ['день', 'вопрос', 'документ', 'минута', 'секунда', 'деньги', 'зайка', 'солнышко']
    for word in sample_words:
        parsed = morph.parse(word)[0]
        for form in parsed.lexeme:
            if 'Dmns' in form.tag:  # Diminutive tag
                diminutives.add(form.word)

    patterns['diminutives'] = list(diminutives)

    return patterns


def check_stopwords(text, stopword_patterns, threshold=85):
    """Проверяет наличие стоп-слов с учетом ошибок и склонений"""
    text = normalize_text(text)
    found_stopwords = set()
    issues = defaultdict(list)

    for base_phrase, variants in stopword_patterns.items():
        for variant in variants:
            # Проверяем как точное вхождение, так и похожие фразы
            if (variant in text) or (fuzz.partial_ratio(variant, text) >= threshold):
                found_stopwords.add(base_phrase)
                break

    return ', '.join(sorted(found_stopwords)) if found_stopwords else None


def check_speech_quality(text, speech_patterns, morph):
    """Проверяет речь на соответствие требованиям"""
    text = normalize_text(text)
    issues = defaultdict(list)

    # Проверяем междометия
    interjs = speech_patterns['interjections'].findall(text)
    if interjs:
        issues['interjections'].extend(interjs)

    # Проверяем слова-паразиты
    for word in speech_patterns['parasites']:
        if re.search(rf'\b{re.escape(word)}\b', text):
            issues['parasites'].append(word)

    # Проверяем сокращения
    abbrevs = speech_patterns['abbreviations'].findall(text)
    if abbrevs:
        issues['abbreviations'].extend(abbrevs)

    # Проверяем сленг
    for word in speech_patterns['slang']:
        if re.search(rf'\b{re.escape(word)}\b', text):
            issues['slang'].append(word)

    # Проверяем неуместные обороты
    for phrase in speech_patterns['inappropriate_phrases']:
        if phrase in text:
            issues['inappropriate_phrases'].append(phrase)

    # УСИЛЕННАЯ проверка уменьшительно-ласкательных форм
    words = re.findall(r'\b[а-яё]+\b', text.lower())

    # Полный список безопасных слов (можно расширять)
    safe_words = {
        'мама', 'папа', 'бабушка', 'дедушка',
        'дочка', 'сынок', 'котик', 'зайка',
        'морсакала', 'римского'  # специальные исключения
    }

    # Основные суффиксы уменьшительных форм с минимальной длиной
    diminutive_suffixes = [
        ('ик', 5),  # документик (8 букв)
        ('чик', 6),  # договорчик
        ('к', 4),  # проблемка
        ('очк', 6),  # бумажечка
        ('ечк', 6),  # печенюшечка
        ('оньк', 6),  # легонький
        ('еньк', 6)  # хорошенький
    ]

    for word in words:
        # Пропускаем короткие слова (меньше 5 букв)
        if len(word) < 5:
            continue

        # Пропускаем слова из белого списка
        if word in safe_words:
            continue

        # Полный морфологический разбор
        parsed = morph.parse(word)[0]

        # Критерии уменьшительной формы:
        # 1. Есть характерный суффикс достаточной длины
        # 2. Нормальная форма отличается от слова
        # 3. Не имя собственное

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


# Добавим после импортов словарь матов и функции для их проверки
MAT_WORDS_BASE = [
    'бля', 'блять', 'еб', 'нахуй', 'пизд', 'хуй', 'хуё', 'заеб', 'ебан',
    'мудак', 'гандон', 'шлюх', 'сука', 'падл', 'долбоёб', 'залуп', 'пидор',
    'ебал', 'выеб', 'ебанут', 'ебис', 'ебуч', 'пизду', 'хуесос', 'хуеплёт',
    'манда', 'мандавош', 'ссанин', 'ссать', 'говн', 'дерьм', 'залупа'
]


def load_mat_words(file_path=None):
    """Загрузка матерных слов из файла или использование базового списка"""
    if file_path:
        try:
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                return df.iloc[:, 0].tolist()
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Ошибка загрузки файла матов: {e}")
    return MAT_WORDS_BASE


def create_mat_patterns(mat_words):
    """Создание regex-паттернов для поиска матов с учетом замен символов"""
    patterns = []
    char_replacements = {
        'а': '[а@aа́]',
        'б': '[б6b]',
        'в': '[вv]',
        'г': '[гg]',
        'д': '[дd]',
        'е': '[еёe]',
        'з': '[з3z]',
        'и': '[иuі]',
        'к': '[кk]',
        'о': '[о0oо́]',
        'п': '[пp]',
        'р': '[рr]',
        'с': '[сc]',
        'т': '[тt]',
        'у': '[уy]',
        'х': '[хx]',
        'ч': '[ч4]'
    }

    for word in mat_words:
        pattern = []
        for char in word:
            pattern.append(char_replacements.get(char, char))
        patterns.append(re.compile(fr'\b{"".join(pattern)}[а-яa-z]*\b', re.IGNORECASE))
    return patterns


def check_mat_words(text, mat_patterns):
    """Поиск матерных слов в тексте"""
    found = set()
    for pattern in mat_patterns:
        matches = pattern.findall(text.lower())
        if matches:
            found.update(matches)
    return ', '.join(sorted(found)) if found else None


def analyze_dialogue_enhanced(dialogue_text, row_id, mat_words_file=None):
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
    mat_words = load_mat_words(mat_words_file)
    mat_patterns = create_mat_patterns(mat_words)
    speech_patterns = create_speech_patterns(morph)
    stopword_patterns = create_stopword_patterns(morph)

    # Белый список допустимых уменьшительных форм
    safe_diminutives = {'мама', 'папа', 'бабушка', 'дедушка', 'дочка', 'сынок', 'котик'}

    # Обработка текста
    text = dialogue_text
    # Основные проверки
    greeting_phrase = find_phrase(text, greeting_phrases)
    farewell_phrase = find_phrase(text, farewell_phrases)
    found_name = extract_valid_names(text, extractor, morph_vocab)

    # Проверка качества речи
    speech_issues = check_speech_quality(text, speech_patterns, morph)
    stopwords_found = check_stopwords(text, stopword_patterns)
    mat_words_found = check_mat_words(text, mat_patterns)

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

    # Формирование результатов
    issues_str = {
        'interjections': speech_issues.get('interjections', '') if speech_issues else '',
        'parasite_words': speech_issues.get('parasites', '') if speech_issues else '',
        'abbreviations': speech_issues.get('abbreviations', '') if speech_issues else '',
        'slang': speech_issues.get('slang', '') if speech_issues else '',
        'inappropriate_phrases': speech_issues.get('inappropriate_phrases', '') if speech_issues else '',
        'diminutives': speech_issues.get('diminutives', '') if speech_issues else '',
        'stop_words': stopwords_found or '',
        'mat_words': mat_words_found or ''
    }

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

    # Prepare arguments
    args = [(row['row_text'], row['id']) for row in rows]

    with Pool(processes=processes) as pool:
        # Process with progress bar
        for result in tqdm(pool.imap(process_row_wrapper, args), total=len(rows)):
            data.append(result)
            count += 1

    return data