from fuzzywuzzy import fuzz

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