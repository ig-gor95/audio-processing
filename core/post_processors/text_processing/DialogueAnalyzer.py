import re
import uuid
from typing import Set, Optional

import pymorphy2
from natasha import MorphVocab, NamesExtractor, NewsEmbedding, NewsMorphTagger, Segmenter, NewsNERTagger, Doc

from core.post_processors.text_processing.criteria_utils import find_phrase
from core.post_processors.text_processing.speech_patterns_detector import SpeechPatternsDetector
from core.post_processors.text_processing.stop_words_detector import StopWordsDetector
from core.post_processors.text_processing.swear_detector import SwearDetector
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.entity.dialog_criteria import DialogCriteria
from yaml_reader import ConfigLoader


class DialogueAnalyzer:
    def __init__(self):
        # Initialize detectors and analyzers
        self.swear_detector = SwearDetector()
        self.stop_words_detector = StopWordsDetector()
        self.speech_patterns_detector = SpeechPatternsDetector()

        # Initialize NLP tools
        self.morph_vocab = MorphVocab()
        self.extractor = NamesExtractor(self.morph_vocab)
        self.morph = pymorphy2.MorphAnalyzer()

        # Initialize Natasha components
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)

        # Load phrase patterns
        self.phrase_patterns = ConfigLoader("post_processors/config/phrase_patterns.yaml").get("patterns")
        self.greeting_phrases = self.phrase_patterns['greetings']
        self.farewell_phrases = self.phrase_patterns['farewell']
        self.name_phrases = self.phrase_patterns['name-phrases']

        # Repository
        self.dialog_criteria_repo = DialogCriteriaRepository()

    def extract_valid_names(self, text: str) -> Optional[str]:
        """Extract names considering introduction context."""
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.tag_ner(self.ner_tagger)

        names = []
        for span in doc.spans:
            if span.type == 'PER':
                start_pos = span.start
                prev_words = ' '.join(token.text for token in doc.tokens[:start_pos]).lower()
                if any(phrase in prev_words for phrase in self.name_phrases):
                    names.append(span.text)

        return names[0] if names else None

    def find_diminutives(self, text: str) -> Set[str]:
        words = re.findall(r'\b[а-яё]+\b', text.lower())
        diminutives = set()

        for word in words:
            try:
                parsed = self.morph.parse(word)[0]
                if 'Dmns' in parsed.tag:
                    diminutives.add(word)
            except IndexError:
                continue

        return diminutives

    def analyze_dialogue(self, text: str, row_id: uuid.UUID) -> DialogCriteria:
        """Analyze dialogue text for various linguistic features."""
        greeting_phrase = find_phrase(text, self.greeting_phrases)
        farewell_phrase = find_phrase(text, self.farewell_phrases)
        found_name = self.extract_valid_names(text)

        speech_issues = self.speech_patterns_detector(text)
        stopwords_found = self.stop_words_detector(text)
        swear_words_found = self.swear_detector(text)

        diminutives = self.find_diminutives(text)
        if diminutives:
            speech_issues = speech_issues or {}
            existing = set(speech_issues.get('diminutives', '').split(', '))
            existing.update(diminutives)
            speech_issues['diminutives'] = ', '.join(sorted(existing))

        return DialogCriteria(
            dialog_criteria_id=uuid.uuid4(),
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
            swear_words=swear_words_found or '',
        )