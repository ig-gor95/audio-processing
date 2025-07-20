import re
import uuid
from typing import Set, Optional

import pymorphy2
from natasha import MorphVocab, NamesExtractor, NewsEmbedding, NewsMorphTagger, Segmenter, NewsNERTagger, Doc

from core.dto.criteria import CriteriaConfig
from core.post_processors.text_processing.criteria_utils import find_phrase
from core.post_processors.text_processing.detector.abbreviations_detector import AbbreviationsDetector
from core.post_processors.text_processing.detector.diminuties_detector import DiminutivesDetector
from core.post_processors.text_processing.detector.inapropriate_phrases_detector import InappropriatePhrasesDetector
from core.post_processors.text_processing.detector.interjections_detector import InterjectionsDetector
from core.post_processors.text_processing.detector.non_professional_patterns_detector import NonProfessionalPatternsDetector
from core.post_processors.text_processing.detector.order_detector import OrderPatternsDetector
from core.post_processors.text_processing.detector.parasites_detector import ParasitesDetector
from core.post_processors.text_processing.detector.slang_detector import SlangDetector
from core.post_processors.text_processing.detector.stop_words_detector import StopWordsDetector
from core.post_processors.text_processing.detector.swear_detector import SwearDetector
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.entity.dialog_criteria import DialogCriteria
from yaml_reader import ConfigLoader


class DialogueAnalyzer:
    def __init__(self):
        # Initialize detectors and analyzers
        self.swear_detector = SwearDetector()
        self.stop_words_detector = StopWordsDetector()
        self.interjections_detector = InterjectionsDetector()
        self.diminutives_detector = DiminutivesDetector()
        self.slang_detector = SlangDetector()
        self.abbreviations_detector = AbbreviationsDetector()
        self.inappropriate_phrases_detector = InappropriatePhrasesDetector()
        self.parasites_detector = ParasitesDetector()
        self.order_pattern_detector = OrderPatternsDetector()
        self.non_professional_patterns_detector = NonProfessionalPatternsDetector()

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
        self.criteria_config = self.load_criteria_config()

        # Repository
        self.dialog_criteria_repo = DialogCriteriaRepository()

    def load_criteria_config(self) -> CriteriaConfig:
        config_data = ConfigLoader("../criteria_detector_config.yaml").get("criteria")
        return CriteriaConfig(
            swear=bool(config_data.get("swear")),
            name=bool(config_data.get("name")),
            order_patterns=bool(config_data.get("order_patterns")),
            interjections=bool(config_data.get("interjections")),
            parasites=bool(config_data.get("parasites")),
            non_professional_patterns=bool(config_data.get("non_professional_patterns")),
            inappropriate_phrases=bool(config_data.get("inappropriate_phrases")),
            greetings=bool(config_data.get("greetings")),
            farewell=bool(config_data.get("farewell")),
            abbreviations=bool(config_data.get("abbreviations")),
            slang=bool(config_data.get("slang")),
            stop_words=bool(config_data.get("stop_words")),
            diminutives=bool(config_data.get("diminutives")),
        )

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

    def analyze_dialogue(self, text: str, row_id: uuid.UUID) -> DialogCriteria:
        """Analyze dialogue text for various linguistic features."""
        existing_criteria = self.dialog_criteria_repo.find_by_row_fk_id(row_id)
        greeting_phrase = find_phrase(text, self.greeting_phrases) if self.criteria_config.greetings else None
        farewell_phrase = find_phrase(text, self.farewell_phrases) if self.criteria_config.farewell else None
        found_name = self.extract_valid_names(text) if self.criteria_config.name else None

        interjections = self.interjections_detector(text) if self.criteria_config.interjections else None
        abbreviations = self.abbreviations_detector(text) if self.criteria_config.abbreviations else None
        inappropriate_phrases = self.inappropriate_phrases_detector(text) if self.criteria_config.abbreviations else None
        slang_words = self.slang_detector(text) if self.criteria_config.slang else None
        parasites = self.parasites_detector(text) if self.criteria_config.parasites else None
        non_professional_patterns = self.non_professional_patterns_detector(text) if self.criteria_config.non_professional_patterns else None
        stopwords_found = self.stop_words_detector(text) if self.criteria_config.stop_words else None
        swear_words_found = self.swear_detector(text) if self.criteria_config.swear else None
        order_result = self.order_pattern_detector(text) if self.criteria_config.order_patterns else None
        diminutives = self.diminutives_detector(text) if self.criteria_config.diminutives else set()

        if existing_criteria:
            if self.criteria_config.name:
                existing_criteria.found_name = found_name
            if self.criteria_config.greetings:
                existing_criteria.greeting_phrase = greeting_phrase
            if self.criteria_config.farewell:
                existing_criteria.farewell_phrase = farewell_phrase
            if self.criteria_config.interjections:
                existing_criteria.interjections = interjections
            if self.criteria_config.parasites:
                existing_criteria.parasite_words = parasites
            if self.criteria_config.abbreviations:
                existing_criteria.abbreviations = abbreviations
            if self.criteria_config.diminutives:
                existing_criteria.diminutives = diminutives
            if self.criteria_config.slang:
                existing_criteria.slang = slang_words
            if self.criteria_config.inappropriate_phrases:
                existing_criteria.inappropriate_phrases = inappropriate_phrases
            if self.criteria_config.order_patterns:
                existing_criteria.order_offer = order_result['offer']
                existing_criteria.order_processing = order_result['processing']
                existing_criteria.order_resume = order_result['resume']

        return DialogCriteria(
            dialog_criteria_id=uuid.uuid4(),
            dialog_row_fk_id=row_id,
            greeting_phrase=greeting_phrase,
            found_name=found_name,
            farewell_phrase=farewell_phrase,
            interjections=interjections,
            parasite_words=parasites,
            abbreviations=abbreviations,
            slang=slang_words,
            inappropriate_phrases=inappropriate_phrases,
            non_professional_phrases=non_professional_patterns,
            diminutives=diminutives,
            stop_words=stopwords_found,
            swear_words=swear_words_found,
            order_offer=order_result['offer'],
            order_processing=order_result['processing'],
            order_resume=order_result['resume']
        )