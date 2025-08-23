import uuid
from typing import Optional

import pandas as pd
import pymorphy2
from natasha import MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter, NewsNERTagger, Doc

from core.dto.criteria import CriteriaConfig
from core.post_processors.text_processing.criteria_utils import find_phrase, find_phrases_in_df, extract_valid_names, \
    normalize_text
from core.post_processors.text_processing.detector.abbreviations_detector import AbbreviationsDetector
from core.post_processors.text_processing.detector.await_request_detector import AwaitRequestPatternsDetector
from core.post_processors.text_processing.detector.diminuties_detector import DiminutivesDetector
from core.post_processors.text_processing.detector.inapropriate_phrases_detector import InappropriatePhrasesDetector
from core.post_processors.text_processing.detector.interjections_detector import InterjectionsDetector
from core.post_processors.text_processing.detector.non_professional_patterns_detector import \
    NonProfessionalPatternsDetector
from core.post_processors.text_processing.detector.order_detector import OrderPatternsDetector
from core.post_processors.text_processing.detector.parasites_detector import ParasitesDetector
from core.post_processors.text_processing.detector.sales_detector import SalesDetector
from core.post_processors.text_processing.detector.slang_detector import SlangDetector
from core.post_processors.text_processing.detector.stop_words_detector import StopWordsDetector
from core.post_processors.text_processing.detector.swear_detector import SwearDetector
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from log_utils import setup_logger
from yaml_reader import ConfigLoader
from natasha import NamesExtractor

logger = setup_logger(__name__)


def parse_bool(bool_str: str) -> bool:
    return bool_str.lower() == 'true'


class DialogueAnalyzerPandas:
    def __init__(self):
        # Initialize detectors and analyzers
        self.swear_detector = SwearDetector()
        self.stop_words_detector = StopWordsDetector()
        self.interjections_detector = InterjectionsDetector()
        self.await_request_detector = AwaitRequestPatternsDetector()
        self.diminutives_detector = DiminutivesDetector()
        self.slang_detector = SlangDetector()
        self.abbreviations_detector = AbbreviationsDetector()
        self.inappropriate_phrases_detector = InappropriatePhrasesDetector()
        self.parasites_detector = ParasitesDetector()
        self.order_pattern_detector = OrderPatternsDetector()
        self.non_professional_patterns_detector = NonProfessionalPatternsDetector()
        self.sales_detector = SalesDetector()

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
        self.farewell_phrases = self.phrase_patterns['farewell']
        self.telling_name_phrases = self.phrase_patterns['telling-name-phrases']
        self.name_phrases = self.phrase_patterns['name-phrases']
        self.criteria_config = self.load_criteria_config()

        # Repository
        self.dialog_criteria_repo = DialogCriteriaRepository()

    def load_criteria_config(self) -> CriteriaConfig:
        config_data = ConfigLoader("../configs/criteria_detector_config.yaml").get("criteria")
        return CriteriaConfig(
            swear=parse_bool(config_data.get("swear")),
            name=parse_bool(config_data.get("name")),
            order_patterns=parse_bool(config_data.get("order_patterns")),
            interjections=parse_bool(config_data.get("interjections")),
            await_requests=parse_bool(config_data.get("await_request_patterns")),
            parasites=parse_bool(config_data.get("parasites")),
            non_professional_patterns=parse_bool(config_data.get("non_professional_patterns")),
            inappropriate_phrases=parse_bool(config_data.get("inappropriate_phrases")),
            greetings=parse_bool(config_data.get("greetings")),
            farewell=parse_bool(config_data.get("farewell")),
            abbreviations=parse_bool(config_data.get("abbreviations")),
            slang=parse_bool(config_data.get("slang")),
            stop_words=parse_bool(config_data.get("stop_words")),
            diminutives=parse_bool(config_data.get("diminutives")),
        )

    def extract_valid_names(self, text: str) -> Optional[str]:
        """Extract names considering introduction context."""
        morph_vocab = MorphVocab()
        extractor = NamesExtractor(morph_vocab)
        matches = extractor(text)
        morph_analyzer = pymorphy2.MorphAnalyzer()

        valid_names = []
        for match in matches:
            if match.fact.first is None:
                continue
            name = match.fact.first
            parsed = morph_analyzer.parse(name)
            if not parsed:
                continue

            is_proper_noun = any('Name' in p.tag or 'Surn' in p.tag for p in parsed)
            if (is_proper_noun and name[0].isupper()):  # should be capitalized
                valid_names.append(name)

        return ', '.join(valid_names) if valid_names else None

    def detect_sales(self, df: pd.DataFrame, text_column='row_text'):
        result = df['speaker_id'].copy()

        normalized_texts = df[text_column].apply(normalize_text)
        sales_mask = normalized_texts.apply(self.sales_detector.detect_by_text)

        result[sales_mask] = 'SALES'

        return result

    def analyze_dialogue(self):
        """Analyze dialogue text for various linguistic features."""
        dialog_criteria_repository = DialogCriteriaRepository()

        unprocessed_rows_pd = dialog_criteria_repository.pd_get_all_unprocessed_rows()
        unprocessed_rows_pd['detected_speaker_id'] = self.detect_sales(unprocessed_rows_pd)

        logger.info(f"Retrieved {len(unprocessed_rows_pd)} dialogue rows")
        texts = unprocessed_rows_pd['row_text'].apply(normalize_text)
        unprocessed_rows_pd['await_requests'] = self.await_request_detector(texts)
        logger.info(f"Recognised await_requests")
        unprocessed_rows_pd['farewell_phrase'] = find_phrases_in_df(texts, self.farewell_phrases)
        logger.info(f"Recognised farewell_phrase")
        unprocessed_rows_pd['telling_name_phrases'] = find_phrases_in_df(texts, self.telling_name_phrases)
        logger.info(f"Recognised telling_name_phrases")
        unprocessed_rows_pd['greeting_phrase'] = find_phrases_in_df(texts, self.greeting_phrases)
        logger.info(f"Recognised greeting_phrase")
        unprocessed_rows_pd['interjections'] = self.interjections_detector(texts)
        logger.info(f"Recognised interjections")
        unprocessed_rows_pd['abbreviations'] = self.abbreviations_detector(texts)
        logger.info(f"Recognised abbreviations")
        unprocessed_rows_pd['inappropriate_phrases'] = self.inappropriate_phrases_detector(texts)
        logger.info(f"Recognised inappropriate_phrases")
        unprocessed_rows_pd['slang'] = self.slang_detector(texts)
        logger.info(f"Recognised slang")
        unprocessed_rows_pd['parasite_words'] = self.parasites_detector(texts)
        logger.info(f"Recognised parasite_words")
        unprocessed_rows_pd['non_professional_phrases'] = self.non_professional_patterns_detector(texts)
        logger.info(f"Recognised non_professional_phrases")
        unprocessed_rows_pd['stop_words'] = self.stop_words_detector(texts)
        logger.info(f"Recognised stop_words")
        unprocessed_rows_pd['swear_words'] = self.swear_detector(texts)
        logger.info(f"Recognised swear_words")
        order, processing, resume = self.order_pattern_detector(texts)
        logger.info(f"Recognised order, processing, resume")
        unprocessed_rows_pd['order_offer'] = order
        logger.info(f"Recognised order_offer")
        unprocessed_rows_pd['order_processing'] = processing
        logger.info(f"Recognised order_processing")
        unprocessed_rows_pd['order_resume'] = resume
        logger.info(f"Recognised order_resume")
        unprocessed_rows_pd['diminutives'] = self.diminutives_detector(texts)
        logger.info(f"Recognised diminutives")
        unprocessed_rows_pd['found_name'] = extract_valid_names(texts)
        logger.info(f"Recognised found_name")
        unprocessed_rows_pd['dialog_criteria_id'] = [str(uuid.uuid4()) for _ in range(len(texts))]
        logger.info(f"Set up dialog_criteria_id")
        unprocessed_rows_pd['dialog_row_fk_id'] = unprocessed_rows_pd.pop('id')
        logger.info(f"Set detected_speaker_id")
        unprocessed_rows_pd['detected_speaker_id'] = self.detect_sales(unprocessed_rows_pd)

        dialog_criteria_pd = unprocessed_rows_pd[
            ['dialog_criteria_id', 'dialog_row_fk_id', 'greeting_phrase', 'found_name',
             'interjections', 'parasite_words', 'abbreviations', 'slang', 'telling_name_phrases',
             'inappropriate_phrases', 'diminutives', 'stop_words', 'swear_words',
             'non_professional_phrases', 'order_offer', 'order_processing', 'order_resume', 'await_requests']]
        logger.info(f"Saving results..")
        dialog_criteria_repository.save_pd(dialog_criteria_pd)
