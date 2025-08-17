import pandas as pd
import pymorphy3
from rapidfuzz import process, fuzz
from yaml_reader import ConfigLoader


class OrderPatternsDetector:
    def __init__(self, config_path: str = "post_processors/config/order_pattern.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy3.MorphAnalyzer()
        self.offer_patterns = self._compile_offer_patterns()
        self.processing_patterns = self._compile_processing_patterns()
        self.resume_patterns = self._compile_resume_patterns()
        self._threshold = 95

    def _compile_offer_patterns(self) -> list[str]:
        return self._config.get('patterns')['offer']

    def _compile_processing_patterns(self) -> list[str]:
        return self._config.get('patterns')['processing']

    def _compile_resume_patterns(self) -> list[str]:
        return self._config.get('patterns')['resume']

    def __call__(self, df: pd.DataFrame, text_column='row_text'):
        texts = df[text_column].str.lower().values

        def batch_match(texts, patterns):
            exact_matches = []
            for t in texts:
                matched_pattern = None
                for p in patterns:
                    pattern_words = set(p.split())
                    text_words = set(t.split())
                    if pattern_words.issubset(text_words):
                        matched_pattern = p
                        break
                exact_matches.append(matched_pattern)

            needs_fuzzy = [i for i, m in enumerate(exact_matches) if m is None]
            fuzzy_results = [None] * len(texts)

            for i in needs_fuzzy:
                result = process.extractOne(
                    texts[i],
                    patterns,
                    scorer=fuzz.token_set_ratio,
                    score_cutoff=self._threshold
                )
                fuzzy_results[i] = result[0] if result else None

            return [exact or fuzzy for exact, fuzzy in zip(exact_matches, fuzzy_results)]

        return (
            pd.Series(batch_match(texts, self.offer_patterns), index=df.index),
            pd.Series(batch_match(texts, self.processing_patterns), index=df.index),
            pd.Series(batch_match(texts, self.resume_patterns), index=df.index)
        )
