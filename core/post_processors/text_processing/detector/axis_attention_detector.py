import pandas as pd
import pymorphy3

from core.post_processors.text_processing.criteria_utils import find_phrases_in_df
from yaml_reader import ConfigLoader


class AxisAttentionPatternsDetector:
    def __init__(self, config_path: str = "post_processors/config/axis_attention_pattern.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy3.MorphAnalyzer()
        self.await_request_patterns = self._compile_patterns()
        self._threshold = 85

    def _compile_patterns(self) -> list[str]:
        return self._config.get('patterns')

    def __call__(self, df: pd.DataFrame, text_column='row_text') -> pd.Series:
        return find_phrases_in_df(df, self.await_request_patterns, threshold=self._threshold)
