import re
from functools import lru_cache, partial

import numpy as np
import pandas as pd
import multiprocessing as mp
import pymorphy3
from typing import List, Pattern

from yaml_reader import ConfigLoader


class NamePatternsDetector:
    def __init__(self, config_path: str = "post_processors/config/name_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy3.MorphAnalyzer()
        self.await_request_patterns = self._compile_patterns()
        self._threshold = 95
        self._compiled_patterns = self._precompile_patterns()
        self._combined_pattern = self._create_combined_pattern()

    def _compile_patterns(self) -> List[str]:
        return self._config.get('patterns')

    def _precompile_patterns(self) -> List[Pattern]:
        return [re.compile(pattern, re.IGNORECASE) for pattern in self.await_request_patterns]

    def _create_combined_pattern(self) -> str:
        return '|'.join(f'({pattern})' for pattern in self.await_request_patterns)

    @lru_cache(maxsize=10000)
    def _normalize_text(self, text: str) -> str:
        return str(text).lower().strip()

    def _process_chunk(self, texts: pd.DataFrame, text_column: str) -> pd.Series:
        """Process a chunk of data"""
        matches = texts.str.contains(self._combined_pattern, regex=True, na=False)
        return matches.astype(int)

    def __call__(self, texts: pd.DataFrame, text_column='row_text') -> pd.Series:
        if len(texts) < 1000:
            matches = texts.str.contains(self._combined_pattern, regex=True, na=False)
            return matches.astype(int)

        num_cores = min(mp.cpu_count(), 8)
        df_split = np.array_split(texts, num_cores)

        with mp.Pool(num_cores) as pool:
            results = pool.map(
                partial(self._process_chunk, text_column=text_column),
                df_split
            )

        return pd.concat(results, ignore_index=True)