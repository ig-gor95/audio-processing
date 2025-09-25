import re
import unicodedata

import pandas as pd
from yaml_reader import ConfigLoader


class InappropriatePhrasesDetector:
    def __init__(self, config_path: str = "post_processors/config/parasites_patterns.yaml"):
        self._config = ConfigLoader(config_path)
        self._patterns = self._compile_patterns()
        self._threshold = 97

    def _compile_patterns(self) -> str:
        return self._config.get('speech_patterns')['inappropriate_phrases']

    def __call__(self, texts: pd.Series):

        patterns = list(self._patterns)

        _dash_rx = re.compile(r"[-–—]+")
        _not_word_space = re.compile(r"[^\w\s]+", re.UNICODE)

        _latin2cyr = str.maketrans({
            "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к",
            "m": "м", "o": "о", "p": "р", "t": "т", "x": "х", "y": "у",
        })

        def norm_ru(s: str) -> str:
            s = unicodedata.normalize("NFKC", s).casefold()
            s = s.replace("ё", "е")
            s = s.translate(_latin2cyr)
            s = _dash_rx.sub(" ", s)
            s = _not_word_space.sub(" ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        prep = []
        for p in patterns:
            p_str = str(p)
            p_norm = norm_ru(p_str)
            tokens = [t for t in p_norm.split(" ") if t]
            prep.append((p_str, tokens))

        order = {p: i for i, p in enumerate(patterns)}

        def match_one(text) -> str | None:
            if text is None:
                return None
            t_norm = norm_ru(str(text))

            found = []
            for p_str, tokens in prep:
                if not tokens:
                    continue
                if len(tokens) == 1:
                    if tokens[0] in t_norm:
                        found.append(p_str)
                else:
                    pos = 0
                    ok = True
                    for tok in tokens:
                        idx = t_norm.find(tok, pos)
                        if idx == -1:
                            ok = False
                            break
                        pos = idx + len(tok)
                    if ok:
                        found.append(p_str)

            if not found:
                return None
            uniq_sorted = list(dict.fromkeys(sorted(found, key=lambda s: order[s])))
            return ", ".join(uniq_sorted)

        return texts.apply(match_one)