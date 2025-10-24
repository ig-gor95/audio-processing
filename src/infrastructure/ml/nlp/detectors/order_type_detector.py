import pandas as pd

from core.post_processors.text_processing.criteria_utils import find_phrases_in_df_full_word
from yaml_reader import ConfigLoader


class OrderTypePatternsDetector:
    def __init__(self, config_path: str = "post_processors/config/order_pattern.yaml"):
        self._config = ConfigLoader(config_path)

        raw = (self._config.get("patterns") or {}).get("order-type") or {}
        if not isinstance(raw, dict) or not raw:
            raise ValueError("Config must contain 'patterns.order-type' mapping.")

        self._type_order: list[str] = list(raw.keys())
        self._patterns_by_type: dict[str, list[str]] = {
            typ: list(phrases or []) for typ, phrases in raw.items()
        }

    def __call__(
        self,
        df: pd.DataFrame,
        text_column: str = "row_text",
        gate_column: str = "order_resume",
    ) -> pd.Series:
        if gate_column in df.columns:
            mask = df[gate_column].notna() & df[gate_column].astype(str).str.strip().ne("")
        else:
            mask = pd.Series(True, index=df.index)

        out = pd.Series([None] * len(df), index=df.index, name="order_type")
        if not mask.any():
            return out

        sub_texts = df.loc[mask, text_column].fillna("").astype(str).str.lower()

        hits_by_row = {idx: [] for idx in sub_texts.index}

        for typ in self._type_order:
            pats = self._patterns_by_type.get(typ, [])
            if not pats:
                continue

            matches = find_phrases_in_df_full_word(sub_texts, pats)

            hit_idx = matches.fillna("").astype(bool)
            hit_idx = hit_idx[hit_idx].index
            if len(hit_idx):
                for idx in hit_idx:
                    hits_by_row[idx].append(typ)

        for idx, cats in hits_by_row.items():
            if cats:
                out.at[idx] = ", ".join(cats)

        return out
