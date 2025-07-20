from dataclasses import dataclass


@dataclass
class CriteriaConfig:
    swear: bool
    inappropriate_phrases: bool
    interjections: bool
    parasites: bool
    abbreviations: bool
    slang: bool
    order_patterns: bool
    stop_words: bool
    non_professional_patterns: bool
    greetings: bool
    farewell: bool
    diminutives: bool
    name: bool
