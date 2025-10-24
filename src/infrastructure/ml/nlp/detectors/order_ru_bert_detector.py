import pandas as pd
import pymorphy3
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

from yaml_reader import ConfigLoader


class OrderRuBertPatternsDetector:
    def __init__(self, config_path: str = "post_processors/config/order_pattern.yaml"):
        self._config = ConfigLoader(config_path)
        self._morph = pymorphy3.MorphAnalyzer()

        # Load ruBERT model with MPS support
        self.model_name = "cointegrated/rubert-tiny2"  # Lightweight version
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        # Detect and use MPS (Apple Silicon) if available
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon) acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA acceleration")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.offer_patterns = self._compile_offer_patterns()
        self.processing_patterns = self._compile_processing_patterns()
        self.resume_patterns = self._compile_resume_patterns()
        self._threshold = 95

        # Precompute pattern embeddings on the correct device
        self.pattern_embeddings = self._precompute_pattern_embeddings()
        self.pattern_arrays = {
            'offer': np.array([self.pattern_embeddings[p] for p in self.offer_patterns]),
            'processing': np.array([self.pattern_embeddings[p] for p in self.processing_patterns]),
            'resume': np.array([self.pattern_embeddings[p] for p in self.resume_patterns])
        }

    def _compile_offer_patterns(self):
        return self._config.get('patterns')['offer']

    def _compile_processing_patterns(self):
        return self._config.get('patterns')['processing']

    def _compile_resume_patterns(self):
        return self._config.get('patterns')['resume']

    def _precompute_pattern_embeddings(self):
        """Precompute embeddings for all patterns in batch on MPS"""
        all_patterns = self.offer_patterns + self.processing_patterns + self.resume_patterns

        if not all_patterns:
            return {}

        # Tokenize all patterns at once
        inputs = self.tokenizer(
            all_patterns,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=32,  # Short length for patterns
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token or mean pooling
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token is faster

        return dict(zip(all_patterns, embeddings))

    def _batch_get_embeddings(self, texts: list) -> np.ndarray:
        """Get embeddings for multiple texts in batch on MPS"""
        if not texts:
            return np.array([])

        # Process in reasonable batch sizes
        batch_size = 32  # Optimal for MPS memory
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token for better performance
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def _batch_cosine_similarity(self, text_embeddings: np.ndarray, pattern_embeddings: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity calculation"""
        if text_embeddings.size == 0:
            return np.array([])

        # Normalize for cosine similarity
        text_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        pattern_norm = pattern_embeddings / np.linalg.norm(pattern_embeddings, axis=1, keepdims=True)

        return np.dot(text_norm, pattern_norm.T)

    def _batch_semantic_match(self, texts: list, pattern_type: str) -> list:
        """Process multiple texts at once for semantic matching"""
        if not texts:
            return []

        text_embeddings = self._batch_get_embeddings(texts)
        pattern_embeddings = self.pattern_arrays[pattern_type]
        patterns = getattr(self, f"{pattern_type}_patterns")

        if text_embeddings.size == 0:
            return [None] * len(texts)

        # Calculate all similarities at once
        similarities = self._batch_cosine_similarity(text_embeddings, pattern_embeddings)

        results = []
        threshold = self._threshold / 100

        for i in range(len(texts)):
            max_idx = np.argmax(similarities[i])
            max_similarity = similarities[i][max_idx]

            results.append(patterns[max_idx] if max_similarity >= threshold else None)

        return results

    def __call__(self, texts: pd.DataFrame):
        text_list = texts.tolist()

        def batch_match(texts, patterns, pattern_type):
            # First: fast exact matching
            exact_matches = []
            pattern_words_list = [set(p.split()) for p in patterns]  # Precompute

            for t in texts:
                text_words = set(t.split())
                matched = None
                for i, pattern_words in enumerate(pattern_words_list):
                    if pattern_words.issubset(text_words):
                        matched = patterns[i]
                        break
                exact_matches.append(matched)

            # Second: semantic matching only for unmatched texts
            needs_semantic = [i for i, m in enumerate(exact_matches) if m is None]

            if not needs_semantic:
                return exact_matches

            semantic_texts = [texts[i] for i in needs_semantic]
            semantic_results = self._batch_semantic_match(semantic_texts, pattern_type)

            # Combine results
            final_results = exact_matches.copy()
            for idx, semantic_result in zip(needs_semantic, semantic_results):
                final_results[idx] = semantic_result

            return final_results

        # Process all pattern types in sequence
        offer_results = batch_match(text_list, self.offer_patterns, 'offer')
        processing_results = batch_match(text_list, self.processing_patterns, 'processing')
        resume_results = batch_match(text_list, self.resume_patterns, 'resume')

        return (
            pd.Series(offer_results, index=texts.index),
            pd.Series(processing_results, index=texts.index),
            pd.Series(resume_results, index=texts.index)
        )