from core.post_processors.text_processing.DialogueAnalyzerPandas import DialogueAnalyzerPandas


if __name__ == "__main__":
    analyzer = DialogueAnalyzerPandas()
    analyzer.analyze_dialogue()