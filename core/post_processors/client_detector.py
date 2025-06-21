from fuzzywuzzy import fuzz

from core.entity.audio_to_text_result import ProcessingResults

MANAGER_KEYWORDS = {
    "как я могу к вам вращаться",
    " переведу вас на другого оператора",
    "как я могу к вам обращаться",
    "ожидайте",
    "policy",
    "Благодарю за обращение", "Благодарю за длительное ожидание",
    "номер телефона, по которому оформляли заказ"
}

THRESHOLD = 80


def detect_manager(dialog: ProcessingResults):
    speaker_scores = {}

    for line in dialog.items:
        phrase = line.text
        speaker = line.speaker_id
        phrase_lower = phrase.lower()
        score = 0

        for manager_phrase in MANAGER_KEYWORDS:
            similarity = fuzz.partial_ratio(manager_phrase.lower(), phrase_lower)
            if similarity >= THRESHOLD:
                score += 1
        if speaker not in speaker_scores:
            speaker_scores[speaker] = score
        else:
            speaker_scores[speaker] += score

    manager = max(speaker_scores.items(), key=lambda x: x[1])[0]
    for line in dialog.items:
        if line.speaker_id == manager:
            line.speaker_id = "SALES"
        else:
            line.speaker_id = "CLIENT"
