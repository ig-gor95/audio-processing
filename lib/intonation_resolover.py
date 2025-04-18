from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import io
import torchaudio


def predict_emotion(audio_segment):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForSequenceClassification.from_pretrained(
        "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned"
    )

    num2emotion = {
        0: 'Нейтрально',
        1: 'Злобно',
        2: 'Позитивно',
        3: 'Грустно',
        4: 'Невнятно'
    }

    buffer = io.BytesIO()
    audio_segment.export(buffer, format="wav")
    buffer.seek(0)

    waveform, sample_rate = torchaudio.load(buffer, normalize=True)
    transform = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = transform(waveform)

    inputs = feature_extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16000 * 10
    )

    with torch.no_grad():
        logits = model(inputs['input_values'][0]).logits

    predicted_emotion = num2emotion[torch.argmax(logits).item()]

    return predicted_emotion
