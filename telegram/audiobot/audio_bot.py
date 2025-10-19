# -*- coding: utf-8 -*-
# Телеграм-бот: ASR (Whisper) + Диаризация (pyannote 3.x) + LLM-анализ (JSON)
import sys, locale
import asyncio
import os
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import httpx
from pydub import AudioSegment
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ====== LOCALE ======
try:
    locale.setlocale(locale.LC_ALL, "C.UTF-8")
except Exception:
    pass

# ====== CONFIG ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
PROXY = os.getenv("OPENAI_HTTP_PROXY", "socks5h://127.0.0.1:1080")  # напр. socks5h://127.0.0.1:1080

# Whisper
OPENAI_STT_MODEL = "whisper-1"
OPENAI_STT_FALLBACK = "gpt-4o-mini-transcribe"

# LLM для анализа
OPENAI_ANALYZE_MODEL = "gpt-4o-mini"

# PyAnnote
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
PYANNOTE_PIPE = os.getenv("PYANNOTE_PIPE", "pyannote/speaker-diarization-3.1")

# I/O limits
MAX_FILE_MB = 45
MAX_AUDIO_DURATION = 600  # сек

# ====== TUNING ======
START_SNAP = 0.15
MIN_DIAR_SPAN = 0.30
DOMINANT_FRAC = 0.85
MERGE_GAP = 0.35
MIN_SEG_DUR = 0.35
HELLO_FIX_WINDOW = 12.0
HELLO_FORCE_CHANGE_T = 3.0

def log(s: str):
    try:
        print(str(s))
    except Exception:
        sys.stdout.buffer.write((str(s) + "\n").encode("utf-8", errors="ignore"))

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ====== utils ======
def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    except Exception:
        raise RuntimeError("ffmpeg не найден. Установи ffmpeg и добавь в PATH.")

def convert_to_wav16k_mono(src_path: str) -> str:
    audio = AudioSegment.from_file(src_path)
    dur = len(audio) / 1000.0
    if dur > MAX_AUDIO_DURATION:
        audio = audio[:MAX_AUDIO_DURATION * 1000]
        log(f"Аудио обрезано до {MAX_AUDIO_DURATION} сек")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    out = src_path.rsplit(".", 1)[0] + "_16k.wav"
    audio.export(out, format="wav", parameters=["-ac", "1", "-ar", "16000"])
    return out

def mb(n: int) -> float:
    return n / (1024 * 1024)

# ====== ASR ======
def openai_transcribe(client: OpenAI, wav_path: str) -> Tuple[str, List[Segment]]:
    text = ""
    segments: List[Segment] = []

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    with open(wav_path, "rb") as f:
        try:
            log("ASR: whisper-1 (verbose_json)")
            resp = client.audio.transcriptions.create(
                model=OPENAI_STT_MODEL,
                file=f,
                response_format="verbose_json",
                temperature=0,
            )
            text = _get(resp, "text", "") or ""
            for s in _get(resp, "segments", []) or []:
                seg_text = (_get(s, "text", "") or "").strip()
                if seg_text:
                    st = float(_get(s, "start", 0.0))
                    en = float(_get(s, "end", 0.0))
                    if en <= st: en = st + 0.10
                    segments.append(Segment(st, en, seg_text))
        except Exception as e:
            log(f"whisper-1 ошибка: {e}; fallback gpt-4o-mini-transcribe")
            f.seek(0)
            try:
                resp = client.audio.transcriptions.create(
                    model=OPENAI_STT_FALLBACK,
                    file=f,
                    response_format="text",
                    temperature=0,
                )
                text = str(resp) or ""
            except Exception as e2:
                log(f"ASR не удался: {e2}")
                return "", []

    if not segments and text:
        try:
            import librosa
            dur = librosa.get_duration(path=wav_path)
        except Exception:
            dur = 0.01
        segments = [Segment(0.0, max(dur, 0.10), text)]
    log(f"ASR ok: {len(segments)} сегментов")
    return text, segments

# ====== PyAnnote diarization ======
def diarize_with_pyannote(wav_path: str) -> List[Tuple[float, float, str]]:
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        log(f"PyAnnote недоступен: {e}")
        return []

    if not HUGGINGFACE_TOKEN:
        log("Не задан HUGGINGFACE_TOKEN — диаризация отключена.")
        return []

    device = _pick_device()
    pipeline = Pipeline.from_pretrained(
        PYANNOTE_PIPE,
        use_auth_token=HUGGINGFACE_TOKEN,
    )
    pipeline.to(device)

    try:
        diar = pipeline(wav_path)
    except Exception as e:
        log(f"PyAnnote ошибка: {e}")
        return []

    spans: List[Tuple[float, float, str]] = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        s, e = float(turn.start), float(turn.end)
        if s is None or e is None:
            continue
        s = max(START_SNAP, s)
        if e - s >= MIN_DIAR_SPAN:
            spans.append((s, e, str(speaker)))
    spans.sort(key=lambda x: (x[0], x[1]))
    log(f"PyAnnote ok: {len(spans)} интервалов")
    return spans

# ====== speaker relabel ======
def relabel_speakers(diar: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    if not diar:
        return diar
    votes = {}
    for s, e, lab in diar:
        if s >= HELLO_FIX_WINDOW:
            break
        ov = min(e, HELLO_FIX_WINDOW) - max(s, 0.0)
        if ov > 0:
            votes[lab] = votes.get(lab, 0.0) + ov
    if not votes:
        return diar
    first_label = max(votes.items(), key=lambda x: x[1])[0]
    others = [lab for _, _, lab in diar if lab != first_label]
    second_label = others[0] if others else None
    mapping = {first_label: "Спикер 1"}
    if second_label:
        mapping[second_label] = "Спикер 2"
    return [(s, e, mapping.get(lab, lab)) for s, e, lab in diar]

# ====== align text to speakers ======
def split_text_by_overlap(asr_segments: List[Segment],
                          diar_spans: List[Tuple[float, float, str]]) -> List[Segment]:
    if not diar_spans:
        return asr_segments

    diar_spans = relabel_speakers(diar_spans)

    def dominant_speaker(a: float, b: float) -> str:
        votes = {}
        for s, e, lab in diar_spans:
            if e <= a or s >= b:
                continue
            ov = min(b, e) - max(a, s)
            if ov > 0:
                votes[lab] = votes.get(lab, 0.0) + ov
        if not votes:
            return "Спикер 1"
        return max(votes.items(), key=lambda x: x[1])[0]

    result: List[Segment] = []
    for seg in asr_segments:
        st, en = seg.start, seg.end
        if en <= st or not seg.text.strip():
            continue

        if st < HELLO_FORCE_CHANGE_T:
            lab = dominant_speaker(st, min(en, HELLO_FIX_WINDOW))
            result.append(Segment(st, en, seg.text, lab))
            continue

        overlaps = []
        for ts, te, lab in diar_spans:
            if en <= ts or st >= te:
                continue
            s0, e0 = max(st, ts), min(en, te)
            if e0 - s0 > 0.20:
                overlaps.append((s0, e0, lab))
        if not overlaps:
            lab = dominant_speaker(st, en)
            result.append(Segment(st, en, seg.text, lab))
            continue

        seg_len = en - st
        best = max(overlaps, key=lambda x: x[1] - x[0])
        if (best[1] - best[0]) / seg_len >= DOMINANT_FRAC:
            result.append(Segment(st, en, seg.text, best[2]))
        else:
            lab = dominant_speaker(st, en)
            result.append(Segment(st, en, seg.text, lab))

    merged: List[Segment] = []
    for s in sorted(result, key=lambda x: x.start):
        if merged and merged[-1].speaker == s.speaker and s.start - merged[-1].end <= MERGE_GAP:
            merged[-1].end = max(merged[-1].end, s.end)
            merged[-1].text = (merged[-1].text + " " + s.text).strip()
        else:
            merged.append(s)
    return merged

# ====== rendering ======
def ts(sec: float) -> str:
    m, s = divmod(int(max(0, sec)), 60)
    return f"{m:02d}:{s:02d}"

def render_transcript(segments: List[Segment]) -> str:
    if not segments:
        return "⚠️ Пусто"
    lines = []
    cur_spk, cur_s, cur_e, buf = None, None, None, []
    for seg in sorted(segments, key=lambda x: x.start):
        if seg.end - seg.start < MIN_SEG_DUR:
            continue
        spk = seg.speaker or "Спикер 1"
        if cur_spk is None:
            cur_spk, cur_s, cur_e, buf = spk, seg.start, seg.end, [seg.text]
            continue
        if spk == cur_spk and seg.start - cur_e <= MERGE_GAP:
            cur_e = max(cur_e, seg.end)
            buf.append(seg.text)
        else:
            lines.append(f"\n🎤 {cur_spk}:\n   [{ts(cur_s)}-{ts(cur_e)}] {' '.join(buf).strip()}")
            cur_spk, cur_s, cur_e, buf = spk, seg.start, seg.end, [seg.text]
    if buf:
        lines.append(f"\n🎤 {cur_spk}:\n   [{ts(cur_s)}-{ts(cur_e)}] {' '.join(buf).strip()}")
    return "\n".join(lines)

def render_stats(segments: List[Segment]) -> str:
    totals, counts = {}, {}
    for s in segments:
        sp = s.speaker or "Спикер 1"
        totals[sp] = totals.get(sp, 0.0) + max(0.0, s.end - s.start)
        counts[sp] = counts.get(sp, 0) + 1
    lines = ["\n📊 Статистика:"]
    for sp in sorted(totals):
        m, ss = divmod(int(totals[sp]), 60)
        lines.append(f"   {sp}: {m:02d}:{ss:02d} ({counts[sp]} реплик)")
    return "\n".join(lines)

# ====== LLM-анализ (JSON) ======
def analyze_dialogue_json(client: OpenAI, transcript: str) -> dict:
    """
    Возвращает dict:
      topic: str
      outcome: one_of["next_step_set","resolved","unresolved","followup_needed"]
      sentiment: one_of["positive","neutral","negative"]
      summary: str
      action_items: list[str]
      quality_flags: list[str]
    """
    try:
        resp = client.chat.completions.create(
            model=OPENAI_ANALYZE_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content":
                 "Ты аналитик колл-центра. Верни строго JSON без пояснений по схеме: "
                 "{"
                 "\"topic\": string, "
                 "\"outcome\": one_of[\"next_step_set\",\"resolved\",\"unresolved\",\"followup_needed\"], "
                 "\"sentiment\": one_of[\"positive\",\"neutral\",\"negative\"], "
                 "\"summary\": string, "
                 "\"action_items\": string[], "
                 "\"quality_flags\": string[]"
                 "}"
                },
                {"role": "user", "content":
                 "Проанализируй транскрипт с пометками спикеров:\n\n" + transcript}
            ],
            temperature=0.2,
        )
        import json
        raw = resp.choices[0].message.content or "{}"
        return json.loads(raw)
    except Exception as e:
        log(f"LLM analyze failed: {e}")
        return {}

# ====== Telegram ======
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"🎙️ Пришлите аудио — расшифрую, разнесу по спикерам и дам краткий анализ.\nДо {MAX_FILE_MB} МБ или {MAX_AUDIO_DURATION//60} минут."
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    log("Получен аудиофайл")
    tg_file, file_name = None, "audio"
    if msg.voice:
        tg_file = await context.bot.get_file(msg.voice.file_id); file_name = "voice.ogg"
    elif msg.audio:
        tg_file = await context.bot.get_file(msg.audio.file_id); file_name = msg.audio.file_name or "audio"
    elif msg.document and (msg.document.mime_type or "").startswith("audio/"):
        tg_file = await context.bot.get_file(msg.document.file_id); file_name = msg.document.file_name or "audio"
    else:
        await msg.reply_text("❌ Пришлите аудио.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, file_name)
        await tg_file.download_to_drive(src)
        if mb(os.path.getsize(src)) > MAX_FILE_MB:
            await msg.reply_text("❌ Файл слишком большой.")
            return

        ensure_ffmpeg()
        wav = convert_to_wav16k_mono(src)

        status = await msg.reply_text("🔄 Обрабатываю...")
        try:
            # HTTP-клиент с прокси (httpx 0.25.2: параметр proxy=)
            http_client = httpx.Client(proxy=PROXY or None, timeout=60.0)
            client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

            await status.edit_text("🔄 Транскрибирую (Whisper)...")
            _, asr_segments = await asyncio.get_event_loop().run_in_executor(
                None, openai_transcribe, client, wav
            )
            if not asr_segments:
                await status.edit_text("❌ Пустая транскрипция.")
                return

            await status.edit_text("🔊 Диаризация (PyAnnote)...")
            diar = await asyncio.get_event_loop().run_in_executor(None, diarize_with_pyannote, wav)

            if diar:
                mapped = split_text_by_overlap(asr_segments, diar)
                transcript = render_transcript(mapped)
                sp_count = len({sp for *_, sp in diar})
                head = f"🎙️ Расшифровка аудио\n🎯 Обнаружено спикеров: {sp_count}\n"
                body = head + transcript + render_stats(mapped)
            else:
                for s in asr_segments:
                    s.speaker = None
                transcript = render_transcript(asr_segments)
                head = "🎙️ Расшифровка аудио\n⚠️ Диаризация недоступна\n"
                body = head + transcript + render_stats(asr_segments)

            # Анализ диалога LLM (JSON)
            await status.edit_text("🧠 Анализ диалога...")
            analysis = analyze_dialogue_json(client, transcript)

            # Отправка результатов
            await status.delete()

            preview = body[:1500] + ("\n\n...(см. файл)" if len(body) > 1500 else "")
            await msg.reply_text(preview)

            import io, json as _json
            # текст расшифровки
            bio_txt = io.BytesIO(body.encode("utf-8")); bio_txt.name = "transcript.txt"
            await msg.reply_document(bio_txt)

            # анализ в человекочитаемом виде
            if analysis:
                analysis_text = (
                    "🧠 Анализ:\n"
                    f"• Тема: {analysis.get('topic','-')}\n"
                    f"• Результат: {analysis.get('outcome','-')}\n"
                    f"• Тональность: {analysis.get('sentiment','-')}\n"
                    f"• Итог: {analysis.get('summary','-')}\n"
                    f"• Действия: " + (", ".join(analysis.get('action_items', [])) or "-") + "\n"
                    f"• Флаги качества: " + (", ".join(analysis.get('quality_flags', [])) or "-")
                )
                await msg.reply_text(analysis_text)

                # сырой JSON-файл
                bio_json = io.BytesIO(_json.dumps(analysis, ensure_ascii=False, indent=2).encode("utf-8"))
                bio_json.name = "analysis.json"
                await msg.reply_document(bio_json)
            else:
                await msg.reply_text("🧠 Анализ недоступен.")

        except Exception as e:
            try:
                await status.edit_text(f"❌ Ошибка: {e}")
            except Exception:
                pass
            log(f"Ошибка: {e}")

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.AUDIO, handle_audio))
    log("Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
