import os
import io
import time
import tempfile
import logging
import asyncio
import threading
import sys
import locale
import json
from logging.handlers import RotatingFileHandler
from typing import List, Optional, Tuple
from dataclasses import dataclass

import httpx
import os
from dotenv import load_dotenv
project_folder = os.path.expanduser('~/mysite')  # adjust as appropriate
load_dotenv(os.path.join(project_folder, '.env'))

from flask import Flask, request, jsonify
from pydub import AudioSegment
from openai import OpenAI
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# Optional torch/pyannote imports are inside functions to avoid hard deps at import time

# ====== LOCALE ======
try:
    locale.setlocale(locale.LC_ALL, "C.UTF-8")
except Exception:
    pass

# ====== CONFIG ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
PYANNOTE_PIPE = os.getenv("PYANNOTE_PIPE", "pyannote/speaker-diarization-3.1")
STATE_FILE = "bot_state.json"

DESCRIPTION_TEXT = (
    "Привет! Это бот для обработки аудио диалогов.\n"
    "Для работы нужно просто отправить диалог или записать голосовое сообщение."
)
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
OPENAI_STT_FALLBACK = os.getenv("OPENAI_STT_FALLBACK", "gpt-4o-mini-transcribe")
OPENAI_ANALYZE_MODEL = os.getenv("OPENAI_ANALYZE_MODEL", "gpt-4o-mini")

MAX_FILE_MB = 45
MAX_AUDIO_DURATION = 300  # seconds

# ====== TUNING (match audio_bot.py) ======
START_SNAP = 0.15
MIN_DIAR_SPAN = 0.30
DOMINANT_FRAC = 0.85
MERGE_GAP = 0.35
MIN_SEG_DUR = 0.35
HELLO_FIX_WINDOW = 12.0
HELLO_FORCE_CHANGE_T = 3.0


# ====== LOGGING ======
def setup_logging():
    logger = logging.getLogger("audiobot-flask")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = RotatingFileHandler(
        "bot.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


def log_action(action: str, **kwargs):
    parts = [f"action={action}"]
    for k, v in kwargs.items():
        parts.append(f"{k}={v}")
    logger.info(" | ".join(parts))


# ====== Utils ======
def ensure_ffmpeg():
    import subprocess

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    except Exception:
        raise RuntimeError("ffmpeg not found. Install ffmpeg and add it to PATH.")


def mb(n: int) -> float:
    return n / (1024 * 1024)


def convert_to_wav16k_mono(src_path: str) -> str:
    audio = AudioSegment.from_file(src_path)
    dur = len(audio) / 1000.0
    if dur > MAX_AUDIO_DURATION:
        audio = audio[: MAX_AUDIO_DURATION * 1000]
        logger.info(f"trimmed_to={MAX_AUDIO_DURATION}s")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    out = src_path.rsplit(".", 1)[0] + "_16k.wav"
    audio.export(out, format="wav", parameters=["-ac", "1", "-ar", "16000"])
    return out


# ====== Bot State Tracking ======
def load_state() -> dict:
    """Load bot state from file"""
    default_state = {
        "start_time": time.time(),
        "messages_processed": 0,
        "audio_files_processed": 0,
        "errors": 0,
        "is_running": False,
        "last_error": None,
        "user_stats": {},  # {user_id: {"username": str, "usage_count": int, "last_used": float}}
        "total_unique_users": 0,
    }
    
    state_file_path = os.path.abspath(STATE_FILE)
    logger.info(f"Looking for state file at: {state_file_path}")
    
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                # Merge with defaults to handle new fields
                default_state.update(loaded)
                # Ensure user_stats structure is correct
                if "user_stats" not in default_state:
                    default_state["user_stats"] = {}
                logger.info(f"State loaded from {state_file_path}, users: {len(default_state.get('user_stats', {}))}")
                return default_state
        except Exception as e:
            logger.error(f"Failed to load state from {state_file_path}: {e}, using defaults", exc_info=True)
    else:
        logger.info(f"State file not found at {state_file_path}, using defaults")
    
    return default_state


def save_state():
    """Save bot state to file"""
    try:
        state_to_save = {
            "start_time": bot_state["start_time"],
            "messages_processed": bot_state["messages_processed"],
            "audio_files_processed": bot_state["audio_files_processed"],
            "errors": bot_state["errors"],
            "is_running": bot_state["is_running"],
            "last_error": bot_state["last_error"],
            "user_stats": bot_state["user_stats"],
            "total_unique_users": bot_state["total_unique_users"],
        }
        state_file_path = os.path.abspath(STATE_FILE)
        logger.debug(f"Saving state to: {state_file_path}")
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state_to_save, f, indent=2, ensure_ascii=False)
        logger.info(f"State saved to file: {state_file_path}, users: {len(bot_state['user_stats'])}")
    except Exception as e:
        logger.error(f"Failed to save state to {STATE_FILE}: {e}", exc_info=True)


def update_user_stat(user_id: int, username: str = None):
    """Update user statistics - increments usage count and updates last_used time"""
    user_id_str = str(user_id)
    logger.info(f"Updating user stat for user_id={user_id}, username={username}")
    
    if user_id_str not in bot_state["user_stats"]:
        bot_state["user_stats"][user_id_str] = {
            "username": username or f"user_{user_id}",
            "usage_count": 1,  # First use
            "last_used": time.time(),
        }
        bot_state["total_unique_users"] = len(bot_state["user_stats"])
        logger.info(f"New user added: {user_id_str}, total users: {bot_state['total_unique_users']}")
    else:
        bot_state["user_stats"][user_id_str]["usage_count"] += 1
        bot_state["user_stats"][user_id_str]["last_used"] = time.time()
        if username:
            bot_state["user_stats"][user_id_str]["username"] = username
        logger.info(f"User {user_id_str} usage updated to: {bot_state['user_stats'][user_id_str]['usage_count']}")
    
    save_state()
    logger.info(f"State saved, current user_stats size: {len(bot_state['user_stats'])}")


# Initialize state
bot_state = load_state()
bot_state["start_time"] = time.time()  # Reset start time on each app start
bot_state["is_running"] = False  # Will be set to True when bot starts

# ====== ASR ======
@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


def openai_transcribe(client: OpenAI, wav_path: str) -> Tuple[str, List[Segment]]:
    text = ""
    segments: List[Segment] = []

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    with open(wav_path, "rb") as f:
        try:
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
                    if en <= st:
                        en = st + 0.10
                    segments.append(Segment(st, en, seg_text))
        except Exception:
            f.seek(0)
            try:
                resp = client.audio.transcriptions.create(
                    model=OPENAI_STT_FALLBACK,
                    file=f,
                    response_format="text",
                    temperature=0,
                )
                text = str(resp) or ""
            except Exception:
                return "", []

    if not segments and text:
        try:
            import librosa

            dur = librosa.get_duration(path=wav_path)
        except Exception:
            dur = 0.01
        segments = [Segment(0.0, max(dur, 0.10), text)]
    return text, segments


# ====== PyAnnote diarization ======
def _pick_device():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return "cpu"


def diarize_with_pyannote(wav_path: str) -> List[Tuple[float, float, str]]:
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        logger.warning(f"pyannote_unavailable error={type(e).__name__}")
        return []

    if not HUGGINGFACE_TOKEN:
        logger.info("hf_token_missing; diarization_disabled")
        return []

    device = _pick_device()
    pipeline = Pipeline.from_pretrained(PYANNOTE_PIPE, use_auth_token=HUGGINGFACE_TOKEN)
    try:
        pipeline.to(device)
    except Exception:
        pass

    try:
        diar = pipeline(wav_path)
    except Exception as e:
        logger.warning(f"pyannote_error type={type(e).__name__}")
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
    return spans


# ====== Align text to speakers ======
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


def split_text_by_overlap(asr_segments: List[Segment], diar_spans: List[Tuple[float, float, str]]) -> List[Segment]:
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


def render_transcript(segments: List[Segment]) -> str:
    if not segments:
        return ""
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
            lines.append({
                "speaker": cur_spk,
                "start": cur_s,
                "end": cur_e,
                "text": " ".join(buf).strip()
            })
            cur_spk, cur_s, cur_e, buf = spk, seg.start, seg.end, [seg.text]
    if buf:
        lines.append({
            "speaker": cur_spk,
            "start": cur_s,
            "end": cur_e,
            "text": " ".join(buf).strip()
        })
    return lines


def render_stats(segments: List[Segment]):
    totals, counts = {}, {}
    for s in segments:
        sp = s.speaker or "Спикер 1"
        totals[sp] = totals.get(sp, 0.0) + max(0.0, s.end - s.start)
        counts[sp] = counts.get(sp, 0) + 1
    out = []
    for sp in sorted(totals):
        out.append({
            "speaker": sp,
            "total_seconds": round(totals[sp], 2),
            "segments": counts[sp],
        })
    return out


def analyze_dialogue_json(client: OpenAI, transcript: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model=OPENAI_ANALYZE_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты аналитик колл-центра. Верни строго JSON без пояснений по схеме: "
                        "{\"topic\": string, \"outcome\": one_of[\"next_step_set\",\"resolved\",\"unresolved\",\"followup_needed\"], "
                        "\"sentiment\": one_of[\"positive\",\"neutral\",\"negative\"], \"summary\": string, \"action_items\": string[]}"
                    ),
                },
                {
                    "role": "user",
                    "content": "Проанализируй транскрипт с пометками спикеров. Значения анализа заполняй по-русски:\n\n" + transcript,
                },
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or "{}"
        import json as _json

        return _json.loads(raw)
    except Exception as e:
        logger.warning(f"llm_analyze_failed type={type(e).__name__}")
        return {}


# ====== Telegram Rendering Functions (text format) ======
def ts(sec: float) -> str:
    m, s = divmod(int(max(0, sec)), 60)
    return f"{m:02d}:{s:02d}"


def render_transcript_text(segments: List[Segment]) -> str:
    """Render transcript as human-readable text for Telegram"""
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


def render_stats_text(segments: List[Segment]) -> str:
    """Render stats as human-readable text for Telegram"""
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


# ====== Telegram Handlers ======
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_state["messages_processed"] += 1
    user = update.effective_user
    if user:
        update_user_stat(user.id, user.username)
    log_action("telegram_text", user_id=user.id if user else "unknown", username=f"@{user.username}" if user and user.username else "no_username")

    if update.message.text == "Показать описание":
        await update.message.reply_text(DESCRIPTION_TEXT)


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_state["messages_processed"] += 1
    user = update.effective_user
    if user:
        update_user_stat(user.id, user.username)
    await update.message.reply_text(DESCRIPTION_TEXT)


async def show_description_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_state["messages_processed"] += 1
    user = update.effective_user
    if user:
        update_user_stat(user.id, user.username)
    q = update.callback_query
    await q.answer()
    await q.message.reply_text(DESCRIPTION_TEXT)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_state["messages_processed"] += 1
    user = update.effective_user
    if user:
        update_user_stat(user.id, user.username)
    log_action("telegram_start", user_id=user.id if user else "unknown", username=f"@{user.username}" if user and user.username else "no_username")

    keyboard = [["Показать описание"]]
    reply_markup = ReplyKeyboardMarkup(
        keyboard,
        resize_keyboard=True,
        one_time_keyboard=False
    )

    await update.message.reply_text(
        "Привет! Это бот для обработки аудио диалогов.\n"
        "Для работы просто отправьте аудио.",
        reply_markup=reply_markup,
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    bot_state["messages_processed"] += 1
    bot_state["audio_files_processed"] += 1

    msg = update.message
    user = update.effective_user
    start_time = time.time()

    if user:
        update_user_stat(user.id, user.username)

    log_action("telegram_audio_received", user_id=user.id if user else "unknown", username=f"@{user.username}" if user and user.username else "no_username")

    tg_file, file_name = None, "audio"
    if msg.voice:
        tg_file = await context.bot.get_file(msg.voice.file_id)
        file_name = "voice.ogg"
    elif msg.audio:
        tg_file = await context.bot.get_file(msg.audio.file_id)
        file_name = msg.audio.file_name or "audio"
    elif msg.document and (msg.document.mime_type or "").startswith("audio/"):
        tg_file = await context.bot.get_file(msg.document.file_id)
        file_name = msg.document.file_name or "audio"
    else:
        await msg.reply_text("❌ Пришлите аудио.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, file_name)
        await tg_file.download_to_drive(src)
        size_mb = mb(os.path.getsize(src))

        if size_mb > MAX_FILE_MB:
            await msg.reply_text("❌ Файл слишком большой.")
            log_action("telegram_audio_rejected", reason="file_too_large", file_size=f"{size_mb:.2f}MB")
            return

        log_action("telegram_audio_processing", user_id=user.id if user else "unknown", file_size=f"{size_mb:.2f}MB")

        ensure_ffmpeg()
        wav = convert_to_wav16k_mono(src)

        status = await msg.reply_text("🔄 Обрабатываю...")
        try:
            http_client = httpx.Client(timeout=60.0)
            client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

            await status.edit_text("🔄 Перевожу диалог в текст...")
            log_action("telegram_asr_started")

            _, asr_segments = await asyncio.get_event_loop().run_in_executor(
                None, openai_transcribe, client, wav
            )
            if not asr_segments:
                await status.edit_text("❌ Ошибка перевода в текст.")
                log_action("telegram_asr_error", reason="no_segments")
                bot_state["errors"] += 1
                bot_state["last_error"] = "ASR failed: no segments"
                save_state()
                return

            log_action("telegram_asr_success", segments=len(asr_segments))

            await status.edit_text("🔊 Определяю говорящих...")
            log_action("telegram_diarization_started")

            diar = await asyncio.get_event_loop().run_in_executor(None, diarize_with_pyannote, wav)
            sp_count = len({sp for *_, sp in diar}) if diar else 0
            log_action("telegram_diarization_success", speakers=sp_count)

            if diar:
                mapped = split_text_by_overlap(asr_segments, diar)
                transcript_text = render_transcript_text(mapped)
                stats_text = render_stats_text(mapped)
                head = f"🎙️ Расшифровка аудио\n🎯 Обнаружено спикеров: {sp_count}\n"
                body = head + transcript_text + stats_text
            else:
                for s in asr_segments:
                    s.speaker = None
                transcript_text = render_transcript_text(asr_segments)
                stats_text = render_stats_text(asr_segments)
                head = "🎙️ Расшифровка аудио\n⚠️ Обнаружение говорящих недоступно\n"
                body = head + transcript_text + stats_text

            await status.edit_text("🧠 Анализ диалога...")
            log_action("telegram_llm_analysis_started")

            # Create human-readable transcript for LLM
            human_text_lines = []
            for seg in sorted(mapped if diar else asr_segments, key=lambda x: x.start):
                spk = seg.speaker or "Спикер 1"
                human_text_lines.append(f"{spk} [{ts(seg.start)}-{ts(seg.end)}]: {seg.text}")
            human_text = "\n".join(human_text_lines)

            analysis = analyze_dialogue_json(client, human_text)
            log_action("telegram_llm_analysis_success", has_analysis=bool(analysis))

            await status.delete()

            preview = body[:1500] + ("\n\n...(см. файл)" if len(body) > 1500 else "")
            await msg.reply_text(preview)

            # Send transcript as file
            bio_txt = io.BytesIO(body.encode("utf-8"))
            bio_txt.name = "transcript.txt"
            await msg.reply_document(bio_txt)

            # Send analysis
            if analysis:
                analysis_text = (
                    "🧠 Анализ:\n"
                    f"• Тема: {analysis.get('topic', '-')}\n"
                    f"• Результат: {analysis.get('outcome', '-')}\n"
                    f"• Тональность: {analysis.get('sentiment', '-')}\n"
                    f"• Итог: {analysis.get('summary', '-')}\n"
                    f"• Действия: " + (", ".join(analysis.get('action_items', [])) or "-") + "\n"
                )
                await msg.reply_text(analysis_text)

                # Send JSON file
                import json as _json
                bio_json = io.BytesIO(_json.dumps(analysis, ensure_ascii=False, indent=2).encode("utf-8"))
                bio_json.name = "analysis.json"
                await msg.reply_document(bio_json)
            else:
                await msg.reply_text("🧠 Анализ недоступен.")

            processing_time = time.time() - start_time
            log_action("telegram_response_sent", processing_time=f"{processing_time:.1f}s")

        except Exception as e:
            try:
                await status.edit_text("❌ Ошибка анализа")
            except Exception:
                pass
            logger.error(f"Telegram audio processing error: {e}")
            bot_state["errors"] += 1
            bot_state["last_error"] = str(e)
            save_state()
            log_action("telegram_audio_error", error_type=type(e).__name__)


_telegram_bot_thread = None


def run_telegram_bot():
    """Run Telegram bot in background thread"""
    global _telegram_bot_thread

    if _telegram_bot_thread and _telegram_bot_thread.is_alive():
        logger.info("Telegram bot already running")
        return

    if not BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set, Telegram bot disabled")
        bot_state["is_running"] = False
        bot_state["last_error"] = "BOT_TOKEN not set"
        save_state()
        return

    async def _run():
        try:
            logger.info(f"Initializing Telegram bot with token: {BOT_TOKEN[:10]}...")
            application = Application.builder().token(BOT_TOKEN).build()
            application.add_handler(CommandHandler("start", start))
            application.add_handler(MessageHandler(filters.TEXT, handle_text))
            application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.AUDIO, handle_audio))
            application.add_handler(CommandHandler("about", about))
            application.add_handler(CallbackQueryHandler(show_description_cb, pattern="^show_desc$"))

            bot_state["is_running"] = True
            save_state()
            logger.info("Telegram bot started successfully, entering polling loop...")
            
            # Use manual start instead of run_polling() to avoid signal handler issues in threads
            await application.initialize()
            await application.start()
            await application.updater.start_polling(drop_pending_updates=True, allowed_updates=None)
            
            logger.info("Telegram bot polling active, waiting for messages...")
            
            # Keep the bot running
            try:
                while bot_state["is_running"]:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.info("Telegram bot cancelled")
            finally:
                logger.info("Stopping Telegram bot...")
                await application.updater.stop()
                await application.stop()
                await application.shutdown()
        except Exception as e:
            logger.error(f"Telegram bot async error: {e}", exc_info=True)
            bot_state["is_running"] = False
            bot_state["last_error"] = f"Async error: {str(e)}"
            save_state()
            raise

    def _thread_target():
        try:
            logger.info("Starting Telegram bot thread...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_run())
            except KeyboardInterrupt:
                logger.info("Telegram bot interrupted")
            except Exception as e:
                logger.error(f"Telegram bot thread error: {e}", exc_info=True)
                bot_state["is_running"] = False
                bot_state["last_error"] = f"Thread error: {str(e)}"
                save_state()
            finally:
                try:
                    loop.close()
                except Exception as e:
                    logger.error(f"Error closing event loop: {e}")
        except Exception as e:
            logger.error(f"Telegram bot thread initialization error: {e}", exc_info=True)
            bot_state["is_running"] = False
            bot_state["last_error"] = f"Thread init error: {str(e)}"
            save_state()

    try:
        _telegram_bot_thread = threading.Thread(target=_thread_target, daemon=True, name="TelegramBot")
        _telegram_bot_thread.start()
        logger.info("Telegram bot thread started, waiting for initialization...")
        # Give it a moment to initialize
        import time
        time.sleep(1)
        if bot_state["is_running"]:
            logger.info("✅ Telegram bot is running!")
        else:
            logger.warning("⚠️ Telegram bot thread started but is_running is still False")
    except Exception as e:
        logger.error(f"Failed to start Telegram bot thread: {e}", exc_info=True)
        bot_state["is_running"] = False
        bot_state["last_error"] = f"Thread start error: {str(e)}"
        save_state()


# ====== Flask app ======
app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello from Flask!"


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/state")
def get_state():
    """Get Telegram bot state and statistics"""
    uptime_seconds = time.time() - bot_state["start_time"]
    uptime_hours = uptime_seconds / 3600

    # Calculate total usage from all users
    total_usage = sum(user["usage_count"] for user in bot_state["user_stats"].values())

    return jsonify({
        "telegram_bot": {
            "is_running": bot_state["is_running"],
            "uptime_seconds": round(uptime_seconds, 2),
            "uptime_hours": round(uptime_hours, 2),
            "messages_processed": bot_state["messages_processed"],
            "audio_files_processed": bot_state["audio_files_processed"],
            "errors": bot_state["errors"],
            "last_error": bot_state["last_error"],
        },
        "user_statistics": {
            "total_unique_users": bot_state["total_unique_users"],
            "total_usage_count": total_usage,
        },
        "config": {
            "has_bot_token": bool(BOT_TOKEN),
            "has_openai_key": bool(OPENAI_API_KEY),
            "has_hf_token": bool(HUGGINGFACE_TOKEN),
            "max_file_mb": MAX_FILE_MB,
            "max_audio_duration_sec": MAX_AUDIO_DURATION,
        }
    })


@app.get("/stats")
def get_stats():
    """Get detailed user statistics - who used the bot and how many times"""
    # Sort users by usage count (descending)
    sorted_users = sorted(
        bot_state["user_stats"].items(),
        key=lambda x: x[1]["usage_count"],
        reverse=True
    )

    # Format user data
    users_data = []
    for user_id, user_info in sorted_users:
        users_data.append({
            "user_id": int(user_id),
            "username": user_info["username"],
            "usage_count": user_info["usage_count"],
            "last_used": user_info["last_used"],
            "last_used_formatted": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(user_info["last_used"])),
        })

    total_usage = sum(user["usage_count"] for user in bot_state["user_stats"].values())

    return jsonify({
        "total_unique_users": bot_state["total_unique_users"],
        "total_usage_count": total_usage,
        "users": users_data,
    })


@app.post("/process")
def process_audio():
    start_time = time.time()
    if "file" not in request.files:
        return jsonify({"error": "file field required"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, f.filename)
            f.save(src)
            size_mb = mb(os.path.getsize(src))
            if size_mb > MAX_FILE_MB:
                log_action("web_audio_rejected", reason="file_too_large", file_size=f"{size_mb:.2f}MB")
                return jsonify({"error": "file too large", "limit_mb": MAX_FILE_MB}), 400

            bot_state["audio_files_processed"] += 1
            log_action("web_audio_received", file_size=f"{size_mb:.2f}MB")

            ensure_ffmpeg()
            wav = convert_to_wav16k_mono(src)

            # HTTP client (proxy optional)
            http_client = httpx.Client(timeout=60.0)
            client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

            # ASR
            log_action("web_asr_processing", status="started")
            _, asr_segments = openai_transcribe(client, wav)
            if not asr_segments:
                log_action("web_asr_processing", status="error", reason="no_segments")
                bot_state["errors"] += 1
                bot_state["last_error"] = "ASR failed: no segments"
                save_state()
                return jsonify({"error": "asr failed"}), 500
            log_action("web_asr_processing", status="success", segments=len(asr_segments))

            # Diarization
            log_action("web_diarization", status="started")
            diar = diarize_with_pyannote(wav)
            sp_count = len({sp for *_, sp in diar}) if diar else 0
            log_action("web_diarization", status="success", speakers=sp_count)

            if diar:
                mapped = split_text_by_overlap(asr_segments, diar)
                structured_transcript = render_transcript(mapped)
                stats = render_stats(mapped)
            else:
                for s in asr_segments:
                    s.speaker = None
                structured_transcript = render_transcript(asr_segments)
                stats = render_stats(asr_segments)

            # Human-readable transcript for LLM prompt
            human_text = []
            for row in structured_transcript:
                human_text.append(
                    f"{row['speaker']} [{row['start']:.2f}-{row['end']:.2f}]: {row['text']}"
                )
            human_text = "\n".join(human_text)

            # LLM analysis
            log_action("web_llm_analysis", status="started")
            analysis = analyze_dialogue_json(client, human_text)
            log_action("web_llm_analysis", status="success", has_analysis=bool(analysis))

            processing_time = time.time() - start_time
            log_action("web_response_ready", status="success", processing_time=f"{processing_time:.1f}s")

            return jsonify(
                {
                    "speakers": sp_count,
                    "transcript": structured_transcript,
                    "stats": stats,
                    "analysis": analysis,
                    "timings": {"processing_time_sec": round(processing_time, 2)},
                }
            )
    except Exception as e:
        logger.error(f"Web audio processing error: {e}")
        bot_state["errors"] += 1
        bot_state["last_error"] = str(e)
        save_state()
        return jsonify({"error": str(e)}), 500


# Initialize Telegram bot when Flask app starts
run_telegram_bot()


