#!/usr/bin/env python3
"""
SubTranslate — Local Anime & Stream VOD Subtitle/Chat Translator

Transcribes audio from media files using Whisper and translates subtitles
or YouTube VOD live chat logs (*.live_chat.json) to English using a local
LLM via Ollama.

Usage:
    python subtranslate.py                              # Process all files in INPUT/
    python subtranslate.py --file INPUT/ep01.mkv        # Process specific video
    python subtranslate.py --file INPUT/vod.live_chat.json  # Translate YT VOD live chat
    python subtranslate.py --whisper-model medium       # Use smaller Whisper model
    python subtranslate.py --ollama-model gemma4:31b    # Use different Ollama model
"""

import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import zlib
from dataclasses import dataclass, field
from pathlib import Path

__version__ = "1.2.0"


# ── Configuration ───────────────────────────────────────────────────────────

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".m4v", ".ts", ".wmv"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt"}
LIVE_CHAT_SUFFIXES = (".live_chat.json", ".live_chat.jsonl")

# Resolve INPUT/OUTPUT relative to the script so it works from any working directory
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "INPUT"
OUTPUT_DIR = SCRIPT_DIR / "OUTPUT"

DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"
TRANSLATION_BATCH_SIZE = 20       # subtitle blocks per LLM call
CHAT_TRANSLATION_BATCH_SIZE = 35  # live chat messages per LLM call
LOOKBACK_COUNT = 5                # preceding translated lines for context continuity
LOOKAHEAD_COUNT = 3               # upcoming untranslated lines for sentence flow
CHAT_LOOKBACK_COUNT = 8           # preceding chat messages for thread continuity
CHAT_LOOKAHEAD_COUNT = 4          # upcoming chat messages for reaction context
MIN_CUE_DURATION_MS = 100         # drop micro-duration artifact cues (e.g. YouTube autogen flashes)
OLLAMA_RETRIES = 2                # extra attempts when an Ollama request fails (timeouts, reloads)
THINK_MODES = ("auto", "off", "on", "low", "medium", "high")

# ASS sidebar chat overlay layout (1920x1080 virtual canvas)
CHAT_DISPLAY_MS = 12_000          # how long each chat message stays visible before expiring
CHAT_MAX_VISIBLE_MESSAGES = 10    # max simultaneous messages in the rolling sidebar window
CHAT_MAX_VISIBLE_LINES = 14       # max wrapped visual lines in the rolling sidebar window
CHAT_WRAP_COLUMNS = 40            # max visual character columns per sidebar line

# Text-based subtitle codecs that can be converted/parsed (image subs like PGS/DVB cannot)
TEXT_SUB_CODECS = {"srt", "subrip", "ass", "ssa", "webvtt", "mov_text", "text"}

# Phrases Whisper commonly hallucinates during silence/music (matched on short segments only)
WHISPER_HALLUCINATIONS = (
    "ご視聴ありがとうございました", "ご視聴ありがとうございます", "チャンネル登録",
    "字幕作成", "字幕提供", "字幕由", "请不吝点赞", "订阅", "訂閱",
    "시청해 주셔서 감사합니다", "구독과 좋아요",
    "thank you for watching", "thanks for watching", "please subscribe",
    "subtitles by", "amara.org",
)


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class SubtitleBlock:
    """A single subtitle entry with index, timing, and text."""
    index: int
    start_time: str  # SRT format: HH:MM:SS,mmm
    end_time: str    # SRT format: HH:MM:SS,mmm
    text: str
    translated_text: str = ""


@dataclass
class LiveChatMessage:
    """A single chat message extracted from a YouTube live_chat.json replay log."""
    index: int
    line_no: int                # 1-based JSONL line number in the source file
    slot_id: str                # deterministic action/renderer path within that JSONL line
    offset_ms: int              # negative for pre-stream waiting room messages
    timestamp_text: str         # e.g. "-0:40", "0:21", "1:02:15"
    author: str                 # e.g. "@Cry4s722-みず"
    role: str                   # "owner" | "moderator" | "member" | "viewer"
    kind: str                   # "text" | "paid" | "sticker" | "membership"
    amount_text: str = ""       # e.g. "¥1,000" for Super Chats / Stickers
    header_text: str = ""       # membership header text if any
    text: str = ""              # flattened message text (with Unicode emojis & :custom-emote: tokens)
    custom_emojis: dict[str, dict] = field(default_factory=dict)
    in_overlay: bool = True     # False for duplicate ticker popup items
    needs_translation: bool = False
    translated_text: str = ""


# ── Embedded Subtitle Detection & Extraction ───────────────────────────────

def detect_embedded_subtitles(media_path: Path) -> list[dict]:
    """Detect embedded subtitle tracks in a media file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "s",  # subtitle streams only
        str(media_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []

    try:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        subs = []
        for s in streams:
            tags = s.get("tags", {}) or {}
            disposition = s.get("disposition", {}) or {}
            subs.append({
                "index": s.get("index", 0),
                "codec": (s.get("codec_name") or "unknown").lower(),
                "language": (tags.get("language") or "und").lower(),
                "title": tags.get("title", ""),
                "forced": bool(disposition.get("forced", 0)),
            })
        return subs
    except (json.JSONDecodeError, KeyError, AttributeError):
        return []


def extract_embedded_subtitles(media_path: Path, track: dict, output_dir: Path) -> Path | None:
    """Extract a subtitle track from a media file using FFmpeg.

    ASS/SSA tracks are copied as-is (so drawing commands and typesetting can be filtered
    by our ASS parser); everything else is converted to SRT. Returns the path of the
    extracted file, or None on failure.
    """
    stream_index = track["index"]
    if track["codec"] in ("ass", "ssa"):
        attempts = [(".ass", ["-c:s", "copy"]), (".srt", ["-c:s", "srt"])]
    else:
        attempts = [(".srt", ["-c:s", "srt"]), (".ass", ["-c:s", "ass"])]

    for suffix, codec_args in attempts:
        out_path = output_dir / f"track{stream_index}{suffix}"
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(media_path),
               "-map", f"0:{stream_index}", *codec_args, str(out_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return out_path
    return None


def _track_is_english(track: dict) -> bool:
    return track["language"] in ("eng", "en")


def _track_is_signs_only(track: dict) -> bool:
    """Heuristic: forced / signs & songs tracks only contain a fraction of the dialogue."""
    title = track.get("title", "").lower()
    if "full" in title or "dialog" in title:
        return False
    return track.get("forced", False) or any(k in title for k in ("sign", "song", "forced", "karaoke"))


def _pick_best_subtitle_track(tracks: list[dict]) -> dict | None:
    """Pick the best subtitle track to translate — prefers full-dialogue, non-English text subs."""
    # Image-based subtitle formats (PGS, DVB, HDMV) can't be converted to text — skip them
    candidates = [t for t in tracks if t["codec"] in TEXT_SUB_CODECS]
    if not candidates:
        return None

    # Prefer non-English tracks (we want to translate, not copy)
    non_english = [t for t in candidates if not _track_is_english(t)]
    if not non_english:
        return None  # already in English

    def score(t: dict) -> tuple:
        return (
            not _track_is_signs_only(t),             # full dialogue over signs/songs
            t["language"] in ("jpn", "ja"),          # prefer Japanese
            t["language"] not in ("und", ""),        # prefer tagged tracks
        )

    return max(non_english, key=score)


def _pick_english_track(tracks: list[dict]) -> dict | None:
    """Pick the best English text subtitle track, if any."""
    english = [t for t in tracks if t["codec"] in TEXT_SUB_CODECS and _track_is_english(t)]
    if not english:
        return None
    return max(english, key=lambda t: not _track_is_signs_only(t))


# ── Audio Extraction ────────────────────────────────────────────────────────

def extract_audio(media_path: Path, output_wav: Path) -> bool:
    """Extract audio from a media file to 16kHz mono WAV using FFmpeg."""
    print(f"  → Extracting audio from {media_path.name}...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(media_path),
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16kHz sample rate
        "-ac", "1",               # mono
        str(output_wav)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ FFmpeg error: {result.stderr[-500:]}")
        return False
    print(f"  ✓ Audio extracted ({output_wav.stat().st_size / 1024 / 1024:.1f} MB)")
    return True


# ── Transcription ───────────────────────────────────────────────────────────

def _unload_ollama_models():
    """Tell Ollama to unload all models from GPU to free VRAM for Whisper."""
    try:
        import ollama
        # List running models and unload them
        running = ollama.ps()
        models = running.get("models", []) or []
        for m in models:
            model_name = m.get("model") or m.get("name") or ""
            if model_name:
                print(f"  → Unloading Ollama model '{model_name}' to free VRAM...")
                ollama.generate(model=model_name, prompt="", keep_alive=0)
        if models:
            time.sleep(2)  # give GPU a moment to release memory
            print("  ✓ Ollama models unloaded")
    except Exception as e:
        print(f"  ⚠ Could not unload Ollama models: {e}")


def _ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT time format (HH:MM:SS,mmm)."""
    ms = max(0, int(ms))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, millis = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm).

    Uses integer milliseconds to avoid float truncation errors (e.g. 1.9999 → 1,999).
    """
    return _ms_to_srt_time(round(seconds * 1000))


def _srt_to_ms(srt_time: str) -> int:
    """Convert a timestamp (HH:MM:SS,mmm / HH:MM:SS.mmm / MM:SS.mmm) to milliseconds."""
    try:
        parts = srt_time.strip().replace(",", ".").split(":")
        if len(parts) == 2:
            parts.insert(0, "0")
        h, m, s = parts
        return round((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)
    except Exception:
        return 0


# ── Language Detection ─────────────────────────────────────────────────────

_KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\uff66-\uff9f]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]")
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CJK_RE = re.compile(
    r"[\u3040-\u309f\u30a0-\u30ff\uff66-\uff9f\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f"
    r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)

_LANGUAGE_NAMES = {
    "ja": "Japanese", "jpn": "Japanese",
    "en": "English", "eng": "English",
    "ko": "Korean", "kor": "Korean",
    "zh": "Chinese", "zho": "Chinese", "chi": "Chinese", "yue": "Cantonese",
    "fr": "French", "fra": "French", "fre": "French",
    "de": "German", "deu": "German", "ger": "German",
    "es": "Spanish", "spa": "Spanish",
    "it": "Italian", "ita": "Italian",
    "ru": "Russian", "rus": "Russian",
    "pt": "Portuguese", "por": "Portuguese",
    "pl": "Polish", "pol": "Polish",
    "nl": "Dutch", "nld": "Dutch", "dut": "Dutch",
    "tr": "Turkish", "tur": "Turkish",
    "ar": "Arabic", "ara": "Arabic",
    "th": "Thai", "tha": "Thai",
    "vi": "Vietnamese", "vie": "Vietnamese",
    "id": "Indonesian", "ind": "Indonesian",
    "uk": "Ukrainian", "ukr": "Ukrainian",
    "hi": "Hindi", "hin": "Hindi",
}
_UNKNOWN_LANG = "the source language"


def _language_code_to_name(code: str | None) -> str | None:
    """Map a language code (ISO 639-1/2) or name to an English language name.
    Returns None for unknown/undetermined codes so callers can fall back to detection."""
    if not code:
        return None
    code = code.strip().lower()
    if code in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[code]
    # Already a full language name (e.g. "japanese" from Whisper or user input)
    if code.title() in _LANGUAGE_NAMES.values():
        return code.title()
    return None


def _detect_language_from_texts(texts: list[str]) -> str | None:
    """Detect CJK languages from script usage. Distinguishes Chinese from Japanese by kana."""
    text = "".join(texts)
    kana = len(_KANA_RE.findall(text))
    hangul = len(_HANGUL_RE.findall(text))
    han = len(_HAN_RE.findall(text))
    if kana >= 3 and kana >= 0.02 * (kana + han):
        return "Japanese"
    if hangul >= 3 and hangul >= kana:
        return "Korean"
    if han >= 3:
        return "Chinese"
    return None


def _detect_language_from_blocks(blocks: list[SubtitleBlock]) -> str | None:
    """Detect CJK languages from subtitle blocks."""
    return _detect_language_from_texts([b.text for b in blocks])


def _resolve_source_lang(code: str | None, blocks: list[SubtitleBlock], override: str | None = None) -> str:
    """Decide the source language: user override > script detection > metadata tag > generic."""
    if override:
        return _language_code_to_name(override) or override
    # Script detection is more reliable than (often missing or wrong) container tags
    detected = _detect_language_from_blocks(blocks)
    if detected:
        return detected
    return _language_code_to_name(code) or _UNKNOWN_LANG


def _contains_cjk(text: str) -> bool:
    """Check if text contains Chinese, Japanese, or Korean characters."""
    return bool(_CJK_RE.search(text))


def _is_english(text: str) -> bool:
    """Quick heuristic to check if text is already mostly English."""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / max(len(text), 1) > 0.85


def _is_file_already_english(blocks: list[SubtitleBlock], source_lang: str = "") -> bool:
    """Check if the subtitle file is genuinely already in English across the entire file."""
    # If source language is identified as a foreign language, do not skip
    if source_lang and source_lang.lower() not in ("english", _UNKNOWN_LANG):
        return False

    # If any block contains CJK characters, it is definitely NOT English
    if any(_contains_cjk(b.text) for b in blocks):
        return False

    if not blocks:
        return True

    # Check that at least 90% of all blocks across the entire file are English
    english_blocks = sum(1 for b in blocks if _is_english(b.text))
    return (english_blocks / len(blocks)) >= 0.90


# ── Subtitle Parsing ───────────────────────────────────────────────────────

_ASS_TAG_RE = re.compile(r"\{[^}]*\}")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_ASS_DRAWING_RE = re.compile(r"\\p[1-9]")


def _read_text_file(path: Path) -> str:
    """Read a subtitle file, detecting its encoding (UTF-8/16, Shift-JIS, GBK, CP1252...)."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        if best is not None:
            print(f"  → Detected subtitle encoding: {best.encoding}")
            return str(best)
    except ImportError:
        pass
    for enc in ("cp932", "gb18030", "cp949", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean_subtitle_text(text: str) -> str:
    """Strip ASS override tags / HTML formatting and flatten line breaks."""
    text = _ASS_TAG_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = text.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    return re.sub(r"\s+", " ", text).strip()


def _build_blocks(cues: list[tuple[int, int, str]]) -> list[SubtitleBlock]:
    """Normalize raw (start_ms, end_ms, text) cues into clean, ordered subtitle blocks.

    - strips formatting tags, drops empty and micro-duration (< MIN_CUE_DURATION_MS) cues
    - removes exact duplicates (e.g. ASS outline/fill layers)
    - sorts by time (ASS events are not guaranteed to be ordered)
    - merges consecutive repeats of the same text (rolling YouTube captions, Whisper loops)
    """
    seen = set()
    cleaned = []
    for start_ms, end_ms, text in cues:
        text = _clean_subtitle_text(text)
        if not text or end_ms - start_ms < MIN_CUE_DURATION_MS:
            continue
        key = (start_ms, end_ms, text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append([start_ms, end_ms, text])

    cleaned.sort(key=lambda c: (c[0], c[1]))

    merged: list[list] = []
    for cue in cleaned:
        if merged and cue[2] == merged[-1][2] and cue[0] <= merged[-1][1] + 500:
            merged[-1][1] = max(merged[-1][1], cue[1])
            continue
        merged.append(cue)

    return [
        SubtitleBlock(index=i, start_time=_ms_to_srt_time(s), end_time=_ms_to_srt_time(e), text=t)
        for i, (s, e, t) in enumerate(merged, 1)
    ]


def parse_srt_file(path: Path) -> list[SubtitleBlock]:
    """Parse an SRT file into subtitle blocks, filtering empty and micro-duration artifact cues."""
    import pysrt
    subs = pysrt.from_string(_read_text_file(path))
    return _build_blocks([(s.start.ordinal, s.end.ordinal, s.text) for s in subs])


def parse_ass_file(path: Path) -> list[SubtitleBlock]:
    """Parse an ASS/SSA subtitle file into subtitle blocks."""
    cues = []
    in_events = False
    format_fields: list[str] = []

    for line in _read_text_file(path).splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_events = line.lower() == "[events]"
            continue
        if not in_events:
            continue
        if line.lower().startswith("format:"):
            format_fields = [f.strip().lower() for f in line.split(":", 1)[1].split(",")]
            continue
        if line.lower().startswith("dialogue:") and format_fields:
            parts = line.split(":", 1)[1].split(",", len(format_fields) - 1)
            if len(parts) < len(format_fields):
                continue
            field_map = dict(zip(format_fields, parts))
            raw_text = field_map.get("text", "")
            # Skip vector drawings (\p1 ...) — they are shapes, not dialogue
            if _ASS_DRAWING_RE.search(raw_text):
                continue
            start = _ass_time_to_ms(field_map.get("start", ""))
            end = _ass_time_to_ms(field_map.get("end", ""))
            cues.append((start, end, raw_text))

    return _build_blocks(cues)


def _ass_time_to_ms(ass_time: str) -> int:
    """Convert ASS time (H:MM:SS.CC) to milliseconds."""
    match = re.match(r"(\d+):(\d{1,2}):(\d{1,2})[.,](\d{1,3})", ass_time.strip())
    if not match:
        return 0
    h, m, s, frac = match.groups()
    frac_ms = int(frac.ljust(3, "0")[:3])  # "5" → 500, "50" → 500, "500" → 500
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1000 + frac_ms


def _ass_time_to_srt(ass_time: str) -> str:
    """Convert ASS time (H:MM:SS.CC) to SRT time (HH:MM:SS,mmm)."""
    return _ms_to_srt_time(_ass_time_to_ms(ass_time))


def parse_vtt_file(path: Path) -> list[SubtitleBlock]:
    """Parse a WebVTT file into subtitle blocks."""
    import webvtt
    vtt = webvtt.from_string(_read_text_file(path))
    return _build_blocks([(_srt_to_ms(c.start), _srt_to_ms(c.end), c.text) for c in vtt])


def parse_subtitle_file(path: Path) -> list[SubtitleBlock]:
    """Auto-detect subtitle format and parse."""
    ext = path.suffix.lower()
    if ext == ".srt":
        return parse_srt_file(path)
    elif ext in (".ass", ".ssa"):
        return parse_ass_file(path)
    elif ext == ".vtt":
        return parse_vtt_file(path)
    else:
        raise ValueError(f"Unsupported subtitle format: {ext}")


# ── Whisper Post-processing ────────────────────────────────────────────────

def _is_whisper_hallucination(text: str) -> bool:
    """Detect stock phrases Whisper produces over silence or music."""
    lower = text.lower()
    return len(text) <= 40 and any(p in lower for p in WHISPER_HALLUCINATIONS)


def _collapse_repetitions(text: str) -> str:
    """Collapse Whisper repetition loops ("あああああああ…", "no no no no no no") to 3 repeats."""
    # Unit must not start with a digit/space so numbers like "100000" are left untouched
    def _collapse(m: re.Match) -> str:
        sep = " " if " " in m.group(0) else ""
        return sep.join([m.group(1)] * 3)
    return re.sub(r"([^\d\s].{0,14}?)(?:\s*\1){4,}", _collapse, text)


# ── Translation ─────────────────────────────────────────────────────────────

def _chat_base_stem(path: Path) -> str:
    """Strip .live_chat.json / .live_chat.jsonl / .chat.json suffixes to recover the VOD base stem."""
    name = path.name
    lower = name.lower()
    for suffix in (*LIVE_CHAT_SUFFIXES, ".chat.json", ".chat.jsonl"):
        if lower.endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def _clean_title_for_context(path: Path) -> str:
    """Clean a media/subtitle/chat filename to extract a human-readable show or stream title."""
    name = _chat_base_stem(path)
    # Remove trailing .live_chat / .chat if still present
    name = re.sub(r"\.(live_chat|chat)$", "", name, flags=re.IGNORECASE)
    # Remove language suffixes like .ja, .en, .jpn
    name = re.sub(r"\.(ja|jpn|en|eng|ko|kor|zh|chi)$", "", name, flags=re.IGNORECASE)
    # Remove resolution and quality tags
    name = re.sub(r"\[?(1080p|720p|480p|4k|2160p|bdrip|webrip|web-dl|h264|x264|hevc|x265)\]?", "", name, flags=re.IGNORECASE)
    # Remove video hashes / IDs in brackets e.g. [0Z-DzI7AJOk]
    name = re.sub(r"\[[A-Za-z0-9_-]{10,}\]", "", name)
    # Remove release group tags like [SubsPlease], [Erai-raws] at beginning
    name = re.sub(r"^\[[^\]]+\]\s*", "", name)
    # Remove CRC32 checksums like [A1B2C3D4] or (A1B2C3D4)
    name = re.sub(r"[\[(][0-9A-Fa-f]{8}[\])]", "", name)
    # Remove brackets left empty by the removals above, e.g. "(1080p)" → "()"
    name = re.sub(r"[\[(]\s*[\])]", "", name)
    # Clean underscores/dots and normalize spaces
    name = re.sub(r"[_.\s]+", " ", name).strip()
    return name or path.stem


def _load_user_context(source_path: Path | None) -> str:
    """Load optional user-supplied context/glossary files.

    A per-file context (`<name>.context.txt` next to the source) is combined with the first
    global context/glossary file found in INPUT/ or the project root.
    """
    sections: list[Path] = []

    if source_path:
        base_stem = _chat_base_stem(source_path)
        for stem_candidate in dict.fromkeys((base_stem, source_path.stem)):
            per_file = source_path.parent / f"{stem_candidate}.context.txt"
            if per_file.is_file() and per_file not in sections:
                sections.append(per_file)

    global_candidates = [
        INPUT_DIR / "context.txt",
        INPUT_DIR / "glossary.txt",
        INPUT_DIR / "glossary.json",
        SCRIPT_DIR / "context.txt",
        SCRIPT_DIR / "glossary.txt",
        SCRIPT_DIR / "glossary.json",
    ]
    for c in global_candidates:
        if c.is_file() and c not in sections:
            sections.append(c)
            break

    contents = []
    for c in sections:
        try:
            content = c.read_text(encoding="utf-8-sig").strip()
            if content:
                print(f"  ✓ Loaded user context/glossary from {c.name}")
                contents.append(content)
        except Exception as e:
            print(f"  ⚠ Could not read user context file {c.name}: {e}")
    return "\n\n".join(contents)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNSUPPORTED_MODELS: set[str] = set()


def _resolve_think_param(think: str) -> bool | str | None:
    """Convert CLI --think mode into the parameter expected by ollama.chat()."""
    mode = (think or "auto").strip().lower()
    if mode == "auto":
        return None
    if mode == "off":
        return False
    if mode == "on":
        return True
    if mode in ("low", "medium", "high"):
        return mode
    return None


def _predict_budget(base_tokens: int, think: str, floor_thinking: int = 4096) -> int:
    """Ensure num_predict has enough room for reasoning tokens when thinking is not disabled."""
    if (think or "auto").strip().lower() == "off":
        return base_tokens
    return max(base_tokens * 2, floor_thinking)


def _ollama_chat(
    model: str,
    messages: list[dict],
    options: dict,
    retries: int = OLLAMA_RETRIES,
    think: str = "auto",
) -> str:
    """Call Ollama chat with retry/backoff and reasoning-model safeguards.

    - Strips inline <think>...</think> blocks from reasoning models.
    - Gracefully drops the `think` parameter if a non-thinking model rejects it.
    - Automatically retries once with `think=False` if a reasoning model exhausts `num_predict`
      inside its thinking trace and emits an empty answer.
    """
    import ollama
    think_param = None if model in _THINK_UNSUPPORTED_MODELS else _resolve_think_param(think)
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            kwargs: dict = {"model": model, "messages": messages, "options": options}
            if think_param is not None:
                kwargs["think"] = think_param
            try:
                response = ollama.chat(**kwargs)
            except (TypeError, ollama.ResponseError) as e:
                # Older ollama client without `think` kwarg, or a model that rejects `think`
                err_msg = str(e).lower()
                if think_param is not None and (
                    isinstance(e, TypeError)
                    or (getattr(e, "status_code", None) == 400 and "think" in err_msg)
                ):
                    _THINK_UNSUPPORTED_MODELS.add(model)
                    think_param = None
                    response = ollama.chat(model=model, messages=messages, options=options)
                else:
                    raise

            msg_obj = response["message"]
            raw_content = msg_obj["content"] or ""
            cleaned = _THINK_RE.sub("", raw_content).strip()

            done_reason = (
                response.get("done_reason")
                if isinstance(response, dict)
                else getattr(response, "done_reason", None)
            )
            thinking_trace = (
                msg_obj.get("thinking")
                if isinstance(msg_obj, dict)
                else getattr(msg_obj, "thinking", None)
            ) or ("<think>" in raw_content.lower())

            # Reasoning model spent its entire token budget thinking and produced no answer
            if not cleaned and (done_reason == "length" or thinking_trace) and think_param is not False:
                if model not in _THINK_UNSUPPORTED_MODELS:
                    try:
                        fallback_resp = ollama.chat(
                            model=model, messages=messages, options=options, think=False
                        )
                        fallback_raw = fallback_resp["message"]["content"] or ""
                        cleaned = _THINK_RE.sub("", fallback_raw).strip()
                    except Exception:
                        pass

            return cleaned
        except ollama.ResponseError as e:
            # Model missing etc. — retrying won't help
            if getattr(e, "status_code", None) == 404:
                raise
            last_error = e
        except Exception as e:
            last_error = e
        if attempt < retries:
            wait = 2 * (attempt + 1)
            print(f"  ⚠ Ollama request failed ({last_error}), retrying in {wait}s...")
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def check_ollama(model: str) -> bool:
    """Verify Ollama is reachable and the model is available before translating."""
    try:
        import ollama
    except ImportError:
        print("  ✗ The 'ollama' Python package is not installed. Run: bash setup.sh")
        return False
    try:
        ollama.show(model)
        return True
    except ollama.ResponseError as e:
        if getattr(e, "status_code", None) == 404:
            print(f"  ✗ Ollama model '{model}' is not installed. Run: ollama pull {model}")
        else:
            print(f"  ✗ Ollama error: {e}")
    except Exception as e:
        print(f"  ✗ Cannot reach Ollama ({e}). Is it running? Start it with: ollama serve")
    return False


def analyze_context(
    blocks: list[SubtitleBlock],
    source_path: Path | None = None,
    model: str = DEFAULT_OLLAMA_MODEL,
    think: str = "auto",
) -> str:
    """Pass 1: Deep show, character, and terminology analysis across the entire episode."""
    user_context = _load_user_context(source_path)

    title_info = ""
    if source_path:
        cleaned_title = _clean_title_for_context(source_path)
        title_info = f"Show / Episode Title (from filename): {cleaned_title}\n"

    # Take representative dialogue samples across the whole episode (prioritizing blocks with actual dialogue)
    dialogue_blocks = [b for b in blocks if _contains_cjk(b.text)]
    if not dialogue_blocks:
        dialogue_blocks = blocks

    sample_size = min(len(dialogue_blocks), 60)
    if len(dialogue_blocks) <= sample_size:
        sample_blocks = dialogue_blocks
    else:
        part = sample_size // 3
        mid_start = (len(dialogue_blocks) - part) // 2
        sample_blocks = dialogue_blocks[:part] + dialogue_blocks[mid_start:mid_start + part] + dialogue_blocks[-part:]

    sample_text = "\n".join(b.text for b in sample_blocks)
    user_section = f"User-provided notes/glossary (authoritative — always prefer these):\n{user_context}\n" if user_context else ""

    prompt = f"""You are a senior anime translation director and localizer.
Analyze this show's title and dialogue to create a comprehensive translation briefing for an expert subtitle translator.

{title_info}Dialogue samples:
{sample_text}

{user_section}
Respond with a structured context guide containing:
1. Show Identification & Premise: (Identify the anime franchise, universe, and episode premise)
2. Character Profiles & Relationships: (Identify who is speaking, their personalities, speaking styles, and relationships to other characters)
3. Terminology & Proper Nouns Glossary: (Official or natural English translations for character names, nicknames, in-universe game/lore mechanics, and abbreviations)
4. Subtitle Tone & Style Guidelines: (Formality level, how to handle honorifics, humor/slang, and emotional tone)

Only state facts supported by the title, the dialogue, or the user notes. If you are not sure about the show or a name, say "unknown" rather than guessing — wrong names are worse than none.
Be concise, concrete, and directly actionable for subtitle translation (max 250 words)."""

    try:
        context = _ollama_chat(
            model,
            [{"role": "user", "content": prompt}],
            {
                "temperature": 0.2,
                "num_ctx": 8192,
                "num_predict": _predict_budget(800, think, floor_thinking=3072),
                "repeat_penalty": 1.1,
            },
            think=think,
        )
        print("  ✓ Deep context & glossary analysis complete")
        if user_context:
            context = f"### User Notes & Glossary (authoritative):\n{user_context}\n\n" + context
        return context
    except Exception as e:
        print(f"  ⚠ Context analysis failed: {e}")
        return user_context


def _needs_retry(source: str, translated: str) -> bool:
    """Return True if a translation is missing, untranslated, or looks like a runaway generation."""
    if not translated:
        return True
    # CJK left in the output (untranslated, or Qwen-style Chinese leakage)
    if _contains_cjk(source) and _contains_cjk(translated):
        return True
    # Runaway output (model rambled or merged several lines into one)
    if len(translated) > 6 * len(source) + 80:
        return True
    return False


def translate_blocks(
    blocks: list[SubtitleBlock],
    context: str = "",
    model: str = DEFAULT_OLLAMA_MODEL,
    source_lang: str = "Japanese",
    batch_size: int = TRANSLATION_BATCH_SIZE,
    think: str = "auto",
) -> list[SubtitleBlock]:
    """Pass 2: Translate subtitle blocks in bidirectional context-aware batches with automatic verification and retry."""
    if not blocks:
        return blocks

    # Check if the entire file is genuinely already English
    if _is_file_already_english(blocks, source_lang):
        print("  → Content appears to already be in English across the entire file, skipping translation")
        for b in blocks:
            b.translated_text = b.text
        return blocks

    batch_size = max(1, batch_size)
    total_batches = (len(blocks) + batch_size - 1) // batch_size
    print(f"  → Translating {len(blocks)} blocks from {source_lang} in {total_batches} batches...")

    context_section = ""
    if context:
        context_section = f"\n\n### Show Context & Glossary (follow these profiles and terms strictly):\n{context}\n### End Context"

    honorific_rule = (
        "- Keep Japanese honorifics (-san, -kun, -chan, -senpai, -sensei, -sama) attached to names unless the glossary localizes them.\n"
        if source_lang == "Japanese" else ""
    )

    system_prompt = f"""You are an expert anime subtitle translator and localizer. Translate {source_lang} dialogue to natural, fluent, character-accurate English subtitles.

Translation Rules:
- Translate ONLY the dialogue text. Output EXACTLY one translation per input line, keeping each line's [N] number prefix.
- Never merge or split lines: output line [N] must correspond to input line [N]. If a sentence spans several lines, translate each fragment so the sequence reads naturally in English.
- Follow the Character Profiles, Voice/Tone Guidelines, and Terminology Glossary provided in the context. Use glossary spellings for names and terms exactly.
- Keep translations natural and concise — subtitles must be easily readable in the video display time.
- Maintain each character's distinct personality, speech style, and emotional nuance.
{honorific_rule}- Translate interjections and reactions naturally (e.g. えっ → "Huh?", ああ → "Yeah." / "Ah.").
- Translate cultural references, idioms, and gaming/fantasy slang naturally into fluent English.
- Infer omitted subjects and pronouns from the surrounding dialogue.
- If an input line is already in English (such as English song lyrics or English signage), keep it as English.
- Output English only — never leave {source_lang} characters in the output.
- Do NOT add notes, explanations, romanization, or commentary.
- Example output format:
[1] Translated line
[2] Translated line{context_section}"""

    unresolved = 0

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(blocks))
        batch = blocks[start:end]

        # Build recent dialogue history (lookback)
        lookback_lines = []
        if start > 0:
            lookback_blocks = blocks[max(0, start - LOOKBACK_COUNT):start]
            for b in lookback_blocks:
                en_text = b.translated_text if b.translated_text else "(pending)"
                lookback_lines.append(f"  • {b.text}  →  {en_text}")

        # Build upcoming dialogue preview (lookahead)
        lookahead_lines = []
        if end < len(blocks):
            lookahead_blocks = blocks[end:min(len(blocks), end + LOOKAHEAD_COUNT)]
            for b in lookahead_blocks:
                lookahead_lines.append(f"  • {b.text}")

        # Build current batch lines to translate
        lines_to_translate = []
        for i, block in enumerate(batch):
            lines_to_translate.append(f"[{i+1}] {block.text}")
        batch_text = "\n".join(lines_to_translate)

        # Assemble prompt with bidirectional context
        prompt_sections = []
        if lookback_lines:
            prompt_sections.append(
                "### Preceding Dialogue History (for speaker continuity, tone, and pronoun targets — do NOT re-translate):\n"
                + "\n".join(lookback_lines)
            )

        prompt_sections.append(
            f"### Lines to Translate ({len(batch)} lines):\n"
            f"Translate these lines into natural English. Output EXACTLY {len(batch)} lines, each prefixed with its number [1] to [{len(batch)}]:\n\n"
            + batch_text
        )

        if lookahead_lines:
            prompt_sections.append(
                "### Upcoming Dialogue Preview (reference only to understand sentence flow and split clauses — do NOT translate these):\n"
                + "\n".join(lookahead_lines)
            )

        user_prompt = "\n\n".join(prompt_sections)

        try:
            response_text = _ollama_chat(
                model,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                {
                    "temperature": 0.2,
                    "num_ctx": 8192,
                    "num_predict": _predict_budget(max(len(batch) * 80, 1000), think, floor_thinking=4096),
                    "repeat_penalty": 1.05,  # high values push the model away from repeating names/terms
                },
                think=think,
            )
            translated_dict = _parse_translation_response(response_text, len(batch))
            for i, block in enumerate(batch, 1):
                block.translated_text = translated_dict.get(i, "")
        except Exception as e:
            print(f"  ⚠ Batch {batch_idx + 1} translation error: {e}")
            for block in batch:
                block.translated_text = ""

        # Check for any missing or untranslated (still CJK) lines in this batch
        missing_indices = [i for i, b in enumerate(batch, 1) if _needs_retry(b.text, b.translated_text)]

        if missing_indices:
            print(f"  ⚠ {len(missing_indices)} line(s) missing or untranslated in batch {batch_idx + 1}, retrying...")
            retry_lines = [f"[{idx}] {batch[idx-1].text}" for idx in missing_indices]
            retry_prompt = (
                "### Full passage for reference (do NOT output these):\n" + batch_text + "\n\n"
                f"### Translate ONLY these {len(retry_lines)} line(s) into natural English.\n"
                f"Output EXACTLY {len(retry_lines)} line(s), each keeping its original [N] number:\n\n"
                + "\n".join(retry_lines)
            )
            try:
                retry_text = _ollama_chat(
                    model,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": retry_prompt},
                    ],
                    {
                        "temperature": 0.1,
                        "num_ctx": 8192,
                        "num_predict": _predict_budget(max(len(retry_lines) * 80, 500), think, floor_thinking=3072),
                        "repeat_penalty": 1.05,
                    },
                    think=think,
                )
                retry_dict = _parse_translation_response(retry_text, len(batch), allow_positional=False)
                for idx in missing_indices:
                    candidate = retry_dict.get(idx, "")
                    if candidate and not _needs_retry(batch[idx-1].text, candidate):
                        batch[idx-1].translated_text = candidate
            except Exception as e:
                print(f"  ⚠ Batch retry error: {e}")

            # Final fallback for any stubborn line: direct single-line translation
            for idx in missing_indices:
                blk = batch[idx-1]
                if not _needs_retry(blk.text, blk.translated_text):
                    continue
                prev_line = batch[idx-2].text if idx >= 2 else (blocks[start-1].text if start > 0 else "")
                next_line = batch[idx].text if idx < len(batch) else (blocks[end].text if end < len(blocks) else "")
                single_user = blk.text
                if prev_line or next_line:
                    single_user = (
                        f"Previous line (context only): {prev_line or '-'}\n"
                        f"Next line (context only): {next_line or '-'}\n\n"
                        f"Line to translate:\n{blk.text}"
                    )
                try:
                    single_text = _ollama_chat(
                        model,
                        [
                            {"role": "system", "content": f"Translate this {source_lang} dialogue line into natural fluent English. Output ONLY the English translation of the requested line with no quotes, notes, or commentary.{context_section}"},
                            {"role": "user", "content": single_user},
                        ],
                        {
                            "temperature": 0.1,
                            "num_ctx": 8192,
                            "num_predict": _predict_budget(200, think, floor_thinking=2048),
                            "repeat_penalty": 1.05,
                        },
                        think=think,
                    )
                    single_text = _clean_translation_line(single_text.split("\n")[0])
                    if not _needs_retry(blk.text, single_text):
                        blk.translated_text = single_text
                except Exception as e:
                    print(f"  ⚠ Single-line fallback failed for block {start + idx}: {e}")

                if _needs_retry(blk.text, blk.translated_text):
                    # Give up: keep the original text rather than an empty or broken cue
                    blk.translated_text = blk.text
                    unresolved += 1

        # Progress
        pct = (batch_idx + 1) / total_batches * 100
        print(f"  → Progress: {pct:.0f}% ({end}/{len(blocks)} blocks)")

    if unresolved:
        print(f"  ⚠ {unresolved} line(s) could not be translated and were kept in the original language")

    return blocks


_INDEX_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\*\*)?(?:\[(\d+)\]|(\d+)[.):\]])(?:\*\*)?:?\s*(.*)$"
)
_ECHO_PREFIX_RE = re.compile(
    r"^(?:\[(?:english|en)\]|(?:english|translation|en)\s*:)\s*", re.IGNORECASE
)
_STOP_RE = re.compile(
    r"^(?:\(?\s*(?:note|notes|explanation|translator'?s? notes?)\b|-{3,}$|#{1,6}\s)", re.IGNORECASE
)


def _clean_translation_line(text: str) -> str:
    """Remove common LLM artifacts from a single translated line."""
    text = text.strip()
    # Model echoed "source → translation"
    if "→" in text:
        left, right = text.rsplit("→", 1)
        if _contains_cjk(left) and right.strip():
            text = right.strip()
    text = _ECHO_PREFIX_RE.sub("", text).strip()
    text = text.replace("**", "").strip()
    # Strip quotes wrapping the whole line (but keep quotes inside dialogue)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”「」" and text.count(text[0]) == 2:
        text = text[1:-1].strip()
    if len(text) >= 2 and text[0] in "“「" and text[-1] in "”」":
        text = text[1:-1].strip()
    return text


def _parse_translation_response(response: str, expected_count: int, allow_positional: bool = True) -> dict[int, str]:
    """Parse the LLM translation response into a dict mapping 1-based index to translation.
    
    Handles:
    - [1] Line text
    - 1. Line text
    - 1) Line text
    - 1: Line text
    - **[1]** Line text (markdown)
    - Multi-line translations under the same index
    - Trailing notes/commentary (ignored)
    - Unnumbered fallback
    """
    indexed_translations: dict[int, str] = {}
    lines = response.strip().split("\n")
    current_idx = None
    current_lines: list[str] = []

    def flush():
        if current_idx is not None and 1 <= current_idx <= expected_count:
            text = _clean_translation_line(" ".join(current_lines))
            # Keep the first non-empty answer if the model repeats an index
            if text and not indexed_translations.get(current_idx):
                indexed_translations[current_idx] = text

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        match = _INDEX_RE.match(line_clean)
        if match:
            flush()
            num_str = match.group(1) or match.group(2)
            current_idx = int(num_str)
            text_part = match.group(3).strip()
            current_lines = [text_part] if text_part else []
        elif _STOP_RE.match(line_clean):
            # Notes / commentary after the translations — stop collecting
            flush()
            current_idx = None
            current_lines = []
        elif current_idx is not None:
            current_lines.append(line_clean)

    flush()

    # If LLM did not use numbers at all, fallback to positional matching
    if not indexed_translations and allow_positional:
        plain_lines = [
            l.strip() for l in lines
            if l.strip() and not l.strip().endswith(":") and not _STOP_RE.match(l.strip())
        ]
        if len(plain_lines) == expected_count:  # only trust positions when counts line up
            for idx, text in enumerate(plain_lines, 1):
                indexed_translations[idx] = _clean_translation_line(text)

    return indexed_translations


# ── SRT Output ──────────────────────────────────────────────────────────────

def write_srt(blocks: list[SubtitleBlock], output_path: Path, use_translated: bool = True):
    """Write subtitle blocks to an SRT file, ensuring only valid non-empty entries are written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for block in blocks:
            text = block.translated_text if (use_translated and block.translated_text) else block.text
            text = text.strip()
            if not text:
                continue
            written_count += 1
            f.write(f"{written_count}\n")
            f.write(f"{block.start_time} --> {block.end_time}\n")
            f.write(f"{text}\n\n")
    print(f"  ✓ Written {written_count} subtitles to {output_path}")


# ── YouTube VOD Live Chat (*.live_chat.json) ────────────────────────────────

_CHAT_RENDERER_KINDS = {
    "liveChatTextMessageRenderer": "text",
    "liveChatPaidMessageRenderer": "paid",
    "liveChatPaidStickerRenderer": "sticker",
    "liveChatMembershipItemRenderer": "membership",
}

# YouTube auto-appends collision suffixes like "-l6o1s" or "-e6j" to @handles
_YT_HANDLE_SUFFIX_RE = re.compile(r"-[a-z]\d[a-z0-9]{1,3}$")
_JP_LAUGH_ONLY_RE = re.compile(r"^[wW]{2,}$")

# Pleasant high-contrast pastel colors in ASS BGR hex (&HBBGGRR&) for viewer handles
_ASS_AUTHOR_PALETTE = (
    "EADE80",  # soft cyan
    "B18FF4",  # soft pink
    "A5E1C5",  # soft lime
    "D893CE",  # soft lavender
    "91ABFF",  # soft peach
    "F9CA90",  # soft sky
    "9DF5FF",  # soft yellow
    "C4CB80",  # soft mint
)
_ASS_ROLE_COLORS = {
    "owner": "4FD5FF",      # warm gold (#FFD54F)
    "moderator": "FF9E5E",  # YouTube mod blue (#5E9EFF)
    "member": "68E34B",     # YouTube member green (#4BE368)
}
_ASS_SUPERCHAT_COLOR = "4370FF"  # bright coral/orange (#FF7043)


def is_live_chat_file(path: Path) -> bool:
    """Return True if path is a yt-dlp YouTube live_chat.json / JSONL replay file."""
    if not path.is_file():
        return False
    lower = path.name.lower()
    if lower in ("glossary.json",):
        return False
    if lower.endswith(LIVE_CHAT_SUFFIXES):
        return True
    if path.suffix.lower() in (".json", ".jsonl"):
        try:
            with open(path, "rb") as f:
                head = f.read(65536).decode("utf-8-sig", errors="ignore")
            for line in head.splitlines():
                line = line.strip()
                if line:
                    return line.startswith("{") and '"replayChatItemAction"' in line
        except OSError:
            return False
    return False


def _parse_chat_offset_ms(video_offset_msec: str | int | None, timestamp_text: str) -> int:
    """Compute signed millisecond offset relative to stream start (0:00).

    YouTube sets videoOffsetTimeMsec="0" on all pre-stream waiting-room messages while storing
    their negative offset in timestampText (e.g. "-20:42"). Parsing the negative timestamp keeps
    waiting-room messages from piling up at 00:00:00 in video overlays.
    """
    raw_ms = 0
    if video_offset_msec is not None:
        try:
            raw_ms = int(video_offset_msec)
        except (TypeError, ValueError):
            raw_ms = 0

    ts = (timestamp_text or "").strip()
    if ts:
        neg = ts.startswith("-")
        clean_ts = ts.lstrip("+-").strip()
        parts = clean_ts.split(":")
        if len(parts) in (2, 3) and all(p.isdigit() for p in parts):
            nums = [int(p) for p in parts]
            secs = (nums[0] * 3600 + nums[1] * 60 + nums[2]) if len(nums) == 3 else (nums[0] * 60 + nums[1])
            parsed_ms = secs * 1000
            if neg and raw_ms <= 0:
                return -parsed_ms
            if not neg and raw_ms == 0 and parsed_ms > 0:
                return parsed_ms

    return raw_ms


def _extract_author_role(renderer: dict) -> str:
    """Identify the chatter's highest badge role: owner > moderator > member > viewer."""
    role = "viewer"
    for badge in renderer.get("authorBadges") or []:
        b = badge.get("liveChatAuthorBadgeRenderer") or {}
        icon_type = ((b.get("icon") or {}).get("iconType") or "").upper()
        label = (
            b.get("tooltip")
            or ((b.get("accessibility") or {}).get("accessibilityData") or {}).get("label")
            or ""
        ).lower()
        if icon_type == "OWNER" or "owner" in label:
            return "owner"
        if icon_type == "MODERATOR" or "moderator" in label:
            role = "moderator"
        elif role == "viewer" and ("member" in label or "customThumbnail" in b):
            role = "member"
    return role


def _extract_runs_text(runs: list[dict]) -> tuple[str, dict[str, dict]]:
    """Flatten YouTube message runs into text plus a map of :custom-emote: tokens to run dicts."""
    parts: list[str] = []
    custom_emojis: dict[str, dict] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        if "text" in run:
            parts.append(str(run["text"]))
        elif "emoji" in run and isinstance(run["emoji"], dict):
            em = run["emoji"]
            if em.get("isCustomEmoji"):
                shortcuts = em.get("shortcuts") or []
                label = (
                    ((em.get("image") or {}).get("accessibility") or {})
                    .get("accessibilityData") or {}
                ).get("label", "")
                raw_token = shortcuts[0] if shortcuts else (f":{label}:" if label else ":emote:")
                clean_inner = re.sub(r"\s+", "-", raw_token.strip(":")) or "emote"
                token = f":{clean_inner}:"
                if parts and not parts[-1].endswith(" "):
                    parts.append(" ")
                parts.append(token)
                parts.append(" ")
                custom_emojis[token] = run
            else:
                emoji_str = (
                    em.get("emojiId")
                    or (
                        ((em.get("image") or {}).get("accessibility") or {})
                        .get("accessibilityData") or {}
                    ).get("label", "")
                )
                parts.append(str(emoji_str))
    text = re.sub(r" +", " ", "".join(parts)).strip()
    return text, custom_emojis


def _rebuild_chat_runs(translated_text: str, custom_emojis: dict[str, dict]) -> list[dict]:
    """Reconstruct YouTube message.runs from translated text, preserving custom emoji run objects."""
    if not custom_emojis:
        return [{"text": translated_text}]

    # Split around known custom emoji tokens (longest token first to avoid prefix collisions)
    tokens_sorted = sorted(custom_emojis.keys(), key=len, reverse=True)
    pattern = re.compile("(" + "|".join(re.escape(t) for t in tokens_sorted) + ")")
    runs: list[dict] = []
    for piece in pattern.split(translated_text):
        if not piece:
            continue
        if piece in custom_emojis:
            runs.append(copy.deepcopy(custom_emojis[piece]))
        else:
            runs.append({"text": piece})
    return runs or [{"text": translated_text}]


def _iter_chat_renderers(record: dict):
    """Yield (slot_id, kind, renderer_dict, in_overlay) for chat items inside a JSONL record."""
    rcia = record.get("replayChatItemAction")
    if not isinstance(rcia, dict):
        return
    for act_idx, act in enumerate(rcia.get("actions") or []):
        if not isinstance(act, dict):
            continue
        if "addChatItemAction" in act:
            item = (act["addChatItemAction"] or {}).get("item") or {}
            for r_key, kind in _CHAT_RENDERER_KINDS.items():
                if r_key in item and isinstance(item[r_key], dict):
                    yield f"{act_idx}:add:{r_key}", kind, item[r_key], True
        if "replaceChatItemAction" in act:
            item = (act["replaceChatItemAction"] or {}).get("replacementItem") or {}
            for r_key, kind in _CHAT_RENDERER_KINDS.items():
                if r_key in item and isinstance(item[r_key], dict):
                    yield f"{act_idx}:rep:{r_key}", kind, item[r_key], False
        if "addLiveChatTickerItemAction" in act:
            t_item = (act["addLiveChatTickerItemAction"] or {}).get("item") or {}
            for t_key, t_val in t_item.items():
                if not isinstance(t_val, dict):
                    continue
                inner = (
                    ((t_val.get("showItemEndpoint") or {}).get("showLiveChatItemEndpoint") or {})
                    .get("renderer") or {}
                )
                for r_key, kind in _CHAT_RENDERER_KINDS.items():
                    if r_key in inner and isinstance(inner[r_key], dict):
                        yield f"{act_idx}:ticker:{t_key}:{r_key}", kind, inner[r_key], False


def _simple_or_runs_text(field_obj: dict | None) -> str:
    """Extract plain text from a YouTube {simpleText: ...} or {runs: [...]} field."""
    if not isinstance(field_obj, dict):
        return ""
    if "simpleText" in field_obj:
        return str(field_obj["simpleText"]).strip()
    if "runs" in field_obj and isinstance(field_obj["runs"], list):
        txt, _ = _extract_runs_text(field_obj["runs"])
        return txt
    return ""


def _chat_message_needs_translation(
    text: str,
    custom_emojis: dict[str, dict],
    source_lang: str,
) -> bool:
    """Return True if a chat message contains foreign-language words that need LLM translation.

    Automatically skips emoji-only, custom-emote-only, numeric/punctuation-only, and (for CJK
    streams) already-English/ASCII messages — including stylized Unicode Latin like '𝑶𝑯 𝑴𝒀 𝑮𝑶𝑫'.
    """
    stripped = text
    for token in custom_emojis:
        stripped = stripped.replace(token, " ")
    norm = unicodedata.normalize("NFKC", stripped).strip()
    if not norm:
        return False

    # Pure emojis, numbers, kaomoji, or punctuation — no letters in any script
    if not any(ch.isalpha() for ch in norm):
        return False

    if source_lang in ("Japanese", "Korean", "Chinese"):
        # Standalone Japanese "www" / "ｗｗｗｗ" laughter should still translate to "lol"
        if source_lang == "Japanese" and _JP_LAUGH_ONLY_RE.match(norm):
            return True
        # Every letter in the normalized message is basic Latin (English viewer chat)
        if all(ord(ch) < 128 for ch in norm if ch.isalpha()):
            return False

    return True


def parse_yt_live_chat(path: Path, language_override: str | None = None) -> tuple[list[LiveChatMessage], str]:
    """Stream-parse a yt-dlp *.live_chat.json file into LiveChatMessage entries and resolve language."""
    messages: list[LiveChatMessage] = []
    malformed = 0

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line_no, raw_line in enumerate(f, 1):
            raw_str = raw_line.strip()
            if not raw_str:
                continue
            try:
                record = json.loads(raw_str)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(record, dict):
                continue

            rcia = record.get("replayChatItemAction") or {}
            video_offset_raw = rcia.get("videoOffsetTimeMsec")

            for slot_id, kind, renderer, in_overlay in _iter_chat_renderers(record):
                author = _simple_or_runs_text(renderer.get("authorName")) or "Chat"
                ts_text = _simple_or_runs_text(renderer.get("timestampText"))
                offset_ms = _parse_chat_offset_ms(video_offset_raw, ts_text)
                role = _extract_author_role(renderer)
                amount_text = _simple_or_runs_text(renderer.get("purchaseAmountText"))
                header_text = (
                    _simple_or_runs_text(renderer.get("headerPrimaryText"))
                    or _simple_or_runs_text(renderer.get("headerSubtext"))
                )

                msg_obj = renderer.get("message")
                runs = msg_obj.get("runs") if isinstance(msg_obj, dict) else None
                text, custom_emojis = _extract_runs_text(runs) if isinstance(runs, list) else ("", {})

                # Skip completely empty items (unless paid Super Chat/Sticker or Membership event)
                if not text and not amount_text and not header_text:
                    continue

                messages.append(LiveChatMessage(
                    index=len(messages) + 1,
                    line_no=line_no,
                    slot_id=slot_id,
                    offset_ms=offset_ms,
                    timestamp_text=ts_text,
                    author=author,
                    role=role,
                    kind=kind,
                    amount_text=amount_text,
                    header_text=header_text,
                    text=text,
                    custom_emojis=custom_emojis,
                    in_overlay=in_overlay,
                ))

    if malformed:
        print(f"  ⚠ Skipped {malformed} malformed JSON line(s) in {path.name}")

    # Resolve primary stream chat language
    if language_override:
        source_lang = _language_code_to_name(language_override) or language_override
    else:
        detected = _detect_language_from_texts([m.text for m in messages if m.text])
        source_lang = detected or _UNKNOWN_LANG

    for m in messages:
        if m.text and m.kind in ("text", "paid", "membership"):
            m.needs_translation = _chat_message_needs_translation(m.text, m.custom_emojis, source_lang)
        else:
            m.needs_translation = False

    return messages, source_lang


def _clean_chat_author(author: str) -> str:
    """Clean a YouTube handle for compact display (strips leading @ and random -x9y collision tags)."""
    clean = (author or "").strip()
    if clean.startswith("@") and len(clean) > 1:
        clean = clean[1:]
    clean = _YT_HANDLE_SUFFIX_RE.sub("", clean)
    return clean or (author or "Chat").strip()


def _clean_chat_translation(text: str, author: str, custom_emojis: dict[str, dict]) -> str:
    """Strip LLM formatting/echoed sender handles and restore any dropped custom emote tokens."""
    cleaned = _clean_translation_line(text)
    if not cleaned:
        return ""

    # Strip echoed sender handle at the start of the message (e.g. "(@user) msg" or "@user: msg")
    raw_author = (author or "").strip()
    short_author = _clean_chat_author(raw_author)
    for handle in dict.fromkeys(h for h in (raw_author, raw_author.lstrip("@"), short_author) if h):
        esc = re.escape(handle)
        cleaned = re.sub(
            rf"^(?:\(\s*@?{esc}\s*\)\s*:?\s*|@?{esc}\s*:\s+)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

    # Never lose custom channel/YouTube emotes that appeared in the original message
    for token in custom_emojis:
        if token not in cleaned:
            cleaned = f"{cleaned} {token}".strip()

    return cleaned


def analyze_chat_context(
    messages: list[LiveChatMessage],
    source_path: Path | None = None,
    model: str = DEFAULT_OLLAMA_MODEL,
    think: str = "auto",
) -> str:
    """Pass 1 (Live Chat): Identify streamers, collab topic, nicknames, and recurring chat slang."""
    user_context = _load_user_context(source_path)

    title_info = ""
    if source_path:
        cleaned_title = _clean_title_for_context(source_path)
        title_info = f"Stream / VOD Title (from filename): {cleaned_title}\n"

    candidates = [m for m in messages if m.in_overlay and m.needs_translation and m.text]
    if not candidates:
        return user_context

    sample_size = min(len(candidates), 75)
    if len(candidates) <= sample_size:
        samples = candidates
    else:
        part = sample_size // 3
        mid_start = (len(candidates) - part) // 2
        samples = candidates[:part] + candidates[mid_start:mid_start + part] + candidates[-part:]

    sample_text = "\n".join(f"({m.author}) {m.text}" for m in samples)
    user_section = f"User-provided notes/glossary (authoritative — always prefer these):\n{user_context}\n" if user_context else ""

    prompt = f"""You are a senior livestream and VTuber/utaite localization director.
Analyze this YouTube stream's title and viewer live chat samples to write a concise translation briefing.

{title_info}Live chat samples:
{sample_text}

{user_section}
Respond with a concise context guide containing:
1. Streamers & Collaborators: (Names, preferred English romanizations, fan nicknames, and honorific usage)
2. Stream Premise & Moments: (What kind of stream this is — collab, karaoke, announcement, gaming, Q&A — and key topics viewers are reacting to)
3. Recurring Chat Terms & Inside Jokes: (Consistent English translations for recurring names, slang, abbreviations, or jokes in the chat)

Only state facts supported by the stream title, chat samples, or user notes. Do not invent lore if unsure.
Keep it concrete and directly actionable for translating viewer chat messages (max 220 words)."""

    try:
        context = _ollama_chat(
            model,
            [{"role": "user", "content": prompt}],
            {
                "temperature": 0.2,
                "num_ctx": 8192,
                "num_predict": _predict_budget(750, think, floor_thinking=3072),
                "repeat_penalty": 1.1,
            },
            think=think,
        )
        print("  ✓ Stream chat context & glossary analysis complete")
        if user_context:
            context = f"### User Notes & Glossary (authoritative):\n{user_context}\n\n" + context
        return context
    except Exception as e:
        print(f"  ⚠ Chat context analysis failed: {e}")
        return user_context


def translate_chat_messages(
    messages: list[LiveChatMessage],
    context: str = "",
    model: str = DEFAULT_OLLAMA_MODEL,
    source_lang: str = "Japanese",
    batch_size: int = CHAT_TRANSLATION_BATCH_SIZE,
    think: str = "auto",
) -> list[LiveChatMessage]:
    """Pass 2 (Live Chat): Translate YouTube VOD chat messages with slang localization and caching."""
    for m in messages:
        if not m.needs_translation:
            m.translated_text = m.text

    translatable = [m for m in messages if m.needs_translation]
    if not translatable:
        print("  → All chat messages are already in English or emoji-only, skipping translation")
        return messages

    batch_size = max(1, batch_size)
    print(
        f"  → Translating {len(translatable)} chat message(s) from {source_lang} "
        f"({len(messages) - len(translatable)} emoji/English message(s) kept as-is)..."
    )

    context_section = ""
    if context:
        context_section = f"\n\n### Stream Context & Glossary (follow strictly):\n{context}\n### End Context"

    jp_slang_guide = (
        "- Localize Japanese livestream & VTuber/utaite chat slang naturally into idiomatic English stream chat:\n"
        "  • trailing w / ww / 草 / 大草原 → \"lol\" / \"lmao\" (never translate 草 literally as \"grass\" or leave raw \"w\" attached to English words)\n"
        "  • trailing （ or （） → playful self-aware aside, e.g. \"(lol)\" or light deadpan tone\n"
        "  • 待機 → \"Waiting!\" / \"Standing by!\", きた / キター → \"Here we go!\" / \"It's here!\"\n"
        "  • てぇてぇ / 尊い / 尊死 → \"so precious\" / \"so wholesome\" / \"dying from how precious this is\"\n"
        "  • 推し → \"oshi\" or \"fave\", 切り抜き → \"clips\", ﾀﾋ → \"die\" (slang)\n"
        "  • ナイスパ / ないすぱ → \"Nice Super Chat!\", 赤スパ → \"Red Super Chat\"\n"
        "  • おつ / お疲れ様でした → \"Thanks for the stream!\" / \"Good work!\", 8888 → applause/clapping\n"
        "- Keep Japanese honorifics (-san, -chan, -kun, -senpai, -sama) on streamer/viewer names unless the glossary specifies otherwise.\n"
        if source_lang == "Japanese" else ""
    )

    system_prompt = f"""You are an expert livestream and VTuber/utaite chat translator. Translate {source_lang} (and any occasional mixed non-English) YouTube live chat messages into natural, lively, idiomatic English livestream chat.

Translation Rules:
- Output EXACTLY one translation per numbered input line, keeping each line's [N] number prefix.
- Input lines include the sender handle in parentheses like `[1] (@user) message` ONLY so you can tell who is talking or replying to whom. Do NOT output `(@user)` — output ONLY `[1] Translated message`.
- Never merge or split messages: output [N] must correspond to input [N].
- Match the casual energy, excitement, humor, and punctuation (!, ?, ~) of real livestream chatters.
{jp_slang_guide}- Preserve all Unicode emojis (🥳, 😭, 👏, 🫶🏻) and colon-wrapped custom emote codes (e.g. `:face-blue-smiling:`) intact.
- If an individual message is written in another non-English language (e.g. Korean `안녕하세요~` inside a Japanese chat), translate it to English as well.
- Output English only — never leave untranslated CJK characters in the output.
- Do NOT add translator notes, explanations, or commentary.
- Example:
Input:
[1] (@userA) 待機！
[2] (@userB) 赤スパで殴られてるw :face-blue-smiling:
Output:
[1] Standing by!
[2] Getting pelted with Red Super Chats lol :face-blue-smiling:{context_section}"""

    # Exact-match cache avoids re-translating repeated greetings/reactions across long VODs
    cache: dict[str, str] = {}
    cache_hits = 0
    unresolved = 0
    processed_count = 0
    cursor = 0
    batch_num = 0

    # Map LiveChatMessage.index (1-based) to 0-based position in `messages`
    while cursor < len(translatable):
        batch: list[LiveChatMessage] = []
        while cursor < len(translatable) and len(batch) < batch_size:
            msg = translatable[cursor]
            cursor += 1
            cached = cache.get(msg.text)
            if cached:
                msg.translated_text = cached
                cache_hits += 1
                processed_count += 1
            else:
                batch.append(msg)

        if not batch:
            break

        batch_num += 1
        first_pos = batch[0].index - 1
        last_pos = batch[-1].index - 1

        # Chronological lookback (includes skipped English messages & cached reactions!)
        lookback_lines = []
        for prev in messages[max(0, first_pos - CHAT_LOOKBACK_COUNT):first_pos]:
            if not prev.text:
                continue
            if prev.needs_translation and prev.translated_text:
                lookback_lines.append(f"  • ({prev.author}) {prev.text}  →  {prev.translated_text}")
            else:
                lookback_lines.append(f"  • ({prev.author}) {prev.text}")

        # Chronological lookahead
        lookahead_lines = []
        for nxt in messages[last_pos + 1:min(len(messages), last_pos + 1 + CHAT_LOOKAHEAD_COUNT)]:
            if nxt.text:
                lookahead_lines.append(f"  • ({nxt.author}) {nxt.text}")

        lines_to_translate = [f"[{i}] ({m.author}) {m.text}" for i, m in enumerate(batch, 1)]
        batch_text = "\n".join(lines_to_translate)

        prompt_sections = []
        if lookback_lines:
            prompt_sections.append(
                "### Preceding Chat History (context only — do NOT re-translate):\n"
                + "\n".join(lookback_lines)
            )
        prompt_sections.append(
            f"### Chat Messages to Translate ({len(batch)} messages):\n"
            f"Output EXACTLY {len(batch)} lines numbered [1] to [{len(batch)}] (without sender handles):\n\n"
            + batch_text
        )
        if lookahead_lines:
            prompt_sections.append(
                "### Upcoming Chat Preview (context only — do NOT translate):\n"
                + "\n".join(lookahead_lines)
            )
        user_prompt = "\n\n".join(prompt_sections)

        try:
            response_text = _ollama_chat(
                model,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                {
                    "temperature": 0.2,
                    "num_ctx": 8192,
                    "num_predict": _predict_budget(max(len(batch) * 65, 1000), think, floor_thinking=4096),
                    "repeat_penalty": 1.05,
                },
                think=think,
            )
            translated_dict = _parse_translation_response(response_text, len(batch))
            for i, m in enumerate(batch, 1):
                cand = _clean_chat_translation(translated_dict.get(i, ""), m.author, m.custom_emojis)
                m.translated_text = cand
                if cand and not _needs_retry(m.text, cand):
                    cache[m.text] = cand
        except Exception as e:
            print(f"  ⚠ Chat batch {batch_num} translation error: {e}")
            for m in batch:
                m.translated_text = ""

        # Fill any duplicate lines inside the same batch that succeeded earlier in the batch
        for m in batch:
            if _needs_retry(m.text, m.translated_text) and m.text in cache:
                m.translated_text = cache[m.text]

        missing_indices = [i for i, m in enumerate(batch, 1) if _needs_retry(m.text, m.translated_text)]
        if missing_indices:
            print(f"  ⚠ {len(missing_indices)} chat line(s) missing/untranslated in batch {batch_num}, retrying...")
            retry_lines = [f"[{idx}] ({batch[idx-1].author}) {batch[idx-1].text}" for idx in missing_indices]
            retry_prompt = (
                "### Surrounding chat batch for reference (do NOT output these):\n" + batch_text + "\n\n"
                f"### Translate ONLY these {len(retry_lines)} chat message(s) into natural English.\n"
                f"Output EXACTLY {len(retry_lines)} line(s), keeping each original [N] number (without sender handles):\n\n"
                + "\n".join(retry_lines)
            )
            try:
                retry_text = _ollama_chat(
                    model,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": retry_prompt},
                    ],
                    {
                        "temperature": 0.1,
                        "num_ctx": 8192,
                        "num_predict": _predict_budget(max(len(retry_lines) * 70, 500), think, floor_thinking=3072),
                        "repeat_penalty": 1.05,
                    },
                    think=think,
                )
                retry_dict = _parse_translation_response(retry_text, len(batch), allow_positional=False)
                for idx in missing_indices:
                    m = batch[idx - 1]
                    cand = _clean_chat_translation(retry_dict.get(idx, ""), m.author, m.custom_emojis)
                    if cand and not _needs_retry(m.text, cand):
                        m.translated_text = cand
                        cache[m.text] = cand
            except Exception as e:
                print(f"  ⚠ Chat batch retry error: {e}")

            # Single-line fallback for any remaining stubborn message
            for idx in missing_indices:
                m = batch[idx - 1]
                if not _needs_retry(m.text, m.translated_text):
                    continue
                if m.text in cache:
                    m.translated_text = cache[m.text]
                    continue
                try:
                    single_raw = _ollama_chat(
                        model,
                        [
                            {
                                "role": "system",
                                "content": (
                                    f"Translate this {source_lang} livestream chat message into natural casual English. "
                                    f"Output ONLY the translated message text with no quotes, sender name, or notes.{context_section}"
                                ),
                            },
                            {"role": "user", "content": m.text},
                        ],
                        {
                            "temperature": 0.1,
                            "num_ctx": 8192,
                            "num_predict": _predict_budget(180, think, floor_thinking=2048),
                            "repeat_penalty": 1.05,
                        },
                        think=think,
                    )
                    single_clean = _clean_chat_translation(single_raw.split("\n")[0], m.author, m.custom_emojis)
                    if single_clean and not _needs_retry(m.text, single_clean):
                        m.translated_text = single_clean
                        cache[m.text] = single_clean
                except Exception as e:
                    print(f"  ⚠ Single-message fallback failed for chat #{m.index}: {e}")

                if _needs_retry(m.text, m.translated_text):
                    m.translated_text = m.text
                    unresolved += 1

        processed_count += len(batch)
        pct = processed_count / len(translatable) * 100
        cache_note = f", {cache_hits} cached" if cache_hits else ""
        print(f"  → Progress: {pct:.0f}% ({processed_count}/{len(translatable)} messages{cache_note})")

    if unresolved:
        print(f"  ⚠ {unresolved} chat message(s) could not be translated and were kept in the original language")

    return messages


def write_translated_live_chat_json(
    source_path: Path,
    messages: list[LiveChatMessage],
    output_path: Path,
):
    """Write a translated *.en.live_chat.json file preserving the full YouTube JSONL structure.

    Lines without translated messages (system notices, emoji-only, already-English lines) are
    copied verbatim; translated lines keep badges, avatars, Super Chat metadata, timestamps,
    and custom emoji run objects intact.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_line: dict[int, dict[str, LiveChatMessage]] = {}
    for m in messages:
        if m.needs_translation and m.translated_text and m.translated_text != m.text:
            by_line.setdefault(m.line_no, {})[m.slot_id] = m

    updated_count = 0
    total_lines = 0
    with (
        open(source_path, "r", encoding="utf-8-sig", errors="replace") as src,
        open(output_path, "w", encoding="utf-8") as dst,
    ):
        for line_no, raw_line in enumerate(src, 1):
            if not raw_line.strip():
                continue
            total_lines += 1
            line_targets = by_line.get(line_no)
            if not line_targets:
                dst.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                dst.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                continue

            modified = False
            for slot_id, _, renderer, _ in _iter_chat_renderers(record):
                msg = line_targets.get(slot_id)
                if msg and isinstance(renderer.get("message"), dict):
                    renderer["message"]["runs"] = _rebuild_chat_runs(
                        msg.translated_text, msg.custom_emojis
                    )
                    modified = True
                    updated_count += 1

            if modified:
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                dst.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")

    print(f"  ✓ Written translated live chat JSONL ({updated_count} translated / {total_lines} lines) to {output_path}")


def _cs_to_ass_time(total_cs: int) -> str:
    """Convert integer centiseconds (1/100s) to ASS timestamp format (H:MM:SS.CC)."""
    total_cs = max(0, int(total_cs))
    total_secs, cs = divmod(total_cs, 100)
    hours, rem = divmod(total_secs, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Escape ASS control characters so arbitrary chat usernames/messages never break tags."""
    text = text.replace("\\", "＼").replace("{", "｛").replace("}", "｝")
    return re.sub(r"\s+", " ", text).strip()


def _visual_char_width(ch: str) -> int:
    """Approximate display column width (2 for CJK/fullwidth/emoji, 1 for narrow ASCII)."""
    if ord(ch) >= 0x1F000 or unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    return 1


def _visual_str_width(s: str) -> int:
    return sum(_visual_char_width(ch) for ch in s)


def _wrap_plain_columns(text: str, first_limit: int, next_limit: int, max_lines: int = 3) -> list[str]:
    """Wrap plain text by visual column width, preferring space boundaries and capping line count."""
    if not text:
        return [""]

    lines: list[str] = []
    cur: list[str] = []
    cur_w = 0
    limit = max(8, first_limit)

    idx = 0
    n = len(text)
    while idx < n:
        ch = text[idx]
        w = _visual_char_width(ch)
        if cur_w + w > limit and cur:
            line_str = "".join(cur)
            # Break cleanly at the last space if one exists in the second half of the line
            space_pos = line_str.rfind(" ")
            if space_pos >= len(line_str) // 2 and ch != " ":
                lines.append(line_str[:space_pos].rstrip())
                leftover = line_str[space_pos + 1:].lstrip()
                cur = list(leftover)
                cur_w = _visual_str_width(leftover)
            else:
                lines.append(line_str.rstrip())
                cur = []
                cur_w = 0
            limit = max(8, next_limit)
            if len(lines) == max_lines - 1:
                # Last allowed line: fit what remains and ellipsize if needed
                rem = ("".join(cur) + text[idx:]).strip()
                if _visual_str_width(rem) <= limit:
                    lines.append(rem)
                else:
                    trimmed: list[str] = []
                    tw = 0
                    for rch in rem:
                        rw = _visual_char_width(rch)
                        if tw + rw > limit - 2:
                            break
                        trimmed.append(rch)
                        tw += rw
                    lines.append("".join(trimmed).rstrip() + "…")
                return lines
            if ch == " " and not cur:
                idx += 1
                continue
        cur.append(ch)
        cur_w += w
        idx += 1

    if cur or not lines:
        lines.append("".join(cur).rstrip())
    return lines


def _author_ass_color(author: str, role: str) -> str:
    """Pick an ASS BGR hex color for a username based on badge role or deterministic hash."""
    if role in _ASS_ROLE_COLORS:
        return _ASS_ROLE_COLORS[role]
    idx = zlib.crc32(author.encode("utf-8")) % len(_ASS_AUTHOR_PALETTE)
    return _ASS_AUTHOR_PALETTE[idx]


def _format_ass_chat_lines(msg: LiveChatMessage, max_cols: int = CHAT_WRAP_COLUMNS) -> list[str]:
    """Format a LiveChatMessage into 1–3 ASS-styled lines for the sidebar chat overlay."""
    short_author = _clean_chat_author(msg.author)
    if _visual_str_width(short_author) > 16:
        acc: list[str] = []
        w = 0
        for ch in short_author:
            cw = _visual_char_width(ch)
            if w + cw > 14:
                break
            acc.append(ch)
            w += cw
        short_author = "".join(acc) + "…"

    color = _author_ass_color(msg.author, msg.role)
    role_prefix = ""
    if msg.role == "owner":
        role_prefix = "[★] "
    elif msg.role == "moderator":
        role_prefix = "[Mod] "

    pre_tag = f"({msg.timestamp_text}) " if (msg.offset_ms < 0 and msg.timestamp_text) else ""
    body = (msg.translated_text if (msg.needs_translation and msg.translated_text) else msg.text).strip()

    badge_plain = ""
    badge_ass = ""
    if msg.kind in ("paid", "sticker") and msg.amount_text:
        label = "Sticker" if msg.kind == "sticker" else "SC"
        badge_plain = f"[{label} {msg.amount_text}] "
        badge_ass = f"{{\\c&H{_ASS_SUPERCHAT_COLOR}&\\b1}}{_escape_ass_text(badge_plain)}{{\\b0\\c&HF2F2F2&}}"
    elif msg.kind == "membership":
        mem_label = msg.header_text or "New Member"
        badge_plain = f"[{mem_label}] "
        badge_ass = f"{{\\c&H{_ASS_ROLE_COLORS['member']}&\\b1}}{_escape_ass_text(badge_plain)}{{\\b0\\c&HF2F2F2&}}"

    header_plain = f"{pre_tag}{role_prefix}{short_author}: "
    first_avail = max(12, max_cols - _visual_str_width(header_plain) - _visual_str_width(badge_plain))

    wrapped = _wrap_plain_columns(body, first_limit=first_avail, next_limit=max_cols - 2, max_lines=3)

    pre_ass = f"{{\\c&HA8A8A8&}}{_escape_ass_text(pre_tag)}" if pre_tag else ""
    author_ass = f"{{\\c&H{color}&\\b1}}{_escape_ass_text(role_prefix + short_author)}:{{\\b0\\c&HF2F2F2&}} "

    first_body = _escape_ass_text(wrapped[0]) if wrapped[0] else ""
    out_lines = [f"{pre_ass}{author_ass}{badge_ass}{first_body}".rstrip()]
    for cont in wrapped[1:]:
        out_lines.append("  " + _escape_ass_text(cont))
    return out_lines


def write_chat_ass(
    messages: list[LiveChatMessage],
    output_path: Path,
    dialogue_blocks: list[SubtitleBlock] | None = None,
):
    """Write a styled 1920x1080 ASS subtitle track rendering a scrolling YouTube chat sidebar.

    Uses centisecond-aligned, coalesced time-slice snapshots so rapid chat bursts scroll smoothly
    in strict chronological order without colliding with bottom-center spoken dialogue subtitles.
    When `dialogue_blocks` is also provided, both bottom spoken dialogue and top-right live chat
    are combined into a single self-contained `.ass` file that works in any video player.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay_msgs = [
        m for m in messages
        if m.in_overlay and (m.text or m.amount_text or m.header_text)
    ]
    pre_msgs = [m for m in overlay_msgs if -120_000 <= m.offset_ms < 0][-6:]
    live_msgs = sorted(
        (m for m in overlay_msgs if m.offset_ms >= 0),
        key=lambda m: (m.offset_ms, m.index),
    )

    # Build (start_cs, end_cs, order_key, formatted_lines) items quantized to 10ms centiseconds
    timed_items: list[tuple[int, int, int, list[str]]] = []
    display_cs = max(100, round(CHAT_DISPLAY_MS / 10))

    for m in pre_msgs:
        timed_items.append((0, min(1000, display_cs), m.index, _format_ass_chat_lines(m)))

    for m in live_msgs:
        start_cs = max(0, round(m.offset_ms / 10))
        end_cs = start_cs + display_cs
        timed_items.append((start_cs, end_cs, m.index, _format_ass_chat_lines(m)))

    # Sweep-line across all arrival/expiration timestamps to build exact rolling window snapshots
    boundaries = sorted({cs for s, e, _, _ in timed_items for cs in (s, e)})
    starts_at: dict[int, list[tuple[int, int, list[str]]]] = {}
    ends_at: dict[int, set[int]] = {}
    for s, e, order_key, fmt_lines in timed_items:
        starts_at.setdefault(s, []).append((order_key, e, fmt_lines))
        ends_at.setdefault(e, set()).add(order_key)

    active: list[tuple[int, int, list[str]]] = []  # (order_key, end_cs, fmt_lines)
    slices: list[list] = []  # [start_cs, end_cs, visible_lines_tuple]

    for idx in range(len(boundaries) - 1):
        t0 = boundaries[idx]
        t1 = boundaries[idx + 1]
        if t0 in ends_at:
            expired = ends_at[t0]
            active = [item for item in active if item[0] not in expired]
        if t0 in starts_at:
            active.extend(sorted(starts_at[t0], key=lambda x: x[0]))

        if t1 <= t0 or not active:
            continue

        # Take the newest messages that fit within both the message cap and visual-line cap
        window_msgs = active[-CHAT_MAX_VISIBLE_MESSAGES:]
        kept_blocks: list[list[str]] = []
        total_lines = 0
        for _, _, m_lines in reversed(window_msgs):
            if kept_blocks and total_lines + len(m_lines) > CHAT_MAX_VISIBLE_LINES:
                break
            kept_blocks.append(m_lines)
            total_lines += len(m_lines)
        kept_blocks.reverse()
        visible_lines = tuple(line for blk in kept_blocks for line in blk)
        if not visible_lines:
            continue

        # Coalesce consecutive slices with identical visible lines to prevent flicker
        if slices and slices[-1][1] == t0 and slices[-1][2] == visible_lines:
            slices[-1][1] = t1
        else:
            slices.append([t0, t1, visible_lines])

    panel_x, panel_y, panel_w = 1408, 24, 488
    text_x, text_y = 1422, 34
    line_height_px = 28

    header = """[Script Info]
; Generated by SubTranslate
Title: YouTube Live Chat Overlay
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ChatPanel,Sans,22,&H78101014,&H78101014,&H552C2C34,&H00000000,0,0,0,0,100,100,0,0,1,1,0,7,0,0,0,1
Style: ChatText,Sans,22,&H00F2F2F2,&H00F2F2F2,&H28000000,&H80000000,0,0,0,0,100,100,0,0,1,1.5,0.8,7,0,0,0,1
Style: Dialogue,Sans,50,&H00FFFFFF,&H00FFFFFF,&H00101010,&H70000000,0,0,0,0,100,100,0,0,1,2.6,1.4,2,60,60,44,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    dialogue_event_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for s_cs, e_cs, vis_lines in slices:
            s_str = _cs_to_ass_time(s_cs)
            e_str = _cs_to_ass_time(e_cs)
            panel_h = len(vis_lines) * line_height_px + 20
            f.write(
                f"Dialogue: 0,{s_str},{e_str},ChatPanel,,0,0,0,,"
                f"{{\\an7\\pos({panel_x},{panel_y})\\p1}}m 0 0 l {panel_w} 0 {panel_w} {panel_h} 0 {panel_h}{{\\p0}}\n"
            )
            joined = "\\N".join(vis_lines)
            f.write(
                f"Dialogue: 1,{s_str},{e_str},ChatText,,0,0,0,,"
                f"{{\\an7\\pos({text_x},{text_y})}}{joined}\n"
            )

        if dialogue_blocks:
            for blk in dialogue_blocks:
                raw_txt = blk.translated_text if blk.translated_text else blk.text
                raw_txt = raw_txt.strip()
                if not raw_txt:
                    continue
                s_cs = round(_srt_to_ms(blk.start_time) / 10)
                e_cs = round(_srt_to_ms(blk.end_time) / 10)
                if e_cs <= s_cs:
                    continue
                ass_lines = "\\N".join(_escape_ass_text(ln) for ln in raw_txt.splitlines() if ln.strip())
                if not ass_lines:
                    continue
                f.write(
                    f"Dialogue: 2,{_cs_to_ass_time(s_cs)},{_cs_to_ass_time(e_cs)},Dialogue,,0,0,0,,{ass_lines}\n"
                )
                dialogue_event_count += 1

    if dialogue_blocks and dialogue_event_count:
        print(
            f"  ✓ Written combined spoken subs ({dialogue_event_count} cues) + chat overlay "
            f"({len(slices)} windows) to {output_path}"
        )
    else:
        print(f"  ✓ Written synchronized ASS chat overlay ({len(slices)} windows) to {output_path}")


# ── Phase 1: Extract & Transcribe ───────────────────────────────────────────

@dataclass
class PendingFile:
    """A file that has been transcribed/parsed and is waiting for translation."""
    source_path: Path
    output_srt: Path
    original_srt: Path | None
    blocks: list[SubtitleBlock]
    source_lang: str


@dataclass
class PendingChatFile:
    """A YouTube live_chat.json file parsed and queued for translation."""
    source_path: Path
    base_stem: str
    output_json: Path
    output_ass: Path
    messages: list[LiveChatMessage]
    source_lang: str


def _whisper_fallback_chain(whisper_model: str) -> list[str]:
    """Requested model followed by smaller models to try if VRAM runs out."""
    order = ["large-v3", "large-v2", "large-v1", "large", "large-v3-turbo", "turbo", "medium", "small", "base", "tiny"]
    chain = [whisper_model]
    if whisper_model in order:
        pos = order.index(whisper_model)
        chain += [m for m in ("medium", "small") if order.index(m) > pos]
    return chain


def _extract_embedded_track(media_path: Path, track: dict) -> list[SubtitleBlock]:
    """Extract and parse an embedded subtitle track into blocks (empty list on failure)."""
    with tempfile.TemporaryDirectory(prefix="subtranslate_") as tmp_dir:
        sub_path = extract_embedded_subtitles(media_path, track, Path(tmp_dir))
        if not sub_path:
            return []
        try:
            return parse_subtitle_file(sub_path)
        except Exception as e:
            print(f"  ⚠ Could not parse extracted subtitles: {e}")
            return []


def phase1_extract_and_transcribe(
    media_files: list[Path],
    subtitle_files: list[Path],
    chat_files: list[Path],
    output_dir: Path,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    ignore_subs: bool = False,
    language: str | None = None,
) -> tuple[list[PendingFile], list[PendingChatFile]]:
    """Phase 1: Extract embedded subs, transcribe audio, and parse chat files.
    Whisper is loaded once and reused across all media files that need it."""

    pending: list[PendingFile] = []
    pending_chat: list[PendingChatFile] = []
    needs_whisper: list[tuple[Path, Path, Path]] = []  # (media_path, wav_path, output_srt)

    # ── Process live chat logs (fast JSONL stream parse, no GPU/Whisper needed) ──
    for chat_path in chat_files:
        base_stem = _chat_base_stem(chat_path)
        output_json = output_dir / f"{base_stem}.en.live_chat.json"
        output_ass = output_dir / f"{base_stem}.chat.en.ass"

        print(f"\n{'━' * 60}")
        print(f"  Parsing YouTube Live Chat: {chat_path.name}")
        print(f"{'━' * 60}")

        try:
            messages, source_lang = parse_yt_live_chat(chat_path, language_override=language)
            print(f"  ✓ Parsed {len(messages)} chat item(s) (detected language: {source_lang})")
            needs_tr = sum(1 for m in messages if m.needs_translation)
            print(f"  → {needs_tr} message(s) need translation ({len(messages) - needs_tr} emoji/English kept as-is)")
            pending_chat.append(PendingChatFile(
                source_path=chat_path,
                base_stem=base_stem,
                output_json=output_json,
                output_ass=output_ass,
                messages=messages,
                source_lang=source_lang,
            ))
        except Exception as e:
            print(f"  ✗ Error parsing {chat_path.name}: {e}")

    # ── Process media files: check for embedded subs first ──
    for media_path in media_files:
        stem = media_path.stem
        output_srt = output_dir / f"{stem}.srt"
        original_srt = output_dir / f"{stem}.original.srt"

        print(f"\n{'━' * 60}")
        print(f"  Processing: {media_path.name}")
        print(f"{'━' * 60}")

        # Check for embedded subtitle tracks (skip if --ignore-subs)
        sub_tracks = []
        if ignore_subs:
            print("\n  ── Ignoring embedded subtitles (--ignore-subs) ──")
        else:
            print("\n  ── Checking for embedded subtitles ──")
            sub_tracks = detect_embedded_subtitles(media_path)

        extracted = False
        if sub_tracks:
            track_info = ", ".join(
                f"#{t['index']} [{t['language']}] {t['codec']}"
                + (f" ({t['title']})" if t['title'] else "")
                + (" [forced]" if t.get("forced") else "")
                for t in sub_tracks
            )
            print(f"  ✓ Found {len(sub_tracks)} subtitle track(s): {track_info}")

            best_track = _pick_best_subtitle_track(sub_tracks)
            eng_track = _pick_english_track(sub_tracks)

            if best_track:
                print(f"  → Using track #{best_track['index']} [{best_track['language']}] for translation")
                blocks = _extract_embedded_track(media_path, best_track)
                if blocks:
                    print(f"  ✓ Extracted {len(blocks)} subtitle blocks")
                    pending.append(PendingFile(
                        source_path=media_path,
                        output_srt=output_srt,
                        original_srt=original_srt,
                        blocks=blocks,
                        source_lang=_resolve_source_lang(best_track["language"], blocks, language),
                    ))
                    extracted = True
                else:
                    print("  ⚠ Could not extract/parse subtitle track, will use Whisper")
            elif eng_track:
                # English subs — no translation needed
                print(f"  → Subtitles are already in English (track #{eng_track['index']}) — extracting as-is")
                blocks = _extract_embedded_track(media_path, eng_track)
                if blocks:
                    for b in blocks:
                        b.translated_text = b.text
                    write_srt(blocks, output_srt)
                    extracted = True  # done, no translation needed
                else:
                    print("  ⚠ Could not extract English subtitle track, will use Whisper")
            else:
                print("  → Only image-based subtitles (e.g. PGS) found — these can't be translated, will use Whisper")

        if not extracted:
            if not sub_tracks and not ignore_subs:
                print("  → No embedded subtitles found, queuing for Whisper")
            # Queue for Whisper transcription
            with tempfile.NamedTemporaryFile(suffix=".wav", prefix="subtranslate_", delete=False) as tmp:
                wav_path = Path(tmp.name)

            if extract_audio(media_path, wav_path):
                needs_whisper.append((media_path, wav_path, output_srt))
            else:
                print("  ✗ Failed to extract audio, skipping")
                wav_path.unlink(missing_ok=True)

    # ── Batch Whisper transcription (load model once) ──
    if needs_whisper:
        try:
            pending.extend(_run_whisper(needs_whisper, output_dir, whisper_model, language))
        finally:
            # Always clean up temp audio, even on Ctrl+C or model load failure
            for _, wav_path, _ in needs_whisper:
                wav_path.unlink(missing_ok=True)

    # ── Parse subtitle files (no Whisper needed) ──
    for sub_path in subtitle_files:
        stem = sub_path.stem
        output_srt = output_dir / f"{stem}.en.srt"

        print(f"\n{'━' * 60}")
        print(f"  Parsing: {sub_path.name}")
        print(f"{'━' * 60}")

        try:
            blocks = parse_subtitle_file(sub_path)
            if blocks:
                print(f"  ✓ Parsed {len(blocks)} valid subtitle blocks")

                # Detect source language from file name suffix (e.g. video.ja.srt) or the text itself
                suffix_match = re.search(r"\.([a-z]{2,3})$", stem.lower())
                code = suffix_match.group(1) if suffix_match else None
                source_lang = _resolve_source_lang(code, blocks, language)
                print(f"  → Source language: {source_lang}")

                pending.append(PendingFile(
                    source_path=sub_path,
                    output_srt=output_srt,
                    original_srt=None,
                    blocks=blocks,
                    source_lang=source_lang,
                ))
            else:
                print(f"  ⚠ No valid subtitles found in {sub_path.name}")
        except Exception as e:
            print(f"  ✗ Error parsing {sub_path.name}: {e}")

    return pending, pending_chat


def _run_whisper(
    needs_whisper: list[tuple[Path, Path, Path]],
    output_dir: Path,
    whisper_model: str,
    language: str | None,
) -> list[PendingFile]:
    """Load Whisper once and transcribe all queued audio files."""
    pending: list[PendingFile] = []

    print(f"\n{'━' * 60}")
    print(f"  ── Whisper Transcription Phase ({len(needs_whisper)} file(s)) ──")
    print(f"{'━' * 60}")

    import whisper
    import torch

    if whisper_model not in whisper.available_models() and not Path(whisper_model).is_file():
        print(f"  ✗ Unknown Whisper model '{whisper_model}'. Available: {', '.join(whisper.available_models())}")
        return pending

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  ✓ Using GPU: {torch.cuda.get_device_name(0)}")
        _unload_ollama_models()
        torch.cuda.empty_cache()
    else:
        print("  ⚠ Using CPU (this will be slow)")

    # Load model once with fallback
    model = None
    used_model = whisper_model
    for try_model in _whisper_fallback_chain(whisper_model):
        try:
            print(f"  → Loading Whisper model '{try_model}'...")
            model = whisper.load_model(try_model, device=device)
            used_model = try_model
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "HIP" in str(e):
                print(f"  ⚠ Not enough VRAM for '{try_model}', trying smaller model...")
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            raise

    if model is None:
        print("  ⚠ All GPU models failed, falling back to CPU...")
        device = "cpu"
        model = whisper.load_model(whisper_model, device="cpu")
        used_model = whisper_model

    if used_model != whisper_model:
        print(f"  ⚠ Using '{used_model}' instead of '{whisper_model}' due to VRAM limits")

    # Whisper expects ISO codes ("ja"); accept full names too ("japanese")
    whisper_lang = language.lower() if language else None

    # Transcribe all files with the loaded model
    for media_path, wav_path, output_srt in needs_whisper:
        stem = media_path.stem
        original_srt = output_dir / f"{stem}.original.srt"

        print(f"\n  → Transcribing: {media_path.name}")
        try:
            start_time = time.time()
            result = model.transcribe(
                str(wav_path),
                task="transcribe",
                language=whisper_lang,
                verbose=False,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                fp16=(device == "cuda"),
            )
            elapsed = time.time() - start_time
            detected_lang = result.get("language", "unknown")
            print(f"  ✓ Transcribed in {elapsed:.0f}s (language: {detected_lang})")

            cues = []
            dropped = 0
            for seg in result["segments"]:
                text = _collapse_repetitions(seg["text"].strip())
                if not text:
                    continue
                if _is_whisper_hallucination(text):
                    dropped += 1
                    continue
                cues.append((round(seg["start"] * 1000), round(seg["end"] * 1000), text))
            if dropped:
                print(f"  → Removed {dropped} likely hallucinated segment(s)")

            blocks = _build_blocks(cues)

            if blocks:
                print(f"  ✓ Generated {len(blocks)} subtitle blocks")
                pending.append(PendingFile(
                    source_path=media_path,
                    output_srt=output_srt,
                    original_srt=original_srt,
                    blocks=blocks,
                    source_lang=_resolve_source_lang(detected_lang, blocks, language),
                ))
            else:
                print(f"  ⚠ No speech detected in {media_path.name}")

        except Exception as e:
            print(f"  ✗ Transcription error for {media_path.name}: {e}")
        finally:
            wav_path.unlink(missing_ok=True)

    # Free Whisper from GPU
    print("\n  → Unloading Whisper model...")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print("  ✓ GPU memory freed")

    return pending


# ── Phase 2: Translate ──────────────────────────────────────────────────────

def phase2_translate_and_write(
    pending: list[PendingFile],
    pending_chat: list[PendingChatFile] | None = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    batch_size: int = TRANSLATION_BATCH_SIZE,
    chat_batch_size: int = CHAT_TRANSLATION_BATCH_SIZE,
    think: str = "auto",
):
    """Phase 2: Translate all pending subtitle blocks and live chat logs using Ollama.
    The model stays loaded across all files."""

    pending = pending or []
    pending_chat = pending_chat or []

    if not pending and not pending_chat:
        return 0, 0

    total_items = len(pending) + len(pending_chat)
    print(f"\n{'━' * 60}")
    print(f"  ── Translation Phase ({total_items} item(s): {len(pending)} subtitle/media, {len(pending_chat)} live chat) ──")
    print(f"{'━' * 60}")

    # Save original-language subtitles first so a translation failure never loses the transcription
    for pf in pending:
        if pf.original_srt:
            write_srt(pf.blocks, pf.original_srt, use_translated=False)

    if not check_ollama(ollama_model):
        return 0, total_items

    success = 0
    failed = 0

    # ── Translate media / subtitle dialogue ──
    for pf in pending:
        print(f"\n  → Translating dialogue: {pf.source_path.name}")

        # Skip if already translated (e.g., English embedded subs)
        if all(b.translated_text for b in pf.blocks):
            write_srt(pf.blocks, pf.output_srt)
            success += 1
            continue

        try:
            context = analyze_context(pf.blocks, source_path=pf.source_path, model=ollama_model, think=think)
            translate_blocks(pf.blocks, context, ollama_model, pf.source_lang, batch_size, think=think)

            # Write translated output
            write_srt(pf.blocks, pf.output_srt)
            success += 1
        except Exception as e:
            print(f"  ✗ Translation error for {pf.source_path.name}: {e}")
            failed += 1

    # ── Translate YouTube live chat logs ──
    for pcf in pending_chat:
        print(f"\n  → Translating live chat: {pcf.source_path.name}")

        try:
            chat_context = analyze_chat_context(
                pcf.messages,
                source_path=pcf.source_path,
                model=ollama_model,
                think=think,
            )
            translate_chat_messages(
                pcf.messages,
                context=chat_context,
                model=ollama_model,
                source_lang=pcf.source_lang,
                batch_size=chat_batch_size,
                think=think,
            )

            # 1. Write translated JSONL replay file (*.en.live_chat.json)
            write_translated_live_chat_json(pcf.source_path, pcf.messages, pcf.output_json)

            # 2. Check if there are matching dialogue blocks to combine into the ASS subtitle track
            matching_pf = next(
                (pf for pf in pending if _chat_base_stem(pf.source_path) == pcf.base_stem or pf.source_path.stem == pcf.base_stem),
                None,
            )
            dialogue_blocks = matching_pf.blocks if (matching_pf and matching_pf.blocks) else None

            # 3. Write ready-to-play ASS sidebar chat overlay (*.chat.en.ass)
            write_chat_ass(pcf.messages, pcf.output_ass, dialogue_blocks=dialogue_blocks)
            success += 1
        except Exception as e:
            print(f"  ✗ Translation error for {pcf.source_path.name}: {e}")
            failed += 1

    return success, failed


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SubTranslate — Local Anime & Stream VOD Subtitle/Chat Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python subtranslate.py                             Process all files in INPUT/
  python subtranslate.py --file INPUT/episode.mkv    Process a specific anime episode
  python subtranslate.py --file INPUT/vod.live_chat.json  Translate a stream VOD chat log
  python subtranslate.py --whisper-model medium       Use smaller/faster Whisper model
  python subtranslate.py --ollama-model gemma4:31b   Use high-accuracy local LLM
  python subtranslate.py --think off                 Disable reasoning tokens on thinking models
        """,
    )
    parser.add_argument("--file", "-f", type=Path, help="Process a specific file (or directory) instead of INPUT/")
    parser.add_argument("--output-dir", "-o", type=Path, default=OUTPUT_DIR,
                        help="Directory for generated subtitles (default: OUTPUT/)")
    parser.add_argument("--whisper-model", "-w", default=DEFAULT_WHISPER_MODEL,
                        help=f"Whisper model to use (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--ollama-model", "-m", default=DEFAULT_OLLAMA_MODEL,
                        help=f"Ollama model for translation (default: {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--batch-size", "-b", type=int, default=TRANSLATION_BATCH_SIZE,
                        help=f"Subtitle dialogue lines per translation request (default: {TRANSLATION_BATCH_SIZE})")
    parser.add_argument("--chat-batch-size", type=int, default=CHAT_TRANSLATION_BATCH_SIZE,
                        help=f"Live chat lines per translation request (default: {CHAT_TRANSLATION_BATCH_SIZE})")
    parser.add_argument("--think", choices=THINK_MODES, default="auto",
                        help="Thinking/reasoning mode for reasoning LLMs (auto, off, on, low, medium, high; default: auto)")
    parser.add_argument("--transcribe-only", action="store_true",
                        help="Only transcribe media / parse chat into ASS overlays, skip LLM translation")
    parser.add_argument("--translate-only", action="store_true",
                        help="Only translate existing subtitle or live chat files (media files are skipped)")
    parser.add_argument("--ignore-subs", "-i", action="store_true",
                        help="Ignore embedded subtitle tracks and force Whisper transcription")
    parser.add_argument("--language", "-l", default=None,
                        help="Force the source language (e.g. ja, ko, zh) instead of auto-detect")
    parser.add_argument("--version", action="version", version=f"SubTranslate {__version__}")

    args = parser.parse_args()

    if args.transcribe_only and args.translate_only:
        parser.error("--transcribe-only and --translate-only are mutually exclusive")

    # Banner
    print("╔══════════════════════════════════════════════════════╗")
    print("║           SubTranslate                               ║")
    print("║           Local Anime & Stream Subtitle Generator    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Check external tools up front instead of failing mid-run
    missing_tools = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing_tools:
        print(f"✗ Required tool(s) not found in PATH: {', '.join(missing_tools)}")
        print("  Install FFmpeg (e.g. sudo apt install ffmpeg / sudo pacman -S ffmpeg)")
        sys.exit(1)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather files to process
    if args.file:
        if not args.file.exists():
            print(f"✗ File not found: {args.file}")
            sys.exit(1)
        files = sorted(args.file.iterdir()) if args.file.is_dir() else [args.file]
    else:
        if not INPUT_DIR.exists():
            print("✗ INPUT/ directory not found. Create it and add media or chat files.")
            sys.exit(1)
        files = sorted(INPUT_DIR.iterdir())

    # Classify files
    chat_files = [f for f in files if is_live_chat_file(f)]
    media_files = [
        f for f in files
        if f.suffix.lower() in MEDIA_EXTENSIONS and f.is_file() and not is_live_chat_file(f)
    ]
    subtitle_files = [
        f for f in files
        if f.suffix.lower() in SUBTITLE_EXTENSIONS and f.is_file() and not is_live_chat_file(f)
    ]

    ignored = [
        f.name for f in files
        if f not in media_files and f not in subtitle_files and f not in chat_files
        and not f.name.startswith(".") and not f.name.endswith((".context.txt",))
        and f.name not in ("context.txt", "glossary.txt", "glossary.json")
    ]
    if ignored:
        print(f"⚠ Skipping {len(ignored)} unsupported file(s): {', '.join(ignored[:5])}")

    if args.translate_only and media_files:
        print(f"→ --translate-only: skipping {len(media_files)} media file(s)")
        media_files = []
    if args.transcribe_only and subtitle_files:
        print(f"→ --transcribe-only: skipping {len(subtitle_files)} subtitle file(s) (nothing to transcribe)")
        subtitle_files = []

    if not media_files and not subtitle_files and not chat_files:
        print("✗ No supported media, subtitle, or live chat files found to process.")
        sys.exit(1)

    print(f"→ Found {len(media_files)} media, {len(subtitle_files)} subtitle, and {len(chat_files)} live chat file(s)")

    # ── Phase 1: Extract, Transcribe, & Parse (Whisper loaded once) ──
    pending, pending_chat = phase1_extract_and_transcribe(
        media_files, subtitle_files, chat_files, output_dir, args.whisper_model,
        ignore_subs=args.ignore_subs,
        language=args.language,
    )

    if args.transcribe_only:
        # Write original transcriptions and raw-language chat overlays
        for pf in pending:
            out = pf.original_srt or pf.output_srt
            write_srt(pf.blocks, out, use_translated=False)
        for pcf in pending_chat:
            write_chat_ass(pcf.messages, pcf.output_ass)
        total_done = len(pending) + len(pending_chat)
        print(f"\n{'━' * 60}")
        print(f"  Done! {total_done} item(s) transcribed/parsed (translation skipped)")
        print(f"  Output directory: {output_dir.resolve()}")
        print(f"{'━' * 60}")
        return

    # ── Phase 2: Translate (Ollama loaded once) ──
    success, failed = phase2_translate_and_write(
        pending,
        pending_chat,
        ollama_model=args.ollama_model,
        batch_size=args.batch_size,
        chat_batch_size=args.chat_batch_size,
        think=args.think,
    )

    # Summary
    print(f"\n{'━' * 60}")
    print(f"  Done! {success} succeeded, {failed} failed")
    print(f"  Output directory: {output_dir.resolve()}")
    print(f"{'━' * 60}")
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        sys.exit(130)
