#!/usr/bin/env python3
"""
SubTranslate — Local Anime Subtitle Generator

Transcribes audio from media files using Whisper and translates
subtitles to English using a local LLM via Ollama.

Usage:
    python subtranslate.py                          # Process all files in INPUT/
    python subtranslate.py --file INPUT/ep01.mkv    # Process specific file
    python subtranslate.py --whisper-model medium    # Use smaller Whisper model
    python subtranslate.py --ollama-model llama3.2   # Use different Ollama model
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

__version__ = "1.1.0"


# ── Configuration ───────────────────────────────────────────────────────────

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".m4v", ".ts", ".wmv"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt"}

# Resolve INPUT/OUTPUT relative to the script so it works from any working directory
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "INPUT"
OUTPUT_DIR = SCRIPT_DIR / "OUTPUT"

DEFAULT_WHISPER_MODEL = "large-v3"
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"
TRANSLATION_BATCH_SIZE = 20  # subtitle blocks per LLM call
LOOKBACK_COUNT = 5           # preceding translated lines for context continuity
LOOKAHEAD_COUNT = 3          # upcoming untranslated lines for sentence flow
MIN_CUE_DURATION_MS = 100    # drop micro-duration artifact cues (e.g. YouTube autogen flashes)
OLLAMA_RETRIES = 2           # extra attempts when an Ollama request fails (timeouts, reloads)

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


def _detect_language_from_blocks(blocks: list[SubtitleBlock]) -> str | None:
    """Detect CJK languages from script usage. Distinguishes Chinese from Japanese by kana."""
    text = "".join(b.text for b in blocks)
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

def _clean_title_for_context(path: Path) -> str:
    """Clean a media/subtitle filename to extract a human-readable show/episode title."""
    name = path.stem
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
    sections = []

    if source_path:
        per_file = source_path.parent / f"{source_path.stem}.context.txt"
        if per_file.is_file():
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
        if c.is_file():
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


def _ollama_chat(model: str, messages: list[dict], options: dict, retries: int = OLLAMA_RETRIES) -> str:
    """Call Ollama chat with retry/backoff. Strips <think> blocks from reasoning models."""
    import ollama
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = ollama.chat(model=model, messages=messages, options=options)
            content = response["message"]["content"] or ""
            return _THINK_RE.sub("", content).strip()
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
            {"temperature": 0.2, "num_ctx": 8192, "num_predict": 800, "repeat_penalty": 1.1},
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
                    "num_predict": max(len(batch) * 80, 1000),
                    "repeat_penalty": 1.05,  # high values push the model away from repeating names/terms
                },
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
                        "num_predict": max(len(retry_lines) * 80, 500),
                        "repeat_penalty": 1.05,
                    },
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
                        {"temperature": 0.1, "num_ctx": 4096, "num_predict": 150, "repeat_penalty": 1.05},
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


# ── Phase 1: Extract & Transcribe ───────────────────────────────────────────

@dataclass
class PendingFile:
    """A file that has been transcribed/parsed and is waiting for translation."""
    source_path: Path
    output_srt: Path
    original_srt: Path | None
    blocks: list[SubtitleBlock]
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
    output_dir: Path,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    ignore_subs: bool = False,
    language: str | None = None,
) -> list[PendingFile]:
    """Phase 1: Extract embedded subs or transcribe audio for all files.
    Whisper is loaded once and reused across all media files that need it."""

    pending: list[PendingFile] = []
    needs_whisper: list[tuple[Path, Path, Path]] = []  # (media_path, wav_path, output_srt)

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

    return pending


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
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    batch_size: int = TRANSLATION_BATCH_SIZE,
):
    """Phase 2: Translate all pending subtitle blocks using Ollama.
    The model stays loaded across all files."""

    if not pending:
        return 0, 0

    print(f"\n{'━' * 60}")
    print(f"  ── Translation Phase ({len(pending)} file(s)) ──")
    print(f"{'━' * 60}")

    # Save original-language subtitles first so a translation failure never loses the transcription
    for pf in pending:
        if pf.original_srt:
            write_srt(pf.blocks, pf.original_srt, use_translated=False)

    if not check_ollama(ollama_model):
        return 0, len(pending)

    success = 0
    failed = 0

    for pf in pending:
        print(f"\n  → Translating: {pf.source_path.name}")

        # Skip if already translated (e.g., English embedded subs)
        if all(b.translated_text for b in pf.blocks):
            write_srt(pf.blocks, pf.output_srt)
            success += 1
            continue

        try:
            context = analyze_context(pf.blocks, source_path=pf.source_path, model=ollama_model)
            translate_blocks(pf.blocks, context, ollama_model, pf.source_lang, batch_size)

            # Write translated output
            write_srt(pf.blocks, pf.output_srt)
            success += 1
        except Exception as e:
            print(f"  ✗ Translation error for {pf.source_path.name}: {e}")
            failed += 1

    return success, failed


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SubTranslate — Generate English subtitles from media files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python subtranslate.py                             Process all files in INPUT/
  python subtranslate.py --file INPUT/episode.mkv    Process a specific file
  python subtranslate.py --whisper-model medium       Use smaller/faster model
  python subtranslate.py --ollama-model llama3.2      Use different LLM
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
                        help=f"Subtitle lines per translation request (default: {TRANSLATION_BATCH_SIZE})")
    parser.add_argument("--transcribe-only", action="store_true",
                        help="Only transcribe, skip translation")
    parser.add_argument("--translate-only", action="store_true",
                        help="Only translate existing subtitle files (media files are skipped)")
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
    print("║           Local Anime Subtitle Generator             ║")
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
            print("✗ INPUT/ directory not found. Create it and add media files.")
            sys.exit(1)
        files = sorted(INPUT_DIR.iterdir())

    # Classify files
    media_files = [f for f in files if f.suffix.lower() in MEDIA_EXTENSIONS and f.is_file()]
    subtitle_files = [f for f in files if f.suffix.lower() in SUBTITLE_EXTENSIONS and f.is_file()]

    ignored = [
        f.name for f in files
        if f not in media_files and f not in subtitle_files
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

    if not media_files and not subtitle_files:
        print("✗ No supported files found to process.")
        sys.exit(1)

    print(f"→ Found {len(media_files)} media file(s) and {len(subtitle_files)} subtitle file(s)")

    # ── Phase 1: Extract & Transcribe (Whisper loaded once) ──
    pending = phase1_extract_and_transcribe(
        media_files, subtitle_files, output_dir, args.whisper_model,
        ignore_subs=args.ignore_subs,
        language=args.language,
    )

    if args.transcribe_only:
        # Just write the original transcriptions
        for pf in pending:
            out = pf.original_srt or pf.output_srt
            write_srt(pf.blocks, out, use_translated=False)
        print(f"\n{'━' * 60}")
        print(f"  Done! {len(pending)} file(s) transcribed (translation skipped)")
        print(f"  Output directory: {output_dir.resolve()}")
        print(f"{'━' * 60}")
        return

    # ── Phase 2: Translate (Ollama loaded once) ──
    success, failed = phase2_translate_and_write(pending, args.ollama_model, args.batch_size)

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
