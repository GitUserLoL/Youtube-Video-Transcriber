#!/usr/bin/env python3
"""
YouTube -> Plain-Text Transcript Utility
======================================

Paste a YouTube video URL (or ID) and this script will try, in order:

1. **Official/creator-provided transcript** via `youtube-transcript-api`.
2. **Auto-generated subtitles** downloaded via `yt-dlp` (WebVTT / SRT parsed & cleaned).
3. **Speech-to-text (STT)** transcription from downloaded audio using Whisper / Faster-Whisper
   (local models) *or* OpenAI ASR API (if `--openai-api-key` supplied & model selected).

All paths lead to a **clean, no-timestamps, plain-text transcript** written to an output file.

---
### Quick Start
```bash
pip install yt-dlp youtube-transcript-api openai-whisper faster-whisper webvtt-py pysrt tqdm requests
# (ffmpeg must be installed & in PATH for yt-dlp + Whisper.)

python youtube_transcriber.py https://www.youtube.com/watch?v=dQw4w9WgXcQ -o rickroll.txt
```

### Minimal Use (auto output name in current dir):
```bash
python youtube_transcriber.py <youtube_url>
```

### Force speech-to-text even if transcripts exist:
```bash
python youtube_transcriber.py <youtube_url> --force-stt
```

### Use Faster-Whisper (recommended for speed/low memory):
```bash
python youtube_transcriber.py <youtube_url> --stt-backend faster-whisper --whisper-model medium
```

### Use OpenAI API Whisper-large-v3 (or other ASR-capable model):
```bash
python youtube_transcriber.py <youtube_url> --stt-backend openai --openai-api-key $OPENAI_API_KEY --openai-model whisper-1
```

---
### Features
- Language preference list (try English variants first, etc.).
- Optional text normalization: strip speaker tags, inline timestamps, [Music], etc.
- Merge short caption fragments into paragraphs with configurable max gap.
- Safe filename generation from video title or ID.
- Progress bars.

---
### Exit Codes
0 success | 1 bad args | 2 download error | 3 transcript error | 4 stt error.

---
### License
MIT. Use responsibly. Heavy usage may violate YouTube TOS; ensure you have rights to download/transcribe.
"""

import argparse
import os
import re
import sys
import json
import tempfile
import subprocess
import shutil
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

# Third-party imports (all optional; script degrades gracefully)
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
except ImportError:  # pragma: no cover - optional dependency
    YouTubeTranscriptApi = None  # type: ignore
    TranscriptsDisabled = NoTranscriptFound = CouldNotRetrieveTranscript = Exception  # fallback

try:
    import webvtt
except ImportError:  # pragma: no cover
    webvtt = None

try:
    import pysrt
except ImportError:  # pragma: no cover
    pysrt = None

# STT backends
# openai-whisper (a.k.a whisper)
try:
    import whisper
except ImportError:  # pragma: no cover
    whisper = None

# faster-whisper
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except ImportError:  # pragma: no cover
    FasterWhisperModel = None

# OpenAI API (for remote ASR)
try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from tqdm import tqdm

########################################################################################
# Utility helpers
########################################################################################

YOUTUBE_ID_REGEX = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?.*?v=|embed/|shorts/|v/))([A-Za-z0-9_-]{11})"
)


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract an 11-char YouTube video ID from a URL or return the string if it looks like one."""
    candidate = url_or_id.strip()
    if len(candidate) == 11 and re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        return candidate
    m = YOUTUBE_ID_REGEX.search(candidate)
    if m:
        return m.group(1)
    return None


########################################################################################
# Transcript retrieval via youtube-transcript-api
########################################################################################

# NOTE ABOUT RECENT youtube-transcript-api CHANGES
# ------------------------------------------------
# Newer versions of `youtube-transcript-api` (≥0.6.x) sometimes return *objects*
# such as `FetchedTranscriptSnippet` instead of plain dicts with `.get()` access.
# The original implementation here assumed dicts and crashed with:
#   "'FetchedTranscriptSnippet' object has no attribute 'get'"
# This updated implementation is defensive: it gracefully extracts text from
# either mapping-style dicts *or* objects exposing a `.text` attribute.
# It also prefers manually-created transcripts before generated ones, and avoids
# touching private attributes like `transcript_list._langs` (which may break
# across versions).

from collections.abc import Mapping


def _coerce_segment_text(seg, preserve_casing: bool = False) -> str:
    """Return clean text from a transcript *segment* that may be a dict or object."""
    if isinstance(seg, Mapping):
        txt = seg.get("text", "")
    else:
        txt = getattr(seg, "text", "")
    txt = txt.strip()
    if not preserve_casing:
        txt = normalize_caption_text(txt)
    return txt


def _fetch_to_text(transcript_obj, preserve_casing: bool = False, join_delim: str = " ") -> Optional[str]:
    """Fetch a transcript object and return merged plain text (or None)."""
    try:
        data = transcript_obj.fetch()
    except Exception:
        return None
    if not data:
        return None
    parts = []
    for seg in data:
        parts.append(_coerce_segment_text(seg, preserve_casing=preserve_casing))
    return join_delim.join([p for p in parts if p]) or None


def try_fetch_transcript(
    video_id: str,
    lang_priority: Optional[List[str]] = None,
    preserve_casing: bool = False,
    join_delim: str = " ",
) -> Tuple[Optional[str], Optional[str]]:
    """Attempt to fetch a transcript via the YouTubeTranscriptApi.

    Returns (text, language_code) or (None, None) if failed/unavailable.
    The function is robust to API version differences and will try, in order:
      1. User-specified language priorities (manual transcripts preferred).
      2. English fallbacks.
      3. Any available generated transcript.
    """
    if YouTubeTranscriptApi is None:
        return None, None

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    except Exception:
        return None, None

    # Build ordered language preference list (dedupe preserving order).
    pref = []
    for lang in (lang_priority or []) + ["en", "en-US", "en-GB"]:
        if lang not in pref:
            pref.append(lang)

    # Helper to attempt manual transcripts for a given language code.
    def _try_manual(lang_code: str):
        try:
            t = transcript_list.find_manually_created_transcript([lang_code])
        except Exception:
            return None
        txt = _fetch_to_text(t, preserve_casing=preserve_casing, join_delim=join_delim)
        if txt:
            return txt, getattr(t, "language_code", lang_code)
        return None

    # Helper to attempt generated transcripts.
    def _try_generated(lang_code: str):
        try:
            t = transcript_list.find_generated_transcript([lang_code])
        except Exception:
            return None
        txt = _fetch_to_text(t, preserve_casing=preserve_casing, join_delim=join_delim)
        if txt:
            return txt, getattr(t, "language_code", lang_code)
        return None

    # 1) Try preferred languages (manual first).
    for lg in pref:
        got = _try_manual(lg)
        if got:
            return got

    # 2) Try preferred languages (generated).
    for lg in pref:
        got = _try_generated(lg)
        if got:
            return got

    # 3) As a last resort, try *any* remaining transcript (manual then generated).
    # The TranscriptList is iterable.
    for t in transcript_list:
        txt = _fetch_to_text(t, preserve_casing=preserve_casing, join_delim=join_delim)
        if txt:
            return txt, getattr(t, "language_code", None)

    return None, None


########################################################################################
# Subtitle parsing (WebVTT / SRT) when direct transcript unavailable
######################################################################################## (WebVTT / SRT) when direct transcript unavailable
########################################################################################

def parse_vtt_to_text(path: Path, preserve_casing: bool = False) -> str:
    if webvtt is None:
        raise RuntimeError("webvtt-py not installed; cannot parse VTT.")
    parts = []
    for caption in webvtt.read(str(path)):
        txt = caption.text.strip()
        if not preserve_casing:
            txt = normalize_caption_text(txt)
        parts.append(txt)
    return " ".join(parts)


def parse_srt_to_text(path: Path, preserve_casing: bool = False) -> str:
    if pysrt is None:
        raise RuntimeError("pysrt not installed; cannot parse SRT.")
    subs = pysrt.open(str(path), encoding='utf-8', errors='ignore')
    parts = []
    for s in subs:
        txt = s.text.strip()
        if not preserve_casing:
            txt = normalize_caption_text(txt)
        parts.append(txt)
    return " ".join(parts)


########################################################################################
# Normalization & cleaning
########################################################################################

INLINE_TIMESTAMP_PAT = re.compile(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?", re.UNICODE)
SQUARE_TAG_PAT = re.compile(r"\[[^\]\n]{0,40}\]")  # [Music], [Applause], etc.
MULTISPACE_PAT = re.compile(r"\s+")


def normalize_caption_text(txt: str) -> str:
    # Remove inline timestamps like [00:12] or 00:12
    txt = INLINE_TIMESTAMP_PAT.sub("", txt)
    # Remove [Music], [Applause], etc.
    txt = SQUARE_TAG_PAT.sub("", txt)
    # Convert HTML entities sometimes present in transcripts
    txt = html_unescape(txt)
    # Normalize unicode
    txt = unicodedata.normalize("NFKC", txt)
    # Collapse whitespace
    txt = MULTISPACE_PAT.sub(" ", txt).strip()
    return txt

# Basic HTML entity unescape without importing html for tight envs; fallback to stdlib if available
try:
    import html
    def html_unescape(s: str) -> str:  # pragma: no cover - trivial
        return html.unescape(s)
except ImportError:  # pragma: no cover
    HTML_ENT_MAP = {
        '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'",
    }
    def html_unescape(s: str) -> str:
        for k,v in HTML_ENT_MAP.items():
            s = s.replace(k,v)
        return s


########################################################################################
# yt-dlp helpers (audio + subtitle download)
########################################################################################

YTDLP_BIN_CANDIDATES = ["yt-dlp", "yt_dlp"]

def find_executable(name_list):
    for n in name_list:
        p = shutil.which(n)
        if p:
            return p
    return None

YTDLP_BIN = find_executable(YTDLP_BIN_CANDIDATES)


def yt_dlp_download(
    url: str,
    out_dir: Path,
    audio_only: bool = True,
    want_subs: bool = True,
    lang_priorities: Optional[List[str]] = None,
) -> Tuple[Optional[Path], List[Path], Optional[str]]:
    """Download audio (& optionally subs) using yt-dlp.

    Returns: (audio_path, [subtitle_paths], video_title)
    """
    if YTDLP_BIN is None:
        raise RuntimeError("yt-dlp executable not found in PATH.")

    out_tpl = str(out_dir / "%(id)s.%(ext)s")

    # Build args
    cmd = [YTDLP_BIN, url, "--no-playlist", "-o", out_tpl]

    if audio_only:
        cmd += ["-f", "bestaudio/best", "--extract-audio", "--audio-format", "mp3", "--audio-quality", "0"]

    if want_subs:
        cmd += ["--write-auto-sub", "--write-sub", "--sub-langs", ",".join(lang_priorities or ['en', 'en.*,live_auto'])]

    cmd += ["--print", "title"]  # We'll capture title from stdout

    # Run
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {proc.stderr.strip()}")

    title = None
    # Parse stdout lines: Last printed 'title' (?) We'll heuristically take last non-empty line.
    for line in proc.stdout.splitlines():
        if line.strip():
            title = line.strip()

    # Find downloaded files in out_dir with matching video ID; We don't know ID w/out parse; fallback: pick newest
    newest_audio = None
    newest_time = -1
    subs_paths = []
    for p in out_dir.iterdir():
        if p.suffix.lower() in {'.mp3', '.m4a', '.aac', '.wav', '.ogg', '.webm'}:
            t = p.stat().st_mtime
            if t > newest_time:
                newest_time = t
                newest_audio = p
        elif p.suffix.lower() in {'.vtt', '.srt', '.ttml', '.srv1', '.srv2'}:
            subs_paths.append(p)

    return newest_audio, subs_paths, title


########################################################################################
# Speech-to-Text Backends
########################################################################################

def transcribe_with_whisper(audio_path: Path, model_name: str = "small", device: Optional[str] = None) -> str:
    if whisper is None:
        raise RuntimeError("openai-whisper not installed.")
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(str(audio_path))
    # result['text'] full text; segments have timestamps but we ignore
    return normalize_caption_text(result.get('text', ''))


def transcribe_with_faster_whisper(audio_path: Path, model_name: str = "small", device: str = "auto", compute_type: str = "int8_float16") -> str:
    if FasterWhisperModel is None:
        raise RuntimeError("faster-whisper not installed.")
    if device == "auto":
        # heuristics: if CUDA avail? faster-whisper does internal detection; pass "cuda" and catch
        device = "cuda" if shutil.which('nvidia-smi') else "cpu"
    model = FasterWhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), beam_size=5)
    parts = []
    for segment in segments:
        txt = segment.text.strip()
        txt = normalize_caption_text(txt)
        parts.append(txt)
    return " ".join(parts)


def transcribe_with_openai(audio_path: Path, api_key: str, model_name: str = "whisper-1", base_url: str = "https://api.openai.com/v1/audio/transcriptions") -> str:
    """Send audio to OpenAI Whisper API-compatible endpoint."""
    if requests is None:
        raise RuntimeError("requests not installed.")
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {
        'file': (audio_path.name, open(audio_path, 'rb'), 'application/octet-stream'),
        'model': (None, model_name),
        'response_format': (None, 'verbose_json'),
    }
    resp = requests.post(base_url, headers=headers, files=files)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI ASR API error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    # verbose_json returns segments; also top-level text maybe included depending on endpoint
    if 'text' in data and data['text']:
        return normalize_caption_text(data['text'])
    segments = data.get('segments') or []
    parts = []
    for seg in segments:
        txt = seg.get('text', '').strip()
        txt = normalize_caption_text(txt)
        parts.append(txt)
    return " ".join(parts)


########################################################################################
# Output helpers
########################################################################################

def safe_filename(name: str, max_len: int = 100, replacement: str = "_") -> str:
    # Remove illegal path chars
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"[\\/*?<>|:\"'\n\r]", replacement, name)
    name = name.strip().strip(replacement)
    if len(name) > max_len:
        name = name[:max_len].rstrip(replacement)
    return name or "untitled"


def write_text(path: Path, text: str):
    path.write_text(text, encoding='utf-8')


########################################################################################
# Main orchestration
########################################################################################

def get_transcript_pipeline(
    url_or_id: str,
    out_path: Optional[Path] = None,
    lang_priority: Optional[List[str]] = None,
    force_stt: bool = False,
    stt_backend: str = "whisper",  # whisper | faster-whisper | openai
    whisper_model: str = "small",
    whisper_device: Optional[str] = None,
    faster_compute_type: str = "int8_float16",
    openai_api_key: Optional[str] = None,
    openai_model: str = "whisper-1",
    openai_base_url: str = "https://api.openai.com/v1/audio/transcriptions",
    keep_temp: bool = False,
    verbose: bool = True,
) -> Path:
    """High-level convenience wrapper. Returns the path to the written transcript file."""
    video_id = extract_video_id(url_or_id)
    if not video_id:
        raise ValueError(f"Could not parse video ID from: {url_or_id}")

    # Step 1: try YouTubeTranscriptApi
    if not force_stt and YouTubeTranscriptApi is not None:
        if verbose:
            print("Attempting direct transcript fetch via API...")
        text, lang_code = try_fetch_transcript(video_id, lang_priority=lang_priority)
        if text:
            if verbose:
                print(f"✔ Got transcript via API (lang={lang_code}).")
            if out_path is None:
                out_path = Path.cwd() / f"{video_id}_transcript.txt"
            write_text(out_path, text)
            return out_path
        else:
            if verbose:
                print("✘ Direct transcript unavailable. Will try subtitles.")

    # Step 2: download subs & audio via yt-dlp
    with tempfile.TemporaryDirectory() as tmpd:
        tmpdir = Path(tmpd)
        if verbose:
            print("Downloading audio/subtitles with yt-dlp...")
        try:
            audio_path, subs_paths, title = yt_dlp_download(
                url_or_id, tmpdir, audio_only=True, want_subs=True, lang_priorities=lang_priority
            )
        except Exception as e:
            if verbose:
                print(f"✘ yt-dlp download failed: {e}")
            audio_path, subs_paths, title = None, [], None

        # Try parse subs first if not forcing STT
        if not force_stt and subs_paths:
            if verbose:
                print(f"Found {len(subs_paths)} subtitle file(s). Parsing...")
            # Choose best subtitle: prefer VTT then SRT
            chosen = None
            for p in subs_paths:
                if p.suffix.lower() == '.vtt':
                    chosen = p; break
            if chosen is None:
                for p in subs_paths:
                    if p.suffix.lower() == '.srt':
                        chosen = p; break
            if chosen is None:
                chosen = subs_paths[0]
            if verbose:
                print(f"Using subtitle file: {chosen.name}")
            try:
                if chosen.suffix.lower() == '.vtt':
                    text = parse_vtt_to_text(chosen)
                elif chosen.suffix.lower() == '.srt':
                    text = parse_srt_to_text(chosen)
                else:
                    # fallback raw read + strip numbers/time ranges
                    raw = chosen.read_text(encoding='utf-8', errors='ignore')
                    text = normalize_caption_text(strip_srt_like(raw))
            except Exception as e:
                if verbose:
                    print(f"Subtitle parse failed: {e}; falling back to STT.")
                text = None

            if text:
                if verbose:
                    print("✔ Parsed subtitles.")
                if out_path is None:
                    base = safe_filename(title) if title else video_id
                    out_path = Path.cwd() / f"{base}_transcript.txt"
                write_text(out_path, text)
                if keep_temp:
                    # copy chosen subtitle for debug
                    shutil.copy2(chosen, out_path.with_suffix(chosen.suffix))
                return out_path

        # Step 3: Speech-to-text from audio
        if verbose:
            print("Invoking speech-to-text backend...")
        if not audio_path:
            raise RuntimeError("No audio file available for STT.")

        if stt_backend == "whisper":
            text = transcribe_with_whisper(audio_path, model_name=whisper_model, device=whisper_device)
        elif stt_backend == "faster-whisper":
            text = transcribe_with_faster_whisper(audio_path, model_name=whisper_model, device=whisper_device or "auto", compute_type=faster_compute_type)
        elif stt_backend == "openai":
            if not openai_api_key:
                raise RuntimeError("--openai-api-key required for openai backend.")
            text = transcribe_with_openai(audio_path, api_key=openai_api_key, model_name=openai_model, base_url=openai_base_url)
        else:
            raise ValueError(f"Unknown stt_backend: {stt_backend}")

        if out_path is None:
            base = safe_filename(title) if title else video_id
            out_path = Path.cwd() / f"{base}_transcript.txt"
        write_text(out_path, text)
        if keep_temp:
            shutil.copy2(audio_path, out_path.with_suffix(audio_path.suffix))
        return out_path


########################################################################################
# Simple SRT-like stripper (fallback parser)
########################################################################################

SRT_INDEX_PAT = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
SRT_TIMERANGE_PAT = re.compile(r"\d\d:\d\d:\d\d,\d\d\d\s*-->.*", re.MULTILINE)

def strip_srt_like(raw: str) -> str:
    raw = SRT_INDEX_PAT.sub("", raw)
    raw = SRT_TIMERANGE_PAT.sub("", raw)
    return raw


########################################################################################
# CLI
########################################################################################

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download or generate a plain-text transcript from a YouTube video.")
    p.add_argument("url", help="YouTube video URL or 11-char video ID.")
    p.add_argument("-o", "--output", help="Output text file path. Default = <videoid>_transcript.txt", default=None)
    p.add_argument("--lang", nargs="*", default=None, help="Preferred language codes in priority order (e.g. en en-US af).")
    p.add_argument("--force-stt", action="store_true", help="Skip transcript/subtitle attempts; go straight to STT.")

    # STT backend opts
    p.add_argument("--stt-backend", choices=["whisper", "faster-whisper", "openai"], default="whisper", help="Speech-to-text engine to use if needed.")
    p.add_argument("--whisper-model", default="small", help="Model name for whisper / faster-whisper (tiny, base, small, medium, large, etc.)")
    p.add_argument("--whisper-device", default=None, help="Device override for whisper backends (cpu, cuda, mps).")
    p.add_argument("--faster-compute-type", default="int8_float16", help="faster-whisper compute precision.")

    # OpenAI remote ASR
    p.add_argument("--openai-api-key", default=None, help="OpenAI API key (required if stt-backend=openai).")
    p.add_argument("--openai-model", default="whisper-1", help="OpenAI ASR model name.")
    p.add_argument("--openai-base-url", default="https://api.openai.com/v1/audio/transcriptions", help="Override ASR endpoint.")

    p.add_argument("--keep-temp", action="store_true", help="Keep downloaded media/subtitle copies next to transcript for debugging.")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages.")
    return p


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_arg_parser().parse_args(argv)

    out_path = Path(args.output).expanduser().resolve() if args.output else None
    try:
        written = get_transcript_pipeline(
            args.url,
            out_path=out_path,
            lang_priority=args.lang,
            force_stt=args.force_stt,
            stt_backend=args.stt_backend,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
            faster_compute_type=args.faster_compute_type,
            openai_api_key=args.openai_api_key,
            openai_model=args.openai_model,
            openai_base_url=args.openai_base_url,
            keep_temp=args.keep_temp,
            verbose=not args.quiet,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # unexpected
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 3

    if not args.quiet:
        print(f"Transcript written to: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
