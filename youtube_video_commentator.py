#!/usr/bin/env python3
"""
YouTube Transcript Downloader + Bible Commentary Generator
=========================================================

Downloads transcripts from YouTube videos or playlists and processes them 
with Google Gemini API to create clean biblical commentaries.

Usage: python youtube_transcriber.py
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
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Third-party imports with graceful fallbacks
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

try:
    import webvtt
except ImportError:
    webvtt = None

try:
    import whisper
except ImportError:
    whisper = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

########################################################################################
# Utility helpers
########################################################################################

YOUTUBE_ID_REGEX = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?.*?v=|embed/|shorts/|v/))([A-Za-z0-9_-]{11})"
)

PLAYLIST_REGEX = re.compile(
    r"youtube\.com/(?:playlist\?list=|watch\?.*list=)([A-Za-z0-9_-]+)"
)

def extract_video_id(url_or_id: str) -> Optional[str]:
    """Extract YouTube video ID from URL or return if already an ID."""
    candidate = url_or_id.strip()
    if len(candidate) == 11 and re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        return candidate
    m = YOUTUBE_ID_REGEX.search(candidate)
    return m.group(1) if m else None

def is_playlist_url(url: str) -> bool:
    """Check if URL is a playlist."""
    return bool(PLAYLIST_REGEX.search(url))

def extract_playlist_id(url: str) -> Optional[str]:
    """Extract playlist ID from URL."""
    m = PLAYLIST_REGEX.search(url)
    return m.group(1) if m else None

########################################################################################
# Transcript retrieval
########################################################################################

def safe_get_text(item) -> str:
    """Safely extract text from transcript item (handles both dict and object types)."""
    if hasattr(item, 'text'):
        return str(item.text).strip()
    elif isinstance(item, dict):
        return str(item.get('text', '')).strip()
    else:
        return str(item).strip()

def try_fetch_transcript(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch transcript via YouTube API."""
    if YouTubeTranscriptApi is None:
        return None, None
    
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try English variants first (manual transcripts preferred)
        for lang in ['en', 'en-US', 'en-GB']:
            try:
                # Try manual transcript first
                transcript = transcript_list.find_manually_created_transcript([lang])
                data = transcript.fetch()
                text_parts = [safe_get_text(item) for item in data]
                text = " ".join([part for part in text_parts if part])
                if text:
                    return normalize_text(text), lang
            except:
                pass
            
            try:
                # Try auto-generated transcript
                transcript = transcript_list.find_generated_transcript([lang])
                data = transcript.fetch()
                text_parts = [safe_get_text(item) for item in data]
                text = " ".join([part for part in text_parts if part])
                if text:
                    return normalize_text(text), lang
            except:
                pass
        
        # Try any available transcript as last resort
        for transcript in transcript_list:
            try:
                data = transcript.fetch()
                text_parts = [safe_get_text(item) for item in data]
                text = " ".join([part for part in text_parts if part])
                if text:
                    lang_code = getattr(transcript, 'language_code', 'unknown')
                    return normalize_text(text), lang_code
            except:
                continue
                
    except Exception as e:
        print(f"Transcript API error: {e}")
    
    return None, None

########################################################################################
# Text normalization
########################################################################################

def normalize_text(text: str) -> str:
    """Clean and normalize transcript text."""
    if not text:
        return ""
    
    # Remove timestamps like [00:12] or 00:12
    text = re.sub(r'\[?\d{1,2}:\d{2}(?::\d{2})?\]?', '', text)
    
    # Remove speaker tags and sound effects
    text = re.sub(r'\[[^\]]*\]', '', text)  # [Music], [Applause], etc.
    text = re.sub(r'\([^)]*\)', '', text)   # (Music), (Applause), etc.
    
    # Remove HTML entities if present
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'")
    
    # Normalize unicode and whitespace
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove duplicate sentences (fix for tripling issue)
    sentences = text.split('. ')
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)
    
    return '. '.join(unique_sentences)

########################################################################################
# yt-dlp helpers
########################################################################################

def find_ytdlp() -> Optional[str]:
    """Find yt-dlp executable."""
    for name in ["yt-dlp", "yt_dlp"]:
        path = shutil.which(name)
        if path:
            return path
    return None

def get_video_title(url: str) -> str:
    """Get video title using yt-dlp."""
    ytdlp_bin = find_ytdlp()
    if not ytdlp_bin:
        video_id = extract_video_id(url)
        return f"Video_{video_id}" if video_id else "Unknown_Video"
    
    try:
        cmd = [ytdlp_bin, "--dump-json", "--no-download", url]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        return info.get('title', 'Unknown_Video')
    except Exception:
        video_id = extract_video_id(url)
        return f"Video_{video_id}" if video_id else "Unknown_Video"

def download_with_ytdlp(url: str, temp_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Download subtitles using yt-dlp."""
    ytdlp_bin = find_ytdlp()
    if not ytdlp_bin:
        return None, None
    
    try:
        title = get_video_title(url)
        
        # Download subtitles
        sub_cmd = [
            ytdlp_bin, url, "--write-auto-sub", "--write-sub", 
            "--sub-langs", "en.*,en", "--skip-download",
            "-o", str(temp_dir / "%(id)s.%(ext)s")
        ]
        subprocess.run(sub_cmd, capture_output=True, check=True)
        
        # Find subtitle files (prefer .en.vtt)
        for pattern in ["*.en.vtt", "*.vtt"]:
            for sub_file in temp_dir.glob(pattern):
                return sub_file, title
            
    except Exception as e:
        print(f"yt-dlp subtitle download failed: {e}")
    
    return None, None

def parse_vtt_file(vtt_path: Path) -> str:
    """Parse VTT subtitle file."""
    if webvtt is not None:
        try:
            captions = webvtt.read(str(vtt_path))
            text_parts = []
            for caption in captions:
                clean_text = caption.text.replace('\n', ' ').strip()
                if clean_text:
                    text_parts.append(clean_text)
            return normalize_text(' '.join(text_parts))
        except Exception as e:
            print(f"WebVTT parsing failed: {e}")
    
    # Fallback: simple text extraction
    try:
        content = vtt_path.read_text(encoding='utf-8', errors='ignore')
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            # Skip VTT headers, timestamps, and empty lines
            if (line and not line.startswith('WEBVTT') and 
                '-->' not in line and not line.isdigit() and
                not line.startswith('NOTE') and not line.startswith('STYLE')):
                lines.append(line)
        return normalize_text(' '.join(lines))
    except Exception as e:
        print(f"Fallback VTT parsing failed: {e}")
        return ""

########################################################################################
# Whisper STT (fallback)
########################################################################################

def transcribe_with_whisper(audio_path: Path) -> str:
    """Transcribe audio using Whisper."""
    if whisper is None:
        raise RuntimeError("Whisper not installed. Install with: pip install openai-whisper")
    
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # Lightweight model
    print("Transcribing audio...")
    result = model.transcribe(str(audio_path))
    return normalize_text(result.get('text', ''))

def download_audio_and_transcribe(url: str, temp_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """Download audio and transcribe with Whisper."""
    ytdlp_bin = find_ytdlp()
    if not ytdlp_bin:
        return None, None
    
    try:
        title = get_video_title(url)
        
        # Download audio
        print("Downloading audio...")
        audio_cmd = [
            ytdlp_bin, url, "-f", "bestaudio", "--extract-audio", 
            "--audio-format", "mp3", "-o", str(temp_dir / "%(id)s.%(ext)s")
        ]
        subprocess.run(audio_cmd, capture_output=True, check=True)
        
        # Find audio file
        for audio_file in temp_dir.glob("*.mp3"):
            text = transcribe_with_whisper(audio_file)
            return text, title
            
    except Exception as e:
        print(f"Audio transcription failed: {e}")
    
    return None, None

########################################################################################
# Google Gemini API Integration
########################################################################################

def setup_gemini_api() -> bool:
    """Setup Gemini API with user's API key."""
    if genai is None:
        print("Google Generative AI library not installed.")
        print("Install with: pip install google-generativeai")
        return False
    
    # Check for API key in environment variable first
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("\nGoogle Gemini API Setup Required")
        print("-" * 35)
        print("You need a Gemini API key to process transcripts.")
        print("Get one at: https://aistudio.google.com/app/apikey")
        api_key = input("Enter your Gemini API key: ").strip()
        
        if not api_key:
            print("No API key provided. Skipping commentary generation.")
            return False
    
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Failed to configure Gemini API: {e}")
        return False

def generate_commentary(transcript: str, title: str) -> Optional[str]:
    """Generate biblical commentary using Gemini API."""
    if genai is None:
        return None
    
    prompt = '''You are an academic Bible‐commentator. I'm going to give you a raw transcript of a sermon or teaching. Produce a clean, verse‐by‐verse (or theme-by-theme) commentary that:

1. **Retains**  
   - All direct Scripture quotations (verbatim).  
   - The speaker's exegetical insights, definitions, theological observations, and any cross-references they introduce.

2. **Omits**  
   - Personal stories, jokes, pop-culture quips, Mother's Day humor, prayer language, in-house announcements, and any non-exegetical "fluff."  
   - Historical anecdotes unrelated to the passage's meaning.  

3. **Organizes**  
   - By verse number or clear thematic headings (e.g. "I. Trials and Perseverance," "II. God's Wisdom," etc.).  
   - Provide brief sub-headings for major points.  

4. **Formats**  
   - Scripture in block quotes or bold.  
   - Commentary in normal paragraphs or bullet points under each heading.  

**Transcript:**  

''' + transcript + '''

**Output:**  
A polished, study-note–ready commentary on the passage.'''

    try:
        print("🤖 Generating biblical commentary with Gemini 2.5 Flash...")
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=8192,
                temperature=0.2,
            ),
        )
        
        if response.text:
            print("✓ Commentary generated successfully!")
            return response.text
        else:
            print("✗ Empty response from Gemini API")
            return None
            
    except Exception as e:
        print(f"✗ Gemini API error: {e}")
        return None

########################################################################################
# Playlist handling
########################################################################################

def get_playlist_videos(playlist_url: str) -> Tuple[List[str], str]:
    """Get video URLs from playlist."""
    ytdlp_bin = find_ytdlp()
    if not ytdlp_bin:
        raise RuntimeError("yt-dlp not found. Install with: pip install yt-dlp")
    
    try:
        # Get playlist info
        cmd = [ytdlp_bin, "--flat-playlist", "--dump-json", playlist_url]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        videos = []
        playlist_title = "Unknown Playlist"
        
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        videos.append(f"https://www.youtube.com/watch?v={data['id']}")
                    if 'playlist_title' in data and data['playlist_title']:
                        playlist_title = data['playlist_title']
                except json.JSONDecodeError:
                    continue
        
        return videos, playlist_title
        
    except Exception as e:
        raise RuntimeError(f"Failed to get playlist videos: {e}")

########################################################################################
# File handling
########################################################################################

def safe_filename(name: str, max_len: int = 100) -> str:
    """Create safe filename from title."""
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip('_')
    return name or "untitled"

def save_transcript(text: str, title: str, output_dir: Path, video_num: Optional[int] = None) -> Path:
    """Save transcript to file."""
    filename = safe_filename(title)
    if video_num:
        filename = f"{video_num:03d}. {filename}"
    
    filepath = output_dir / f"{filename}.txt"
    
    # Add metadata header
    content = f"Title: {title}\n"
    content += f"=" * 50 + "\n\n"
    content += text
    
    filepath.write_text(content, encoding='utf-8')
    return filepath

def save_commentary(commentary: str, title: str, output_dir: Path, video_num: Optional[int] = None) -> Path:
    """Save commentary to file."""
    filename = safe_filename(title)
    if video_num:
        filename = f"{video_num:03d}. {filename}"
    
    filepath = output_dir / f"{filename}_commentary.txt"
    
    # Add metadata header
    content = f"Biblical Commentary: {title}\n"
    content += f"Generated by Gemini 2.5 Pro\n"
    content += f"=" * 50 + "\n\n"
    content += commentary
    
    filepath.write_text(content, encoding='utf-8')
    return filepath

########################################################################################
# Main processing
########################################################################################

def process_single_video(url: str, transcript_dir: Path, commentary_dir: Path, 
                        video_num: Optional[int] = None, total: Optional[int] = None, 
                        use_gemini: bool = True) -> bool:
    """Process a single video and save transcript + commentary."""
    video_id = extract_video_id(url)
    if not video_id:
        print(f"✗ Invalid video URL: {url}")
        return False
    
    progress = f"[{video_num}/{total}] " if video_num and total else ""
    
    # Get video title first
    title = get_video_title(url)
    print(f"{progress}Processing: {title}")
    
    transcript_text = None
    
    # Method 1: Try official transcript
    text, lang = try_fetch_transcript(video_id)
    if text:
        transcript_text = text
        filepath = save_transcript(text, title, transcript_dir, video_num)
        print(f"✓ Official transcript ({lang}) saved: {filepath.name}")
    else:
        # Method 2: Try subtitles via yt-dlp
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            sub_file, _ = download_with_ytdlp(url, temp_path)
            if sub_file:
                text = parse_vtt_file(sub_file)
                if text:
                    transcript_text = text
                    filepath = save_transcript(text, title, transcript_dir, video_num)
                    print(f"✓ Subtitle transcript saved: {filepath.name}")
                else:
                    # Method 3: Try Whisper STT (if available)
                    if whisper is not None:
                        print(f"{progress}Trying speech-to-text...")
                        text, _ = download_audio_and_transcribe(url, temp_path)
                        if text:
                            transcript_text = text
                            filepath = save_transcript(text, title, transcript_dir, video_num)
                            print(f"✓ STT transcript saved: {filepath.name}")
    
    if not transcript_text:
        print(f"✗ Failed to get transcript for: {title}")
        return False
    
    # Generate commentary if Gemini is available and enabled
    if use_gemini and transcript_text:
        commentary = generate_commentary(transcript_text, title)
        if commentary:
            commentary_filepath = save_commentary(commentary, title, commentary_dir, video_num)
            print(f"✓ Commentary saved: {commentary_filepath.name}")
        else:
            print(f"✗ Failed to generate commentary for: {title}")
    
    return True

def process_playlist(playlist_url: str, transcript_dir: Path, commentary_dir: Path, use_gemini: bool = True) -> None:
    """Process entire playlist."""
    try:
        videos, playlist_title = get_playlist_videos(playlist_url)
        
        if not videos:
            print("No videos found in playlist")
            return
        
        # Create playlist subdirectories
        playlist_transcript_dir = transcript_dir / safe_filename(playlist_title)
        playlist_commentary_dir = commentary_dir / safe_filename(playlist_title)
        playlist_transcript_dir.mkdir(exist_ok=True)
        if use_gemini:
            playlist_commentary_dir.mkdir(exist_ok=True)
        
        print(f"Playlist: {playlist_title}")
        print(f"Videos found: {len(videos)}")
        print(f"Transcript directory: {playlist_transcript_dir}")
        if use_gemini:
            print(f"Commentary directory: {playlist_commentary_dir}")
        print("-" * 50)
        
        successful = 0
        for i, video_url in enumerate(videos, 1):
            if process_single_video(video_url, playlist_transcript_dir, 
                                  playlist_commentary_dir, i, len(videos), use_gemini):
                successful += 1
            print()  # Add spacing between videos
            
            # Add small delay to avoid rate limiting
            if use_gemini and i < len(videos):
                time.sleep(2)
        
        print("=" * 50)
        print(f"Playlist processing complete!")
        print(f"Successful: {successful}/{len(videos)}")
        
    except Exception as e:
        print(f"Playlist processing failed: {e}")

########################################################################################
# Main function
########################################################################################

def main():
    """Main interactive function."""
    print("YouTube Transcript Downloader + Biblical Commentary Generator")
    print("Powered by Google Gemini 2.5 Pro")
    print("-" * 60)
    
    # Check dependencies
    missing_deps = []
    if YouTubeTranscriptApi is None:
        missing_deps.append("youtube-transcript-api")
    if find_ytdlp() is None:
        missing_deps.append("yt-dlp")
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return 1
    
    # Setup Gemini API
    use_gemini = setup_gemini_api()
    if not use_gemini:
        print("Continuing with transcript download only...")
    
    print()
    
    # Get URL from user
    url = input("Enter YouTube video or playlist URL: ").strip()
    if not url:
        print("No URL provided.")
        return 1
    
    # Create output directories
    transcript_dir = Path("transcripts")
    commentary_dir = Path("commentaries")
    transcript_dir.mkdir(exist_ok=True)
    if use_gemini:
        commentary_dir.mkdir(exist_ok=True)
    
    # Process based on URL type
    if is_playlist_url(url):
        print("Playlist detected!")
        process_playlist(url, transcript_dir, commentary_dir, use_gemini)
    else:
        print("Single video detected!")
        process_single_video(url, transcript_dir, commentary_dir, use_gemini=use_gemini)
    
    print("\n✅ Processing complete!")
    print(f"📁 Transcripts saved in: {transcript_dir}")
    if use_gemini:
        print(f"📖 Commentaries saved in: {commentary_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())