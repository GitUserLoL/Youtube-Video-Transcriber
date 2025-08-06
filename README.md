# YouTube Transcript Downloader + Bible Commentary Generator

A command-line tool that downloads transcripts from YouTube videos or playlists and uses Google’s Gemini API to generate clean, study-note-ready biblical commentaries (KJV only).

---

## Features

- **Transcript Retrieval**  
  - Official or auto-generated YouTube transcripts via `youtube-transcript-api`.  
  - Subtitle download & parsing with `yt-dlp` + WebVTT.  
  - Whisper-based fallback transcription if no subtitle/transcript exists.

- **Text Normalization**  
  - Cleans timestamps, speaker tags, sound-effects, HTML entities.  
  - Unicode normalization, whitespace cleanup, duplicate removal.

- **Commentary Generation**  
  - Integrates Google Gemini (gemini-2.5-flash) to produce verse-by-verse or theme-by-theme commentary.  
  - Strictly KJV quotations, faithful to speaker’s exegetical insights, omits non-exegetical fluff.  
  - Retry logic for robust API calls (configurable in second script).

- **Playlist Support**  
  - Batch process an entire YouTube playlist, preserving order and creating subfolders.

- **Safe Filenames & Metadata**  
  - Sanitizes titles for filesystem compatibility.  
  - Adds metadata headers to transcript & commentary files.

---

## Requirements

- Python 3.8+  
- [youtube-transcript-api](https://pypi.org/project/youtube-transcript-api/)  
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)  
- [webvtt-py](https://pypi.org/project/webvtt-py/) (optional, for better VTT parsing)  
- [openai-whisper](https://pypi.org/project/openai-whisper/) (optional, for audio fallback)  
- [google-generativeai](https://pypi.org/project/google-generativeai/)  

```bash
pip install youtube-transcript-api yt-dlp webvtt-py openai-whisper google-generativeai

