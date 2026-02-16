import os
import mimetypes
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SOURCE_DIR = os.path.expanduser("/Users/kian/Library/Mobile Documents/com~apple~CloudDocs/Music")
OBSIDIAN_BASE = os.path.expanduser("~/obsidian")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

def process_audio(file_path):
    # Send inline audio bytes directly (no Files API upload required for small requests).
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        ext = os.path.splitext(file_path)[1].lower()
        mime_type = {
            ".m4a": "audio/mp4",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".aac": "audio/aac",
        }.get(ext, "application/octet-stream")

    with open(file_path, 'rb') as f:
        audio_bytes = f.read()

    # Prompt to both transcribe and classify
    prompt = """Transcribe the audio faithfully.
Do NOT summarize.
Reformat into concise bullet points.
Preserve intent and phrasing.
Output markdown bullets only.

Additionally, determine if the speaker is talking about "course à pied" (running). 
At the very end of your response, add a single line with exactly [COURSE] if it is about running, or [NOTE] if it is not."""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            prompt,
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
    )
    
    full_text = response.text.strip()
    
    # Determine target based on tag
    if "[COURSE]" in full_text:
        target_dir = os.path.join(OBSIDIAN_BASE, "course")
        clean_transcript = full_text.replace("[COURSE]", "").strip()
    else:
        target_dir = os.path.join(OBSIDIAN_BASE, "notes")
        clean_transcript = full_text.replace("[NOTE]", "").strip()
        
    # Append to Obsidian
    date_str = datetime.now().strftime("%Y-%m-%d")
    target_file = os.path.join(target_dir, f"{date_str}.md")
    os.makedirs(target_dir, exist_ok=True)
    
    with open(target_file, "a") as f:
        f.write(f"\n\n### Transcribed at {datetime.now().strftime('%H:%M:%S')}\n")
        f.write(clean_transcript)
        f.write("\n")
    
    # Delete the local file
    os.remove(file_path)

def main():
    if not os.path.exists(SOURCE_DIR):
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.m4a', '.mp3', '.wav', '.aac')) and not f.startswith(".")]
    
    for filename in files:
        file_path = os.path.join(SOURCE_DIR, filename)
        if os.path.getsize(file_path) < 100:
            continue
            
        try:
            process_audio(file_path)
        except Exception:
            pass

if __name__ == "__main__":
    main()
