import os
from pydantic import BaseModel, Field
import re
from datetime import datetime
from google import genai
from google.genai import errors, types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SOURCE_DIR = os.path.expanduser("/Users/kian/Library/Mobile Documents/com~apple~CloudDocs/Music")
OBSIDIAN_BASE = os.path.expanduser("~/obsidian")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_FALLBACKS = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-flash-lite"]

client = genai.Client(api_key=GEMINI_API_KEY)


class TranscriptResult(BaseModel):
    response: str = Field(description="The transcript markdown text")
    course: bool = Field(description="True only if the recording starts with 'course a pied', else false")

def extract_recorded_datetime(file_path):
    filename = os.path.basename(file_path)
    match = re.search(r"(\d{4}-\d{2}-\d{2}).*?(\d{2}\.\d{2}\.\d{2})", filename)
    return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H.%M.%S")


def process_audio(file_path):
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()

    # Prompt with structured JSON output.
    prompt = """Transcribe the audio faithfully.
Reformat into markdown bullet points.
Preserve intent and phrasing."""

    response = None
    contents = [
        prompt,
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp4"),
    ]
    for model_name in MODEL_FALLBACKS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": {
                        "type": "object",
                        "properties": {
                            "response": {"type": "string"},
                            "course": {"type": "boolean"},
                        },
                        "required": ["response", "course"],
                        "additionalProperties": False,
                    },
                },
            )
            break
        except errors.APIError as err:
            if err.code != 429:
                raise
            continue

    response = TranscriptResult.model_validate_json(response.text)
    print(response)
    # Append to Obsidian
    recorded_at = extract_recorded_datetime(file_path)
    date_str = recorded_at.strftime("%Y-%m-%d")
    target_file = os.path.join(OBSIDIAN_BASE, f"{date_str}.md")
    
    with open(target_file, "a") as f:
        f.write(response.response)
    
    # Delete the local file
    os.remove(file_path)

def main():
    for filename in os.listdir(SOURCE_DIR):
        file_path = os.path.join(SOURCE_DIR, filename)
        process_audio(file_path)

if __name__ == "__main__":
    main()
