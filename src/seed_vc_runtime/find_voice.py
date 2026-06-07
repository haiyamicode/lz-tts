import base64
import json
import os.path
import re
from resemblyzer import VoiceEncoder, preprocess_wav
from lzb_core import jsonrpc
import os
import pickle
import numpy as np
import sys
from pathlib import Path

encoder = VoiceEncoder()
RUNTIME_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUNTIME_ROOT.parents[1]
VOICES_FINAL_PATH = Path(
  os.environ.get("SEED_VC_VOICES_FINAL_PATH", PROJECT_ROOT / "data" / "seed-vc" / "voices_final.pkl")
)
VOICES_JSON_PATH = RUNTIME_ROOT / "voices.json"

def log(*args, **kwargs):
    if os.environ.get("DEBUG") == "1":
        print(*args, **kwargs)

def data_url_to_file(data_url, output_file_path):
    # Extract the Base64 part of the Data URL
    match = re.match(r"data:(.*?);base64,(.*)", data_url)
    if not match:
        raise ValueError("Invalid Data URL format")

    mime_type, base64_data = match.groups()

    # Decode the Base64 data
    file_data = base64.b64decode(base64_data)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Save the decoded data to a file
    with open(output_file_path, "wb") as file:
        file.write(file_data)

    log(f"File saved to: {output_file_path}")

def get_sample(voice):
  id = voice["id"]
  sample_mp3_path = "premade-voice-samples/{}.mp3".format(id)
  sample_wav_path = "premade-voice-samples/{}.wav".format(id)
  if os.path.isfile(sample_wav_path):
    return sample_wav_path
  
  response = jsonrpc({
    "method": "synthesizeSpeech",
    "input": {
      "voiceId": voice["id"],
      "text": "Hi, my name is Unknown. I am an English voice from the United States.",
    }
  })

  log("result:", response)
  data_url = response["result"]["audioUrl"]
  data_url_to_file(data_url, sample_mp3_path)
  os.system("ffmpeg -i {} -y -f wav {} && rm {}".format(sample_mp3_path, sample_wav_path, sample_mp3_path))
  return sample_wav_path

def analyze_voices():
  voices = json.load(open(VOICES_JSON_PATH))
  data = {}

  for voice in voices:
    if not voice.get("secondaryLocaleList"):
      continue

    id = voice["id"]
    log("Analyzing voice: {}".format(id))
    sample_wav_path = get_sample(voice)
    reference_wav = preprocess_wav(sample_wav_path)
    reference_embedding = encoder.embed_utterance(reference_wav)
    data[id] = voice
    data[id]["embedding"] = reference_embedding
    log("Analyzed voice: {}".format(id))

  VOICES_FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
  pickle.dump(data, open(VOICES_FINAL_PATH, "wb"))
  return data

def find_base_voice(input_file_path):
  voices_data = None
  if not os.path.isfile(VOICES_FINAL_PATH):
    voices_data = analyze_voices()
  else:
    voices_data = pickle.load(open(VOICES_FINAL_PATH, "rb"))
  
  log("Analzying input sample: ", input_file_path)
  sample_wav = preprocess_wav(input_file_path)
  sample_embedding = encoder.embed_utterance(sample_wav)
  
  log("Total voices:", len(voices_data))
  best_similarity = 0
  best_voice = None

  for id, voice in voices_data.items():
    embedding = voice["embedding"]
    similarity = np.dot(sample_embedding, embedding)
    log("compared with {}: {}".format(id, similarity))
    if similarity > best_similarity:
      best_similarity = similarity
      best_voice = voice
  
  log("Best voice: {}, similarity: {}".format(best_voice["id"], best_similarity))
  return best_voice["id"]

def main():
  input_file_path = sys.argv[1]
  best_voice_id = find_base_voice(input_file_path)
  print(best_voice_id)

if __name__ == "__main__":
  main()
