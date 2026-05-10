# Text-to-speech

Conduct can act as a TTS dispatcher for audiobook-style workloads — POST a
chunk of text, it synthesizes locally via [Piper](https://github.com/rhasspy/piper),
writes an MP3 to a shared `output/` directory, and returns a URL. The use
case it was built for: another machine on the LAN rsyncs the chunks and
stitches them into a finished audiobook.

## Setup

```bash
# 1. ffmpeg is required for WAV → MP3
brew install ffmpeg                          # host-side dev
# (containers ship ffmpeg in the image — no action needed)

# 2. Download at least one voice (~25-60MB per voice)
make download-voice                          # default: en_US-amy-medium
make download-voice v=en_GB-alan-medium      # alternative

# Browse the full catalog at:
# https://huggingface.co/rhasspy/piper-voices
```

Voice files land in `voices/`. The directory is gitignored and mounted into
the API + worker containers as a read-only volume.

## Configuration

```
TTS_VOICES_DIR=./voices             # where .onnx + .onnx.json files live
TTS_DEFAULT_VOICE=en_US-amy-medium  # fallback when request omits `voice`
TTS_OUTPUT_DIR=./output             # where MP3s are written
TTS_MAX_CHARS=10000                 # per-request input limit
```

`TTS_MAX_CHARS=10000` is roughly 75 seconds of audio at typical narration
pace. Larger inputs return `413 Request Entity Too Large` — chunk them on
the caller side.

## API

```bash
# Submit a chunk
curl -X POST http://localhost:8000/tts \
  -H "Authorization: Bearer cdt_..." \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "voice": "en_US-amy-medium"}'

# → 202 Accepted
# {
#   "job_id": "...",
#   "status": "pending",
#   "poll_url": "/jobs/{id}",
#   "expected_output_url": "/output/{id}.mp3",
#   "voice": "en_US-amy-medium"
# }

# Poll
curl http://localhost:8000/jobs/{id} -H "Authorization: Bearer cdt_..."

# When status=complete, response field holds the URL:
# {
#   "status": "complete",
#   "response": "/output/{id}.mp3",
#   "model_used": "en_US-amy-medium",
#   "tokens_in": 205,        # input character count
#   "tokens_out": 155212,    # MP3 byte size
#   "latency_ms": 3759,
#   "metadata": { "tts": { ... } }
# }

# Fetch the file (admin auth)
curl -O http://localhost:8000/output/{id}.mp3 \
  -H "Authorization: Bearer $CONDUCT_ADMIN_KEY"
```

## Delivery patterns

The intended workflow is **text in via API, MP3 in `output/`, fetched out
of band**. Two delivery options:

- **HTTP** — `GET /output/{filename}.mp3` returns the file. Admin-auth.
  Suitable for one-off downloads or a single sync agent that holds the key.
- **Filesystem rsync** — the `output/` directory is mounted on the host
  (and thus on the LAN). `rsync -avz user@conduct-host:/Users/dave/conduct/output/ ./local-chunks/`
  pulls everything; works without API auth, scales for batch transfer.

Files persist until manually deleted. There's no automatic TTL — if you're
generating a full book's worth of chunks, plan a cleanup pass.

## Performance notes

On an M5 Mac with `en_US-amy-medium`:
- Synthesis: ~0.5s per 200 chars (CPU-bound, ONNX runtime)
- MP3 encode: ~0.04s per chunk after first ffmpeg invocation (cold start ~3s)
- Memory: voice model stays cached in the worker process across jobs

For a 100k-word book chunked into 10k-char pieces (~10 chunks):
- Total wall time: ~1-2 minutes generation + transfer
- Total disk: ~15-20MB MP3

## Engine swap

`tts/piper_engine.py` is the only engine today. The executor calls
`synthesize_to_mp3(text, voice_name, voices_dir, output_path)` —
implementing the same signature in a `tts/<other>_engine.py` and switching
the dispatch is the minimal path to an alternative (Coqui XTTS, F5-TTS,
Kokoro, OpenAI TTS, ElevenLabs). The Job → file → URL contract stays the same.
