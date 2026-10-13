# Reelferry

Turn long videos into captioned short clips that your AI agent can cut, caption and (soon) post everywhere, all on your own machine.

<!-- DEMO VIDEO PLACEHOLDER: drop the demo video (GitHub user-attachments link) here before launch. -->

> **Status:** early alpha.
> This release is the engine: sources, transcription, clip suggestions, rendering and captions.
> Posting to social channels, the content calendar and the web UI come next.

## Why Reelferry

- **Agent first.** Every feature is one operation, exposed the same way over MCP, the CLI and a local HTTP API.
- **Local and cheap.** Videos never leave your machine. Transcription runs locally with Whisper for free; only clip selection calls an LLM through your own OpenRouter key (a few cents per hour of video).
- **Captions you can trust.** Word timings come from Whisper large-v3-turbo refined by CTC forced alignment. Fix any word before rendering, export SRT, VTT (with per-word timestamps), ASS or JSON, and burn captions in only if you want to.

## How it works

```
add_source -> transcribe -> suggest_clips -> create_clip -> render_clip
  (file/link)   (local Whisper   (LLM picks      (from a        (MP4 9:16 or 16:9,
                 + alignment)     moments)        suggestion)     captions burned or soft)
```

Long steps run as durable background jobs stored in SQLite.
They survive restarts and resume at the last finished step.

## Install

You need [uv](https://docs.astral.sh/uv/) and FFmpeg.

```sh
uvx --from git+https://github.com/BatuCani/reelferry reelferry system check
```

Or install the command once:

```sh
uv tool install git+https://github.com/BatuCani/reelferry
reelferry system check
```

From a clone:

```sh
git clone https://github.com/BatuCani/reelferry && cd reelferry
uvx --from . reelferry system check
```

### FFmpeg with libass

Burned-in captions need an FFmpeg build that includes libass (the `ass` filter).
`reelferry system check` tells you whether yours has it.

| System | Install |
| --- | --- |
| macOS | `brew install ffmpeg-full` (the plain `ffmpeg` formula has no libass), or any static build that lists libass |
| Debian / Ubuntu | `sudo apt install ffmpeg` |
| Windows | a "full" build from gyan.dev or BtbN |

If the FFmpeg you want is not the first one on your `PATH`, point Reelferry at it:

```sh
reelferry settings update --ffmpeg-path /path/to/ffmpeg
```

Without libass you can still render with `--no-burn-captions`: captions are added as a soft subtitle track and written next to the video.

### Speech models

The first transcription downloads Whisper large-v3-turbo (about 1.6 GB) and a small alignment model into the Hugging Face cache.
Forced alignment is available for English, German, French, Spanish, Dutch and Polish; other languages use Whisper's own word timings.
On machines that are too slow for local Whisper, switch to the OpenRouter fallback (about one cent per hour of audio):

```sh
reelferry settings update --transcription-engine openrouter
```

## Configure the OpenRouter key

Clip suggestions use one OpenRouter key.
Create one at [openrouter.ai/keys](https://openrouter.ai/keys), then either store it in your system keychain:

```sh
reelferry config set-key
```

or set it in the environment of the process that runs Reelferry:

```sh
export OPENROUTER_API_KEY=...
```

The key is never written to settings files, logs or command output.
Pick the model per call or as a default:

| Preset | Model | Use |
| --- | --- | --- |
| `quality` (default) | `anthropic/claude-opus-5.5` | best hooks and picks |
| `budget` | `google/gemini-3.8-flash` | about five times cheaper |

```sh
reelferry settings update --suggest-model budget
```

Any other OpenRouter model id works too.

## Connect an agent over MCP

Claude Code:

```sh
claude mcp add reelferry -- uvx --from git+https://github.com/BatuCani/reelferry reelferry mcp
```

Any MCP client that launches stdio servers:

```json
{
  "mcpServers": {
    "reelferry": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/BatuCani/reelferry", "reelferry", "mcp"],
      "env": { "OPENROUTER_API_KEY": "..." }
    }
  }
}
```

Agents on another machine can use MCP over HTTP: run `reelferry serve --host 0.0.0.0` and connect to `http://<host>:7878/mcp` with the header `Authorization: Bearer <token>` (see below).
Only do that on a network you trust; the token is the only lock.

Then just ask, for example: *"Take ~/Videos/podcast.mp4, find the 5 strongest moments about pricing, and render them as 9:16 clips with pop captions."*

### Tools

| Tool | What it does |
| --- | --- |
| `add_source` | Add a local file or a public link (downloaded with yt-dlp) |
| `transcribe` | Local transcription with exact word timings (job) |
| `get_transcript` | Read segments, words with indexes, or plain text, optionally for a time range |
| `edit_transcript` | Fix words by index or replace names everywhere; timings stay intact |
| `suggest_clips` | LLM finds moments for a goal, with hook title, score and reason (job) |
| `create_clip` / `update_clip` | Clip from a suggestion or from start/end; change bounds, format, framing, style |
| `render_clip` | MP4 render, 9:16 or 16:9, crop or blurred fit, captions burned or soft (job) |
| `preview_clip` | Contact sheet of frames, returned as an image so the agent can check framing |
| `list_clips` / `get_clip` | Browse the library with scores, status and file paths |
| `list_caption_styles` / `export_captions` | Six caption styles; export SRT, VTT, ASS or JSON |
| `get_job` / `wait_job` / `cancel_job` / `list_jobs` | Follow and control background jobs |
| `system_check` / `get_settings` / `update_settings` | Health check and settings |

## CLI

Every tool is also a command: `reelferry <area> <verb>`.
Add `--json` for machine-readable output and `--wait` to wait for the job a command starts.

```sh
reelferry source add ~/Videos/talk.mp4 --title "Conference talk" --json
reelferry transcript create src_abc123 --vocabulary Reelferry --wait
reelferry transcript edit src_abc123 --replacements '[{"find": "reel ferry", "replace": "Reelferry"}]'
reelferry clip suggest src_abc123 --goal "the 5 funniest moments" --count 5 --max-seconds 45 --wait
reelferry clip create --candidate-id cand_xyz --caption-style karaoke
reelferry clip render clip_def456 --format 9:16 --wait
reelferry clip preview clip_def456
reelferry caption export --clip-id clip_def456 --format vtt
reelferry job list
```

Any command also accepts its whole input as JSON with `--input '{...}'`, `--input @file.json` or `--input -`.

## Local API

```sh
reelferry serve            # http://127.0.0.1:7878, with MCP at /mcp
reelferry token            # prints the bearer token
```

The OpenAPI description is at `http://127.0.0.1:7878/openapi.json` and interactive docs at `/docs`.
Routes mirror the tools:

```sh
TOKEN=$(reelferry token)
curl -s -X POST http://127.0.0.1:7878/v1/sources \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"uri": "https://www.youtube.com/watch?v=..."}'
curl -s -X POST http://127.0.0.1:7878/v1/sources/src_abc123/transcript \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{}'
curl -s -X POST http://127.0.0.1:7878/v1/jobs/job_abc123/wait \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"timeout": 120}'
curl -s http://127.0.0.1:7878/v1/clips -H "Authorization: Bearer $TOKEN"
```

Errors always look like `{"error": {"code": "not_found", "message": "...", "hint": "..."}}`, with the same codes on every interface.

## Caption styles

| Style | Look |
| --- | --- |
| `pop` | Bold white words; the spoken word turns yellow and pops |
| `karaoke` | Two calm lines; color sweeps through each word as it is spoken |
| `box` | The spoken word sits on a bright pill |
| `bold` | One or two huge words in the middle of the frame |
| `neon` | White words with a soft glow; the spoken word lights up |
| `minimal` | Plain sentence-case subtitles for interviews and talks |

Names and numbers, and the words the LLM marks as the punchline, are highlighted automatically.

## Where data lives

| What | Default | Override |
| --- | --- | --- |
| Database, settings, API token | the user data dir (`~/Library/Application Support/reelferry` on macOS) | `REELFERRY_DATA_DIR` |
| Downloaded sources and rendered clips | `~/Movies/Reelferry` on macOS, `~/Videos/Reelferry` elsewhere | `REELFERRY_MEDIA_DIR` |
| Speech models | the Hugging Face cache | `HF_HOME` |

Local source files are referenced in place, never copied.

## Development

```sh
uv sync
uv run poe check   # format check, lint, type check, tests
```

Tests generate their own synthetic media with FFmpeg.

## License

[Apache-2.0](LICENSE)
