# freellm

**Pool every free LLM API key you have, rotate between them, and fall back automatically when one is rate-limited.**
把手上所有免費 LLM 金鑰集中輪流用，哪一把被限流或掛掉就自動換下一把。

Groq, NVIDIA NIM, Google Gemini, OpenRouter `:free` models, GitHub Models, Pollinations… all offer an
OpenAI-compatible `/chat/completions` endpoint with a free tier — and each free tier has tight per-minute limits.
For background jobs (summaries, tagging, classification, chat bots) you can stay at **$0** by spreading calls across
all of them. freellm does that in one file with no dependencies.

We have run this in production for months behind a LINE bot and several background workers.

## Features

- **Round-robin** across every key of every provider (state kept in a tiny file, written atomically).
- **Fallback** on HTTP 429 / 5xx / timeouts / **empty replies** (reasoning models sometimes return empty content).
- `important=False` → try at most 2 keys for cheap, skippable calls.
- `chat_vision()` → only routes to image-capable models.
- Keys as `"env:VAR_NAME"`, so secrets never sit in the config file. Missing env vars are skipped.
- `--check` pings every key and shows which ones still work.
- Optional JSONL usage log (`FREELLM_USAGE_LOG`), safe under `pythonw` (no stderr crash).
- Standard library only, Python 3.8+.

## Quick start

```bash
cp freellm.example.json freellm.json      # edit providers / order
export GROQ_API_KEY=...  NVIDIA_API_KEY=...
python freellm.py --check                  # which keys work?
python freellm.py "Summarise this in one line: ..."
```

```python
from freellm import chat, AllProvidersFailed
try:
    r = chat([{"role": "user", "content": "Say hi"}], max_tokens=200)
    print(r["text"], "<-", r["provider"], "key", r["key"], "attempts", r["attempts"])
except AllProvidersFailed as e:
    print("everything is down:", e.errors)
```

Example `--check` output:

```
OK   groq           key#1 openai/gpt-oss-120b                      0.6s 'ok'
OK   groq           key#2 openai/gpt-oss-120b                      0.6s 'ok'
OK   nvidia         key#1 nvidia/nemotron-3-ultra-550b-a55b        0.5s 'ok'
OK   pollinations   key#1 openai                                   3.5s 'ok'
```

## Config

```json
{
  "order": ["groq", "nvidia"],
  "providers": {
    "groq":   {"endpoint": "https://api.groq.com/openai/v1/chat/completions",
               "model": "openai/gpt-oss-120b", "keys": ["env:GROQ_API_KEY", "env:GROQ_API_KEY_2"]},
    "nvidia": {"endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
               "model": "nvidia/nemotron-3-ultra-550b-a55b", "keys": ["env:NVIDIA_API_KEY"],
               "enabled": true, "extra": {"top_p": 0.9}}
  }
}
```

| Field | Meaning |
|---|---|
| `order` | provider order for the rotation |
| `keys` | literal keys or `env:NAME` |
| `enabled` | `false` to park a provider without deleting it |
| `extra` | extra JSON fields merged into every request |
| `keyless` | `true` for endpoints that need no key |

Env: `FREELLM_CONFIG` (config path), `FREELLM_STATE_DIR` (where the rotation file lives),
`FREELLM_USAGE_LOG` (JSONL path), `FREELLM_QUIET=1` (no stderr logs).

Free tiers, model names and terms change often — check each provider's docs and respect their terms of use.

## Tests

```bash
python tests/test_freellm.py   # offline: spins up a fake OpenAI-compatible server
```

## License

MIT
