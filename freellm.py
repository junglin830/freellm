#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
freellm — rotate free LLM API keys across providers, with automatic fallback.

Many providers (Groq, NVIDIA NIM, GitHub Models, Google Gemini, OpenRouter free models,
Pollinations...) expose an OpenAI-compatible /chat/completions endpoint with a free tier.
Each free tier has small rate limits. freellm pools every key you have, spreads calls
round-robin, and when one key is rate-limited or down it moves on to the next.

    from freellm import chat
    r = chat([{"role": "user", "content": "Say hi"}])
    print(r["text"], r["provider"], r["key"])

Config: freellm.json (path via FREELLM_CONFIG). Keys may be literal strings or
"env:NAME" to read from an environment variable (recommended).

Pure standard library, Python 3.8+. MIT License.
"""
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request

__version__ = "1.0.0"

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.environ.get("FREELLM_CONFIG", os.path.join(HERE, "freellm.json"))
STATE_DIR = os.environ.get("FREELLM_STATE_DIR", HERE)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) freellm/" + __version__

VISION_HINTS = ("gpt-4o", "gpt-4.1", "vision", "gemini", "llava", "qwen-vl", "qwen2-vl",
                "qwen2.5-vl", "pixtral", "llama-4", "-vl")


class AllProvidersFailed(RuntimeError):
    """Raised when every key that was tried failed. ``errors`` lists each attempt."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__("all providers failed: " + "; ".join(errors[-5:]) if errors else "no providers configured")


def _log(msg):
    # pythonw has sys.stderr = None; never let logging crash the caller.
    try:
        if sys.stderr is not None and not os.environ.get("FREELLM_QUIET"):
            sys.stderr.write("[freellm] " + msg + "\n")
    except Exception:
        pass


def _resolve_key(k):
    if isinstance(k, str) and k.startswith("env:"):
        return os.environ.get(k[4:], "")
    return k or ""


def load_pool(config=None):
    """Flatten the config into a list of {provider, key_idx, key, endpoint, model}."""
    path = config or CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    pool = []
    for name in cfg.get("order") or list(cfg.get("providers", {})):
        p = cfg["providers"].get(name)
        if not p or p.get("enabled", True) is False:
            continue
        for i, k in enumerate(p.get("keys") or [""]):
            key = _resolve_key(k)
            if not key and not p.get("keyless"):
                continue  # missing env var: skip instead of sending an empty bearer token
            pool.append({"provider": name, "key_idx": i + 1, "key": key,
                         "endpoint": p["endpoint"], "model": p["model"],
                         "extra": p.get("extra") or {}})
    return pool


def _state_path(config):
    tag = os.path.splitext(os.path.basename(config or CONFIG))[0]
    return os.path.join(STATE_DIR, "." + tag + ".rr")


def _read_rr(config):
    try:
        with open(_state_path(config)) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def _write_rr(config, n):
    path = _state_path(config)
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".")
        with os.fdopen(fd, "w") as f:
            f.write(str(n))
        os.replace(tmp, path)  # atomic: concurrent callers never read a half-written number
    except Exception:
        pass


def _usage_log(entry, model, data, purpose):
    path = os.environ.get("FREELLM_USAGE_LOG")
    if not path:
        return
    try:
        u = (data.get("usage") or {}) if isinstance(data, dict) else {}
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "provider": entry["provider"], "model": model,
               "key": entry["key_idx"], "purpose": purpose,
               "in_tok": u.get("prompt_tokens"), "out_tok": u.get("completion_tokens")}
        if os.path.exists(path) and os.path.getsize(path) > 2_000_000:
            os.replace(path, path + ".1")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _call(entry, model, messages, max_tokens, temperature, timeout):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if "gpt-oss" in model:
        # gpt-oss reasoning models can spend the whole budget on reasoning and return empty content
        payload["reasoning_effort"] = "low"
    payload.update(entry["extra"])
    headers = {"Content-Type": "application/json", "User-Agent": UA}
    if entry["key"]:
        headers["Authorization"] = "Bearer " + entry["key"]
    req = urllib.request.Request(entry["endpoint"], data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    if not text.strip():
        raise ValueError("empty content")
    return text, data


def _run(pool, messages, model, tries, max_tokens, temperature, timeout, purpose, config, rotate):
    if not pool:
        raise AllProvidersFailed([])
    n = len(pool)
    start = _read_rr(config) % n if rotate else 0
    errors = []
    for j in range(min(tries, n)):
        e = pool[(start + j) % n]
        m = model or e["model"]
        try:
            text, data = _call(e, m, messages, max_tokens, temperature, timeout)
        except urllib.error.HTTPError as he:
            errors.append("%s#%d HTTP %s" % (e["provider"], e["key_idx"], he.code))
        except Exception as ex:
            errors.append("%s#%d %s" % (e["provider"], e["key_idx"], str(ex)[:80]))
        else:
            if rotate:
                _write_rr(config, (start + j + 1) % n)
            _usage_log(e, m, data, purpose)
            _log("%s key#%d (%s) ok" % (e["provider"], e["key_idx"], m))
            return {"text": text, "provider": e["provider"], "key": e["key_idx"], "model": m,
                    "usage": data.get("usage"), "attempts": j + 1}
        _log(errors[-1] + " -> next")
    raise AllProvidersFailed(errors)


def chat(messages, model=None, important=True, max_tokens=1024, temperature=0.6, timeout=60,
         purpose=None, config=None):
    """Send an OpenAI-style chat request through the key pool.

    important=True tries every key before giving up; False tries at most two (cheap calls).
    model overrides the provider's default model (only useful when all providers share names).
    Returns {"text", "provider", "key", "model", "usage", "attempts"}.
    """
    pool = load_pool(config)
    tries = len(pool) if important else 2
    return _run(pool, messages, model, tries, max_tokens, temperature, timeout,
                purpose or os.path.basename(sys.argv[0] or "py"), config, rotate=True)


def is_vision(model):
    m = (model or "").lower()
    return any(v in m for v in VISION_HINTS)


def chat_vision(messages, model=None, max_tokens=512, temperature=0.2, timeout=90, purpose=None, config=None):
    """Like chat() but only uses providers whose model looks image-capable.

    messages use the OpenAI vision format:
    [{"role": "user", "content": [{"type": "text", "text": "..."},
                                  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]
    """
    pool = [e for e in load_pool(config) if is_vision(model or e["model"])]
    return _run(pool, messages, model, len(pool), max_tokens, temperature, timeout,
                purpose or "vision", config, rotate=False)


def check_all(config=None, timeout=30):
    """Ping every key once. Returns a list of (provider, key_idx, model, ok, detail)."""
    out = []
    for e in load_pool(config):
        t0 = time.time()
        try:
            text, _ = _call(e, e["model"], [{"role": "user", "content": "Reply with: ok"}], 128, 0, timeout)
            out.append((e["provider"], e["key_idx"], e["model"], True, "%.1fs %r" % (time.time() - t0, text[:20])))
        except urllib.error.HTTPError as he:
            out.append((e["provider"], e["key_idx"], e["model"], False, "HTTP %s" % he.code))
        except Exception as ex:
            out.append((e["provider"], e["key_idx"], e["model"], False, str(ex)[:60]))
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="freellm", description="Rotate free LLM keys with fallback")
    p.add_argument("prompt", nargs="?", help="prompt to send (reads stdin if '-')")
    p.add_argument("--config")
    p.add_argument("--check", action="store_true", help="ping every configured key")
    p.add_argument("--max-tokens", type=int, default=1024)
    a = p.parse_args(argv)
    if a.check:
        rows = check_all(a.config)
        for prov, idx, model, ok, detail in rows:
            print("%-4s %-14s key#%d %-40s %s" % ("OK" if ok else "FAIL", prov, idx, model, detail))
        return 0 if rows and all(r[3] for r in rows) else 1
    if not a.prompt:
        p.print_help()
        return 2
    prompt = sys.stdin.read() if a.prompt == "-" else a.prompt
    try:
        r = chat([{"role": "user", "content": prompt}], max_tokens=a.max_tokens, config=a.config)
    except AllProvidersFailed as ex:
        print("ERROR: %s" % ex, file=sys.stderr)
        return 1
    print(r["text"])
    _log("answered by %s key#%d (%s), attempt %d" % (r["provider"], r["key"], r["model"], r["attempts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
