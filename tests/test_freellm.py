"""Offline tests: a local fake OpenAI-compatible server, no network or real keys needed."""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ["FREELLM_QUIET"] = "1"
import freellm  # noqa: E402

CALLS = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        mode = self.path.strip("/")
        CALLS.append((mode, self.headers.get("Authorization"), body))
        if mode == "429":
            self.send_response(429); self.end_headers(); return
        if mode == "500":
            self.send_response(500); self.end_headers(); return
        text = "" if mode == "empty" else "hello from " + mode
        out = json.dumps({"choices": [{"message": {"content": text}}],
                          "usage": {"prompt_tokens": 3, "completion_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


srv = HTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = "http://127.0.0.1:%d/" % srv.server_port


def cfg(providers, order=None):
    d = tempfile.mkdtemp()
    os.environ["FREELLM_STATE_DIR"] = d
    freellm.STATE_DIR = d
    p = os.path.join(d, "c.json")
    json.dump({"order": order or list(providers), "providers": providers}, open(p, "w"))
    return p


def prov(mode, model="m", keys=("k",), **kw):
    return dict(endpoint=BASE + mode, model=model, keys=list(keys), **kw)


def test_fallback_on_429_and_500():
    c = cfg({"a": prov("429"), "b": prov("500"), "c": prov("ok")})
    r = freellm.chat([{"role": "user", "content": "hi"}], config=c)
    assert r["provider"] == "c" and r["attempts"] == 3 and r["text"] == "hello from ok"


def test_round_robin_spreads_keys():
    c = cfg({"a": prov("okA", keys=["k1", "k2"]), "b": prov("okB")})
    seen = [(r["provider"], r["key"]) for r in
            (freellm.chat([{"role": "user", "content": "x"}], config=c) for _ in range(4))]
    assert seen == [("a", 1), ("a", 2), ("b", 1), ("a", 1)], seen


def test_empty_content_counts_as_failure():
    c = cfg({"a": prov("empty"), "b": prov("ok")})
    assert freellm.chat([{"role": "user", "content": "x"}], config=c)["provider"] == "b"


def test_model_override_is_reported():
    c = cfg({"a": prov("ok", model="default")})
    r = freellm.chat([{"role": "user", "content": "x"}], model="override", config=c)
    assert r["model"] == "override" and CALLS[-1][2]["model"] == "override"


def test_not_important_tries_two():
    c = cfg({"a": prov("429"), "b": prov("429"), "c": prov("ok")})
    try:
        freellm.chat([{"role": "user", "content": "x"}], important=False, config=c)
        assert False, "should fail after 2 tries"
    except freellm.AllProvidersFailed as e:
        assert len(e.errors) == 2


def test_env_keys_and_missing_env_skipped():
    os.environ["FREELLM_TEST_KEY"] = "secret123"
    c = cfg({"a": prov("ok", keys=["env:FREELLM_NOPE", "env:FREELLM_TEST_KEY"])})
    pool = freellm.load_pool(c)
    assert len(pool) == 1 and pool[0]["key"] == "secret123"
    freellm.chat([{"role": "user", "content": "x"}], config=c)
    assert CALLS[-1][1] == "Bearer secret123"


def test_disabled_and_empty_pool():
    c = cfg({"a": prov("ok", enabled=False)})
    try:
        freellm.chat([{"role": "user", "content": "x"}], config=c)
        assert False
    except freellm.AllProvidersFailed:
        pass  # old version crashed here with ZeroDivisionError


def test_vision_only_uses_vision_models():
    c = cfg({"text": prov("okText", model="llama-3.3-70b"), "vis": prov("okVis", model="gpt-4o-mini")})
    r = freellm.chat_vision([{"role": "user", "content": "x"}], config=c)
    assert r["provider"] == "vis"


def test_gpt_oss_reasoning_low():
    c = cfg({"a": prov("ok", model="openai/gpt-oss-120b")})
    freellm.chat([{"role": "user", "content": "x"}], config=c)
    assert CALLS[-1][2]["reasoning_effort"] == "low"


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); n += 1; print("OK", name)
    print("%d passed" % n)
