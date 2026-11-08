"""End-to-end: `swarm-agent fix` against a fake Ollama HTTP server."""

import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from swarm_agent import cli

BUGGY = "def add(a, b):\n    return a - b\n"

CANNED_REPLY = """Here is the fix:

```diff
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
```
"""

TRACEBACK = """Traceback (most recent call last):
  File "calc.py", line 2, in <module>
    assert add(2, 3) == 5
AssertionError
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": "qwen2.5-coder:7b"}]})
        elif self.path == "/":
            body = b"Ollama is running"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path == "/api/chat":
            self._json({"message": {"role": "assistant", "content": CANNED_REPLY},
                        "done": True})
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture()
def fake_ollama():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    (tmp_path / "calc.py").write_text(BUGGY)
    monkeypatch.setattr(sys, "stdin", io.StringIO(TRACEBACK))
    return tmp_path


def test_fix_dry_run_prints_diff_and_changes_nothing(fake_ollama, repo, capsys):
    rc = cli.main(["--peers", fake_ollama, "fix", "--path", str(repo)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "return a + b" in out
    assert "dry run" in out
    assert (repo / "calc.py").read_text() == BUGGY


def test_fix_apply_writes_and_tests_pass(fake_ollama, repo, capsys):
    check = f'{sys.executable} -c "from calc import add; assert add(2, 3) == 5"'
    rc = cli.main([
        "--peers", fake_ollama, "fix", "--path", str(repo),
        "--apply", "--test-cmd", check,
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tests pass" in out
    assert (repo / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_swarm_lists_peer(fake_ollama, capsys):
    rc = cli.main(["--peers", fake_ollama, "swarm"])
    out = capsys.readouterr().out
    assert rc == 0
    assert fake_ollama in out
    assert "qwen2.5-coder:7b" in out


def test_no_peer_friendly_error(tmp_path, capsys):
    rc = cli.main(["--peers", "http://127.0.0.1:1", "swarm"])
    err = capsys.readouterr()
    assert rc == 1
    assert "ollama serve" in (err.out + err.err)
