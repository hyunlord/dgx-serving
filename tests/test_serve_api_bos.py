#!/usr/bin/env python3
"""CPU-only completion BOS regression checks; no model assets required."""
import sys
import unittest
import json
import threading
import urllib.request
import contextlib
import io
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import serve_api


class Tokenizer:
    bos_token_id = 2
    chat_template = "template"

    def __init__(self, chat_bos=True, raw_bos=False):
        self.chat_bos = chat_bos
        self.raw_bos = raw_bos

    def __call__(self, text):
        prefix = [2] if self.raw_bos or text.startswith("<bos>") else []
        return SimpleNamespace(input_ids=prefix + [10, 11])

    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": ([2] if self.chat_bos else []) + [105, 10, 106]}


class CompletionBosTests(unittest.TestCase):
    def test_chat_template_bos_is_added_to_raw_completion(self):
        self.assertEqual(serve_api.completion_ids(Tokenizer(), "hello"), [2, 10, 11])

    def test_explicit_bos_is_not_duplicated(self):
        self.assertEqual(serve_api.completion_ids(Tokenizer(), "<bos>hello"), [2, 10, 11])

    def test_automatic_tokenizer_bos_is_not_duplicated(self):
        self.assertEqual(serve_api.completion_ids(Tokenizer(raw_bos=True), "hello"), [2, 10, 11])

    def test_bos_property_without_chat_bos_is_unchanged(self):
        self.assertEqual(serve_api.completion_ids(Tokenizer(chat_bos=False), "hello"), [10, 11])

    def test_tokenizer_without_bos_is_unchanged(self):
        tok = Tokenizer(chat_bos=False)
        tok.bos_token_id = None
        self.assertEqual(serve_api.completion_ids(tok, "hello"), [10, 11])

    def test_no_chat_template_keeps_completion_working(self):
        tok = Tokenizer()
        tok.chat_template = None
        self.assertEqual(serve_api.completion_ids(tok, "hello"), [10, 11])


class HttpBosTests(unittest.TestCase):
    def test_http_passes_correct_ids_to_generation(self):
        class CaptureHandler(serve_api.Handler):
            tok = Tokenizer()

            def _generate(self, ids, *args, **kwargs):
                self._json(200, {"ids": ids})

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            cases = [
                ("/v1/completions", {"prompt": "hello"}, [2, 10, 11]),
                ("/v1/completions", {"prompt": "<bos>hello"}, [2, 10, 11]),
                ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}, [2, 105, 10, 106]),
            ]
            for path, body, expected in cases:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_port}{path}",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request) as response:
                    self.assertEqual(json.load(response)["ids"], expected)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


class DaemonLogTests(unittest.TestCase):
    def test_opt_in_forwards_post_ready_stderr_without_blocking_protocol(self):
        child = (
            "import sys\n"
            "print('[online] ready', file=sys.stderr, flush=True)\n"
            "line = sys.stdin.readline()\n"
            "print('INPUT_IDS req=0 count=3 ids=2,10,11', file=sys.stderr, flush=True)\n"
            "print('0 END length', flush=True)\n"
        )
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured):
            daemon = serve_api.OnlineDaemon(
                [sys.executable, "-c", child], env=dict(os.environ, DGX_LOG_INPUT_IDS="1")
            )
            try:
                self.assertEqual(list(daemon.stream([2, 10, 11], 1)), [])
                daemon.p.wait(timeout=5)
                daemon.stderr_reader.join(timeout=5)
                self.assertFalse(daemon.stderr_reader.is_alive())
            finally:
                if daemon.p.poll() is None:
                    daemon.p.terminate()
                daemon.p.wait(timeout=5)
                daemon.reader.join(timeout=5)
                for stream in (daemon.p.stdin, daemon.p.stdout, daemon.p.stderr):
                    stream.close()
        self.assertIn("INPUT_IDS req=0 count=3 ids=2,10,11", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
