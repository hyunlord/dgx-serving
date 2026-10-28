#!/usr/bin/env python3
"""Write R14 prompt token files (int32) for serving-path gates.

Usage: python tools/r14_make_prompts.py <out_dir> [--tokenizer google/gemma-4-E4B]
Tokenizes with serve_api.completion_ids, i.e. exactly the /v1/completions path
(the serve_api default tokenizer; no chat template is required).
Writes short.bin, medium.bin, long.bin (long >= 1000 tokens) and prompts.json
with the message text, token count and SHA-256 of each file. Existing files
are never overwritten.
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    from serve_api import completion_ids  # same tokenization as /v1/completions
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    with open(os.path.join(REPO, "docs", "serving-setup.md"), encoding="utf-8") as f:
        doc = f.read()
    texts = {
        "short": "Hello! Briefly introduce yourself.",
        "medium": "Summarize the following setup notes in five bullet points.\n\n" + doc[:1200],
        "long": "Read the following document carefully, then explain step by step how a new user "
                "should start the API server and what each option controls.\n\n" + doc,
    }
    os.makedirs(a.out, exist_ok=True)
    meta = {}
    for name, text in texts.items():
        ids = completion_ids(tok, text)
        if name == "long":  # shortest whole-line prefix of the document reaching 1000 tokens
            lines, cut = doc.splitlines(keepends=True), 0
            while True:
                cut += 1
                text = texts[name].split("\n\n", 1)[0] + "\n\n" + "".join(lines[:cut])
                ids = completion_ids(tok, text)
                if len(ids) >= 1000 or cut >= len(lines):
                    break
            if len(ids) < 1000:
                sys.exit(f"long prompt has only {len(ids)} tokens; need >= 1000")
        path = os.path.join(a.out, name + ".bin")
        data = b"".join(int(i).to_bytes(4, "little", signed=True) for i in ids)
        with open(path, "xb") as f:
            f.write(data)
        meta[name] = {"tokens": len(ids), "sha256": hashlib.sha256(data).hexdigest(), "text": text}
        print(f"{name}: {len(ids)} tokens")
    with open(os.path.join(a.out, "prompts.json"), "x", encoding="utf-8") as f:
        json.dump({"tokenizer": a.tokenizer, "prompts": meta}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
