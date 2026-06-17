"""Browser-side tokenizer helpers, executed inside Pyodide.

OpenAI's real `tiktoken` library runs here in WebAssembly. The BPE vocab files
are bundled in the repo and injected into tiktoken's on-disk cache, so
`get_encoding()` reads them locally instead of downloading from OpenAI's blob
storage (which blocks browser CORS requests)."""

import os
import json
import hashlib

import tiktoken

# tiktoken looks in this dir before hitting the network. Pyodide gives us an
# in-memory virtual filesystem, so /tmp is fine.
_CACHE_DIR = "/tmp/tiktoken_cache"
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = _CACHE_DIR

# The exact blob URLs tiktoken uses internally. The cache filename is the
# sha1 of the URL, so these must match openai_public.py verbatim.
_BLOB_URLS = {
    "o200k_base": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "p50k_base": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    "r50k_base": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
}

_encoders = {}


def install_vocab(name, data):
    """Write a bundled .tiktoken file into the cache under the name tiktoken expects."""
    url = _BLOB_URLS[name]
    key = hashlib.sha1(url.encode()).hexdigest()
    path = os.path.join(_CACHE_DIR, key)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(bytes(data))
    return True


def _encoder(name):
    enc = _encoders.get(name)
    if enc is None:
        enc = tiktoken.get_encoding(name)
        _encoders[name] = enc
    return enc


def tokenize(name, text):
    """Return token count, ids, and the text piece for each token as a JSON string."""
    enc = _encoder(name)
    # disallowed_special=() means strings like "<|endoftext|>" are tokenized as
    # ordinary text rather than raising — we just want to count arbitrary input.
    ids = enc.encode(text, disallowed_special=())
    pieces = [enc.decode_single_token_bytes(i).decode("utf-8", "replace") for i in ids]
    return json.dumps({"count": len(ids), "ids": ids, "pieces": pieces})
